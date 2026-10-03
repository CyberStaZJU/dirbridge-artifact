import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from models.local_update import LocalSGD
from utils.state_dict_ops import load_param_dict_, model_param_dict, sd_copy, sd_sub, sd_zero_like


def _getattr(args, name, default):
    if not hasattr(args, name):
        return default
    value = getattr(args, name)
    return default if value is None else value


def _ablation_value(args) -> str:
    value = str(_getattr(args, 'dirbridge_ablation', 'none'))
    return value.lower().replace('-', '_')


def _ablation_enabled(args, name: str) -> bool:
    name = name.lower().replace('-', '_')
    flag_names = {
        'no_ema_cache': 'dirbridge_disable_ema_cache',
        'random_grouping': 'dirbridge_random_grouping',
        'no_staleness_filter': 'dirbridge_disable_staleness_filter',
    }
    return _ablation_value(args) == name or bool(getattr(args, flag_names.get(name, ''), False))


def _float_tensor_items(sd: Dict[str, torch.Tensor]):
    return [
        (key, value)
        for key, value in sd.items()
        if torch.is_tensor(value) and torch.is_floating_point(value)
    ]


def build_count_sketch_plan(ref_sd: Dict[str, torch.Tensor], feature_dim: int, seed: int):
    feature_dim = max(1, int(feature_dim))
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed) % (2**63 - 1))

    plan = []
    for key, value in _float_tensor_items(ref_sd):
        size = int(value.numel())
        if size <= 0:
            continue
        buckets = torch.randint(
            low=0,
            high=feature_dim,
            size=(size,),
            generator=generator,
            dtype=torch.long,
        )
        signs = torch.randint(
            low=0,
            high=2,
            size=(size,),
            generator=generator,
            dtype=torch.int8,
        )
        signs = signs.mul_(2).sub_(1)
        plan.append((key, buckets, signs))
    return plan


def encode_delta_raw_count_sketch(
    delta: Dict[str, torch.Tensor],
    count_sketch_plan,
    feature_dim: int,
) -> torch.Tensor:
    if not count_sketch_plan:
        return torch.ones(1, dtype=torch.float32)

    feature = torch.zeros(max(1, int(feature_dim)), dtype=torch.float32)
    for key, buckets, signs in count_sketch_plan:
        flat = delta[key].detach().float().reshape(-1).cpu()
        signed_values = flat * signs.to(dtype=torch.float32)
        feature.index_add_(0, buckets, signed_values)
    return feature


def _normalize_feature(feature: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(feature, p=2)
    if norm > 0:
        return feature / norm
    return feature


def encode_client_feature(state, args, idx: int) -> torch.Tensor:
    raw_delta = encode_delta_raw_count_sketch(
        state['delta'][idx],
        state['count_sketch_plan'],
        state['group_proj_dim'],
    )
    return _normalize_feature(raw_delta)


def _normalize_rows(X: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(X, p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return X / norms


def _init_spherical_centroids(
    X: torch.Tensor,
    num_clusters: int,
    init_centroids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_points = X.shape[0]
    centroids = []

    if (
        init_centroids is not None
        and init_centroids.ndim == 2
        and init_centroids.shape[1] == X.shape[1]
        and init_centroids.shape[0] > 0
    ):
        seeded = _normalize_rows(init_centroids.detach().float().cpu())
        take = min(int(seeded.shape[0]), num_clusters)
        centroids.extend(seeded[:take])

    if len(centroids) >= num_clusters:
        return torch.stack(centroids[:num_clusters], dim=0)

    if centroids:
        existing = torch.stack(centroids, dim=0)
        sims = X @ existing.t()
        candidate_order = torch.argsort(sims.max(dim=1).values, descending=False)
    else:
        candidate_order = torch.randperm(num_points)

    for point_idx in candidate_order.tolist():
        centroids.append(X[point_idx].clone())
        if len(centroids) >= num_clusters:
            break

    if len(centroids) < num_clusters:
        fallback_idx = np.random.choice(num_points, num_clusters - len(centroids), replace=True)
        for point_idx in fallback_idx.tolist():
            centroids.append(X[int(point_idx)].clone())

    return torch.stack(centroids[:num_clusters], dim=0)


def spherical_kmeans(
    X: torch.Tensor,
    num_clusters: int,
    num_iters: int = 20,
    init_centroids: Optional[torch.Tensor] = None,
) -> Tuple[List[int], torch.Tensor]:
    num_points, dim = X.shape
    num_clusters = min(max(1, int(num_clusters)), num_points)

    centroids = _init_spherical_centroids(X, num_clusters, init_centroids=init_centroids)

    for _ in range(num_iters):
        sims = X @ centroids.t()
        assign = torch.argmax(sims, dim=1)

        new_centroids = []
        for cluster_id in range(num_clusters):
            mask = assign == cluster_id
            if mask.sum().item() == 0:
                ridx = np.random.randint(0, num_points)
                centroid = X[ridx].clone()
            else:
                centroid = X[mask].mean(dim=0)
                centroid = centroid / (torch.norm(centroid, p=2) + 1e-12)
            new_centroids.append(centroid)
        centroids = torch.stack(new_centroids, dim=0)

    sims = X @ centroids.t()
    assign = torch.argmax(sims, dim=1)
    return assign.cpu().tolist(), centroids.cpu()


def _recompute_group_centroids(
    X: torch.Tensor,
    group_ids: List[int],
    num_groups: int,
    fallback_centroids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    centroids = []
    for group_id in range(num_groups):
        member_positions = [
            idx for idx, assigned_group in enumerate(group_ids)
            if int(assigned_group) == group_id
        ]

        if member_positions:
            centroid = X[member_positions].mean(dim=0)
            centroid = centroid / (torch.norm(centroid, p=2) + 1e-12)
        elif fallback_centroids is not None and group_id < int(fallback_centroids.shape[0]):
            centroid = fallback_centroids[group_id].detach().float().cpu().clone()
        else:
            centroid = X[np.random.randint(0, int(X.shape[0]))].clone()

        centroids.append(centroid)

    return torch.stack(centroids, dim=0).cpu()


def random_grouping(X: torch.Tensor, num_groups: int) -> Tuple[List[int], torch.Tensor]:
    num_points = int(X.shape[0])
    num_groups = min(max(1, int(num_groups)), num_points)
    perm = torch.randperm(num_points).tolist()
    group_ids = [0] * num_points

    for rank, point_idx in enumerate(perm):
        group_ids[int(point_idx)] = rank % num_groups

    centroids = _recompute_group_centroids(X, group_ids, num_groups)
    return group_ids, centroids


def grouped_buffered_aggregation(selected_items: List[dict], w_glob: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    aggregated_diff = sd_zero_like(w_glob)
    for item in selected_items:
        weight = item['weight']
        delta = item['delta']
        for key in aggregated_diff.keys():
            aggregated_diff[key] += weight * delta[key]
    return aggregated_diff


def mean_state_dict(dicts: List[Dict[str, torch.Tensor]], ref_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    avg = sd_zero_like(ref_model)
    if not dicts:
        return avg

    scale = 1.0 / float(len(dicts))
    for delta in dicts:
        for key in avg.keys():
            avg[key] += scale * delta[key]
    return avg


def blend_state_dict_by_count(
    base: Dict[str, torch.Tensor],
    base_count: float,
    cache: Dict[str, torch.Tensor],
    cache_count: float,
) -> Dict[str, torch.Tensor]:
    total_count = float(base_count) + float(cache_count)
    if total_count <= 0:
        return sd_copy(base)

    base_weight = float(base_count) / total_count
    cache_weight = float(cache_count) / total_count
    out = {}
    for key in base.keys():
        out[key] = base_weight * base[key] + cache_weight * cache[key]
    return out


def local_train_delta(state, args, idx: int):
    net = copy.deepcopy(state['net_glob']).to(args.device)
    w_start = model_param_dict(state['net_glob'], device=args.device)
    local = LocalSGD(
        args=args,
        dataset=state['dataset_train'],
        idxs=state['dict_users'][idx],
        iters=args.local_period,
        nums=state['num_samples'][idx],
    )
    w_local = local.train(net=net)
    delta = sd_sub(w_local, w_start)
    del w_start, w_local, net
    return delta


def client_delay(state, idx: int) -> float:
    stamp = state['stamp'][idx]
    if stamp < 0:
        return float('inf')
    return max(0.0, float(state['iterations'] - stamp))


def _buffer_delay_limit(args) -> float:
    return float(args.concurrency) / float(max(1, int(args.buffer_size)))


def sync_client_delays(state):
    for idx in range(len(state['delays'])):
        state['delays'][idx] = client_delay(state, idx)


def should_rebuild_groups(state, args) -> bool:
    if state['iterations'] <= 0:
        return False

    if _ablation_enabled(args, 'random_grouping'):
        return False

    interval = max(1, int(_getattr(
        args,
        'dirbridge_recluster_interval',
        1,
    )))

    last_rebuild = int(state.get('last_group_rebuild', 0))
    rounds_since_rebuild = int(state['iterations']) - last_rebuild

    if rounds_since_rebuild >= interval:
        state['last_group_rebuild_reason'] = 'interval'
        return True

    return False


def rebuild_direction_groups(state, args):
    num_users = int(args.num_users)
    target_groups = max(1, int(_getattr(args, 'dirbridge_num_groups', 5)))
    kmeans_iters = int(_getattr(args, 'dirbridge_kmeans_iters', 20))

    features = []
    for idx in range(num_users):
        feature = state['client_features'][idx]
        if feature is None:
            feature = encode_client_feature(
                state,
                args,
                idx,
            )
            state['client_features'][idx] = feature
        features.append(feature)
    Z = torch.stack(features, dim=0)

    state['group_proj_dim'] = int(Z.shape[1]) if Z.ndim == 2 else 1
    state['group_kmeans_iters'] = kmeans_iters

    init_centroids = state.get('group_centroids')
    if _ablation_enabled(args, 'random_grouping'):
        group_ids, group_centroids = random_grouping(Z, target_groups)
        state['last_grouping_method'] = 'random'
    else:
        group_ids, group_centroids = spherical_kmeans(
            Z,
            target_groups,
            num_iters=kmeans_iters,
            init_centroids=init_centroids,
        )
        state['last_grouping_method'] = 'spherical_kmeans'

    actual_num_groups = int(group_centroids.shape[0])

    state['num_groups'] = actual_num_groups
    state['group_ids'] = group_ids
    state['group_centroids'] = group_centroids
    state['client_embeds'] = [feature.clone() for feature in features]
    state['group_members'] = [[] for _ in range(actual_num_groups)]
    state['group_counts'] = [0] * actual_num_groups

    for idx, gid in enumerate(group_ids):
        state['group_members'][gid].append(idx)
        state['group_counts'][gid] += 1

    state['last_group_rebuild'] = int(state['iterations'])


def _refresh_ema_cache_delays(state):
    num_groups = int(state['num_groups'])
    for group_id in range(num_groups):
        stamp = state['group_cache_stamp'][group_id]
        if stamp < 0:
            state['group_cache_delay'][group_id] = float('inf')
        else:
            state['group_cache_delay'][group_id] = max(0.0, float(state['iterations'] - stamp))


def _init_empty_ema_group_cache(state):
    num_groups = int(state['num_groups'])
    state['group_cache'] = [sd_zero_like(state['w_glob']) for _ in range(num_groups)]
    state['group_cache_stamp'] = [-1] * num_groups
    state['group_cache_delay'] = [float('inf')] * num_groups
    _refresh_ema_cache_delays(state)


def _capture_ema_group_cache(state):
    group_ids = list(state.get('group_ids', []))
    caches = state.get('group_cache')
    stamps = list(state.get('group_cache_stamp', []))

    if not group_ids or caches is None or len(caches) != len(stamps):
        return None

    copied_caches = [
        sd_copy(cache) if int(stamp) >= 0 else None
        for cache, stamp in zip(caches, stamps)
    ]
    return {
        'group_ids': group_ids,
        'group_cache': copied_caches,
        'group_cache_stamp': stamps,
    }


def _remap_ema_group_cache_from_previous(state, previous):
    _init_empty_ema_group_cache(state)
    if previous is None:
        return

    old_group_ids = previous['group_ids']
    old_caches = previous['group_cache']
    old_stamps = previous['group_cache_stamp']
    num_old_groups = len(old_caches)

    for new_group_id, members in enumerate(state['group_members']):
        overlap_counts = {}
        for idx in members:
            if idx < 0 or idx >= len(old_group_ids):
                continue
            old_group_id = int(old_group_ids[idx])
            if old_group_id < 0 or old_group_id >= num_old_groups:
                continue
            if old_caches[old_group_id] is None or int(old_stamps[old_group_id]) < 0:
                continue
            overlap_counts[old_group_id] = overlap_counts.get(old_group_id, 0) + 1

        total_overlap = sum(overlap_counts.values())
        if total_overlap <= 0:
            continue

        remapped_cache = sd_zero_like(state['w_glob'])
        for old_group_id, count in overlap_counts.items():
            weight = float(count) / float(total_overlap)
            old_cache = old_caches[old_group_id]
            for key in remapped_cache.keys():
                remapped_cache[key] += weight * old_cache[key]

        state['group_cache'][new_group_id] = remapped_cache
        state['group_cache_stamp'][new_group_id] = min(
            int(old_stamps[old_group_id])
            for old_group_id in overlap_counts
        )

    _refresh_ema_cache_delays(state)


def _build_ema_group_updates(state, args, buffer_list):
    delay_limit = float('inf') if _ablation_enabled(args, 'no_staleness_filter') else _buffer_delay_limit(args)
    use_ema_cache = not _ablation_enabled(args, 'no_ema_cache')
    valid_buffer_list = []
    invalid_buffer_list = []

    for idx in buffer_list:
        if client_delay(state, idx) > delay_limit:
            invalid_buffer_list.append(idx)
        else:
            valid_buffer_list.append(idx)

    state['last_buffer_delay_limit'] = delay_limit
    state['last_valid_buffer_list'] = list(valid_buffer_list)
    state['last_invalid_buffer_list'] = list(invalid_buffer_list)
    state['last_valid_buffer_count'] = len(valid_buffer_list)
    state['last_invalid_buffer_count'] = len(invalid_buffer_list)

    group_to_members = {}
    for idx in valid_buffer_list:
        group_id = state['group_ids'][idx]
        group_to_members.setdefault(group_id, []).append(idx)

    selected_items = []
    num_groups = int(state['num_groups'])
    group_counts = list(state.get('group_counts', [1] * num_groups))
    total_group_count = float(sum(max(0, int(count)) for count in group_counts))
    if total_group_count <= 0.0:
        total_group_count = float(max(1, num_groups))
        group_counts = [1] * num_groups

    for group_id in range(num_groups):
        group_size = max(0, int(group_counts[group_id])) if group_id < len(group_counts) else 0
        group_weight = float(group_size) / total_group_count
        if group_weight <= 0.0:
            continue

        target_buffer_count = float(len(valid_buffer_list)) * group_weight
        member_ids = group_to_members.get(group_id, [])
        cache_ready = use_ema_cache and state['group_cache_stamp'][group_id] >= 0
        member_count = len(member_ids)

        base_delta = None
        if member_count > 0:
            base_delta = mean_state_dict(
                [state['delta'][idx] for idx in member_ids],
                state['w_glob'],
            )
            group_delta = base_delta
            cache_fill_count = 0.0

            if cache_ready and float(member_count) < target_buffer_count:
                cache_fill_count = target_buffer_count - float(member_count)
                group_delta = blend_state_dict_by_count(
                    base_delta,
                    float(member_count),
                    state['group_cache'][group_id],
                    cache_fill_count,
                )
        elif cache_ready and target_buffer_count > 0:
            group_delta = sd_copy(state['group_cache'][group_id])
            cache_fill_count = target_buffer_count
        else:
            continue

        selected_items.append({
            'kind': 'group',
            'group': group_id,
            'delta': group_delta,
            'base_delta': base_delta,
            'delay': (
                float(np.mean([client_delay(state, idx) for idx in member_ids]))
                if member_ids
                else state['group_cache_delay'][group_id]
            ),
            'member_count': member_count,
            'cache_fill_count': cache_fill_count,
            'target_buffer_count': target_buffer_count,
            'valid_buffer_count': len(valid_buffer_list),
            'invalid_buffer_count': len(invalid_buffer_list),
            'delay_limit': delay_limit,
            'group_size': group_size,
            'weight': group_weight,
        })

    return selected_items


def _update_ema_group_cache(state, args, selected_items):
    beta = float(_getattr(args, 'dirbridge_cache_beta', 0.9))
    beta = float(min(max(beta, 0.0), 0.999))

    for item in selected_items:
        if item['kind'] != 'group':
            continue

        group_id = int(item['group'])
        observed_delta = item.get('base_delta')
        if observed_delta is None:
            continue

        if state['group_cache_stamp'][group_id] < 0:
            state['group_cache'][group_id] = sd_copy(observed_delta)
        else:
            prev_cache = state['group_cache'][group_id]
            next_cache = sd_zero_like(state['w_glob'])
            for key in next_cache.keys():
                next_cache[key] = beta * prev_cache[key] + (1.0 - beta) * observed_delta[key]
            state['group_cache'][group_id] = next_cache

        state['group_cache_stamp'][group_id] = int(state['iterations'])
        state['group_cache_delay'][group_id] = 0.0


def _schedule_new_clients(state, args, active_list, num_to_launch, random_cost):
    sampled_idx = [idx for idx in range(args.num_users) if idx not in active_list]
    np.random.shuffle(sampled_idx)
    sampled_idx = sampled_idx[:num_to_launch]

    for idx in sampled_idx:
        delta = local_train_delta(state, args, idx)
        state['delta'][idx] = delta
        state['client_features'][idx] = encode_client_feature(
            state,
            args,
            idx,
        )
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = state['global_cost'] + random_cost(idx)


def init_state(state, args, random_cost):
    num_users = int(args.num_users)

    state['delta'] = [sd_zero_like(state['w_glob']) for _ in range(num_users)]
    state['stamp'] = [0] * num_users
    state['cost'] = [-1] * num_users
    state['delays'] = [0.0] * num_users
    state['global_cost'] = 0
    state['iterations'] = 0

    state['num_groups'] = max(1, int(_getattr(args, 'dirbridge_num_groups', 5)))
    state['group_ids'] = [-1] * num_users
    state['group_proj_dim'] = int(_getattr(args, 'dirbridge_sketch_dim', 2048))
    state['group_kmeans_iters'] = int(_getattr(args, 'dirbridge_kmeans_iters', 20))
    feature_mode = str(_getattr(args, 'dirbridge_feature_mode', 'count_sketch')).lower().replace('-', '_')
    if feature_mode not in {'count_sketch', 'countsketch'}:
        raise ValueError("DirBridge now supports only --dirbridge_feature_mode count_sketch")
    count_sketch_seed = _getattr(
        args,
        'dirbridge_count_sketch_seed',
        int(getattr(args, 'seed', 0)) + 9176,
    )
    state['count_sketch_plan'] = build_count_sketch_plan(
        state['w_glob'],
        state['group_proj_dim'],
        count_sketch_seed,
    )
    state['count_sketch_seed'] = int(count_sketch_seed)
    state['group_centroids'] = None
    state['group_cache'] = [sd_zero_like(state['w_glob']) for _ in range(state['num_groups'])]
    state['group_cache_stamp'] = [-1] * state['num_groups']
    state['group_cache_delay'] = [float('inf')] * state['num_groups']
    state['group_members'] = [[] for _ in range(state['num_groups'])]
    state['group_counts'] = [0] * state['num_groups']
    state['client_features'] = [None] * num_users
    state['client_embeds'] = [None] * num_users
    state['last_group_rebuild'] = 0
    state['last_group_rebuild_reason'] = 'init'

    for idx in range(num_users):
        delta = local_train_delta(state, args, idx)
        state['delta'][idx] = delta
        state['client_features'][idx] = encode_client_feature(
            state,
            args,
            idx,
        )

    sync_client_delays(state)
    state['recluster_count'] = 0
    state['last_recluster_time_sec'] = 0.0
    state['total_recluster_time_sec'] = 0.0
    state['max_recluster_time_sec'] = 0.0
    start_time = time.perf_counter()
    rebuild_direction_groups(state, args)
    elapsed = time.perf_counter() - start_time
    state['recluster_count'] = 1
    state['last_recluster_time_sec'] = elapsed
    state['total_recluster_time_sec'] = elapsed
    state['max_recluster_time_sec'] = elapsed
    _init_empty_ema_group_cache(state)

    sampled_idx = random.sample(range(num_users), min(int(args.concurrency), num_users))
    for idx in sampled_idx:
        state['cost'][idx] = random_cost(idx)

    return state


def run_round(state, args, random_cost):
    active_list = [idx for idx, cost in enumerate(state['cost']) if cost > 0]

    buffer_list = []
    for _ in range(args.buffer_size):
        filtered = [
            (idx, cost)
            for idx, cost in enumerate(state['cost'])
            if idx not in buffer_list and cost > 0
        ]
        if not filtered:
            break
        idx, _ = min(filtered, key=lambda item: item[1])
        buffer_list.append(idx)

    if not buffer_list:
        return state

    state['global_cost'] = state['cost'][buffer_list[-1]]
    for idx in buffer_list:
        state['cost'][idx] = -1
    state['last_buffer_list'] = list(buffer_list)
    state['last_selected_item_summary'] = []
    state['last_regrouped'] = False

    sync_client_delays(state)

    regrouped = False
    state['last_recluster_time_sec'] = 0.0
    if should_rebuild_groups(state, args):
        previous_cache = _capture_ema_group_cache(state)
        start_time = time.perf_counter()
        rebuild_direction_groups(state, args)
        _remap_ema_group_cache_from_previous(state, previous_cache)
        elapsed = time.perf_counter() - start_time
        state['last_recluster_time_sec'] = elapsed
        state['total_recluster_time_sec'] = float(state.get('total_recluster_time_sec', 0.0)) + elapsed
        state['max_recluster_time_sec'] = max(float(state.get('max_recluster_time_sec', 0.0)), elapsed)
        state['recluster_count'] = int(state.get('recluster_count', 0)) + 1
        regrouped = True
    elif (
        'group_cache' not in state
        or len(state['group_cache']) != int(state['num_groups'])
        or len(state['group_cache_stamp']) != int(state['num_groups'])
    ):
        _init_empty_ema_group_cache(state)
    state['last_regrouped'] = bool(regrouped)

    _refresh_ema_cache_delays(state)

    selected_items = _build_ema_group_updates(state, args, buffer_list)
    state['last_selected_item_summary'] = [
        {
            'group': int(item.get('group', -1)),
            'member_count': int(item.get('member_count', 0)),
            'cache_fill_count': float(item.get('cache_fill_count', 0.0)),
            'target_buffer_count': float(item.get('target_buffer_count', 0.0)),
        }
        for item in selected_items
        if item.get('kind') == 'group'
    ]
    if not selected_items:
        state['iterations'] += 1
        sync_client_delays(state)
        _refresh_ema_cache_delays(state)
        _schedule_new_clients(state, args, active_list, len(buffer_list), random_cost)
        return state

    aggregated_diff = grouped_buffered_aggregation(selected_items, state['w_glob'])
    with torch.no_grad():
        for key in aggregated_diff.keys():
            state['w_glob'][key] += aggregated_diff[key]
    load_param_dict_(state['net_glob'], state['w_glob'])

    _update_ema_group_cache(state, args, selected_items)

    state['iterations'] += 1
    sync_client_delays(state)
    _refresh_ema_cache_delays(state)
    _schedule_new_clients(state, args, active_list, len(buffer_list), random_cost)

    if regrouped:
        _refresh_ema_cache_delays(state)

    return state
