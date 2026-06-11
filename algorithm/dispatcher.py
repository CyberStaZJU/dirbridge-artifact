import numpy as np

from utils.aggregation import (
    DELAY_CATEGORIES,
    build_client_delay_profile,
    normalize_delay_profile,
    random_cost,
    sample_client_delay,
)
from utils.fedscale_trace import ensure_fedscale_trace_sampler
from algorithm import ca2fl, casa, dirbridge, fadas, fedasmu, fedbuff, fedbuffma

DIRSKEW_COSTS = {
    'dir-skew',
}
FEDSCALE_TRACE_COSTS = {
    'fedscale_trace',
}
DIRSKEW_SPEED_CATEGORIES = DELAY_CATEGORIES


SUPPORTED_ALGORITHMS = {
    'FedBuff',
    'CA2FL',
    'FedBuffMA',
    'FedBuffMALight',
    'CASA',
    'FADAS',
    'FedASMU',
    'DirBridge',
}


def _is_dirskew(args):
    return normalize_delay_profile(getattr(args, 'random_cost', 'dir-skew')) in DIRSKEW_COSTS




def _is_fedscale_trace(args):
    return normalize_delay_profile(getattr(args, 'random_cost', 'dir-skew')) in FEDSCALE_TRACE_COSTS


def _build_dirskew_delay_profile(state, args, seed):
    if state is None:
        raise ValueError("dir-skew delay requires dataset state")

    group_ids = getattr(args, 'label_correlated_group_ids', None)
    num_blocks = int(getattr(args, 'label_correlated_num_groups', 0) or 0)

    if group_ids is None:
        raise ValueError(
            "dir-skew delay requires label-derived client groups; "
            "use --distribution noniid or --distribution label_correlated"
        )

    group_ids = np.asarray(group_ids, dtype=np.int64)
    num_users = int(args.num_users)
    if group_ids.size != num_users:
        raise ValueError(
            f"dir-skew expected {num_users} client group ids, found {group_ids.size}"
        )

    if num_blocks <= 0:
        num_blocks = int(group_ids.max()) + 1
    speed_categories = DIRSKEW_SPEED_CATEGORIES
    if num_blocks != len(speed_categories):
        raise ValueError(
            f"dir-skew currently expects exactly {len(speed_categories)} label groups; "
            f"found {num_blocks}"
        )

    rng = np.random.default_rng(int(seed) + 104729)
    speed_order = rng.permutation(num_blocks)
    block_to_speed = {
        int(block_id): speed_categories[rank]
        for rank, block_id in enumerate(speed_order)
    }
    delay_groups = [block_to_speed[int(group_id)] for group_id in group_ids]
    speed_counts = [delay_groups.count(category) for category in speed_categories]
    probs = [count / float(max(1, num_users)) for count in speed_counts]

    block_counts = np.bincount(group_ids, minlength=num_blocks)
    args.dirskew_direction_group_ids = group_ids.astype(int).tolist()
    args.dirskew_direction_group_counts = block_counts.astype(int).tolist()
    args.dirskew_speed_order = speed_order.astype(int).tolist()
    args.dirskew_block_to_speed = {
        int(block_id): speed_name for block_id, speed_name in block_to_speed.items()
    }
    return delay_groups, probs


def _ensure_client_delay_profile(args, state=None):
    num_users = int(args.num_users)
    groups = getattr(args, 'client_delay_groups', None)
    if groups is not None and len(groups) == num_users:
        if not hasattr(args, 'client_delay_rng'):
            seed = getattr(args, 'delay_seed', None)
            if seed is None:
                seed = int(getattr(args, 'seed', 0))
            args.client_delay_rng = np.random.default_rng(seed + 1)
        return groups

    seed = getattr(args, 'delay_seed', None)
    if seed is None:
        seed = int(getattr(args, 'seed', 0))

    if _is_dirskew(args):
        groups, probs = _build_dirskew_delay_profile(state, args, seed)
    else:
        groups, probs = build_client_delay_profile(
            num_users=num_users,
            gamma=getattr(args, 'delay_gamma', 1.0),
            seed=seed,
        )
    args.client_delay_groups = groups
    args.client_delay_group_probs = probs
    args.client_delay_rng = np.random.default_rng(seed + 1)
    return groups


def _paper_uniform_cost(args, idx):
    if idx is None:
        raise ValueError("client-indexed delay profile requires a client index")
    groups = _ensure_client_delay_profile(args)
    return sample_client_delay(
        groups,
        idx,
        profile=getattr(args, 'random_cost', 'dir-skew'),
        rng=args.client_delay_rng,
    )


def _attach_delay_metadata(state, args):
    if _is_fedscale_trace(args):
        ensure_fedscale_trace_sampler(args, state)
        return

    if not _is_dirskew(args):
        return
    _ensure_client_delay_profile(args, state)
    state['client_delay_groups'] = list(args.client_delay_groups)
    state['client_delay_group_probs'] = list(args.client_delay_group_probs)
    if hasattr(args, 'dirskew_direction_group_ids'):
        state['dirskew_direction_group_ids'] = list(args.dirskew_direction_group_ids)
        state['dirskew_direction_group_counts'] = list(args.dirskew_direction_group_counts)
        state['dirskew_speed_order'] = list(args.dirskew_speed_order)
        state['dirskew_block_to_speed'] = dict(args.dirskew_block_to_speed)


def _random_cost(args, state=None):
    if _is_fedscale_trace(args):
        sampler = ensure_fedscale_trace_sampler(args, state)
        return lambda idx=None: sampler.sample_client_delay(
            idx,
            float(state.get('global_cost', 0.0)) if state is not None else 0.0,
        )
    if _is_dirskew(args):
        _ensure_client_delay_profile(args, state)
        return lambda idx=None: _paper_uniform_cost(args, idx)
    return lambda idx=None: random_cost(getattr(args, 'random_cost', 'dir-skew'))


def init_state(state, args):
    random_cost = _random_cost(args, state)
    _attach_delay_metadata(state, args)
    if args.algo == 'FedBuff':
        return fedbuff.init_state(state, args, random_cost)
    if args.algo == 'FADAS':
        return fadas.init_state(state, args, random_cost)
    if args.algo == 'FedASMU':
        return fedasmu.init_state(state, args, random_cost)
    if args.algo in ['FedBuffMA', 'FedBuffMALight']:
        return fedbuffma.init_state(state, args, random_cost)
    if args.algo == 'CA2FL':
        return ca2fl.init_state(state, args, random_cost)
    if args.algo == 'DirBridge':
        return dirbridge.init_state(state, args, random_cost)
    if args.algo == 'CASA':
        return casa.init_state(state, args, random_cost)
    raise ValueError(f"Unsupported algorithm: {args.algo}. Supported: {sorted(SUPPORTED_ALGORITHMS)}")


def run_one_round(args, state, dataset_train, dict_users, num_samples):
    random_cost = _random_cost(args, state)
    if args.algo == 'FedBuff':
        return fedbuff.run_round(state=state, args=args, random_cost=random_cost)
    if args.algo == 'FADAS':
        return fadas.run_round(state=state, args=args, random_cost=random_cost)
    if args.algo in ['FedBuffMA', 'FedBuffMALight']:
        return fedbuffma.run_round(state=state, args=args, random_cost=random_cost)
    if args.algo == 'FedASMU':
        return fedasmu.run_round(state=state, args=args, random_cost=random_cost)
    if args.algo == 'CA2FL':
        return ca2fl.run_round(state, args, dataset_train, dict_users, num_samples, random_cost)
    if args.algo == 'DirBridge':
        return dirbridge.run_round(state=state, args=args, random_cost=random_cost)
    if args.algo == 'CASA':
        return casa.run_round(state=state, args=args, random_cost=random_cost)
    raise ValueError(f"Unsupported algorithm: {args.algo}. Supported: {sorted(SUPPORTED_ALGORITHMS)}")
