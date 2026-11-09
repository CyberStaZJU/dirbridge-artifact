"""仅按设备 profile 构建标签组与 FedScale 设备簇的固定耦合。"""
import math
from collections import defaultdict
from types import SimpleNamespace

import numpy as np


def _kmeans(x, k, seed, iterations=100):
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iterations):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        assignments = distances.argmin(1)
        if any(not np.any(assignments == j) for j in range(k)):
            raise ValueError('profile coupling KMeans produced an empty cluster')
        updated = np.array([x[assignments == j].mean(0) for j in range(k)])
        if np.allclose(updated, centers):
            break
        centers = updated
    return assignments, centers


def build_profile_coupling(
    profile_path, label_group_ids, batch_size, local_steps, upload_size=1.0,
    download_size=None, seed=0, num_groups=3, augmentation_factor=3.0,
):
    from utils.fedscale_trace import (
        _as_profile_mapping, _profile_duration, load_fedscale_payload,
    )

    if num_groups != 3:
        raise ValueError('profile coupling requires three device clusters')
    if label_group_ids is None:
        raise ValueError('profile coupling requires label-derived client groups')
    groups = []
    for group in label_group_ids:
        try:
            group_id = int(group)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'unknown label group: {group!r}') from exc
        if group_id not in range(num_groups) or group != group_id:
            raise ValueError(f'unknown label group: {group!r}')
        groups.append(group_id)
    if not groups:
        raise ValueError('profile coupling requires nonempty label groups')

    download_size = upload_size if download_size is None else download_size
    if any(not math.isfinite(float(value)) or float(value) < 0
           for value in (upload_size, download_size)):
        raise ValueError('profile coupling payload sizes must be finite and nonnegative')
    if (batch_size <= 0 or local_steps <= 0
            or int(batch_size) != batch_size or int(local_steps) != local_steps
            or not math.isfinite(float(augmentation_factor)) or augmentation_factor <= 0):
        raise ValueError('profile coupling requires positive batch size, steps and augmentation')

    profiles = _as_profile_mapping(load_fedscale_payload(profile_path))
    rows = []
    for profile_id, record in profiles.items():
        if not isinstance(record, dict):
            continue
        try:
            computation = float(record['computation'])
            communication = float(record['communication'])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if (computation > 0 and communication > 0
                and math.isfinite(computation) and math.isfinite(communication)):
            rows.append((profile_id, computation, communication))
    if len(rows) < num_groups:
        raise ValueError('profile coupling requires at least three valid positive device profiles')

    features = np.array([[math.log(c), math.log(b)] for _, c, b in rows])
    mean = features.mean(0)
    std = features.std(0)
    standardized = (features - mean) / np.where(std > 0, std, 1)
    assignments, _ = _kmeans(standardized, num_groups, seed)
    duration_args = SimpleNamespace(
        fedscale_batch_size=batch_size,
        fedscale_local_steps=local_steps,
        fedscale_upload_size_mb=upload_size,
        fedscale_download_size_mb=download_size,
        fedscale_augmentation_factor=augmentation_factor,
    )
    # payload 与 rate 沿用 sampler 的同一单位，不额外转换通信量。
    durations = [float(_profile_duration(profiles[pid], duration_args)) for pid, _, _ in rows]
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError('profile coupling requires finite positive service durations')
    by_class = defaultdict(list)
    for index, cluster in enumerate(assignments):
        by_class[int(cluster)].append(index)
    by_class = {
        cluster: sorted(indices, key=lambda index: durations[index])
        for cluster, indices in by_class.items()
    }
    order = sorted(
        range(num_groups),
        key=lambda cluster: float(np.median([durations[i] for i in by_class[cluster]])),
    )
    permutation = np.random.default_rng(seed + 104729).permutation(num_groups)
    label_to_class = {group: int(permutation[group]) for group in range(num_groups)}
    positions = defaultdict(int)
    mapping = []
    reuse = 0
    for client, group in enumerate(groups):
        cluster = label_to_class[group]
        candidates = by_class[cluster]
        position = positions[group]
        index = candidates[position % len(candidates)]
        cycled = position >= len(candidates)
        reuse += cycled
        positions[group] += 1
        mapping.append({
            'client_index': client,
            'label_group': group,
            'profile_id': rows[index][0],
            'profile_class': cluster,
            'duration': durations[index],
            'cycled': cycled,
        })
    return {
        'mapping': mapping,
        'profile_cluster_sizes': {str(j): len(by_class[j]) for j in range(num_groups)},
        'profile_cluster_order_fast_to_slow': [int(j) for j in order],
        'label_to_profile_class': label_to_class,
        'profile_reuse_count': int(reuse),
        'features': ['log_computation', 'log_communication'],
        'kmeans': 'custom_lloyd',
    }
