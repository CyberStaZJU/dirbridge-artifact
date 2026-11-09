"""在线初始化的覆盖率、当前分组权重与到达诊断工具。"""

from typing import List, Sequence

INIT_MODES = ('full', 'online')
WEIGHT_SOURCES = ('full_count', 'arrival_freq', 'unique_client')


def _getattr(args, name, default):
    value = getattr(args, name, default)
    return default if value is None else value


def e2_init_mode(args) -> str:
    value = str(_getattr(args, 'e2_init_mode', 'full')).lower()
    if value not in INIT_MODES:
        raise ValueError(f'不支持的初始化模式: {value}')
    return value


def e2_online_init(args) -> bool:
    return e2_init_mode(args) == 'online'


def e2_weight_source(args) -> str:
    value = str(_getattr(args, 'e2_weight_source', 'full_count')).lower()
    if value not in WEIGHT_SOURCES:
        raise ValueError(f'不支持的分组权重来源: {value}')
    return value


def coverage_bound(num_seen: int, num_total: int) -> float:
    if num_total <= 0:
        return float('nan')
    seen = max(0, min(int(num_seen), int(num_total)))
    return 2.0 * (1.0 - float(seen) / float(num_total))


def normalize_counts(counts: Sequence[float]) -> List[float]:
    values = [max(0.0, float(count)) for count in counts]
    total = sum(values)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values) if values else []
    return [value / total for value in values]


def group_refresh_gap(group_members, client_last_arrival, iterations: int) -> float:
    gaps = []
    for members in group_members:
        stamps = [int(client_last_arrival[idx]) for idx in members if int(client_last_arrival[idx]) >= 0]
        if stamps:
            gaps.append(float(int(iterations) - max(stamps)))
    return float(sum(gaps) / len(gaps)) if gaps else float('nan')
