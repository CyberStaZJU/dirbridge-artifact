#!/usr/bin/env python3
"""核算补充材料 S-XI 的七方法解析服务器状态预算，不测量模拟器驻留内存。

全部预留槽位按已填充计数，排除瞬时分配、Python 对象及分配器开销。
这不是运行时或峰值内存测量；实际模拟器中的别名、额外副本和状态寿命可不同。
默认将 CSV 写到 stdout；--output 仅接受仓库外的绝对新文件路径。
"""

import argparse
import csv
import sys
from dataclasses import dataclass, fields
from pathlib import Path


MIB = 1 << 20
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BudgetConfig:
    parameters: int = 11_173_962
    bn_bytes: int = 38_560
    clients: int = 100
    buffer_size: int = 10
    groups: int = 4
    sketch_dim: int = 2048
    rounds: int = 500
    launch_snapshots: int = 100
    casa_dim: int = 128
    feature_records: int = 100

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            minimum = 0 if field.name in {'bn_bytes', 'launch_snapshots', 'feature_records'} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f'{field.name} 必须为不小于 {minimum} 的整数')
        if self.buffer_size > self.clients:
            raise ValueError('buffer_size 必须满足 1 <= B <= N')
        if self.groups > self.clients:
            raise ValueError('groups 必须满足 1 <= K <= N')


def calculate_budget(config):
    """按 S-XI 公式返回七行；所有字节数均由整数项求和。"""
    p = config.parameters
    u = config.bn_bytes
    n = config.clients
    b = config.buffer_size
    k = config.groups
    d = config.sketch_dim
    t = config.rounds
    launch = config.launch_snapshots
    dc = config.casa_dim
    nfeat = config.feature_records
    s = 4 * p
    common = (b + 1) * (s + u)
    variants = [
        ('FedBuff', {}, '0', 'Common model/update device',
         '标量簿记不增加本表所列张量或数组预算'),
        ('FADAS', {'adaptive_moments': 3 * s}, '3*S', 'Model device',
         '三个 FP32 参数大小的自适应矩向量'),
        ('MA-Light', {'momentum': s, 'history_t': 8 * t * t,
                      'history_t_plus_1': 8 * (t + 1) * (t + 1), 'coefficients': 8 * t},
         'S+8*(T^2+(T+1)^2+T)', 'Model device and CPU',
         '动量 FP32；两个历史矩阵及系数向量 FP64'),
        ('FedASMU', {'launch_snapshots': launch * (s + u), 'q_tables': 4 * n * 10 * 3},
         'L*(S+U)+4*N*10*3', 'Model device and CPU',
         '完整启动快照含 BN；每客户端 Q 表固定为 10x3 FP32'),
        ('CASA-inspired buffered variant',
         {'feature_plan': 8 * dc, 'client_features': 4 * n * dc,
          'similarities_and_timestamps': 12 * n * n, 'feature_records': 4 * nfeat * dc},
         '8*dc+4*N*dc+12*N^2+4*nfeat*dc', 'Common device and CPU',
         '特征索引 int64；相似度 FP32 与时间戳 int64'),
        ('CA2FL', {'client_update_caches': n * s}, 'N*S', 'Model device',
         'N 个 FP32 客户端更新缓存'),
        ('DirBridge', {'group_memories': k * s, 'explicit_sketch_plan': 9 * p,
                       'client_features': 4 * n * d, 'clustering_feature_copy': 4 * n * d,
                       'centroids': 4 * k * d, 'centroid_copy': 4 * k * d},
         'K*S+9*P+4*(2*N+2*K)*d', 'Model device and CPU',
         '显式计划每坐标 int64 桶索引和 int8 符号共 9P；含客户端特征及其聚类副本、质心及其副本'),
    ]
    rows = []
    for method, components, formula, location, note in variants:
        method_bytes = sum(components.values())
        total = common + method_bytes
        rows.append({
            'method': method,
            'P': p, 'U': u, 'N': n, 'B': b, 'K': k, 'd': d, 'T': t,
            'L': launch, 'dc': dc, 'nfeat': nfeat, 'S': s,
            'common_formula': '(B+1)*(S+U)',
            'common_bytes': common,
            'method_formula': formula,
            'method_components_bytes': ';'.join(f'{name}={size}' for name, size in components.items()),
            'method_bytes': method_bytes,
            'total_bytes': total,
            'total_mib': f'{total / MIB:.6f}',
            'table_mib': f'{total / MIB:.1f}',
            'state_location': location,
            'scope': 'analytical_populated_slots_transients_excluded',
            'note': note,
        })
    return rows


def external_output_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('--output 必须是绝对路径')
    resolved = path.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError('--output 必须位于本仓库之外')
    return path


def write_csv(handle, rows):
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = BudgetConfig()
    for field in fields(defaults):
        parser.add_argument('--' + field.name.replace('_', '-'), type=int,
                            default=getattr(defaults, field.name))
    parser.add_argument('--output', help='仓库外的绝对新文件路径；已有文件不会被覆盖')
    args = parser.parse_args(argv)
    try:
        config = BudgetConfig(**{field.name: getattr(args, field.name) for field in fields(defaults)})
        rows = calculate_budget(config)
        if args.output is None:
            write_csv(sys.stdout, rows)
        else:
            path = external_output_path(args.output)
            with path.open('x', newline='', encoding='utf-8') as handle:
                write_csv(handle, rows)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
