#!/usr/bin/env python3
"""汇总 Weak 七方法结果与 DirBridge 同状态 FIFO 选择诊断。"""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np

ALGORITHMS = ('FedBuff', 'DirBridge', 'FADAS', 'CASA', 'CA2FL', 'FedBuffMALight', 'FedASMU')
ALPHAS = (.1, .5)
SEEDS = tuple(range(1, 6))


def _file(root, *parts):
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError('输入文件不得位于指定 root 之外')
    return path


def _json(path):
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f'{name} 必须是有限数值')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'{name} 必须是有限数值')
    return number


def _integer(value, expected, name):
    if type(value) is not int or value != expected:
        raise ValueError(f'{name} 应为 {expected}')


def _stats(values):
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values))


def _evaluation(path, rounds):
    accuracy = []
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = {'round', 'test_accuracy', 'test_loss', 'simulated_time'}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f'{path}: 缺少评估列')
        for number, row in enumerate(reader, 1):
            if row['round'] != str(number) or number > rounds:
                raise ValueError(f'{path}: 评估轮次必须完整、有序且唯一')
            accuracy.append(_finite(row['test_accuracy'], 'test_accuracy'))
            _finite(row['test_loss'], 'test_loss')
            _finite(row['simulated_time'], 'simulated_time')
    if len(accuracy) != rounds:
        raise ValueError(f'{path}: 需要 {rounds} 行评估，实际 {len(accuracy)}')
    return statistics.mean(accuracy[-10:])


def _ages(record, decision, field, number):
    costs, stamps = record['raw_costs'], record['stamps']
    ids = decision['selected_ids']
    if (not isinstance(costs, list) or not isinstance(stamps, list)
            or len(costs) != len(stamps) or not isinstance(ids, list)
            or len(ids) != 10 or any(type(i) is not int for i in ids)
            or len(set(ids)) != 10):
        raise ValueError('选择记录需要十个唯一客户端及对应成本和时间戳')
    ages = []
    for client in ids:
        if not 0 <= client < len(stamps):
            raise ValueError('选择客户端超出时间戳范围')
        stamp = stamps[client]
        if type(stamp) is not int or not 0 <= stamp < number:
            raise ValueError('选择时间戳违反启动轮次约束')
        if _finite(costs[client], 'raw_cost') <= 0:
            raise ValueError('选择客户端必须为活跃更新')
        ages.append(number - 1 - stamp)
    reported = record[field]
    if (not isinstance(reported, list) or len(reported) != 10
            or any(type(age) is not int for age in reported) or reported != ages):
        raise ValueError(f'{field} 与原始时间戳不一致')
    return ages


def _selection(path, rounds):
    metrics = {policy: {name: [] for name in ('phi', 'mean_age', 'p95_age', 'max_age')}
               for policy in ('weak', 'fifo')}
    count = 0
    with path.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            record = json.loads(line)
            _integer(record['round'], number, 'selection.round')
            if number > rounds:
                raise ValueError(f'{path}: 选择记录轮次超出预期')
            for policy, key, age_field in (
                    ('weak', 'decision', 'selected_staleness'),
                    ('fifo', 'fifo_counterfactual', 'fifo_counterfactual_staleness')):
                decision = record[key]
                _integer(decision['round_number'], number, 'decision.round_number')
                if decision['policy'] != policy:
                    raise ValueError('选择策略与记录字段不一致')
                phi = _finite(decision['selected_phi'], 'selected_phi')
                if phi < 0:
                    raise ValueError('方向偏斜不能为负')
                ages = _ages(record, decision, age_field, number)
                values = (phi, statistics.mean(ages), float(np.percentile(ages, 95)), max(ages))
                for name, value in zip(metrics[policy], values):
                    metrics[policy][name].append(value)
            count = number
    if count != rounds:
        raise ValueError(f'{path}: 需要 {rounds} 行选择记录，实际 {count}')
    return {policy: {name: statistics.mean(values) for name, values in fields.items()}
            for policy, fields in metrics.items()}


def summarize(root, expected_rounds=500):
    root = Path(root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError('root 必须为已存在的绝对目录')
    root = root.resolve()
    if type(expected_rounds) is not int or expected_rounds < 1:
        raise ValueError('expected_rounds 必须为正整数')
    expected = {(alpha, seed, algorithm) for alpha in ALPHAS
                for seed in SEEDS for algorithm in ALGORITHMS}
    matrix = _json(_file(root, 'matrix.json'))
    if not isinstance(matrix, list) or len(matrix) != 70:
        raise ValueError('matrix.json 必须包含完整的 70 项矩阵')
    observed = []
    for item in matrix:
        if (not isinstance(item, dict) or set(item) != {'alpha', 'seed', 'algorithm'}
                or type(item['alpha']) not in (int, float) or type(item['seed']) is not int
                or not isinstance(item['algorithm'], str)):
            raise ValueError('无效矩阵条目')
        observed.append((item['alpha'], item['seed'], item['algorithm']))
    if len(set(observed)) != 70 or set(observed) != expected:
        raise ValueError('矩阵必须恰好为两个 alpha、五个 seed、七个方法')
    completed = _json(_file(root, 'completion.json'))
    if completed['status'] != 'completed':
        raise ValueError('根矩阵尚未完成')
    _integer(completed['runs'], 70, 'completion.runs')
    runs, diagnostics = [], []
    for alpha in ALPHAS:
        for seed in SEEDS:
            cell = f'alpha={alpha:g}_seed={seed}'
            for algorithm in ALGORITHMS:
                marker = _json(_file(root, cell, algorithm, 'completion.json'))
                if marker['status'] != 'completed':
                    raise ValueError(f'{cell}/{algorithm} 尚未完成')
                _integer(marker['rounds'], expected_rounds, 'completion.rounds')
                tail10 = _evaluation(_file(root, cell, algorithm, 'evaluation.csv'), expected_rounds)
                selection = _selection(_file(root, cell, algorithm, 'selection.jsonl'), expected_rounds)
                runs.append(dict(alpha=alpha, seed=seed, algorithm=algorithm, tail10_accuracy=tail10))
                if algorithm == 'DirBridge':
                    for policy, values in selection.items():
                        diagnostics.append(dict(alpha=alpha, seed=seed, policy=policy, **values))
    accuracy = []
    same_state = []
    for alpha in ALPHAS:
        for algorithm in ALGORITHMS:
            values = [run['tail10_accuracy'] for run in runs
                      if run['alpha'] == alpha and run['algorithm'] == algorithm]
            accuracy.append(dict(alpha=alpha, algorithm=algorithm, seeds=5,
                                 tail10_accuracy=_stats(values)))
        for policy in ('weak', 'fifo'):
            rows = [row for row in diagnostics if row['alpha'] == alpha and row['policy'] == policy]
            same_state.append(dict(alpha=alpha, policy=policy, seeds=5,
                **{name: _stats([row[name] for row in rows])
                   for name in ('phi', 'mean_age', 'p95_age', 'max_age')}))
    return dict(status='completed', runs=70, rounds=expected_rounds,
                tail_window=min(10, expected_rounds), accuracy_unit='percent',
                fifo_interpretation='same_state_selection_only_not_independent_training',
                diagnostic_aggregation='per_round_then_round_mean_then_five_seed_mean_and_sample_sd',
                accuracy=accuracy, dirbridge_same_state=same_state,
                per_seed_accuracy=runs, per_seed_dirbridge_same_state=diagnostics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Weak 结果矩阵的绝对目录')
    parser.add_argument('--expected-rounds', type=int, default=500, help='短程测试轮次；正式协议为 500')
    args = parser.parse_args(argv)
    try:
        result = summarize(args.root, args.expected_rounds)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
