#!/usr/bin/env python3
"""Summarize table-script and revision results with per-seed tail accuracy."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = re.compile(r'(?:^|/)table(?P<table>[1-6])/(?P<setting>[^/]+)/(?P<label>[^/]+)/seed=(?P<seed>\d+)/[^/]+-test_acc\.txt$')
IGNORED = {'seed', 'delay_seed', 'run_tag', 'output_dir', 'system_metrics_log_dir',
           'direction_skew_log_dir', 'fedscale_client_profile_path'}


def finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}: expected a number') from exc
    if not math.isfinite(number):
        raise ValueError(f'{name}: expected a finite value')
    return number


def read_accuracy(path, expected_rounds=500):
    lines = Path(path).read_text().splitlines()
    if len(lines) != expected_rounds:
        raise ValueError(f'{path}: expected {expected_rounds} rounds, found {len(lines)}')
    return [accuracy_value(line, f'{path}:{i}') for i, line in enumerate(lines, 1)]


def accuracy_value(value, name):
    number = finite(value, name)
    if not 0 <= number <= 100:
        raise ValueError(f'{name}: accuracy must be in [0, 100]')
    return number


def one(paths, label):
    paths = sorted({p.resolve() for p in paths if p.is_file()})
    if len(paths) > 1:
        raise ValueError(f'{label}: multiple matching files')
    return paths[0] if paths else None


def associated(root, run_dir, params, accuracy, kind):
    flag = 'system_metrics_log_dir' if kind == 'system_metrics' else 'direction_skew_log_dir'
    if params.get(flag):
        directory = Path(params[flag]).expanduser()
        if not directory.is_absolute():
            directory = run_dir / directory
        return one(directory.glob(f'*-{kind}.csv'), flag)
    name = accuracy.name.replace('-test_acc.txt', f'-{kind}.csv')
    local = one(run_dir.rglob(f'*-{kind}.csv'), kind)
    if local:
        return local
    match = TABLE.search(accuracy.relative_to(root).as_posix())
    if match:
        table = 'table' + match.group('table')
        seed = 'seed=' + match.group('seed')
        if kind == 'system_metrics' and table in ('table4', 'table5'):
            return one((root / 'system_metrics' / table / match.group('label') / seed / name,), name)
        if kind == 'direction_skew' and table == 'table3':
            setting = match.group('setting').removesuffix('_dirskew')
            return one((root / 'direction_skew' / setting / seed / name,), name)
        return None
    return None


def table_metadata(match):
    item = match.groupdict()
    setting, label = item['setting'], item['label']
    parsed = re.match(r'^(cifar100|cifar|tinyimagenet|femnist|gspeech)(?:_alpha([0-9.]+))?', setting)
    if not parsed:
        raise ValueError(f'Unknown table setting: {setting}')
    dataset, alpha = parsed.groups()
    algo = 'DirBridge' if label.startswith('DirBridge_') else label
    params = dict(dataset=dataset, alpha=float(alpha) if alpha else .5, algo=algo,
                  interval=1, total_rounds=500, variant=label,
                  bn_buffer_policy='legacy_delta' if algo in ('FedBuffMALight', 'FedASMU') else 'endpoint_mean')
    return params, 'table' + item['table'], setting, label, int(item['seed'])


def collect_runs(root, expected_rounds=500, include_direction_skew=False):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Not a directory: {root}')
    manifests = {}
    for path in root.rglob('manifest.json'):
        value = json.loads(path.read_text())
        if 'resolved_parameters' in value:
            manifests[path.parent.resolve()] = value
    runs, used = [], set()
    for accuracy in sorted(root.rglob('*-test_acc.txt')):
        directory = next((p for p in accuracy.parents if p.resolve() in manifests), None)
        match = TABLE.search(accuracy.relative_to(root).as_posix())
        if directory:
            directory = directory.resolve()
            if directory in used:
                raise ValueError(f'{directory}: multiple accuracy files for one run')
            used.add(directory)
            manifest = manifests[directory]
            params = dict(manifest['resolved_parameters'])
            suite = manifest.get('suite', 'revision')
            setting, label, seed = '', '', int(params['seed'])
        elif match:
            params, suite, setting, label, seed = table_metadata(match)
            directory = accuracy.parent
        else:
            continue
        if int(params.get('interval', 1)) != 1:
            raise ValueError('This summary requires interval=1')
        rounds = int(params.get('total_rounds', expected_rounds))
        if not directory.joinpath('manifest.json').is_file():
            rounds = expected_rounds
        values = read_accuracy(accuracy, rounds)
        completion = directory / 'completion.json'
        exit_status = 'not_recorded'
        if completion.is_file():
            marker = json.loads(completion.read_text())
            if marker.get('status') != 'completed' or marker.get('returncode', 0) != 0:
                raise ValueError(f'{directory}: process did not complete successfully')
            exit_status = 'completed'
        metrics = associated(root, directory, params, accuracy, 'system_metrics')
        direction = associated(root, directory, params, accuracy, 'direction_skew') if include_direction_skew else None
        runs.append(dict(path=str(accuracy), run_dir=str(directory), params=params,
                         suite=suite, setting=setting, label=label, seed=seed,
                         dataset=params['dataset'], alpha=params.get('alpha', ''), algo=params['algo'],
                         rounds=rounds, values=values, metrics_path=metrics, direction_path=direction,
                         process_exit_status=exit_status))
    if not runs:
        raise ValueError('No recognized paper runs')
    missing = set(manifests) - used
    if missing:
        raise ValueError(f'Missing accuracy output: {sorted(missing)[0]}')
    return runs


def read_rows(path, rounds):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if [int(row['round']) for row in rows] != list(range(1, rounds + 1)):
        raise ValueError(f'{path}: incomplete or duplicate round sequence')
    return rows


def mean_or_none(values):
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def health(rows, channel):
    if not rows:
        return 'not_recorded'
    fields = (['e2_eval_loss_finite', 'e2_eval_logits_finite', 'e2_eval_inputs_finite', 'e2_eval_predictions_valid']
              if channel == 'test' else ['validation_loss_finite', 'validation_logits_finite', 'validation_predictions_valid'])
    observed = []
    for row in rows:
        for field in fields:
            value = row.get(field, '').strip().lower()
            if value:
                if value not in ('true', '1', 'false', '0'):
                    raise ValueError(f'Invalid evaluation health field: {field}')
                observed.append(value in ('true', '1'))
    if observed and not all(observed):
        raise ValueError(f'Nonfinite or invalid {channel} evaluation')
    return 'finite' if len(observed) == len(rows) * len(fields) else 'not_recorded'


def summarize_runs(runs, channel='test', expected_seeds=5, expected_rounds=500, tail=10):
    if channel not in ('test', 'validation', 'auto') or expected_seeds < 1 or tail < 1:
        raise ValueError('Invalid summary options')
    groups = defaultdict(list)
    for run in runs:
        params = dict(run['params'])
        metrics = read_rows(run['metrics_path'], run['rounds']) if run['metrics_path'] else []
        policies = {row['bn_buffer_policy'] for row in metrics if row.get('bn_buffer_policy')}
        if len(policies) > 1:
            raise ValueError('BN policy changed within a run')
        if policies:
            params['bn_buffer_policy'] = next(iter(policies))
        actual_channel = channel
        if channel == 'auto':
            actual_channel = 'validation' if params.get('validation_fraction', 0) else 'test'
        status = health(metrics, actual_channel)
        if actual_channel == 'validation':
            if not metrics or any(not row.get('validation_accuracy', '').strip() for row in metrics):
                raise ValueError('validation_accuracy missing; test_acc cannot replace validation')
            if any(row.get('validation_round', str(i)) != str(i) for i, row in enumerate(metrics, 1)):
                raise ValueError('Validation round does not match evaluation round')
            values = [accuracy_value(row['validation_accuracy'], 'validation_accuracy') for row in metrics]
        else:
            values = run['values']
        if len(values) < tail:
            raise ValueError('Metric sequence shorter than tail window')
        threshold = 50 if run['dataset'] == 'cifar' else 80 if run['dataset'] == 'femnist' else None
        threshold_round = next((i for i, value in enumerate(values, 1) if threshold is not None and value >= threshold), None)
        threshold_time = None
        if threshold_round and metrics:
            raw = metrics[threshold_round - 1].get('simulated_wall_time', '')
            threshold_time = finite(raw, 'simulated_wall_time') if raw != '' else None
        runtime = [finite(row['round_runtime_sec'], 'round_runtime_sec') for row in metrics if row.get('round_runtime_sec', '') != '']
        recluster = [finite(row['last_recluster_time_sec'], 'last_recluster_time_sec') for row in metrics
                     if row.get('regrouped', '').lower() in ('1', 'true') and row.get('last_recluster_time_sec', '') != '']
        direction = read_rows(run['direction_path'], run['rounds']) if run['direction_path'] else []
        record = dict(seed=run['seed'], tail=statistics.mean(values[-tail:]), final=values[-1],
                      threshold_round=threshold_round, threshold_simulated_time=threshold_time,
                      round_runtime_mean=mean_or_none(runtime), recluster_mean=mean_or_none(recluster),
                      evaluation_health=status, process_exit_status=run['process_exit_status'],
                      buffer_phi_mean=mean_or_none([finite(row['buffer_phi'], 'buffer_phi') for row in direction]),
                      top2_share_mean=mean_or_none([finite(row['top2_share'], 'top2_share') for row in direction]))
        protocol = {key: value for key, value in params.items() if key not in IGNORED}
        key = json.dumps([run['suite'], run['setting'], run['label'], actual_channel, protocol], sort_keys=True)
        groups[key].append((run, record, params, actual_channel))
    output = []
    for key, items in sorted(groups.items()):
        seeds = [item[1]['seed'] for item in items]
        if len(seeds) != len(set(seeds)):
            raise ValueError('Duplicate seed within one protocol')
        if len(seeds) != expected_seeds:
            raise ValueError(f'Expected {expected_seeds} seeds, found {len(seeds)}')
        run, _, params, actual_channel = items[0]
        records = sorted([item[1] for item in items], key=lambda item: item['seed'])
        tails = [item['tail'] for item in records]
        row = dict(suite=run['suite'], dataset=run['dataset'], alpha=run['alpha'], setting=run['setting'],
                   algo=run['algo'], variant=run['label'], bn_buffer_policy=params.get('bn_buffer_policy'),
                   channel=actual_channel, seed_count=len(seeds), seeds=sorted(seeds),
                   tail_mean=statistics.mean(tails), tail_std=statistics.stdev(tails) if len(tails) > 1 else 0.,
                   final_mean=statistics.mean([item['final'] for item in records]),
                   threshold_reached_seeds=sum(item['threshold_round'] is not None for item in records),
                   protocol={k: v for k, v in params.items() if k not in IGNORED}, runs=records)
        for field, source in [('threshold_round_mean', 'threshold_round'), ('threshold_simulated_time_mean', 'threshold_simulated_time'),
                              ('round_runtime_mean', 'round_runtime_mean'), ('dirbridge_recluster_mean_sec', 'recluster_mean'),
                              ('buffer_phi_mean', 'buffer_phi_mean'), ('top2_share_mean', 'top2_share_mean')]:
            row[field] = mean_or_none([item[source] for item in records])
        if run['algo'] != 'DirBridge':
            row['dirbridge_recluster_mean_sec'] = None
        output.append(row)
    def baseline_key(row):
        # Systems tables compare each method's recorded implementation to FedBuff.
        policy = None if row['suite'] in ('table4', 'table5') else row['bn_buffer_policy']
        return row['suite'], row['dataset'], row['alpha'], row['setting'], row['channel'], policy

    baselines = {}
    for row in output:
        key = baseline_key(row)
        if row['algo'] == 'FedBuff':
            if key in baselines:
                raise ValueError('Multiple FedBuff runtime baselines; select a single protocol')
            baselines[key] = row['round_runtime_mean']
    for row in output:
        baseline = baselines.get(baseline_key(row))
        row['round_runtime_vs_fedbuff'] = row['round_runtime_mean'] / baseline if baseline and row['round_runtime_mean'] is not None else None
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--channel', choices=('test', 'validation', 'auto'), default='auto')
    parser.add_argument('--expected-rounds', type=int, default=500)
    parser.add_argument('--expected-seeds', type=int, default=5)
    parser.add_argument('--tail', type=int, default=10)
    parser.add_argument('--include-direction-skew', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not args.root.is_absolute():
        parser.error('--root must be absolute')
    try:
        rows = summarize_runs(collect_runs(args.root, args.expected_rounds, args.include_direction_skew),
                              args.channel, args.expected_seeds, args.expected_rounds, args.tail)
        payload = json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            path = args.output.resolve()
            if not args.output.is_absolute() or path == ROOT or ROOT in path.parents:
                raise ValueError('--output must be an absolute path outside the repository')
            with path.open('x') as handle:
                handle.write(payload + '\n')
        else:
            print(payload)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
