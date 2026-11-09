#!/usr/bin/env python3
"""构造论文修订实验矩阵；默认只打印，不训练也不写运行文件。"""

import argparse
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / 'configs' / 'revision_experiments.json'


def load_config():
    with CONFIG_PATH.open(encoding='utf-8') as handle:
        return json.load(handle)


def parse_seeds(value):
    try:
        seeds = [int(part) for part in value.replace(',', ' ').split()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('seeds 必须为逗号或空格分隔的整数') from exc
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError('seeds 必须非空、非负且不重复')
    return seeds


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=tuple(load_config()['suites']), required=True)
    parser.add_argument('--seeds', type=parse_seeds, help='覆盖协议默认种子，例如 1,2,3')
    parser.add_argument('--python', default=sys.executable, help='训练用 Python 可执行程序')
    parser.add_argument('--state-root', default=os.environ.get(
        'DIRBRIDGE_STATE_ROOT', str(Path.home() / '.local' / 'state' / 'dirbridge')),
        help='仓库外的运行状态及共享 data 目录父路径')
    parser.add_argument('--fedscale-profile', default='', help='用户提供的真实 FedScale profile 文件')
    parser.add_argument('--execute', action='store_true', help='明确执行训练；默认仅打印命令')
    return parser


def external_state_root(value):
    root = Path(value).expanduser().resolve()
    # 同时拒绝仓库子目录和仓库祖先，避免清理或运行状态混入源码树。
    if root == REPO_ROOT or REPO_ROOT in root.parents or root in REPO_ROOT.parents:
        raise ValueError('state-root 必须位于仓库外，且不能是仓库祖先目录')
    return root


def expand_suite(suite, seeds=None, config=None):
    config = config if config is not None else load_config()
    spec = config['suites'][suite]
    seeds = spec['default_seeds'] if seeds is None else seeds
    grid = spec.get('grid', {})
    keys = list(grid)
    combinations = list(itertools.product(*(grid[key] for key in keys)))
    runs = []
    for variant, values, seed in itertools.product(
        spec.get('variants', [{}]), combinations, seeds
    ):
        params = dict(config['common'])
        params.update(spec.get('fixed', {}))
        params.update(variant)
        params.update(zip(keys, values))
        params.update(spec.get('algorithm_overrides', {}).get(params['algo'], {}))
        params['seed'] = seed
        runs.append(params)
    return runs


def flag_arguments(params):
    command = []
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                command.append('--' + key)
        elif value is not None:
            command.extend(['--' + key, str(value)])
    return command


def build_plan(args, config=None, invocation_id=None):
    config = config if config is not None else load_config()
    root = external_state_root(args.state_root)
    profile = str(Path(args.fedscale_profile).expanduser().resolve()) if args.fedscale_profile else ''
    if args.execute and args.suite == 'profile-coupling' and not (
        profile and Path(profile).is_file()
    ):
        raise ValueError('执行 profile-coupling 必须提供存在的 --fedscale-profile 文件')
    invocation_id = invocation_id or uuid.uuid4().hex
    batch_dir = root / 'revision' / args.suite / invocation_id
    plans = []
    for index, original in enumerate(expand_suite(args.suite, args.seeds, config), 1):
        params = dict(original)
        run_dir = external_state_root(
            batch_dir / ('run-%04d-%s-seed%d' % (index, params['algo'], params['seed']))
        )
        params.update({
            'output_dir': str(run_dir / 'acc'),
            'system_metrics_log_dir': str(run_dir / 'metrics'),
            'direction_skew_log_dir': str(run_dir / 'direction_skew'),
            'run_tag': '%s-%04d' % (invocation_id, index),
        })
        if args.suite == 'profile-coupling':
            params['fedscale_client_profile_path'] = profile
        command = [args.python, str(REPO_ROOT / 'main_fed.py')] + flag_arguments(params)
        plans.append({
            'suite': args.suite,
            'invocation_id': invocation_id,
            'run_dir': str(run_dir),
            'cwd': str(root),
            'command': command,
            'resolved_parameters': params,
            'metric_split': 'validation' if params.get('validation_fraction', 0) else 'test',
            'validation_sampling': config['validation_protocol']['sampling'] if params.get('validation_fraction', 0) else None,
            'protocol_origin': config['suites'][args.suite].get('protocol_origin', 'paper'),
            'evidence': config['suites'][args.suite]['evidence'],
            'notes': config['suites'][args.suite].get('notes', []),
            'dir_skew_alias': config['dir_skew_alias'] if params['random_cost'] == 'dir-skew' else None,
        })
    return plans


def write_manifest(plan, execute):
    run_dir = Path(plan['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(plan, execution_requested=execute)
    with (run_dir / 'manifest.json').open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return run_dir


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    try:
        plans = build_plan(args, config)
    except ValueError as exc:
        parser.error(str(exc))
    print('%s：%s，共 %d 个运行' % ('执行' if args.execute else 'DRY-RUN', args.suite, len(plans)))
    for note in config['suites'][args.suite].get('notes', []):
        print('注意：' + note)
    for plan in plans:
        print('工作目录：' + plan['cwd'])
        print(shlex.join(plan['command']))
        if args.execute:
            run_dir = write_manifest(plan, execute=True)
            # 数据读取器采用相对路径；在外部 state-root 运行，避免下载/缓存写入仓库。
            try:
                with (run_dir / 'stdout.log').open('x') as stdout, (run_dir / 'stderr.log').open('x') as stderr:
                    subprocess.run(plan['command'], cwd=plan['cwd'], stdout=stdout, stderr=stderr, check=True)
            except subprocess.CalledProcessError as exc:
                with (run_dir / 'completion.json').open('x', encoding='utf-8') as handle:
                    json.dump({'status': 'failed', 'returncode': exc.returncode}, handle)
                raise
            else:
                with (run_dir / 'completion.json').open('x', encoding='utf-8') as handle:
                    json.dump({'status': 'completed', 'returncode': 0}, handle)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
