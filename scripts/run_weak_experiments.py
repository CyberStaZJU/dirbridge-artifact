#!/usr/bin/env python3
"""Weak CIFAR-10 两个 alpha、七方法、五种子；默认只打印计划。

每 alpha/seed 共享初始化前参考方向、模型和分区，各算法独立子进程。
FIFO 仅为同状态选择反事实；不迁移硬件控制器或独立 FIFO 训练。
"""

import argparse
import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALGORITHMS = ('FedBuff', 'DirBridge', 'FADAS', 'CASA', 'CA2FL', 'FedBuffMALight', 'FedASMU')
BASE = dict(dataset='cifar', model='resnet', distribution='noniid', random_cost='dir-skew',
            num_users=100, concurrency=40, buffer_size=10, total_rounds=500,
            local_bs=100, local_period=10, lr=.01, global_lr=1., momentum=0., weight_decay=0.,
            interval=1, bs=500, num_workers=0, bn_buffer_policy='endpoint_mean',
            dirbridge_num_groups=4, dirbridge_feature_mode='count_sketch', dirbridge_sketch_dim=2048,
            dirbridge_kmeans_iters=20, dirbridge_recluster_interval=5, dirbridge_cache_beta=.9,
            e2_init_mode='full', e2_weight_source='full_count')
OVERRIDES = {
    'FedBuff': {}, 'DirBridge': {},
    'FADAS': dict(global_lr=.0001, fadas_beta1=.9, fadas_beta2=.99, fadas_eps=1e-8,
                  fadas_tau_c=4, fadas_delay_adaptive=True),
    'CASA': dict(casa_num_groups=4, casa_alpha0=.3, casa_omega=.01, casa_partition_gamma=.15,
                 casa_cluster_bias=3., casa_eigengap_topk=10, casa_cluster_interval=5,
                 casa_min_cluster_size=8, casa_sketch_dim=128, casa_buffer_budget=10,
                 casa_start_gap=10, casa_max_splits=4, casa_kmeans_iters=20),
    'CA2FL': {}, 'FedBuffMALight': dict(momentum=.9, ma_p=0., ma_staleness_limit=20,
                                      weak_extra_baseline_endpoints=True),
    'FedASMU': dict(asmu_mu_alpha=1., asmu_mu_beta=1., asmu_lambda0=1., asmu_sigma0=1.,
                    asmu_iota0=0., asmu_gamma0=1., asmu_nu0=0., asmu_eta_lambda=.01,
                    asmu_eta_sigma=.01, asmu_eta_iota=.01, asmu_eta_gamma=.01, asmu_eta_nu=.01,
                    asmu_eps_greedy=.1, asmu_q_lr=.1, asmu_q_gamma=.9, asmu_staleness_limit=40,
                    weak_extra_baseline_endpoints=True),
}


def matrix():
    return [dict(alpha=alpha, seed=seed, algorithm=algorithm)
            for alpha in (.1, .5) for seed in range(1, 6) for algorithm in ALGORITHMS]


def external_root(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('输出父目录必须为绝对路径')
    path = path.resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError('输出必须在代码仓库外')
    if not path.is_dir():
        raise ValueError('输出父目录必须已存在')
    return path


def json_value(value):
    import numpy as np
    import torch
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, np.random.Generator):
        return {'bit_generator': value.bit_generator.__class__.__name__,
                'state': json_value(value.bit_generator.state)}
    return value


def write_json(path, value):
    payload = json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(payload + '\n')


def native_args(alpha, seed, algorithm, gpu):
    from utils.options import args_parser
    values = dict(BASE, **OVERRIDES[algorithm], alpha=alpha, seed=seed, algo=algorithm, gpu=gpu)
    argv = []
    for name, value in values.items():
        if isinstance(value, bool):
            if value:
                argv.append('--' + name)
        else:
            argv.extend(['--' + name, str(value)])
    with patch.object(sys, 'argv', [sys.argv[0], *argv]):
        args = args_parser()
    args.reference_seed = 2081041
    return args


def prepare_cell(cell, alpha, seed, gpu):
    import main_fed as main
    from algorithm import dirbridge
    from utils.weak_protocol import build_preinit_state, build_reference, original_tiers
    args = native_args(alpha, seed, 'DirBridge', gpu)
    main.configure_runtime(args)
    state, *_ = build_preinit_state(main, args)
    reference = build_reference(state, args, dirbridge)
    reference.update(alpha=alpha, seed=seed,
                     tiers=original_tiers(args.label_correlated_group_ids, seed),
                     label_group_ids=[int(g) for g in args.label_correlated_group_ids])
    write_json(cell / 'reference.json', reference)


def train_cell(cell, alpha, seed, algorithm, gpu):
    import main_fed as main
    from algorithm import dispatcher
    from utils.weak_protocol import (BACKENDS, ServiceBank, WeakScheduler, build_preinit_state,
                                     model_digest, original_tiers, partition_record)
    reference = json.loads((cell / 'reference.json').read_text())
    if reference['alpha'] != alpha or reference['seed'] != seed:
        raise ValueError('参考方向属于另一 alpha/seed 单元')
    args = native_args(alpha, seed, algorithm, gpu)
    main.configure_runtime(args)
    state, train, test, partition, counts = build_preinit_state(main, args)
    if (partition_record(state) != reference['partition']
            or model_digest(state['net_glob']) != reference['initial_model_sha256']
            or original_tiers(args.label_correlated_group_ids, seed) != reference['tiers']):
        raise RuntimeError('共同初始化模型、BN、分区或服务关联不匹配')
    if getattr(args, 'resolved_bn_buffer_policy', '') != 'endpoint_mean':
        raise RuntimeError(f'{algorithm} requires endpoint_mean for the Weak protocol')
    bank = ServiceBank(reference['tiers'], seed)
    scheduler = WeakScheduler(reference['group_ids'])
    module = importlib.import_module('algorithm.' + BACKENDS[algorithm])
    run = cell / algorithm
    run.mkdir(exist_ok=False)
    write_json(run / 'identity.json', dict(alpha=alpha, seed=seed, algorithm=algorithm,
               pairing_model_and_partition_verified=True, reference='common_initial_model_before_algorithm_init',
               service_rule='SeedSequence([seed, source_id=client_id, launch_index])',
               fifo='same_state_selection_only_not_independent_training', native_parameters=vars(args)))

    def random_cost(native_args, current=None):
        if current is None:
            raise RuntimeError('固定服务轨迹要求明确的实时状态')
        return lambda client: bank.draw(int(client), current)

    with patch.object(dispatcher, '_random_cost', random_cost):
        if bank.events:
            raise RuntimeError('参考探针提前消费服务轨迹')
        state = main.init_state(state, args)
        if len(bank.events) != args.concurrency:
            raise RuntimeError('初始化必须恰好启动 concurrency 个客户端')
        with (run / 'selection.jsonl').open('x', encoding='utf-8') as selections, \
                (run / 'evaluation.csv').open('x', newline='', encoding='utf-8') as evaluation:
            writer = csv.DictWriter(evaluation, fieldnames=['round', 'test_accuracy', 'test_loss', 'simulated_time'])
            writer.writeheader()
            while state['iterations'] < args.total_rounds:
                before_events = len(bank.events)
                state, record = scheduler.step(module, args, state,
                    lambda: main.run_one_round(args, state, train, partition, counts))
                if len(bank.events) - before_events != args.buffer_size:
                    raise RuntimeError('每轮必须恰好补启动 B 个客户端')
                latest = {event['client_id']: event for event in bank.events}
                for client, cost in enumerate(state['cost']):
                    if cost > 0 and (cost != latest[client]['completion_time']
                                     or state['stamp'][client] != latest[client]['round']):
                        raise RuntimeError('活跃更新的真实时钟或启动版本与服务 bank 不同')
                selections.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                selections.flush()
                accuracy, loss, diagnostics = main.evaluate_global_model(state, test, args)
                if (not diagnostics.get('eval_loss_finite', True)
                        or not diagnostics.get('eval_logits_finite', True)
                        or not diagnostics.get('eval_inputs_finite', True)
                        or not diagnostics.get('eval_predictions_valid', True)):
                    raise RuntimeError('评估出现非有限损失、输入、logits 或无效预测')
                writer.writerow(dict(round=state['iterations'], test_accuracy=float(accuracy),
                                     test_loss=float(loss), simulated_time=float(state['global_cost'])))
                evaluation.flush()
    write_json(run / 'launch_trace.json', bank.events)
    write_json(run / 'completion.json', dict(status='completed', rounds=state['iterations']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='显式启动训练子进程')
    parser.add_argument('--output-root', help='已存在的仓库外绝对父目录；在其中创建 UUID 新目录')
    parser.add_argument('--gpu', default='-1', help='沿用目标原生运行时设备选项')
    parser.add_argument('--worker', choices=['reference', *ALGORITHMS], help=argparse.SUPPRESS)
    parser.add_argument('--cell', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--alpha', type=float, choices=[.1, .5], help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, choices=range(1, 6), help=argparse.SUPPRESS)
    settings = parser.parse_args(argv)
    if settings.worker:
        if not settings.execute or settings.cell is None or settings.alpha is None or settings.seed is None:
            parser.error('内部 worker 缺少执行授权或单元参数')
        cell = external_root(str(settings.cell))
        if settings.worker == 'reference':
            prepare_cell(cell, settings.alpha, settings.seed, settings.gpu)
        else:
            train_cell(cell, settings.alpha, settings.seed, settings.worker, settings.gpu)
        return
    if not settings.execute:
        print(json.dumps(dict(mode='dry_run', run_count=70, parameters=BASE, overrides=OVERRIDES,
                              runs=matrix(), reference_seed=2081041,
                              fifo='same_state_selection_only_not_independent_training'),
                         ensure_ascii=False, indent=2))
        return
    if settings.output_root is None:
        parser.error('--execute 必须指定 --output-root')
    root = external_root(settings.output_root) / ('weak-' + str(uuid.uuid4()))
    root.mkdir(exist_ok=False)
    write_json(root / 'matrix.json', matrix())
    for alpha in (.1, .5):
        for seed in range(1, 6):
            cell = root / f'alpha={alpha:g}_seed={seed}'
            cell.mkdir()
            for worker in ('reference', *ALGORITHMS):
                command = [sys.executable, str(Path(__file__).resolve()), '--execute', '--worker', worker,
                           '--cell', str(cell), '--alpha', str(alpha), '--seed', str(seed), '--gpu', settings.gpu]
                with (cell / f'{worker}.log').open('x', encoding='utf-8') as handle:
                    subprocess.run(command, cwd=Path(settings.output_root).resolve(), stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT, check=True)
    write_json(root / 'completion.json', dict(status='completed', runs=70))
    print(root)


if __name__ == '__main__':
    main()
