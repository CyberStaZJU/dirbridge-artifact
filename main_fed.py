#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import gc
import csv
import os
import random
import time
import tracemalloc
import resource
import sys

import matplotlib as mpl
import numpy as np
import torch
try:
    import torch_npu
except ImportError:
    pass

from algorithm.dispatcher import init_state, run_one_round
from builders.dataset_builder import build_dataset
from builders.model_builder import build_model
from models.test import test_img
from utils.direction_skew_logging import (
    log_direction_skew_metrics,
    setup_direction_skew_monitor,
)
from utils.options import args_parser

mpl.use('Agg')
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

def _fraction_token(value):
    text = f"{float(value):.3g}"
    return text.replace('.', 'p')


def npu_is_available():
    return hasattr(torch, 'npu') and torch.npu.is_available()


def configure_runtime(args):
    if args.gpu == '-1':
        args.device = 'cpu'
    else:
        if npu_is_available():
            args.device = f'npu:{args.gpu}'
            torch.npu.set_device(args.device)
        elif torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            args.device = 'cuda'
        else:
            args.device = 'cpu'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if npu_is_available():
        torch.npu.manual_seed_all(args.seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def build_training_state(args):
    init_start = time.perf_counter()
    dataset_train, dataset_test, dict_users, num_samples, dataset_meta = build_dataset(args)
    net_glob, w_glob = build_model(args)
    state = {
        'net_glob': net_glob,
        'w_glob': w_glob,
        'dataset_train': dataset_train,
        'dict_users': dict_users,
        'num_samples': num_samples,
        'global_cost': 0,
        'dataset_meta': dataset_meta,
        'validation_dataset': dataset_meta.get('validation'),
        'validation_indices': dataset_meta.get('validation_indices', []),
    }
    state = init_state(state, args)
    state['e2_init_time_sec'] = time.perf_counter() - init_start
    state['e2_init_cost_status'] = 'process_wall_time_includes_dataset_model_and_init'
    if state.get('e2_init_mode') != 'online':
        state['e2_online_bootstrap_time_sec'] = 0.0
    return state, dataset_train, dataset_test, dict_users, num_samples


def build_testacc_filename(args):
    parts = [args.dataset, args.algo]
    random_cost_name = getattr(args, 'random_cost', 'dir-skew')
    append_random_cost = True

    if args.distribution == 'iid':
        parts.append(args.distribution)
    elif args.distribution == 'label_block':
        parts.append(args.distribution)
    elif args.distribution == 'label_correlated':
        parts.append(f"{args.distribution}_p{_fraction_token(getattr(args, 'label_primary_fraction', 0.5))}")
    elif args.distribution == 'speed_label_correlated_dirichlet':
        parts.append(
            f"{args.distribution}_p{_fraction_token(getattr(args, 'label_primary_fraction', 0.9))}"
            f"_alpha{_fraction_token(getattr(args, 'alpha', 0.1))}"
        )
    else:
        parts.append(str(args.alpha))

    if append_random_cost:
        parts.append(random_cost_name)
    parts.extend([str(args.concurrency), str(args.buffer_size), str(args.seed)])
    if getattr(args, 'run_tag', ''):
        parts.append(str(args.run_tag))
    return '-'.join(parts) + '-test_acc.txt'


def build_output_path(args):
    filename = build_testacc_filename(args)
    output_dir = getattr(args, 'output_dir', '')
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    return filename


def _current_device_memory_mb(args):
    if getattr(args, 'device', 'cpu') != 'cpu':
        if npu_is_available():
            try:
                return torch.npu.max_memory_allocated() / (1024 ** 2)
            except Exception:
                return ''
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return ''


def _current_process_memory_mb():
    current_mb = 0.0
    peak_mb = 0.0
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / (1024 ** 2)
        peak_mb = peak / (1024 ** 2)
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss / (1024 ** 2 if sys.platform == 'darwin' else 1024.0)
        peak_mb = max(peak_mb, rss_mb)
    except Exception:
        pass
    return current_mb, peak_mb


def _system_metrics_fieldnames():
    return [
        'round',
        'algo',
        'dataset',
        'alpha',
        'random_cost',
        'concurrency',
        'buffer_size',
        'seed',
        'run_tag',
        'bn_buffer_policy',
        'simulated_wall_time',
        'runtime_sec',
        'round_runtime_sec',
        'process_memory_mb',
        'process_peak_memory_mb',
        'device_peak_memory_mb',
        'regrouped',
        'last_recluster_time_sec',
        'total_recluster_time_sec',
        'max_recluster_time_sec',
        'recluster_count',
        'e2_init_mode',
        'e2_weight_source',
        'e2_seen_clients',
        'e2_seen_ratio',
        'e2_clustered_clients',
        'e2_assigned_clients',
        'e2_raw_arrivals',
        'e2_valid_updates',
        'e2_used_updates',
        'e2_group_version',
        'e2_weight_update_round',
        'e2_weight_l1_oracle',
        'e2_weight_l1_status',
        'e2_coverage_bound',
        'e2_unseen_client_ratio',
        'e2_uninitialized_group_mass',
        'e2_cache_ready_ratio',
        'evaluation_round',
        'e2_eval_loss_finite',
        'e2_eval_logits_finite',
        'e2_eval_inputs_finite',
        'e2_eval_predictions_valid',
        'e2_eval_nonfinite_loss_batches',
        'e2_eval_nonfinite_logit_batches',
        'e2_init_time_sec',
        'e2_online_bootstrap_time_sec',
        'e2_oracle_measurement_time_sec',
        'e2_init_cost_status',
        'validation_round',
        'validation_accuracy',
        'validation_loss',
        'validation_loss_finite',
        'validation_logits_finite',
        'validation_predictions_valid',
    ]


def build_system_metrics_path(args, output_path):
    log_dir = getattr(args, 'system_metrics_log_dir', '')
    if not log_dir:
        return None
    os.makedirs(log_dir, exist_ok=True)
    output_name = os.path.basename(output_path)
    if output_name.endswith('-test_acc.txt'):
        output_name = output_name[:-len('-test_acc.txt')] + '-system_metrics.csv'
    else:
        output_name = os.path.splitext(output_name)[0] + '-system_metrics.csv'
    path = os.path.join(log_dir, output_name)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_system_metrics_fieldnames())
        writer.writeheader()
    print(f"System metrics log: {path}", flush=True)
    return path


def log_system_metrics(args, state, csv_path, train_start_time, round_runtime_sec):
    if not csv_path:
        return
    process_mb, process_peak_mb = _current_process_memory_mb()
    e2 = state.get('e2_metrics') or {}
    current_round = int(state.get('iterations', 0))
    evaluated = state.get('last_evaluation_round') == current_round
    validated = state.get('last_validation_round') == current_round
    eval_diag = (state.get('last_eval_diagnostics') or {}) if evaluated else {}
    validation_diag = (state.get('last_validation_diagnostics') or {}) if validated else {}
    row = {
        'round': int(state.get('iterations', 0)),
        'algo': getattr(args, 'algo', ''),
        'dataset': getattr(args, 'dataset', ''),
        'alpha': getattr(args, 'alpha', ''),
        'random_cost': getattr(args, 'random_cost', ''),
        'concurrency': getattr(args, 'concurrency', ''),
        'buffer_size': getattr(args, 'buffer_size', ''),
        'seed': getattr(args, 'seed', ''),
        'run_tag': getattr(args, 'run_tag', ''),
        'bn_buffer_policy': getattr(args, 'resolved_bn_buffer_policy', getattr(args, 'bn_buffer_policy', 'legacy_delta')),
        'simulated_wall_time': float(state.get('global_cost', 0.0)),
        'runtime_sec': time.perf_counter() - train_start_time,
        'round_runtime_sec': float(round_runtime_sec),
        'process_memory_mb': process_mb,
        'process_peak_memory_mb': process_peak_mb,
        'device_peak_memory_mb': _current_device_memory_mb(args),
        'regrouped': int(bool(state.get('last_regrouped', False))),
        'last_recluster_time_sec': float(state.get('last_recluster_time_sec', 0.0)),
        'total_recluster_time_sec': float(state.get('total_recluster_time_sec', 0.0)),
        'max_recluster_time_sec': float(state.get('max_recluster_time_sec', 0.0)),
        'recluster_count': int(state.get('recluster_count', 0)),
        'e2_init_mode': state.get('e2_init_mode', ''),
        'e2_weight_source': state.get('e2_last_weight_source', state.get('e2_weight_source', '')),
        'e2_seen_clients': int(e2.get('seen_clients', 0)),
        'e2_seen_ratio': e2.get('seen_ratio', ''),
        'e2_clustered_clients': int(e2.get('clustered_clients', 0)),
        'e2_assigned_clients': int(sum(1 for gid in state.get('group_ids', []) if int(gid) >= 0)),
        'e2_raw_arrivals': int(sum(state.get('arrival_count', []))),
        'e2_valid_updates': int(state.get('last_valid_buffer_count', 0)),
        'e2_used_updates': int(sum(
            int(item.get('member_count', 0))
            for item in state.get('last_selected_item_summary', [])
        )),
        'e2_group_version': int(state.get('recluster_count', 0)),
        'e2_weight_update_round': state.get('e2_weight_update_round', ''),
        'e2_weight_l1_oracle': e2.get('weight_l1_oracle', ''),
        'e2_weight_l1_status': e2.get('weight_l1_status', 'reference_unavailable'),
        'e2_coverage_bound': e2.get('coverage_bound', ''),
        'e2_unseen_client_ratio': (
            1.0 - float(e2['seen_ratio']) if e2.get('seen_ratio', '') != '' else ''
        ),
        'e2_uninitialized_group_mass': e2.get('uninit_group_mass', ''),
        'e2_cache_ready_ratio': state.get('e2_cache_ready_ratio', ''),
        'evaluation_round': current_round if evaluated else '',
        'e2_eval_loss_finite': eval_diag.get('eval_loss_finite', ''),
        'e2_eval_logits_finite': eval_diag.get('eval_logits_finite', ''),
        'e2_eval_inputs_finite': eval_diag.get('eval_inputs_finite', ''),
        'e2_eval_predictions_valid': eval_diag.get('eval_predictions_valid', ''),
        'e2_eval_nonfinite_loss_batches': eval_diag.get('eval_nonfinite_loss_batches', ''),
        'e2_eval_nonfinite_logit_batches': eval_diag.get('eval_nonfinite_logit_batches', ''),
        'e2_init_time_sec': state.get('e2_init_time_sec', ''),
        'e2_online_bootstrap_time_sec': state.get('e2_online_bootstrap_time_sec', ''),
        'e2_oracle_measurement_time_sec': state.get('e2_init_oracle_time_sec', ''),
        'e2_init_cost_status': state.get('e2_init_cost_status', 'not_measured'),
        'validation_round': current_round if validated else '',
        'validation_accuracy': state.get('last_validation_accuracy', '') if validated else '',
        'validation_loss': state.get('last_validation_loss', '') if validated else '',
        'validation_loss_finite': validation_diag.get('eval_loss_finite', ''),
        'validation_logits_finite': validation_diag.get('eval_logits_finite', ''),
        'validation_predictions_valid': validation_diag.get('eval_predictions_valid', ''),
    }
    with open(csv_path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_system_metrics_fieldnames())
        writer.writerow(row)


def evaluate_global_model(state, dataset_test, args, diagnostics_key='last_eval_diagnostics'):
    net_glob = state['net_glob']
    was_training = net_glob.training
    net_glob.eval()
    try:
        with torch.no_grad():
            acc_test, loss_test, diagnostics = test_img(
                net_glob, dataset_test, args, return_diagnostics=True
            )
    finally:
        net_glob.train(was_training)
    state[diagnostics_key] = diagnostics
    if diagnostics_key == 'last_eval_diagnostics':
        state['last_evaluation_round'] = state['iterations']
    return acc_test, loss_test, diagnostics


def log_evaluation(state, dataset_test, args, output_path):
    acc_test, loss_test, diagnostics = evaluate_global_model(state, dataset_test, args)
    iterations = state['iterations']

    print('Round {:3d}, Test loss {:.3f}'.format(iterations, loss_test), flush=True)
    print('Round {:3d}, Test acc {:.3f}'.format(iterations, acc_test), flush=True)

    with open(output_path, 'a') as handle:
        handle.write(str(acc_test))
        handle.write('\n')
    return acc_test, loss_test, diagnostics


def log_validation_evaluation(state, args):
    dataset = state.get('validation_dataset')
    if dataset is None:
        return {'accuracy': '', 'loss': '', 'diagnostics': {}}
    accuracy, loss, diagnostics = evaluate_global_model(
        state, dataset, args, diagnostics_key='last_validation_diagnostics'
    )
    state['last_validation_accuracy'] = accuracy
    state['last_validation_loss'] = loss
    state['last_validation_round'] = state['iterations']
    return {'accuracy': accuracy, 'loss': loss, 'diagnostics': diagnostics}


def should_evaluate(state, args):
    return state['iterations'] % args.interval == 0


def run_training(
    args,
    state,
    dataset_train,
    dict_users,
    num_samples,
    dataset_test,
    output_path,
    direction_skew_path=None,
    system_metrics_path=None,
):
    train_start_time = time.perf_counter()
    if system_metrics_path and not tracemalloc.is_tracing():
        tracemalloc.start()

    while state["iterations"] < args.total_rounds:
        round_start_time = time.perf_counter()
        state = run_one_round(args, state, dataset_train, dict_users, num_samples)
        round_runtime_sec = time.perf_counter() - round_start_time
        log_direction_skew_metrics(args, state, direction_skew_path)
        if should_evaluate(state, args):
            log_evaluation(state, dataset_test, args, output_path)
            if state.get('validation_dataset') is not None:
                log_validation_evaluation(state, args)

        log_system_metrics(args, state, system_metrics_path, train_start_time, round_runtime_sec)
        gc.collect()

    return state


def main():
    args = args_parser()
    configure_runtime(args)

    state, dataset_train, dataset_test, dict_users, num_samples = build_training_state(args)
    output_path = build_output_path(args)
    direction_skew_path = setup_direction_skew_monitor(
        args=args,
        state=state,
        dataset_train=dataset_train,
        dict_users=dict_users,
        output_path=output_path,
    )
    system_metrics_path = build_system_metrics_path(args, output_path)

    state = run_training(
        args=args,
        state=state,
        dataset_train=dataset_train,
        dict_users=dict_users,
        num_samples=num_samples,
        dataset_test=dataset_test,
        output_path=output_path,
        direction_skew_path=direction_skew_path,
        system_metrics_path=system_metrics_path,
    )


if __name__ == '__main__':
    main()
