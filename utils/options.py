#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import sys


SUPPORTED_ALGORITHMS = [
    'FedBuff',
    'CA2FL',
    'FedBuffMA',
    'FedBuffMALight',
    'CASA',
    'FADAS',
    'FedASMU',
    'DirBridge',
]


def _flag_was_provided(argv, flags):
    return any(
        token == flag or token.startswith(flag + '=')
        for token in argv
        for flag in flags
    )


def _maybe_set_default(args, argv, attr, flags, value):
    if _flag_was_provided(argv, flags):
        return
    resolved = value(args) if callable(value) else value
    setattr(args, attr, resolved)


def _apply_overrides(args, argv, overrides):
    for attr, flags, value in overrides:
        _maybe_set_default(args, argv, attr, flags, value)
    return args


def _is_femnist_task(args):
    return str(args.dataset).lower() == 'femnist'


def _is_gspeech_task(args):
    return str(args.dataset).lower() == 'gspeech'


def _apply_femnist_defaults(args, argv):
    if not _is_femnist_task(args):
        return args
    args.dataset = 'femnist'
    return _apply_overrides(
        args,
        argv,
        (
            ('model', ['--model'], 'cnn'),
            ('num_users', ['--num_users'], 1000),
            ('femnist_min_samples_per_client', ['--femnist_min_samples_per_client'], 250),
            ('num_classes', ['--num_classes'], 62),
            ('num_channels', ['--num_channels'], 1),
        ),
    )


def _apply_gspeech_defaults(args, argv):
    if not _is_gspeech_task(args):
        return args
    args.dataset = 'gspeech'
    return _apply_overrides(
        args,
        argv,
        (
            ('model', ['--model'], 'melcnn'),
            ('num_users', ['--num_users'], 1000),
            ('gspeech_min_samples_per_client', ['--gspeech_min_samples_per_client'], 30),
            ('num_classes', ['--num_classes'], 35),
            ('num_channels', ['--num_channels'], 1),
        ),
    )


def _apply_casa_defaults(args, argv):
    if args.algo != 'CASA':
        return args
    return _apply_overrides(
        args,
        argv,
        (
            ('casa_num_groups', ['--casa_num_groups'], 4),
            ('casa_alpha0', ['--casa_alpha0'], 0.3),
            ('casa_partition_gamma', ['--casa_partition_gamma'], 0.15),
            ('casa_cluster_bias', ['--casa_cluster_bias'], 3.0),
            ('casa_eigengap_topk', ['--casa_eigengap_topk'], 10),
            ('casa_cluster_interval', ['--casa_cluster_interval'], 5),
            ('casa_min_cluster_size', ['--casa_min_cluster_size'], 8),
            ('casa_sketch_dim', ['--casa_sketch_dim'], 128),
            (
                'casa_buffer_budget',
                ['--casa_buffer_budget'],
                lambda parsed: max(int(parsed.num_users), int(parsed.buffer_size) * 4),
            ),
            ('casa_start_gap', ['--casa_start_gap'], lambda parsed: parsed.local_period),
        ),
    )


def _apply_dirbridge_defaults(args, argv):
    if args.algo != 'DirBridge':
        return args

    dataset_num_classes = {
        'cifar': 10,
        'cifar100': 100,
        'femnist': 62,
        'gspeech': 35,
        'tinyimagenet': 200,
    }

    def num_groups(parsed):
        num_classes = dataset_num_classes.get(
            str(parsed.dataset).lower(),
            int(getattr(parsed, 'num_classes', 10)),
        )
        return max(1, int(math.ceil(math.log2(float(max(1, int(num_classes)))))))

    return _apply_overrides(
        args,
        argv,
        (
            ('dirbridge_num_groups', ['--dirbridge_num_groups'], num_groups),
            ('dirbridge_feature_mode', ['--dirbridge_feature_mode'], 'count_sketch'),
            ('dirbridge_sketch_dim', ['--dirbridge_sketch_dim'], 2048),
            ('dirbridge_kmeans_iters', ['--dirbridge_kmeans_iters'], 20),
            ('dirbridge_recluster_interval', ['--dirbridge_recluster_interval'], 5),
            ('dirbridge_cache_beta', ['--dirbridge_cache_beta'], 0.9),
        ),
    )


def _apply_fedbuffma_defaults(args, argv):
    if args.algo not in ['FedBuffMA', 'FedBuffMALight']:
        return args
    _maybe_set_default(args, argv, 'ma_staleness_limit', ['--ma_staleness_limit'], 20)
    return args


def _apply_fadas_defaults(args, argv):
    if args.algo != 'FADAS':
        return args
    lr, global_lr = 0.1, 0.0001
    _maybe_set_default(args, argv, 'lr', ['--lr'], lr)
    _maybe_set_default(args, argv, 'global_lr', ['--global_lr'], global_lr)
    _maybe_set_default(args, argv, 'fadas_beta1', ['--fadas_beta1'], 0.9)
    _maybe_set_default(args, argv, 'fadas_beta2', ['--fadas_beta2'], 0.99)
    _maybe_set_default(args, argv, 'fadas_eps', ['--fadas_eps'], 1e-8)
    _maybe_set_default(
        args,
        argv,
        'fadas_tau_c',
        ['--fadas_tau_c'],
        lambda parsed: max(1.0, float(parsed.concurrency) / float(max(1, int(parsed.buffer_size)))),
    )
    return args


def _apply_algo_defaults(args, argv):
    args = _apply_femnist_defaults(args, argv)
    args = _apply_gspeech_defaults(args, argv)
    if (
        not _flag_was_provided(argv, ['--distribution'])
        and not _is_femnist_task(args)
        and not _is_gspeech_task(args)
        and getattr(args, 'random_cost', '') == 'dir-skew'
    ):
        args.distribution = 'noniid'

    for apply_defaults in (
        _apply_casa_defaults,
        _apply_dirbridge_defaults,
        _apply_fedbuffma_defaults,
        _apply_fadas_defaults,
    ):
        args = apply_defaults(args, argv)
    return args


def _add_misc_args(parser):
    group = parser.add_argument_group('Misc')
    group.add_argument('--momentum', type=float, default=0.0, help='SGD momentum used by local training')
    group.add_argument('--verbose', action='store_true', help='verbose print')
    group.add_argument('--num_workers', type=int, default=0, help='number of workers when loading data')
    group.add_argument('--output_dir', type=str, default='', help='directory for test accuracy output')
    group.add_argument('--run_tag', type=str, default='', help='optional tag appended to output filenames')
    group.add_argument('--direction_skew_log_dir', type=str, default='', help='directory for per-round direction-skew CSV logs')
    group.add_argument('--direction_skew_log_every', type=int, default=1, help='log direction-skew metrics every N server rounds')
    group.add_argument('--system_metrics_log_dir', type=str, default='', help='directory for per-round system-metrics CSV logs')
    group.add_argument('--direction_skew_groups', type=int, default=4, help='fixed monitor groups for direction-skew logging')


def _add_federated_args(parser):
    group = parser.add_argument_group('Federated')
    group.add_argument('--total_rounds', type=int, default=500, help='rounds of training')
    group.add_argument('--num_users', type=int, default=100, help='number of users')
    group.add_argument('--local_bs', type=int, default=100, help='local batch size')
    group.add_argument('--bs', type=int, default=500, help='test batch size')
    group.add_argument('--interval', type=int, default=1, help='number of rounds between evaluations')
    group.add_argument('--local_period', type=int, default=10, help='number of local SGD steps')
    group.add_argument('--local_epochs', type=float, default=None, help='local epochs; overrides local_period when positive')
    group.add_argument('--algo', type=str, default='FedBuff', choices=SUPPORTED_ALGORITHMS, help='federated learning algorithm')


def _add_model_and_runtime_args(parser):
    group = parser.add_argument_group('Model and Runtime')
    group.add_argument('--model', type=str, default='resnet', help='model name')
    group.add_argument('--dataset', type=str, default='cifar', help='dataset name')
    group.add_argument('--num_classes', type=int, default=10, help='number of classes')
    group.add_argument('--num_channels', type=int, default=3, help='number of input channels')
    group.add_argument('--gpu', type=str, default='0', help='GPU ID, or -1 for CPU')
    group.add_argument('--seed', type=int, default=123, help='random seed')


def _add_data_args(parser):
    group = parser.add_argument_group('Data')
    group.add_argument('--distribution', type=str, default='noniid', help='iid/noniid/label_block/label_correlated')
    group.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha for non-IID partitioning')
    group.add_argument('--label_block_groups', type=int, default=5, help='number of label-disjoint client groups')
    group.add_argument('--label_primary_fraction', type=float, default=0.5, help='fraction of each label-correlated group drawn from its primary label block')
    group.add_argument('--concurrency', type=int, default=20, help='number of concurrent clients')
    group.add_argument('--buffer_size', type=int, default=10, help='server buffer size')
    group.add_argument('--femnist_data_dir', type=str, default='./data/femnist_pt', help='FEMNIST data directory')
    group.add_argument('--femnist_min_samples_per_client', type=int, default=0, help='minimum train samples per FEMNIST client')
    group.add_argument('--gspeech_min_samples_per_client', type=int, default=0, help='minimum train samples per GSpeech client')
    group.add_argument(
        '--random_cost',
        type=str,
        default='dir-skew',
        choices=[
            'dir-skew',
            'fedscale_trace',
        ],
        help='client delay profile',
    )
    group.add_argument('--delay_gamma', type=float, default=1.0, help='delay-profile compatibility parameter')
    group.add_argument('--delay_seed', type=int, default=None, help='optional seed for fixed client delay groups')
    group.add_argument('--fedscale_client_profile_path', type=str, default='', help='FedScale client profile path')
    group.add_argument('--fedscale_availability_trace_path', type=str, default='', help='optional FedScale availability trace path')
    group.add_argument('--fedscale_time_scale', type=float, default=1.0, help='FedScale time multiplier')
    group.add_argument('--fedscale_default_duration', type=float, default=1.0, help='fallback FedScale client duration')
    group.add_argument('--fedscale_min_duration', type=float, default=1e-6, help='minimum positive FedScale duration')
    group.add_argument('--fedscale_upload_size_mb', type=float, default=1.0, help='FedScale upload size in MB')
    group.add_argument('--fedscale_download_size_mb', type=float, default=None, help='FedScale download size in MB')
    group.add_argument('--fedscale_batch_size', type=int, default=0, help='FedScale computation batch size')
    group.add_argument('--fedscale_local_steps', type=int, default=0, help='FedScale computation local steps')
    group.add_argument('--fedscale_augmentation_factor', type=float, default=3.0, help='FedScale backward-pass multiplier')
    group.add_argument('--fedscale_profile_sample', type=str, default='random', choices=['random', 'sorted'], help='FedScale profile sampling mode')
    group.add_argument('--fedscale_no_trace_wrap', action='store_true', help='do not cyclically wrap FedScale traces')
    group.add_argument('--fedscale_trace_exhausted_penalty', type=float, default=3600.0, help='FedScale exhausted-trace wait penalty')


def _add_optimization_args(parser):
    group = parser.add_argument_group('Optimization')
    group.add_argument('--lr', type=float, default=0.01, help='local learning rate')
    group.add_argument('--global_lr', type=float, default=None, help='server/global learning rate')
    group.add_argument('--weight_decay', type=float, default=0.0, help='weight decay used by local SGD')


def _add_grouping_args(parser):
    group = parser.add_argument_group('DirBridge')
    group.add_argument('--dirbridge_num_groups', type=int, default=5, help='number of DirBridge direction groups')
    group.add_argument('--dirbridge_feature_mode', type=str, default='count_sketch', choices=['count_sketch'], help='DirBridge grouping feature')
    group.add_argument('--dirbridge_sketch_dim', type=int, default=2048, help='DirBridge Count Sketch dimension')
    group.add_argument('--dirbridge_count_sketch_seed', type=int, default=None, help='DirBridge Count Sketch seed')
    group.add_argument('--dirbridge_kmeans_iters', type=int, default=20, help='spherical k-means iterations')
    group.add_argument('--dirbridge_recluster_interval', type=int, default=None, help='DirBridge reclustering interval')
    group.add_argument('--dirbridge_cache_beta', type=float, default=0.9, help='EMA retention factor for group caches')
    group.add_argument(
        '--dirbridge_ablation',
        type=str,
        default='none',
        choices=['none', 'no_ema_cache', 'random_grouping', 'no_staleness_filter'],
        help='single DirBridge ablation variant',
    )
    group.add_argument('--dirbridge_disable_ema_cache', action='store_true', help='ablation: disable EMA cache refill')
    group.add_argument('--dirbridge_random_grouping', action='store_true', help='ablation: use random grouping')
    group.add_argument('--dirbridge_disable_staleness_filter', action='store_true', help='ablation: aggregate all buffered updates')


def _add_momentum_approximation_args(parser):
    group = parser.add_argument_group('MA-Light')
    group.add_argument('--ma_p', type=float, default=0.0, help='staleness down-scaling exponent')
    group.add_argument('--ma_staleness_limit', type=int, default=None, help='maximum tolerated staleness; negative disables dropping')


def _add_fedasmu_args(parser):
    group = parser.add_argument_group('FedASMU')
    group.add_argument('--asmu_staleness_limit', type=int, default=None, help='maximum tolerated staleness tau')
    group.add_argument('--asmu_mu_alpha', type=float, default=1.0, help='server aggregation weight exponent')
    group.add_argument('--asmu_mu_beta', type=float, default=1.0, help='device-side fresh-model aggregation weight exponent')
    group.add_argument('--asmu_lambda0', type=float, default=1.0, help='initial lambda control')
    group.add_argument('--asmu_sigma0', type=float, default=1.0, help='initial sigma control')
    group.add_argument('--asmu_iota0', type=float, default=0.0, help='initial iota control')
    group.add_argument('--asmu_gamma0', type=float, default=1.0, help='initial gamma control')
    group.add_argument('--asmu_nu0', type=float, default=0.0, help='initial nu control')
    group.add_argument('--asmu_eta_lambda', type=float, default=0.01, help='lambda-control learning rate')
    group.add_argument('--asmu_eta_sigma', type=float, default=0.01, help='sigma-control learning rate')
    group.add_argument('--asmu_eta_iota', type=float, default=0.01, help='iota-control learning rate')
    group.add_argument('--asmu_eta_gamma', type=float, default=0.01, help='gamma-control learning rate')
    group.add_argument('--asmu_eta_nu', type=float, default=0.01, help='nu-control learning rate')
    group.add_argument('--asmu_eps_greedy', type=float, default=0.1, help='epsilon-greedy exploration rate')
    group.add_argument('--asmu_q_lr', type=float, default=0.1, help='Q-learning rate')
    group.add_argument('--asmu_q_gamma', type=float, default=0.9, help='Q-learning discount factor')


def _add_casa_args(parser):
    group = parser.add_argument_group('CASA')
    group.add_argument('--casa_num_groups', type=int, default=4, help='maximum number of CASA clusters')
    group.add_argument('--casa_alpha0', type=float, default=1.0, help='initial cluster-level decay')
    group.add_argument('--casa_omega', type=float, default=0.01, help='exponential decay factor')
    group.add_argument('--casa_buffer_budget', type=int, default=None, help='total similarity-buffer budget')
    group.add_argument('--casa_start_gap', type=int, default=None, help='maximum start-round gap')
    group.add_argument('--casa_partition_gamma', type=float, default=1.0, help='eigengap scaling')
    group.add_argument('--casa_max_splits', type=int, default=4, help='maximum sub-clusters from one split')
    group.add_argument('--casa_min_cluster_size', type=int, default=4, help='minimum eligible cluster size')
    group.add_argument('--casa_cluster_interval', type=int, default=1, help='round interval for repartition checks')
    group.add_argument('--casa_sketch_dim', type=int, default=256, help='low-dimensional sketch size')
    group.add_argument('--casa_kmeans_iters', type=int, default=20, help='number of k-means iterations')
    group.add_argument('--casa_cluster_bias', type=float, default=3.0, help='cluster-size bias in decay denominator')
    group.add_argument('--casa_eigengap_topk', type=int, default=10, help='number of eigenvalues considered')


def _add_fadas_args(parser):
    group = parser.add_argument_group('FADAS')
    group.add_argument('--fadas_beta1', type=float, default=0.9, help='first-moment coefficient')
    group.add_argument('--fadas_beta2', type=float, default=0.99, help='second-moment coefficient')
    group.add_argument('--fadas_eps', type=float, default=1e-8, help='numerical epsilon')
    group.add_argument('--fadas_tau_c', type=float, default=None, help='delay threshold for delay-adaptive FADAS')
    group.add_argument('--fadas_delay_adaptive', action='store_true', help='enable delay-adaptive FADAS learning rate')


def args_parser():
    parser = argparse.ArgumentParser()

    _add_misc_args(parser)
    _add_federated_args(parser)
    _add_model_and_runtime_args(parser)
    _add_data_args(parser)
    _add_optimization_args(parser)
    _add_grouping_args(parser)
    _add_momentum_approximation_args(parser)
    _add_fedasmu_args(parser)
    _add_casa_args(parser)
    _add_fadas_args(parser)

    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    return _apply_algo_defaults(args, argv)
