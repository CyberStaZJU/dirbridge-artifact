"""Weak 固定参考方向、原关联服务轨迹及真实 backend 选择 hook。

仅供探索性 Weak 协议；FIFO 是同一训练状态的选择反事实，不是独立训练对照。
"""

import builtins
import copy
from contextlib import contextmanager
import hashlib
import random
import sys
from unittest.mock import patch

import numpy as np
import torch

from utils.direction_balanced_buffer import select_buffer

REFERENCE_SEED = 2081041
MILD_RANGES = {'Small': (1.0, 2.0), 'Medium': (3.0, 5.0), 'Large': (5.0, 8.0)}
BACKENDS = {'FedBuff': 'fedbuff', 'DirBridge': 'dirbridge', 'FADAS': 'fadas',
            'CASA': 'casa', 'CA2FL': 'ca2fl', 'FedBuffMALight': 'fedbuffma', 'FedASMU': 'fedasmu'}


def original_tiers(label_groups, seed):
    labels = [int(group) for group in label_groups]
    if not labels or any(group not in range(3) for group in labels):
        raise ValueError('原关联服务轨迹需要三组标签分组')
    order = np.random.default_rng(int(seed) + 104729).permutation(3)
    mapping = {int(block): tier for block, tier in zip(order, MILD_RANGES)}
    return [mapping[group] for group in labels]


class ServiceBank:
    """按 (delay_seed, source_id, launch_index) 重算，原关联 source_id=client_id。"""

    def __init__(self, tiers, delay_seed):
        self.tiers = tuple(tiers)
        if not self.tiers or any(tier not in MILD_RANGES for tier in self.tiers):
            raise ValueError('未知服务层级')
        if isinstance(delay_seed, bool) or not isinstance(delay_seed, int) or delay_seed < 0:
            raise ValueError('delay_seed 必须为非负整数')
        self.delay_seed = delay_seed
        self.counts = [0] * len(self.tiers)
        self.events = []

    def recompute(self, source, index):
        if (type(source) is not int or not 0 <= source < len(self.tiers)
                or type(index) is not int or index < 0):
            raise ValueError('source/index 超出范围')
        rng = np.random.default_rng(np.random.SeedSequence([self.delay_seed, source, index]))
        return float(rng.uniform(*MILD_RANGES[self.tiers[source]]))

    def draw(self, client, state):
        if type(client) is not int or not 0 <= client < len(self.tiers):
            raise ValueError('client 超出范围')
        index = self.counts[client]
        duration = self.recompute(client, index)
        self.counts[client] += 1
        stamp = int(state.get('iterations', 0))
        clock = float(state.get('global_cost', 0))
        state.setdefault('stamp', [0] * len(self.tiers))[client] = stamp
        self.events.append(dict(client_id=client, source_id=client, launch_index=index,
                                tier=self.tiers[client], duration=duration, round=stamp,
                                launch_time=clock, completion_time=clock + duration))
        return duration


def _accelerator(device):
    device = torch.device(device)
    if device.type == 'cpu':
        return None
    backend = getattr(torch, device.type, None)
    if device.type not in ('cuda', 'npu') or backend is None:
        raise ValueError('参考探针仅支持 cpu 或已分配的 cuda/npu 设备')
    index = device.index if device.index is not None else backend.current_device()
    return backend, index


@contextmanager
def isolated_rng(seed, device='cpu'):
    """只触及分配的单个加速设备，成功或异常均恢复全局 RNG。"""
    python = random.getstate()
    numpy = np.random.get_state()
    cpu = torch.get_rng_state().clone()
    accelerator = _accelerator(device)
    device_rng = accelerator[0].get_rng_state(accelerator[1]).clone() if accelerator else None
    try:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.default_generator.manual_seed(seed)
        if accelerator:
            backend, index = accelerator
            with backend.device(index):
                backend.manual_seed(seed)
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if accelerator:
            accelerator[0].set_rng_state(device_rng, accelerator[1])


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def partition_record(state):
    return {str(client): [int(index) for index in state['dict_users'][client]]
            for client in range(len(state['num_samples']))}


def build_preinit_state(main, args):
    """跳过 main 唯一算法初始化点；finally 恢复，不启动 worker 或消费服务轨迹。"""
    with patch.object(main, 'init_state', lambda state, unused_args: state):
        result = main.build_training_state(args)
    if 'cost' in result[0] or 'iterations' in result[0]:
        raise RuntimeError('基础状态已提前执行算法初始化')
    return result


def build_reference(state, args, backend):
    """100 客户端均从相同参数和 BN 独立启动；探针不共享可变原状态。"""
    if int(args.num_users) != 100:
        raise ValueError('Weak 参考方向需要全部 100 客户端')
    if 'iterations' in state or 'cost' in state:
        raise ValueError('参考方向必须在算法初始化前生成')
    before = model_digest(state['net_glob'])
    modes = [module.training for module in state['net_glob'].modules()]
    partition = partition_record(state)
    if len(partition) != 100 or any(
            int(state['num_samples'][client]) <= 0
            or int(state['num_samples'][client]) != len(partition[str(client)]) for client in range(100)):
        raise ValueError('参考方向要求 100 个非空且计数一致的客户端分区')
    local_args = copy.deepcopy(args)
    local_args.lr = .01
    local_args.weight_decay = 0.
    local_args.local_period = 10
    local_args.local_epochs = None
    local_args.num_workers = 0
    features = []
    with isolated_rng(REFERENCE_SEED, args.device):
        plan = backend.build_count_sketch_plan(
            {name: tensor.detach().clone() for name, tensor in state['w_glob'].items()}, 2048, REFERENCE_SEED)
    for client in range(100):
        with isolated_rng(REFERENCE_SEED + client, args.device):
            local = dict(net_glob=copy.deepcopy(state['net_glob']),
                         w_glob={name: tensor.detach().clone() for name, tensor in state['w_glob'].items()},
                         dataset_train=state['dataset_train'],
                         dict_users=copy.deepcopy(state['dict_users']),
                         num_samples=copy.deepcopy(state['num_samples']), buffer_endpoints=[None] * 100)
            delta = backend.local_train_delta(local, copy.deepcopy(local_args), client)
            if delta.keys() != state['w_glob'].keys() or any(
                    tensor.shape != state['w_glob'][name].shape or not torch.isfinite(tensor).all()
                    for name, tensor in delta.items()):
                raise ValueError('无效参考客户端更新')
            feature = backend.encode_delta_raw_count_sketch(delta, plan, 2048).detach().float().cpu()
            if feature.shape != (2048,) or not torch.isfinite(feature).all():
                raise ValueError('参考方向必须是有限的 2048 维 CountSketch')
            norm = feature.norm()
            features.append((feature / norm if norm > 0 else feature).clone())
    with isolated_rng(REFERENCE_SEED, args.device):
        groups, centroids = backend.spherical_kmeans(torch.stack(features), 4, num_iters=20)
    groups = [int(group) for group in groups]
    if len(groups) != 100 or any(group not in range(4) for group in groups):
        raise ValueError('无效参考分组')
    if (model_digest(state['net_glob']) != before or partition_record(state) != partition
            or [module.training for module in state['net_glob'].modules()] != modes):
        raise RuntimeError('参考探针修改了原始模型、BN、模式或分区')
    return dict(group_ids=groups, population_weights=[groups.count(g) / 100 for g in range(4)],
                initial_model_sha256=before, partition=partition,
                protocol=dict(reference_seed=REFERENCE_SEED, phase='common_initial_model_before_algorithm_init',
                              clients=100, local_steps=10, lr=.01, weight_decay=0., groups=4,
                              sketch_dim=2048, kmeans_iters=20, client_seed_rule='reference_seed+client_id',
                              local_epochs=None, num_workers=0, count_sketch_seed=REFERENCE_SEED,
                              kmeans_seed=REFERENCE_SEED, population_weighting='uniform_clients'),
                isolation=dict(rng=True, parameters=True, bn=True, modes=True, service_bank_unconsumed=True))


@contextmanager
def planned_selection(module, state, record):
    """拦截限定 code frame 的真实 min selector；验证实际消费与单调时钟。"""
    selected = list(record['selected_ids'])
    raw = list(state['cost'])
    prior_clock = float(state['global_cost'])
    clock = float(record['decision_clock'])
    if (clock < prior_clock or len(selected) != 10 or len(set(selected)) != 10
            or selected != sorted(selected, key=lambda i: (raw[i], i))
            or not all(0 < raw[i] <= clock for i in selected)):
        raise ValueError('无效选择计划')
    present, previous = 'min' in vars(module), vars(module).get('min')
    fallback = previous if present else builtins.min
    selector_code = getattr(module, '_select_buffer', module.run_round).__code__
    calls = 0

    def scoped_min(*args, **kwargs):
        nonlocal calls
        if (calls < len(selected) and len(args) == 1 and set(kwargs) == {'key'}
                and callable(kwargs['key']) and isinstance(args[0], list)):
            candidates = args[0]
            expected = [(i, cost) for i, cost in enumerate(state['cost'])
                        if cost > 0 and i not in selected[:calls]]
            if (candidates == expected and candidates
                    and sys._getframe(1).f_code is selector_code
                    and (selector_code is module.run_round.__code__
                         or sys._getframe(2).f_code is module.run_round.__code__)):
                result = next(item for item in candidates if item[0] == selected[calls])
                calls += 1
                return result
        return fallback(*args, **kwargs)

    try:
        for client in selected:
            state['cost'][client] = clock
        module.min = scoped_min
        yield
        if calls != len(selected):
            raise RuntimeError('原生 selector 未完整消费计划')
        if state.get('last_buffer_list') != selected:
            raise RuntimeError('实际 buffer 与计划不同')
        if float(state['global_cost']) != clock:
            raise RuntimeError('实际时钟与计划不同')
        if any(state['cost'][client] != -1 for client in selected):
            raise RuntimeError('选中更新未消费或错误地立即重启')
        if any(state['cost'][i] != value for i, value in enumerate(raw) if value > 0 and i not in selected):
            raise RuntimeError('修改了未选中活跃更新的真实完成时间')
    except BaseException:
        state['cost'][:] = raw
        state['global_cost'] = prior_clock
        raise
    finally:
        if present:
            module.min = previous
        else:
            del module.min


class WeakScheduler:
    def __init__(self, groups):
        self.groups = list(groups)
        self.weights = [self.groups.count(g) / len(self.groups) for g in range(4)]
        self.state = None

    def step(self, module, args, state, run_round):
        raw, stamps = list(state['cost']), list(state['stamp'])
        number, clock = int(state['iterations']) + 1, float(state['global_cost'])
        decision, next_state = select_buffer(raw, stamps, self.groups, self.weights,
                                            number, clock, policy='weak', state=self.state)
        fifo, _ = select_buffer(raw, stamps, self.groups, self.weights, number, clock,
                                policy='fifo', state=self.state)
        if args.algo == 'FedBuffMALight' and any(
                number - 1 - stamps[client] > 20 for client in decision['selected_ids']):
            raise RuntimeError('MA-Light 选中 age>20，不能静默丢弃或继续额外消费')
        previous = copy.deepcopy(self.state)
        with planned_selection(module, state, decision):
            result = run_round()
        if int(state['iterations']) != number:
            raise RuntimeError('一次选择必须完成一个服务器更新轮次')
        self.state = next_state
        return result, dict(round=number, raw_costs=raw, stamps=stamps, prior_state=previous,
                            decision=decision, next_state=next_state,
                            fifo_counterfactual=fifo,
                            fifo_counterfactual_staleness=[number - 1 - stamps[i] for i in fifo['selected_ids']],
                            selected_staleness=[number - 1 - stamps[i] for i in decision['selected_ids']],
                            interpretation='same_state_selection_only_not_independent_training')
