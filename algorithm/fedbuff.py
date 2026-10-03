import copy
import numpy as np
import torch
try:
    import torch_npu
except ImportError:
    pass
import random
from models.local_update import LocalSGD
from utils.state_dict_ops import (
    sd_sub, model_param_dict, load_param_dict_, sd_average, sd_zero_like,
    init_buffer_endpoints, capture_buffer_endpoint, apply_buffer_endpoints_,
)
from utils.numerical import require_finite_tensor, validate_model


def init_state(state, args, random_cost):
    state['delta'] = [sd_zero_like(state['w_glob']) for _ in range(args.num_users)]
    state['stamp'] = [0] * int(args.num_users)
    state['cost'] = [-1] * int(args.num_users)
    state['iterations'] = 0
    state['last_update_l2_norm'] = 0.0
    init_buffer_endpoints(state, args.num_users)

    sampled_idx = random.sample(range(args.num_users), args.concurrency)
    for idx in sampled_idx:
        net = copy.deepcopy(state['net_glob']).to(args.device)
        w_start = model_param_dict(state['net_glob'])
        local = LocalSGD(
            args=args,
            dataset=state['dataset_train'],
            idxs=state['dict_users'][idx],
            iters=args.local_period,
            nums=state['num_samples'][idx],
        )
        w_local = local.train(net=net)
        delta = sd_sub(w_local, w_start)
        capture_buffer_endpoint(state, idx, net)
        del w_start, w_local, net

        state['delta'][idx] = delta
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = random_cost(idx)

    return state


def run_round(state, args, random_cost):
    active_list = [i for i, c in enumerate(state['cost']) if c > 0]
    buffer_list = []
    for _ in range(args.buffer_size):
        filtered = [(i, c) for i, c in enumerate(state['cost']) if i not in buffer_list and c > 0]
        idx, _ = min(filtered, key=lambda x: x[1])
        buffer_list.append(idx)
    state['global_cost'] = state['cost'][buffer_list[-1]]

    for idx in buffer_list:
        state['cost'][idx] = -1
    state['last_buffer_list'] = list(buffer_list)
    state['last_valid_buffer_list'] = list(buffer_list)
    state['last_invalid_buffer_list'] = []
    state['last_selected_item_summary'] = []
    state['last_regrouped'] = False

    for client in buffer_list:
        for key, value in state['delta'][client].items():
            require_finite_tensor(key, value, 'selected_update', client=client, round=state['iterations'])
    selected_updates = [state['delta'][i] for i in buffer_list]
    aggregated_diff = sd_average(selected_updates)

    with torch.no_grad():
        for key in aggregated_diff.keys():
            state['w_glob'][key] += aggregated_diff[key]
    load_param_dict_(state['net_glob'], state['w_glob'])
    apply_buffer_endpoints_(state, buffer_list)
    validate_model(state['net_glob'], 'server_aggregate', round=state['iterations'])
    state['iterations'] += 1

    sampled_idx = [i for i in range(args.num_users) if i not in active_list]
    np.random.shuffle(sampled_idx)
    sampled_idx = sampled_idx[:args.buffer_size]

    for idx in sampled_idx:
        net = copy.deepcopy(state['net_glob']).to(args.device)
        w_start = model_param_dict(state['net_glob'])
        local = LocalSGD(
            args=args,
            dataset=state['dataset_train'],
            idxs=state['dict_users'][idx],
            iters=args.local_period,
            nums=state['num_samples'][idx],
        )
        w_local = local.train(net=net)
        delta = sd_sub(w_local, w_start)
        capture_buffer_endpoint(state, idx, net)
        del w_start, w_local, net

        state['delta'][idx] = delta
        state['stamp'][idx] = state['iterations']
        state['cost'][idx] = state['global_cost'] + random_cost(idx)

    return state
