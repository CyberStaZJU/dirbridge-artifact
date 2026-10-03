import copy
import math
import random

import numpy as np
import torch
try:
    import torch_npu
except ImportError:
    pass
from torch import nn
from torch.utils.data import DataLoader

from models.local_update import DatasetSplit, LocalFedASMU
from utils.state_dict_ops import load_param_dict_, model_param_dict, sd_copy, sd_lerp


# Repo-compatible buffered FedASMU:
# 1. keep the existing completion-cost simulator instead of adding true threaded execution,
# 2. replay local training at completion time so a stale client can merge one fresh global model,
# 3. consume a FedBuff-style buffer of finished clients per round for fairer comparisons,
# 4. use a per-client Q-table for request-slot adaptation in place of the paper's LSTM meta-initializer.

ACTION_MINUS = 0
ACTION_STAY = 1
ACTION_ADD = 2
ACTION_DELTAS = {
    ACTION_MINUS: -1,
    ACTION_STAY: 0,
    ACTION_ADD: 1,
}


def _num_local_steps(args):
    return max(1, int(args.local_period))


def _mid_slot(args):
    steps = _num_local_steps(args)
    return max(1, (steps + 1) // 2)


def _apply_action(slot, action, max_slot):
    next_slot = slot + ACTION_DELTAS[action]
    return int(min(max(next_slot, 1), max_slot))


def _select_buffer(costs, buffer_size):
    buffer_list = []
    for _ in range(buffer_size):
        filtered = [
            (idx, cost)
            for idx, cost in enumerate(costs)
            if idx not in buffer_list and cost >= 0
        ]
        if not filtered:
            break
        idx, _ = min(filtered, key=lambda item: item[1])
        buffer_list.append(idx)
    return buffer_list


def _alpha_from_controls(round_idx, staleness, lambda_ctrl, sigma_ctrl, iota_ctrl, mu_alpha):
    round_term = math.sqrt(max(1.0, float(round_idx)))
    stale_term = math.pow(max(1.0, float(staleness)), max(0.0, float(sigma_ctrl)) / 2.0)
    xi = float(lambda_ctrl) / (round_term * stale_term) + float(iota_ctrl)
    scaled = float(mu_alpha) * xi
    if scaled <= 0:
        return 0.0
    alpha = scaled / (1.0 + scaled)
    return float(max(0.0, min(1.0, alpha)))


def _sample_batch(args, dataset, idxs):
    subset = DatasetSplit(dataset, idxs)
    if len(subset) == 0:
        return None

    loader = DataLoader(
        subset,
        batch_size=min(args.local_bs, len(subset)),
        shuffle=True,
        num_workers=args.num_workers,
    )
    return next(iter(loader))


def _evaluate_loss(net_template, params, batch, device):
    if batch is None:
        return 0.0

    net = copy.deepcopy(net_template).to(device)
    load_param_dict_(net, params)
    criterion = nn.CrossEntropyLoss()
    was_training = net.training
    net.eval()

    with torch.no_grad():
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        logits = net(images)
        loss = criterion(logits, labels).item()

    if was_training:
        net.train()
    del net
    return float(loss)


def _pick_request_slot(state, args, idx):
    max_slot = _num_local_steps(args)
    current_slot = int(state['slot_pref'][idx])

    if state['slot_count'][idx] == 0:
        return current_slot, ACTION_STAY, current_slot

    if random.random() < float(args.asmu_eps_greedy):
        action = random.choice([ACTION_MINUS, ACTION_STAY, ACTION_ADD])
    else:
        action = int(np.argmax(state['slot_q'][idx][current_slot - 1]))

    next_slot = _apply_action(current_slot, action, max_slot)
    return current_slot, action, next_slot


def _launch_client(idx, state, args, start_time):
    anchor_slot, action, request_slot = _pick_request_slot(state, args, idx)
    state['start_models'][idx] = sd_copy(state['w_glob'])
    state['stamp'][idx] = state['iterations']
    state['cost'][idx] = start_time + state['random_cost'](idx)
    state['slot_anchor'][idx] = anchor_slot
    state['slot_action'][idx] = action
    state['request_slot'][idx] = request_slot


def _update_slot_q(state, args, idx, reward):
    anchor_slot = int(state['slot_anchor'][idx])
    action = int(state['slot_action'][idx])
    next_slot = int(state['request_slot'][idx])
    q_values = state['slot_q'][idx]

    current_q = q_values[anchor_slot - 1, action]
    next_q = float(np.max(q_values[next_slot - 1]))
    target = reward + float(args.asmu_q_gamma) * next_q
    q_values[anchor_slot - 1, action] = current_q + float(args.asmu_q_lr) * (target - current_q)

    state['slot_pref'][idx] = next_slot
    state['slot_count'][idx] += 1


def _update_server_controls(state, args, idx, reward):
    state['lambda_ctrl'][idx] = max(
        0.0,
        float(state['lambda_ctrl'][idx]) + float(args.asmu_eta_lambda) * reward,
    )
    state['sigma_ctrl'][idx] = max(
        0.0,
        float(state['sigma_ctrl'][idx]) - float(args.asmu_eta_sigma) * reward,
    )
    state['iota_ctrl'][idx] = max(
        0.0,
        float(state['iota_ctrl'][idx]) + float(args.asmu_eta_iota) * reward,
    )


def _train_completed_client(idx, state, args, fresh_global, fresh_version):
    net = copy.deepcopy(state['net_glob']).to(args.device)
    load_param_dict_(net, state['start_models'][idx])

    local = LocalFedASMU(
        args=args,
        dataset=state['dataset_train'],
        idxs=state['dict_users'][idx],
        iters=args.local_period,
        nums=state['num_samples'][idx],
        fresh_global=sd_copy(fresh_global),
        start_version=state['stamp'][idx],
        fresh_version=fresh_version,
        request_step=state['request_slot'][idx],
        gamma_ctrl=state['gamma_ctrl'][idx],
        nu_ctrl=state['nu_ctrl'][idx],
        mu_beta=args.asmu_mu_beta,
        eta_gamma=args.asmu_eta_gamma,
        eta_nu=args.asmu_eta_nu,
    )
    w_local = local.train(net=net)

    state['gamma_ctrl'][idx] = float(local.gamma_ctrl)
    state['nu_ctrl'][idx] = float(local.nu_ctrl)
    reward = float(local.last_reward) if local.did_merge else 0.0
    _update_slot_q(state, args, idx, reward)

    del net
    return w_local


def init_state(state, args, random_cost):
    state['random_cost'] = random_cost
    state['cost'] = [-1.0] * int(args.num_users)
    state['stamp'] = [0] * int(args.num_users)
    state['start_models'] = [None for _ in range(args.num_users)]
    state['iterations'] = 0
    state['global_cost'] = 0.0

    state['lambda_ctrl'] = [float(args.asmu_lambda0) for _ in range(args.num_users)]
    state['sigma_ctrl'] = [float(args.asmu_sigma0) for _ in range(args.num_users)]
    state['iota_ctrl'] = [float(args.asmu_iota0) for _ in range(args.num_users)]
    state['gamma_ctrl'] = [float(args.asmu_gamma0) for _ in range(args.num_users)]
    state['nu_ctrl'] = [float(args.asmu_nu0) for _ in range(args.num_users)]

    init_slot = _mid_slot(args)
    max_slot = _num_local_steps(args)
    state['slot_pref'] = [init_slot for _ in range(args.num_users)]
    state['slot_anchor'] = [init_slot for _ in range(args.num_users)]
    state['request_slot'] = [init_slot for _ in range(args.num_users)]
    state['slot_action'] = [ACTION_STAY for _ in range(args.num_users)]
    state['slot_count'] = [0 for _ in range(args.num_users)]
    state['slot_q'] = [np.zeros((max_slot, 3), dtype=np.float32) for _ in range(args.num_users)]

    sampled_idx = random.sample(range(args.num_users), args.concurrency)
    for idx in sampled_idx:
        _launch_client(idx, state, args, start_time=0.0)

    return state


def run_round(state, args, random_cost):
    del random_cost

    active_list = [idx for idx, cost in enumerate(state['cost']) if cost >= 0]
    if not active_list:
        refill = min(args.buffer_size, args.num_users)
        sampled_idx = random.sample(range(args.num_users), refill)
        for idx in sampled_idx:
            _launch_client(idx, state, args, start_time=state['global_cost'])
        return state

    buffer_list = _select_buffer(state['cost'], args.buffer_size)
    if not buffer_list:
        return state

    current_version = int(state['iterations'])
    stale_limit = getattr(args, 'asmu_staleness_limit', None)
    if stale_limit is None:
        stale_limit = max(1, int(args.concurrency))

    state['global_cost'] = float(state['cost'][buffer_list[-1]])
    shared_fresh_global = sd_copy(state['w_glob'])
    aggregated_global = sd_copy(state['w_glob'])
    round_idx = current_version + 1

    for idx in buffer_list:
        state['cost'][idx] = -1.0

    for idx in buffer_list:
        staleness = current_version - int(state['stamp'][idx]) + 1
        if staleness > int(stale_limit):
            continue

        w_local = _train_completed_client(
            idx=idx,
            state=state,
            args=args,
            fresh_global=shared_fresh_global,
            fresh_version=current_version,
        )

        batch = _sample_batch(args, state['dataset_train'], state['dict_users'][idx])
        alpha = _alpha_from_controls(
            round_idx=round_idx,
            staleness=staleness,
            lambda_ctrl=state['lambda_ctrl'][idx],
            sigma_ctrl=state['sigma_ctrl'][idx],
            iota_ctrl=state['iota_ctrl'][idx],
            mu_alpha=args.asmu_mu_alpha,
        )

        mixed_before = sd_lerp(aggregated_global, w_local, alpha)
        loss_before = _evaluate_loss(state['net_glob'], aggregated_global, batch, args.device)
        loss_after = _evaluate_loss(state['net_glob'], mixed_before, batch, args.device)
        reward = loss_before - loss_after
        _update_server_controls(state, args, idx, reward)

        alpha = _alpha_from_controls(
            round_idx=round_idx,
            staleness=staleness,
            lambda_ctrl=state['lambda_ctrl'][idx],
            sigma_ctrl=state['sigma_ctrl'][idx],
            iota_ctrl=state['iota_ctrl'][idx],
            mu_alpha=args.asmu_mu_alpha,
        )
        aggregated_global = sd_lerp(aggregated_global, w_local, alpha)

    state['w_glob'] = aggregated_global
    load_param_dict_(state['net_glob'], state['w_glob'])

    state['iterations'] += 1

    sampled_idx = [i for i in range(args.num_users) if i not in active_list]
    np.random.shuffle(sampled_idx)
    sampled_idx = sampled_idx[:len(buffer_list)]
    for idx in sampled_idx:
        _launch_client(idx, state, args, start_time=state['global_cost'])

    return state

# We implemented FedASMU by extending our existing FedBuff-style asynchronous federated learning framework. 
# Our implementation retains the same two conceptual components as the original method: a dynamic staleness-aware server update and an adaptive local model adjustment with one fresh global-model merge during local training. 
# However, to make the method compatible with our codebase and to enable a fair comparison with buffered baselines, we adopted a buffered variant of FedASMU.
# Specifically, instead of updating the global model immediately after each single client upload as in the original event-driven design, the server aggregates the first \(B\) completed client updates in each round. 
# In addition, because our simulator is step-based rather than thread-based, the epoch-level request slot in the original paper is mapped to local SGD steps, and the fresh-model request is emulated by replaying local training against a shared pre-buffer global snapshot. 
# We also replace the original LSTM-based meta-initializer with a lightweight per-client epsilon-greedy Q-learning policy for request-slot selection. Finally, the control-parameter updates derived from loss partial derivatives in the original formulation are approximated in our implementation by reward-driven updates based on the mini-batch loss change before and after local/global model mixing. 
# Therefore, our method should be viewed as a codebase-compatible buffered implementation of FedASMU rather than an exact system-level reproduction of the original paper.
