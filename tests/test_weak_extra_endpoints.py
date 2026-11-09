import copy
import random
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from algorithm import casa, fedasmu, fedbuffma
from builders import model_builder
from utils.options import args_parser
from utils.state_dict_ops import model_param_dict


torch.set_num_threads(1)


def _args(algo, enabled=False, **overrides):
    argv = [
        'test', '--algo', algo, '--num_users', '2', '--concurrency', '2',
        '--buffer_size', '2', '--local_bs', '4', '--local_period', '1',
        '--total_rounds', '10', '--momentum', '0.9',
    ]
    if enabled:
        argv.append('--weak_extra_baseline_endpoints')
    with patch.object(sys, 'argv', argv):
        args = args_parser()
    args.device = 'cpu'
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _state(args):
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
    with patch.object(model_builder, 'ResNet18', return_value=model):
        net, weights = model_builder.build_model(args)
    base = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    data = torch.cat([base + 10 * idx for idx in range(args.num_users)])
    labels = torch.tensor([0, 1, 0, 1]).repeat(args.num_users)
    return dict(
        net_glob=net, w_glob=weights, dataset_train=TensorDataset(data, labels),
        dict_users={idx: list(range(4 * idx, 4 * idx + 4)) for idx in range(args.num_users)},
        num_samples=[4] * args.num_users, global_cost=0.,
    )


def _init(module, state, args):
    with patch.object(module.random, 'sample', return_value=list(range(args.concurrency))):
        module.init_state(state, args, lambda idx: float(idx + 1))


def _assert_endpoint(model, endpoints):
    for name, buffer in model.named_buffers():
        values = [endpoint[name] for endpoint in endpoints]
        expected = torch.stack(values).mean(0) if buffer.is_floating_point() else torch.stack(values).amax(0)
        torch.testing.assert_close(buffer, expected)


@pytest.mark.parametrize('algo', ['FedBuffMALight', 'FedASMU'])
def test_flag_is_explicit_store_true(algo):
    assert _args(algo).weak_extra_baseline_endpoints is False
    assert _args(algo, enabled=True).weak_extra_baseline_endpoints is True


def test_flag_does_not_accept_implicit_boolean_values():
    with patch.object(sys, 'argv', ['test', '--algo', 'FedASMU', '--weak_extra_baseline_endpoints=true']):
        with pytest.raises(SystemExit):
            args_parser()


@pytest.mark.parametrize('algo', ['FedBuff', 'FedBuffMA', 'DirBridge'])
def test_flag_rejects_unsupported_algorithm(algo):
    with pytest.raises(SystemExit):
        _args(algo, enabled=True)


@pytest.mark.parametrize('algo', ['FedBuffMALight', 'FedASMU'])
@pytest.mark.parametrize('enabled,policy,expected', [
    (False, 'endpoint_mean', False), (True, 'endpoint_mean', True),
    (True, 'legacy_delta', False),
])
def test_builder_requires_both_opt_in_and_endpoint_policy(algo, enabled, policy, expected):
    args = _args(algo, enabled, bn_buffer_policy=policy)
    state = _state(args)
    assert state['net_glob']._asyncbuffer_trainable_state_only is expected
    assert args.resolved_bn_buffer_policy == ('endpoint_mean' if expected else 'legacy_delta')
    qualification = 'endpoint_repair' if expected else ('diagnostic_reference_only' if enabled else 'unqualified')
    assert args.bn_buffer_policy_qualification == qualification
    assert ('0.running_mean' not in state['w_glob']) is expected


@pytest.mark.parametrize('module,algo', [(fedbuffma, 'FedBuffMALight'), (fedasmu, 'FedASMU')])
def test_absent_flag_matches_explicit_false_historical_path(module, algo):
    results = []
    for omit_flag in (False, True):
        args = _args(algo, asmu_eta_lambda=0., asmu_eta_sigma=0., asmu_eta_iota=0.)
        if omit_flag:
            del args.weak_extra_baseline_endpoints
        state = _state(args)
        _init(module, state, args)
        module.run_round(state, args, lambda idx: float(idx + 1))
        assert 'buffer_endpoints' not in state
        assert 'start_buffers' not in state
        assert int(state['net_glob'][0].num_batches_tracked) == 0
        assert '0.running_mean' in state['w_glob']
        expected_mean = torch.tensor([0.9, 1.0]) if module is fedbuffma else torch.tensor([0.8, 0.875])
        torch.testing.assert_close(state['net_glob'][0].running_mean, expected_mean)
        results.append(copy.deepcopy(state['net_glob'].state_dict()))
    for key in results[0]:
        assert torch.equal(results[0][key], results[1][key])


def test_ma_captures_real_local_bn_and_applies_uniform_accepted_endpoints():
    args = _args('FedBuffMALight', enabled=True)
    state = _state(args)
    initial_params = model_param_dict(state['net_glob'])
    _init(fedbuffma, state, args)
    endpoints = copy.deepcopy(state['buffer_endpoints'])
    for idx, endpoint in enumerate(endpoints):
        torch.testing.assert_close(endpoint['0.running_mean'], torch.tensor([0.4, 0.5]) + idx)
        assert int(endpoint['0.num_batches_tracked']) == 1
    assert torch.count_nonzero(state['net_glob'][0].running_mean) == 0
    fedbuffma.run_round(state, args, lambda idx: float(idx + 1))
    _assert_endpoint(state['net_glob'], endpoints)
    assert any(not torch.equal(initial_params[key], value) for key, value in state['w_glob'].items())


def test_ma_filters_rejected_endpoint_before_validation_and_application():
    args = _args('FedBuffMALight', enabled=True, ma_staleness_limit=1)
    state = _state(args)
    _init(fedbuffma, state, args)
    state['iterations'] = 2
    state['stamp'] = [0, 2]
    state['buffer_endpoints'][0]['0.running_var'].fill_(float('nan'))
    accepted = copy.deepcopy(state['buffer_endpoints'][1])
    fedbuffma.run_round(state, args, lambda idx: float(idx + 1))
    assert state['last_invalid_buffer_list'] == [0]
    assert state['last_valid_buffer_list'] == [1]
    _assert_endpoint(state['net_glob'], [accepted])


@pytest.mark.parametrize('reject_all', [False, True])
def test_ma_captures_endpoint_on_both_refill_training_branches(reject_all):
    args = _args('FedBuffMALight', enabled=True, num_users=3, buffer_size=1, ma_staleness_limit=1)
    state = _state(args)
    _init(fedbuffma, state, args)
    assert state['buffer_endpoints'][2] is None
    if reject_all:
        state['iterations'] = 3
    fedbuffma.run_round(state, args, lambda idx: float(idx + 1))
    assert state['buffer_endpoints'][2] is not None
    expected = state['net_glob'][0].running_mean * 0.9 + torch.tensor([2.4, 2.5])
    torch.testing.assert_close(state['buffer_endpoints'][2]['0.running_mean'], expected)
    expected_count = int(state['net_glob'][0].num_batches_tracked) + 1
    assert int(state['buffer_endpoints'][2]['0.num_batches_tracked']) == expected_count
    if reject_all:
        assert state['last_valid_buffer_list'] == []
        assert torch.count_nonzero(state['net_glob'][0].running_mean) == 0


def test_asmu_captures_completed_replay_before_delete_and_restores_launch_buffers():
    args = _args('FedASMU', enabled=True)
    state = _state(args)
    _init(fedasmu, state, args)
    assert state['buffer_endpoints'] == [None, None]
    state['net_glob'][0].running_mean.fill_(50.)
    state['net_glob'][0].running_var.fill_(30.)
    state['net_glob'][0].num_batches_tracked.fill_(9)
    state['iterations'] = 1
    fedasmu.run_round(state, args, lambda idx: float(idx + 1))
    endpoints = state['buffer_endpoints']
    for idx, endpoint in enumerate(endpoints):
        torch.testing.assert_close(endpoint['0.running_mean'], torch.tensor([0.4, 0.5]) + idx)
        assert int(endpoint['0.num_batches_tracked']) == 1
    _assert_endpoint(state['net_glob'], endpoints)
    torch.testing.assert_close(state['net_glob'][0].running_mean, torch.tensor([0.9, 1.0]))


def test_asmu_only_trains_and_applies_accepted_clients():
    args = _args('FedASMU', enabled=True, asmu_staleness_limit=1)
    state = _state(args)
    _init(fedasmu, state, args)
    state['iterations'] = 2
    state['stamp'] = [0, 2]
    state['buffer_endpoints'][0] = {'invalid': torch.tensor(float('nan'))}
    with patch.object(fedasmu, '_train_completed_client', wraps=fedasmu._train_completed_client) as train:
        fedasmu.run_round(state, args, lambda idx: float(idx + 1))
    assert train.call_count == 1
    assert train.call_args.kwargs['idx'] == 1
    assert state['last_buffer_list'] == [0, 1]
    assert state['last_invalid_buffer_list'] == [0]
    assert state['last_valid_buffer_list'] == [1]
    _assert_endpoint(state['net_glob'], [state['buffer_endpoints'][1]])


def test_asmu_all_rejected_does_not_apply_or_replay():
    args = _args('FedASMU', enabled=True, asmu_staleness_limit=1)
    state = _state(args)
    _init(fedasmu, state, args)
    state['iterations'] = 2
    before = copy.deepcopy(state['net_glob'].state_dict())
    with patch.object(fedasmu, '_train_completed_client') as train, patch.object(fedasmu, 'apply_buffer_endpoints_') as apply:
        fedasmu.run_round(state, args, lambda idx: float(idx + 1))
    train.assert_not_called()
    apply.assert_not_called()
    for key, value in before.items():
        assert torch.equal(value, state['net_glob'].state_dict()[key])


@pytest.mark.parametrize('module,algo', [(fedbuffma, 'FedBuffMALight'), (fedasmu, 'FedASMU')])
def test_enabled_aggregate_validates_nonfinite_endpoint(module, algo):
    args = _args(algo, enabled=True)
    state = _state(args)
    _init(module, state, args)
    if module is fedbuffma:
        state['buffer_endpoints'][0]['0.running_mean'].fill_(float('nan'))
        with pytest.raises(FloatingPointError):
            module.run_round(state, args, lambda idx: float(idx + 1))
    else:
        original_train = module.LocalFedASMU.train

        def invalid_train(local, net):
            result = original_train(local, net)
            net[0].running_var.fill_(-1.)
            return result

        with patch.object(module.LocalFedASMU, 'train', invalid_train), pytest.raises(FloatingPointError):
            module.run_round(state, args, lambda idx: float(idx + 1))


def test_casa_endpoint_policy_captures_real_bn_and_aggregates_selected_clients():
    args = _args('CASA', num_users=3)
    state = _state(args)
    assert not args.weak_extra_baseline_endpoints
    assert args.resolved_bn_buffer_policy == 'endpoint_mean'
    assert args.bn_buffer_policy_qualification == 'endpoint_repair'
    _init(casa, state, args)
    endpoints = copy.deepcopy(state['buffer_endpoints'])
    for idx, endpoint in enumerate(endpoints):
        torch.testing.assert_close(endpoint['0.running_mean'], torch.tensor([0.4, 0.5]) + idx)
        assert int(endpoint['0.num_batches_tracked']) == 1
    state['buffer_endpoints'][2]['0.running_var'].fill_(float('nan'))
    initial_weights = copy.deepcopy(state['w_glob'])
    casa.run_round(state, args, lambda idx: float(idx + 1))
    assert state['last_buffer_list'] == [0, 1]
    _assert_endpoint(state['net_glob'], endpoints[:2])
    torch.testing.assert_close(state['net_glob'][0].running_mean, torch.tensor([0.9, 1.0]))
    assert any(not torch.equal(initial_weights[key], value) for key, value in state['w_glob'].items())
    torch.testing.assert_close(state['buffer_endpoints'][2]['0.running_mean'], torch.tensor([3.21, 3.4]))
    assert int(state['buffer_endpoints'][2]['0.num_batches_tracked']) == 2


def test_casa_legacy_policy_matches_original_hooks_disabled():
    results = []
    for disable_hooks in (False, True):
        args = _args('CASA', bn_buffer_policy='legacy_delta')
        state = _state(args)
        assert args.resolved_bn_buffer_policy == 'legacy_delta'
        assert args.bn_buffer_policy_qualification == 'diagnostic_reference_only'
        if disable_hooks:
            with patch.object(casa, 'init_buffer_endpoints'), patch.object(casa, 'capture_buffer_endpoint'), patch.object(casa, 'apply_buffer_endpoints_'):
                _init(casa, state, args)
                casa.run_round(state, args, lambda idx: float(idx + 1))
        else:
            _init(casa, state, args)
            casa.run_round(state, args, lambda idx: float(idx + 1))
        assert 'buffer_endpoints' not in state
        assert int(state['net_glob'][0].num_batches_tracked) == 0
        torch.testing.assert_close(state['net_glob'][0].running_mean, torch.tensor([0.9, 1.0]))
        results.append(copy.deepcopy(state['net_glob'].state_dict()))
    for key in results[0]:
        assert torch.equal(results[0][key], results[1][key])


def test_casa_selected_endpoint_is_validated():
    args = _args('CASA')
    state = _state(args)
    _init(casa, state, args)
    state['buffer_endpoints'][0]['0.running_var'].fill_(-1.)
    with pytest.raises(FloatingPointError):
        casa.run_round(state, args, lambda idx: float(idx + 1))
