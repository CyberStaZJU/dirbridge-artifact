import copy
import importlib
import io
import json
import random
import tempfile
from pathlib import Path
from contextlib import ExitStack, redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts.run_weak_experiments import external_root, main as runner_main, matrix, native_args
from utils.state_dict_ops import model_param_dict
from utils.weak_protocol import (
    BACKENDS, REFERENCE_SEED, ServiceBank, WeakScheduler, build_preinit_state,
    build_reference, isolated_rng, model_digest, original_tiers, planned_selection,
)


torch.set_num_threads(1)


class WeakProtocolTests(unittest.TestCase):
    def test_bank_exact_seedsequence_launch_prefix_independent_of_order(self):
        tiers = ['Small', 'Medium', 'Large'] * 10
        first, second = ServiceBank(tiers, 3), ServiceBank(tiers, 3)
        state = {'iterations': 0, 'global_cost': 0.}
        for client in [2, 0, 2, 1]:
            first.draw(client, state)
        for client in [0, 1, 2, 2]:
            second.draw(client, state)
        self.assertEqual(first.counts, second.counts)
        self.assertEqual([e['duration'] for e in first.events if e['client_id'] == 2],
                         [e['duration'] for e in second.events if e['client_id'] == 2])
        expected = np.random.default_rng(np.random.SeedSequence([3, 2, 0])).uniform(5., 8.)
        self.assertEqual(first.events[0]['duration'], expected)
        self.assertEqual(first.events[2]['launch_index'], 1)
        snapshot = copy.deepcopy(first.counts)
        first.recompute(2, 100)
        self.assertEqual(first.counts, snapshot)

    def test_original_association_uses_frozen_block_seed_rule(self):
        order = np.random.default_rng(5 + 104729).permutation(3)
        tiers = original_tiers([0, 1, 2], 5)
        for rank, block in enumerate(order):
            self.assertEqual(tiers[int(block)], ['Small', 'Medium', 'Large'][rank])

    def test_rng_isolation_restores_on_failure_and_does_not_seed_all_devices(self):
        python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        with patch.object(torch, 'manual_seed', side_effect=AssertionError('禁止全部加速器重播种')):
            with self.assertRaisesRegex(RuntimeError, '预期失败'):
                with isolated_rng(REFERENCE_SEED):
                    random.random()
                    np.random.random()
                    torch.rand(3)
                    raise RuntimeError('预期失败')
        self.assertEqual(random.getstate(), python)
        self.assertEqual(np.random.get_state()[0], numpy[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy[1])
        self.assertEqual(np.random.get_state()[2:], numpy[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu))

    def test_assigned_accelerator_rng_does_not_touch_other_devices(self):
        from contextlib import nullcontext
        backend = SimpleNamespace(get_rng_state=lambda index: torch.tensor([index]),
                                  set_rng_state=unittest.mock.Mock(),
                                  manual_seed=unittest.mock.Mock(), device=lambda index: nullcontext())
        with patch('utils.weak_protocol._accelerator', return_value=(backend, 3)):
            with isolated_rng(17, 'cuda:3'):
                backend.manual_seed.assert_called_once_with(17)
        backend.set_rng_state.assert_called_once()
        self.assertTrue(torch.equal(backend.set_rng_state.call_args.args[0], torch.tensor([3])))
        self.assertEqual(backend.set_rng_state.call_args.args[1], 3)

    def test_preinit_skips_real_init_and_always_restores_hook(self):
        calls = []
        main = SimpleNamespace(init_state=lambda *a: calls.append('initialized'))
        main.build_training_state = lambda args: (main.init_state({'net': 1}, args),)
        original = main.init_state
        self.assertEqual(build_preinit_state(main, object())[0], {'net': 1})
        self.assertEqual(calls, [])
        self.assertIs(main.init_state, original)
        main.build_training_state = lambda args: (_ for _ in ()).throw(RuntimeError('故意失败'))
        with self.assertRaises(RuntimeError):
            build_preinit_state(main, object())
        self.assertIs(main.init_state, original)

    def test_reference_preserves_model_bn_modes_args_rng_and_unconsumed_bank(self):
        from algorithm import dirbridge
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        model._asyncbuffer_trainable_state_only = True
        model[1].eval()
        state = dict(net_glob=model, w_glob=model_param_dict(model), dataset_train=[],
                     dict_users={i: [i] for i in range(100)}, num_samples=[1] * 100)
        args = SimpleNamespace(num_users=100, lr=.2, weight_decay=.7, local_period=2,
                               local_bs=100, device='cpu', seed=1)
        bank = ServiceBank(['Small'] * 100, 1)
        before, modes = model_digest(model), [m.training for m in model.modules()]
        original_args, original_partition = copy.deepcopy(vars(args)), copy.deepcopy(state['dict_users'])
        rng = torch.get_rng_state().clone()
        starts, seeds = [], []

        def train(local, local_args, client):
            self.assertEqual(local_args.lr, .01)
            self.assertEqual(local_args.weight_decay, 0.)
            self.assertEqual(local_args.local_period, 10)
            starts.append(model_digest(local['net_glob']))
            seeds.append(torch.initial_seed())
            random.random()
            np.random.random()
            with torch.no_grad():
                for tensor in local['net_glob'].state_dict().values():
                    tensor.add_(1)
            local['dict_users'][client].append(999)
            return {name: torch.ones_like(tensor) * (client + 1) for name, tensor in local['w_glob'].items()}

        with patch.object(dirbridge, 'local_train_delta', side_effect=train):
            reference = build_reference(state, args, dirbridge)
        self.assertEqual(len(reference['group_ids']), 100)
        self.assertTrue(all(value == before for value in starts))
        self.assertEqual(seeds, list(range(REFERENCE_SEED, REFERENCE_SEED + 100)))
        self.assertEqual(model_digest(model), before)
        self.assertEqual([m.training for m in model.modules()], modes)
        self.assertEqual(vars(args), original_args)
        self.assertEqual(state['dict_users'], original_partition)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(bank.events, [])
        self.assertEqual(bank.counts, [0] * 100)
        with self.assertRaises(ValueError):
            build_reference(dict(state, iterations=0), args, dirbridge)

    def test_default_runner_is_70_run_dry_plan_without_backend_import(self):
        output = io.StringIO()
        with redirect_stdout(output), patch('subprocess.run') as run:
            runner_main([])
        value = json.loads(output.getvalue())
        self.assertEqual(value['run_count'], 70)
        self.assertEqual(len(matrix()), 70)
        self.assertEqual(len({(v['alpha'], v['seed'], v['algorithm']) for v in matrix()}), 70)
        self.assertEqual(value['mode'], 'dry_run')
        self.assertEqual(value['reference_seed'], REFERENCE_SEED)
        run.assert_not_called()

    def test_output_parent_must_be_existing_external_absolute_directory(self):
        from utils.weak_protocol import __file__ as protocol_file
        root = Path(protocol_file).resolve().parents[1]
        for value in ('relative', str(root), str(root / 'tests')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                external_root(value)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(external_root(directory), Path(directory).resolve())
            link = Path(directory) / 'source'
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                external_root(str(link))

    def test_all_seven_native_argument_profiles_parse(self):
        for algorithm in BACKENDS:
            with self.subTest(algorithm=algorithm):
                args = native_args(.1, 1, algorithm, '-1')
                self.assertEqual(args.algo, algorithm)
                self.assertEqual((args.num_users, args.concurrency, args.buffer_size), (100, 40, 10))
                self.assertEqual(args.total_rounds, 500)
                self.assertEqual(args.local_period, 10)

    def test_seven_profiles_resolve_real_model_endpoint_qualification(self):
        from builders.model_builder import build_model
        for algorithm in BACKENDS:
            with self.subTest(algorithm=algorithm):
                args = native_args(.1, 1, algorithm, '-1')
                args.device = 'cpu'
                with patch('builders.model_builder.ResNet18', side_effect=lambda: nn.Linear(2, 2)):
                    model, _ = build_model(args)
                self.assertEqual(args.resolved_bn_buffer_policy, 'endpoint_mean')
                self.assertTrue(model._asyncbuffer_trainable_state_only)
                if algorithm in ('FedBuffMALight', 'FedASMU'):
                    self.assertTrue(args.weak_extra_baseline_endpoints)

    def test_nonfifo_plan_enables_seven_actual_backend_selectors(self):
        # 保留真实 run_round、聚合与补启动逻辑；仅把本地训练换成微型张量更新。
        from utils.direction_balanced_buffer import select_buffer
        class Local:
            def __init__(self, args, **kwargs):
                pass

            def train(self, net):
                return {name: tensor + .001 for name, tensor in model_param_dict(net).items()}

        for algorithm, backend_name in BACKENDS.items():
            with self.subTest(algorithm=algorithm):
                module = importlib.import_module('algorithm.' + backend_name)
                args = native_args(.1, 1, algorithm, '-1')
                args.device = 'cpu'
                args.num_users, args.concurrency = 40, 20
                args.total_rounds = 10
                args.dirbridge_sketch_dim = 8
                args.dirbridge_kmeans_iters = 2
                args.casa_kmeans_iters = 2
                model = nn.Linear(2, 2)
                model._asyncbuffer_trainable_state_only = True
                state = dict(net_glob=model, w_glob=model_param_dict(model),
                             dataset_train=[], dict_users={i: [i] for i in range(40)},
                             num_samples=[1] * 40, global_cost=0.)
                with patch.object(module, 'LocalSGD', Local, create=True):
                    state = module.init_state(state, args, lambda client: float(client + 1))
                active = [i for i, cost in enumerate(state['cost']) if cost > 0]
                for index, client in enumerate(active):
                    state['cost'][client] = float(index + 1)
                state['iterations'] = 4
                state.setdefault('stamp', [0] * 40)
                groups = [0] * 40
                for client in active[10:15]:
                    groups[client] = 1
                decision, _ = select_buffer(state['cost'], state['stamp'], groups,
                                             [.5, .5, 0., 0.], 5, 0., 'weak')
                self.assertNotEqual(decision['selected_ids'], decision['fifo_ids'])
                def completed(idx, state, **kwargs):
                    from utils.state_dict_ops import capture_buffer_endpoint
                    capture_buffer_endpoint(state, idx, state['net_glob'])
                    return {name: tensor + .001 for name, tensor in state['w_glob'].items()}

                with patch.object(module, 'LocalSGD', Local, create=True), \
                        patch.object(module, '_sample_batch', return_value=None, create=True), \
                        patch.object(module, '_evaluate_loss', return_value=0., create=True), \
                        patch.object(module, '_train_completed_client',
                                     side_effect=completed, create=True):
                    with planned_selection(module, state, decision):
                        if algorithm == 'CA2FL':
                            module.run_round(state, args, [], state['dict_users'], state['num_samples'],
                                             lambda client: float(client + 1))
                        else:
                            module.run_round(state, args, lambda client: float(client + 1))
                self.assertEqual(state['last_buffer_list'], decision['selected_ids'])
                self.assertEqual(state['global_cost'], decision['decision_clock'])
                self.assertEqual(state['iterations'], 5)

    def test_runner_seven_algorithm_preinit_bank_pairing_end_to_end(self):
        import main_fed as main
        from algorithm import dirbridge, fedasmu
        from scripts.run_weak_experiments import prepare_cell, train_cell, ALGORITHMS
        from utils.state_dict_ops import capture_buffer_endpoint

        class Local:
            def __init__(self, args, **kwargs):
                pass

            def train(self, net):
                return {name: tensor + .001 for name, tensor in model_param_dict(net).items()}

        def configure(args):
            args.device = 'cpu'
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.default_generator.manual_seed(args.seed)

        def dataset(args):
            args.label_correlated_group_ids = np.arange(100, dtype=np.int64) % 3
            args.label_correlated_num_groups = np.int64(3)
            args.dataset_metadata = {'counts': np.ones(100, dtype=np.int64),
                                     'rng': np.random.default_rng(args.seed)}
            return [], [], {i: np.array([i]) for i in range(100)}, [1] * 100, {}

        from builders.model_builder import build_model

        def completed(idx, state, **kwargs):
            capture_buffer_endpoint(state, idx, state['net_glob'])
            return {name: tensor + .001 for name, tensor in state['w_glob'].items()}

        from scripts.run_weak_experiments import native_args as original_native_args
        def short_args(*args):
            result = original_native_args(*args)
            result.total_rounds = 6
            return result

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(main, 'configure_runtime', side_effect=configure))
            stack.enter_context(patch.object(main, 'build_dataset', side_effect=dataset))
            stack.enter_context(patch.object(main, 'build_model', side_effect=build_model))
            stack.enter_context(patch('builders.model_builder.ResNet18',
                                      side_effect=lambda: nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))))
            stack.enter_context(patch.object(main, 'evaluate_global_model', return_value=(50., 1., {})))
            stack.enter_context(patch('scripts.run_weak_experiments.native_args', side_effect=short_args))
            for backend_name in BACKENDS.values():
                module = importlib.import_module('algorithm.' + backend_name)
                stack.enter_context(patch.object(module, 'LocalSGD', Local, create=True))
            stack.enter_context(patch.object(fedasmu, '_train_completed_client', side_effect=completed))
            stack.enter_context(patch.object(fedasmu, '_sample_batch', return_value=None))
            stack.enter_context(patch.object(fedasmu, '_evaluate_loss', return_value=0.))
            cell = Path(directory)
            prepare_cell(cell, .1, 1, '-1')
            reference = json.loads((cell / 'reference.json').read_text())
            self.assertEqual(reference['protocol']['phase'], 'common_initial_model_before_algorithm_init')
            for algorithm in ALGORITHMS:
                with self.subTest(algorithm=algorithm):
                    train_cell(cell, .1, 1, algorithm, '-1')
                    selections = [json.loads(line) for line in
                                  (cell / algorithm / 'selection.jsonl').read_text().splitlines()]
                    self.assertEqual(len(selections), 6)
                    self.assertEqual([item['round'] for item in selections], list(range(1, 7)))
                    self.assertTrue(selections[4]['decision']['balance_round'])
                    launches = json.loads((cell / algorithm / 'launch_trace.json').read_text())
                    self.assertEqual(len(launches), 40 + 6 * 10)
                    self.assertEqual([event['round'] for event in launches[:40]], [0] * 40)
                    self.assertEqual(sum(event['round'] == 1 for event in launches), 10)
                    self.assertTrue((cell / algorithm / 'completion.json').exists())
                    identity = json.loads((cell / algorithm / 'identity.json').read_text())
                    parameters = identity['native_parameters']
                    self.assertEqual(parameters['resolved_bn_buffer_policy'], 'endpoint_mean')
                    self.assertEqual(parameters['label_correlated_group_ids'], [i % 3 for i in range(100)])
                    self.assertEqual(parameters['dataset_metadata']['counts'], [1] * 100)
                    self.assertEqual(parameters['dataset_metadata']['rng']['bit_generator'], 'PCG64')

    def test_controller_workers_use_external_runtime_directory(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('scripts.run_weak_experiments.subprocess.run') as run, \
                redirect_stdout(io.StringIO()):
            runner_main(['--execute', '--output-root', directory])
            self.assertEqual(run.call_count, 80)
            for call in run.call_args_list:
                self.assertEqual(call.kwargs['cwd'], Path(directory).resolve())
                self.assertTrue(call.kwargs['check'])
            roots = list(Path(directory).glob('weak-*'))
            self.assertEqual(len(roots), 1)
            self.assertEqual(json.loads((roots[0] / 'completion.json').read_text())['runs'], 70)

    def test_json_metadata_serialization_fails_before_creating_file(self):
        from scripts.run_weak_experiments import write_json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'identity.json'
            with self.assertRaises(ValueError):
                write_json(path, {'bad': np.float64(float('nan'))})
            self.assertFalse(path.exists())

    def test_plan_hook_rejects_clamp_only_and_restores_module_on_failure(self):
        from algorithm import fedbuff
        from utils.direction_balanced_buffer import select_buffer
        state = dict(cost=list(range(1, 31)), stamp=[0] * 30, iterations=4, global_cost=0.)
        groups = [0] * 10 + [1] * 20
        decision, _ = select_buffer(state['cost'], state['stamp'], groups,
                                     [.5, .5], 5, 0., 'weak')
        raw = list(state['cost'])
        with self.assertRaisesRegex(RuntimeError, '未完整消费'):
            with planned_selection(fedbuff, state, decision):
                pass
        self.assertNotIn('min', vars(fedbuff))
        self.assertEqual(state['cost'], raw)
        self.assertEqual(state['global_cost'], 0.)

    def test_ma_age_over20_fails_before_backend_consumption(self):
        groups = [i % 4 for i in range(30)]
        scheduler = WeakScheduler(groups)
        state = dict(cost=list(range(1, 31)), stamp=[0] * 30, iterations=21, global_cost=0.)
        with self.assertRaisesRegex(RuntimeError, 'age>20'):
            scheduler.step(SimpleNamespace(), SimpleNamespace(algo='FedBuffMALight'), state,
                           lambda: self.fail('不应消费超龄计划'))


if __name__ == '__main__':
    unittest.main()
