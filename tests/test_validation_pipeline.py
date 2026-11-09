import csv
import math
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision import transforms

import main_fed
from builders import dataset_builder
from models.test import test_img as evaluate_image


class SyntheticCifar:
    def __init__(self, root='', train=True, download=False, transform=None):
        size = 50000 if train else 20
        self.data = np.zeros((size, 32, 32, 3), dtype=np.uint8)
        self.targets = [i % 10 for i in range(size)]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.data[index])), self.targets[index]


class ValidationSplitTests(unittest.TestCase):
    def _args(self, **overrides):
        values = dict(
            distribution='iid', num_users=5, validation_fraction=0.1,
            validation_seed=20260922, random_cost='dir-skew',
            fedscale_profile_coupling='none', label_block_groups=5,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_cifar_builders_hold_out_5000_without_augmentation(self):
        datasets = SimpleNamespace(CIFAR10=SyntheticCifar, CIFAR100=SyntheticCifar)
        for builder in (dataset_builder._build_cifar, dataset_builder._build_cifar100):
            with self.subTest(builder=builder.__name__), patch.object(
                dataset_builder, '_load_torchvision', return_value=(datasets, transforms)
            ):
                train, test, clients, counts, meta = builder(self._args())
                heldout = set(meta['validation_indices'])
                expected = np.sort(np.random.default_rng(20260922).choice(
                    50000, size=5000, replace=False
                )).tolist()
                self.assertEqual(meta['validation_indices'], expected)
                self.assertEqual(len(meta['validation']), 5000)
                self.assertEqual(int(counts.sum()), 45000)
                assigned = [int(i) for indices in clients.values() for i in indices]
                self.assertEqual(len(set(assigned)), 45000)
                self.assertTrue(heldout.isdisjoint(assigned))
                self.assertEqual(set(assigned) | heldout, set(range(50000)))
                self.assertEqual(len(train), 50000)
                self.assertEqual(len(test), 20)
                self.assertTrue(any(isinstance(t, transforms.RandomCrop) for t in train.transform.transforms))
                self.assertEqual(
                    [type(t) for t in meta['validation'].transform.transforms],
                    [transforms.ToTensor, transforms.Normalize],
                )
                self.assertTrue(torch.equal(meta['validation'][0][0], meta['validation'][0][0]))

    def test_empty_client_after_validation_is_rejected(self):
        datasets = SimpleNamespace(CIFAR10=SyntheticCifar, CIFAR100=SyntheticCifar)
        heldout = np.sort(np.random.default_rng(20260922).choice(
            50000, size=5000, replace=False
        )).tolist()
        partition = {0: np.array([heldout[0]]), 1: np.arange(50000)}
        for builder in (dataset_builder._build_cifar, dataset_builder._build_cifar100):
            with self.subTest(builder=builder.__name__), patch.object(
                dataset_builder, '_load_torchvision', return_value=(datasets, transforms)
            ), patch.object(dataset_builder, '_partition_vision_dataset', return_value=partition):
                with self.assertRaisesRegex(ValueError, 'empty training client'):
                    builder(self._args(num_users=2))

    def test_direction_metadata_is_frozen_before_heldout_removal(self):
        datasets = SimpleNamespace(CIFAR10=SyntheticCifar, CIFAR100=SyntheticCifar)
        heldout = set(np.random.default_rng(20260922).choice(
            50000, size=5000, replace=False
        ).tolist())
        removed_low_labels = [i for i in sorted(heldout) if i % 10 < 4][:20]
        retained_high_label = next(i for i in range(50000) if i not in heldout and i % 10 == 9)
        all_first = set(removed_low_labels + [retained_high_label])
        partition = {
            0: np.array(sorted(all_first)),
            1: np.array([i for i in range(50000) if i not in all_first]),
        }
        for builder in (dataset_builder._build_cifar, dataset_builder._build_cifar100):
            args = self._args(distribution='noniid', num_users=2, alpha=0.1)
            with self.subTest(builder=builder.__name__), patch.object(
                dataset_builder, '_load_torchvision', return_value=(datasets, transforms)
            ), patch.object(dataset_builder, 'generate_noniid', return_value=partition):
                _, _, clients, _, _ = builder(args)
                self.assertEqual(args.label_correlated_group_ids[0], 0)
                self.assertEqual(clients[0].tolist(), [retained_high_label])
                self.assertEqual(retained_high_label % 10, 9)

    def test_exact_historical_numpy_choice_indices(self):
        train = SyntheticCifar(transform=transforms.ToTensor())
        _, indices = dataset_builder._server_validation_subset(train, self._args())
        expected = np.sort(np.random.default_rng(20260922).choice(
            len(train), size=5000, replace=False
        )).tolist()
        self.assertEqual(indices, expected)
        torch_indices = sorted(torch.randperm(
            len(train), generator=torch.Generator().manual_seed(20260922)
        ).tolist()[:5000])
        self.assertNotEqual(indices, torch_indices)

    def test_seed_is_independent_of_training_rng_and_input_is_unchanged(self):
        train = SyntheticCifar(transform=transforms.ToTensor())
        args = self._args()
        torch.manual_seed(1)
        np.random.seed(1)
        first, indices = dataset_builder._server_validation_subset(train, args)
        torch.manual_seed(999)
        np.random.seed(999)
        second, second_indices = dataset_builder._server_validation_subset(train, args)
        self.assertEqual(indices, second_indices)
        self.assertEqual(len(first), len(second))
        self.assertEqual(len(train), 50000)
        self.assertIsNot(first, train)
        different = dataset_builder._server_validation_subset(
            train, self._args(validation_seed=20260923)
        )[1]
        self.assertNotEqual(indices, different)

    def test_disabled_and_invalid_fraction(self):
        train = SyntheticCifar()
        self.assertEqual(dataset_builder._server_validation_subset(
            train, self._args(validation_fraction=0)
        ), (None, []))
        for fraction in (-0.1, 1.0, float('nan')):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                dataset_builder._server_validation_subset(train, self._args(validation_fraction=fraction))

    def test_dir_skew_and_profile_label_groups_keep_three_blocks(self):
        dataset = SyntheticCifar()
        clients = {0: np.arange(0, 10), 1: np.array([4, 5]), 2: np.array([8, 9])}
        for random_cost, coupling, groups in (
            ('dir-skew', 'none', 3), ('fedscale', 'label_group', 3),
            ('fedscale', 'none', 5),
        ):
            args = self._args(num_users=3, random_cost=random_cost, fedscale_profile_coupling=coupling)
            dataset_builder._attach_dirichlet_direction_groups(dataset, clients, args)
            self.assertEqual(args.label_correlated_num_groups, groups)

    def test_femnist_and_gspeech_profile_groups_without_validation(self):
        class ClientDataset:
            client_ids = ['a', 'b', 'c']
            targets = [0, 1, 3, 4, 6, 7]
            num_classes = 8
            label_map = {str(i): i for i in range(8)}

            def __getitem__(self, index):
                return torch.zeros(1, 2, 2), self.targets[index]

            def get_dict_clients(self):
                return {0: np.array([0, 1]), 1: np.array([2, 3]), 2: np.array([4, 5])}

        dataset = ClientDataset()
        args = self._args(num_users=3, fedscale_profile_coupling='label_group', random_cost='fedscale')
        with patch.object(dataset_builder, '_load_torchvision', return_value=(None, transforms)), patch.object(
            dataset_builder.os.path, 'isdir', return_value=True
        ), patch.object(dataset_builder, '_select_client_ids_from_leaf_files', return_value=dataset.client_ids), patch.object(
            dataset_builder.femnist, 'FEMNIST', return_value=dataset
        ), patch.object(dataset_builder, 'FedScaleGSpeech', return_value=dataset), patch.object(
            dataset_builder, '_server_validation_subset', side_effect=AssertionError('不应构建验证集')
        ):
            for builder in (dataset_builder._build_femnist, dataset_builder._build_gspeech):
                with self.subTest(builder=builder.__name__):
                    _, _, clients, counts, meta = builder(args)
                    self.assertEqual(args.label_correlated_num_groups, 3)
                    self.assertEqual(int(counts.sum()), 6)
                    self.assertEqual(len(clients), 3)
                    self.assertNotIn('validation', meta)

    def test_noniid_profile_attachment_preserves_partition(self):
        dataset = SyntheticCifar()
        clients = {0: np.array([0, 1]), 1: np.array([4, 5]), 2: np.array([8, 9])}
        args = self._args(distribution='noniid', num_users=3, alpha=0.1,
                          random_cost='fedscale', fedscale_profile_coupling='label_group')
        with patch.object(dataset_builder, 'generate_noniid', return_value=clients):
            result = dataset_builder._partition_vision_dataset(dataset, args)
        self.assertIs(result, clients)
        self.assertEqual(args.label_correlated_num_groups, 3)


class EvaluationDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(bs=2, num_workers=0, device='cpu', verbose=False)
        self.finite = TensorDataset(torch.tensor([[2., 0.], [0., 2.]]), torch.tensor([0, 1]))
        self.nonfinite = TensorDataset(torch.tensor([[float('nan'), 0.], [0., 2.]]), torch.tensor([0, 1]))

    def test_legacy_return_and_finite_diagnostics(self):
        model = nn.Identity()
        accuracy, loss = evaluate_image(model, self.finite, self.args)
        self.assertEqual(accuracy, 100.)
        self.assertTrue(math.isfinite(loss))
        _, _, diagnostics = evaluate_image(model, self.finite, self.args, return_diagnostics=True)
        self.assertTrue(diagnostics['eval_loss_finite'])
        self.assertTrue(diagnostics['eval_logits_finite'])
        self.assertTrue(diagnostics['eval_predictions_valid'])
        self.assertEqual(diagnostics['eval_batches'], 1)

    def test_nonfinite_values_are_recorded_not_replaced(self):
        _, loss, diagnostics = evaluate_image(nn.Identity(), self.nonfinite, self.args, return_diagnostics=True)
        self.assertFalse(math.isfinite(loss))
        for key in ('eval_inputs_finite', 'eval_logits_finite', 'eval_loss_finite', 'eval_predictions_valid'):
            self.assertFalse(diagnostics[key])
        for key in ('eval_nonfinite_input_batches', 'eval_nonfinite_logit_batches',
                    'eval_nonfinite_loss_batches', 'eval_nonfinite_prediction_batches'):
            self.assertEqual(diagnostics[key], 1)

    def test_empty_evaluation_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_image(nn.Identity(), TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)), self.args)

    def test_validation_diagnostics_do_not_overwrite_test(self):
        model = nn.Identity()
        state = {'net_glob': model, 'iterations': 2, 'validation_dataset': self.nonfinite}
        main_fed.evaluate_global_model(state, self.finite, self.args)
        diagnostics = state['last_eval_diagnostics']
        result = main_fed.log_validation_evaluation(state, self.args)
        self.assertIs(state['last_eval_diagnostics'], diagnostics)
        self.assertTrue(diagnostics['eval_loss_finite'])
        self.assertFalse(result['diagnostics']['eval_loss_finite'])
        self.assertEqual(state['last_evaluation_round'], 2)
        self.assertEqual(state['last_validation_round'], 2)
        self.assertTrue(model.training)
        model.eval()
        main_fed.evaluate_global_model(state, self.finite, self.args)
        self.assertFalse(model.training)

    def test_system_log_uses_current_round_and_keeps_bn_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(**vars(self.args), system_metrics_log_dir=directory,
                                   bn_buffer_policy='endpoint_mean')
            path = main_fed.build_system_metrics_path(args, os.path.join(directory, 'run-test_acc.txt'))
            state = {'net_glob': nn.Identity(), 'iterations': 2, 'validation_dataset': self.nonfinite,
                     'e2_metrics': {'seen_clients': 2, 'seen_ratio': 0.5, 'coverage_bound': 0.2},
                     'group_ids': [0, -1, 1], 'arrival_count': [2, 0, 1]}
            main_fed.evaluate_global_model(state, self.finite, args)
            main_fed.log_validation_evaluation(state, args)
            main_fed.log_system_metrics(args, state, path, time.perf_counter(), 0.1)
            state['iterations'] = 3
            main_fed.log_system_metrics(args, state, path, time.perf_counter(), 0.1)
            with open(path, newline='') as handle:
                first, second = list(csv.DictReader(handle))
            self.assertEqual(first['evaluation_round'], '2')
            self.assertEqual(first['validation_round'], '2')
            self.assertEqual(first['e2_eval_loss_finite'], 'True')
            self.assertEqual(first['validation_loss_finite'], 'False')
            self.assertEqual(first['bn_buffer_policy'], 'endpoint_mean')
            self.assertEqual(first['e2_raw_arrivals'], '3')
            self.assertEqual(first['e2_assigned_clients'], '2')
            self.assertEqual(first['e2_unseen_client_ratio'], '0.5')
            for key in ('evaluation_round', 'validation_round', 'e2_eval_loss_finite',
                        'validation_loss_finite', 'validation_accuracy', 'validation_loss'):
                self.assertEqual(second[key], '')

    def test_weight_update_round_tracks_refresh_not_reclustering(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(**vars(self.args), system_metrics_log_dir=directory)
            path = main_fed.build_system_metrics_path(args, os.path.join(directory, 'run-test_acc.txt'))
            state = {'iterations': 8, 'last_group_rebuild': 2, 'e2_weight_update_round': 8}
            main_fed.log_system_metrics(args, state, path, time.perf_counter(), 0.1)
            state.update(iterations=9, e2_weight_update_round=9)
            main_fed.log_system_metrics(args, state, path, time.perf_counter(), 0.1)
            del state['e2_weight_update_round']
            main_fed.log_system_metrics(args, state, path, time.perf_counter(), 0.1)
            with open(path, newline='') as handle:
                first, second, missing = list(csv.DictReader(handle))
            self.assertEqual(first['e2_weight_update_round'], '8')
            self.assertEqual(second['e2_weight_update_round'], '9')
            self.assertEqual(missing['e2_weight_update_round'], '')
            self.assertEqual(state['last_group_rebuild'], 2)

    def test_training_logs_diagnostics_after_current_evaluation(self):
        state = {'iterations': 0, 'net_glob': nn.Identity(), 'validation_dataset': self.nonfinite}
        args = SimpleNamespace(**vars(self.args), interval=2, total_rounds=3)
        snapshots = []

        def advance(args, state, *unused):
            state['iterations'] += 1
            return state

        def record(args, state, *unused):
            snapshots.append((state['iterations'], state.get('last_evaluation_round'),
                              state.get('last_validation_round')))

        with tempfile.TemporaryDirectory() as directory, patch.object(
            main_fed, 'run_one_round', side_effect=advance
        ), patch.object(main_fed, 'log_direction_skew_metrics'), patch.object(
            main_fed, 'log_system_metrics', side_effect=record
        ):
            main_fed.run_training(args, state, None, {}, [], self.finite,
                                  os.path.join(directory, 'accuracy.txt'))
        self.assertEqual(snapshots, [(1, None, None), (2, 2, 2), (3, 2, 2)])

    def test_peak_rss_platform_units(self):
        for platform, rss in (('darwin', 100 * 1024 ** 2), ('linux', 100 * 1024)):
            with self.subTest(platform=platform), patch.object(main_fed.sys, 'platform', platform), patch.object(
                main_fed.resource, 'getrusage', return_value=SimpleNamespace(ru_maxrss=rss)
            ), patch.object(main_fed.tracemalloc, 'is_tracing', return_value=False):
                self.assertEqual(main_fed._current_process_memory_mb(), (0., 100.))


if __name__ == '__main__':
    unittest.main()
