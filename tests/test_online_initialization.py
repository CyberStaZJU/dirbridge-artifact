import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from algorithm import dirbridge
from utils.e2_online import coverage_bound, normalize_counts
from utils.state_dict_ops import model_param_dict


torch.set_num_threads(1)


class OnlineInitializationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)

    def _args(self, num_users=4, **overrides):
        values = dict(
            num_users=num_users, concurrency=2, buffer_size=1, device='cpu',
            e2_init_mode='online', e2_weight_source='unique_client',
            dirbridge_num_groups=2, dirbridge_sketch_dim=8,
            dirbridge_kmeans_iters=3, dirbridge_recluster_interval=100,
            local_period=1, local_bs=4, num_workers=0, lr=0.1,
            seed=7, global_lr=None,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def _state(self, num_users=4, batchnorm=False):
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)) if batchnorm else nn.Linear(2, 2)
        if batchnorm:
            model._asyncbuffer_trainable_state_only = True
        x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]])
        dataset = TensorDataset(x.repeat(num_users, 1), torch.tensor([0, 1, 0, 1]).repeat(num_users))
        return dict(
            net_glob=model, w_glob=model_param_dict(model), dataset_train=dataset,
            dict_users={idx: list(range(4 * idx, 4 * idx + 4)) for idx in range(num_users)},
            num_samples=[4] * num_users,
        )

    def test_firstwave_unknown_until_real_delta_arrives(self):
        args = self._args()
        state = self._state()
        with patch.object(dirbridge, 'local_train_delta', wraps=dirbridge.local_train_delta) as train:
            dirbridge.init_state(state, args, lambda idx: idx + 1)
        self.assertEqual(train.call_count, args.concurrency)
        self.assertEqual(len(state['inflight']), args.concurrency)
        self.assertEqual(state['seen_set'], set())
        self.assertEqual(state['group_ids'], [-1] * args.num_users)
        self.assertTrue(all(feature is None for feature in state['client_features']))
        self.assertTrue(all(stamp == -1 for stamp in state['stamp']))
        self.assertTrue(all(all(torch.count_nonzero(v) == 0 for v in delta.values()) for delta in state['delta']))
        idx = min(state['inflight'], key=lambda client: state['cost'][client])
        record = state['inflight'][idx]
        self.assertGreater(sum(float(value.norm()) for value in record['delta'].values()), 0.)
        with patch.object(dirbridge, '_schedule_new_clients'):
            dirbridge.run_round(state, args, lambda client: client + 1)
        self.assertEqual(state['seen_set'], {idx})
        self.assertEqual(state['num_groups'], 1)
        self.assertEqual(state['group_ids'][idx], 0)
        self.assertIs(state['delta'][idx], record['delta'])
        self.assertEqual(state['last_selected_item_summary'][0]['member_count'], 1)
        self.assertEqual(state['last_unassigned_valid_count'], 0)
        with self.assertRaises(RuntimeError):
            dirbridge._materialize_arrivals(state, [idx])

    def test_materialize_rejects_duplicate_batch_without_consuming_flight(self):
        state = self._state()
        dirbridge.init_state(state, self._args(), lambda idx: idx + 1)
        idx = next(iter(state['inflight']))
        with self.assertRaises(RuntimeError):
            dirbridge._materialize_arrivals(state, [idx, idx])
        self.assertIn(idx, state['inflight'])

    def test_arrival_frequency_unique_counts_and_reassignment(self):
        args = self._args(num_users=3)
        state = self._state(3)
        dirbridge.init_state(state, args, lambda idx: idx + 1)
        state.update(
            seen_set={0, 1, 2}, client_features=[torch.tensor([1., 0.]), torch.tensor([0., 1.]), torch.tensor([1., 0.])],
            client_embeds=[None] * 3, group_ids=[0, 1, 0], num_groups=2,
            group_centroids=torch.eye(2), group_members=[[0, 2], [1]], group_counts=[2, 1],
            arrival_count=[4, 1, 1],
        )
        dirbridge._refresh_group_weights_from_current_assignments(state, args)
        self.assertEqual(state['group_weights'], [2 / 3, 1 / 3])
        args.e2_weight_source = 'arrival_freq'
        dirbridge._refresh_group_weights_from_current_assignments(state, args)
        self.assertEqual(state['group_weights'], [5 / 6, 1 / 6])
        with patch.object(dirbridge, 'spherical_kmeans', return_value=([1, 0, 1], torch.eye(2))):
            dirbridge.rebuild_direction_groups(state, args)
        self.assertEqual(state['group_counts'], [1, 2])
        self.assertEqual(state['group_members'], [[1], [0, 2]])
        self.assertEqual(state['group_weights'], [1 / 6, 5 / 6])
        args.e2_init_mode, args.e2_weight_source = 'full', 'full_count'
        dirbridge._refresh_group_weights_from_current_assignments(state, args)
        self.assertEqual(state['group_weights'], [1 / 3, 2 / 3])

    def test_newly_seen_assignment_between_reclusters_updates_counts_once(self):
        state = self._state(3)
        args = self._args(3)
        dirbridge.init_state(state, args, lambda idx: idx + 1)
        state.update(num_groups=2, group_centroids=torch.eye(2), group_ids=[0, 1, -1],
                     group_members=[[0], [1]], group_counts=[1, 1],
                     client_features=[torch.tensor([1., 0.]), torch.tensor([0., 1.]), torch.tensor([0., 1.])])
        self.assertEqual(dirbridge._assign_arrivals_to_existing_groups(state, [2]), [2])
        self.assertEqual(dirbridge._assign_arrivals_to_existing_groups(state, [2]), [])
        dirbridge._refresh_group_weights_from_current_assignments(state, args)
        self.assertEqual(state['group_counts'], [1, 2])
        self.assertEqual(state['group_weights'], [1 / 3, 2 / 3])

    def test_refill_reads_previous_cache_before_refresh(self):
        args = self._args()
        state = dict(
            iterations=0, stamp=[0] * 4, w_glob={'w': torch.tensor([0.])},
            delta=[{'w': torch.tensor([value])} for value in (2., 4., 10., 0.)],
            group_ids=[0, 0, 1, 1], num_groups=2, group_counts=[2, 2], group_weights=[0.5, 0.5],
            group_cache=[{'w': torch.tensor([20.])}, {'w': torch.tensor([30.])}],
            group_cache_stamp=[0, 0], group_cache_delay=[0., 0.],
        )
        items = dirbridge._build_ema_group_updates(state, args, [0, 1, 2])
        group1 = next(item for item in items if item['group'] == 1)
        self.assertAlmostEqual(float(group1['delta']['w']), (10. + 0.5 * 30.) / 1.5, places=5)
        self.assertEqual(float(state['group_cache'][1]['w']), 30.)
        dirbridge._update_ema_group_cache(state, args, items)
        self.assertAlmostEqual(float(state['group_cache'][1]['w']), 28.)

    def test_staleness_filters_update_but_not_observation_or_bn_endpoint(self):
        args = self._args(2, concurrency=2)
        state = self._state(2, batchnorm=True)
        dirbridge.init_state(state, args, lambda idx: idx + 1)
        state['iterations'] = 5
        before = model_param_dict(state['net_glob'])
        bn_before = state['net_glob'][1].running_mean.clone()
        with patch.object(dirbridge, '_schedule_new_clients'):
            dirbridge.run_round(state, args, lambda idx: idx + 1)
        self.assertEqual(state['last_invalid_buffer_count'], 1)
        self.assertEqual(state['last_valid_buffer_count'], 0)
        self.assertEqual(len(state['seen_set']), 1)
        self.assertEqual(sum(state['arrival_count']), 1)
        self.assertEqual(state['e2_metrics']['used_updates'], 0)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, state['w_glob'][key]))
        self.assertTrue(torch.equal(state['net_glob'][1].running_mean, bn_before))

    def test_bn_aggregation_uses_only_valid_arrival_endpoints(self):
        args = self._args(2, concurrency=2, buffer_size=2)
        state = self._state(2, batchnorm=True)
        dirbridge.init_state(state, args, lambda idx: idx + 1)
        state['iterations'] = 5
        state['inflight'][0]['stamp'] = 0
        state['inflight'][1]['stamp'] = 5
        state['inflight'][0]['buffer_endpoint']['1.running_mean'].fill_(2.)
        state['inflight'][1]['buffer_endpoint']['1.running_mean'].fill_(8.)
        with patch.object(dirbridge, '_schedule_new_clients'):
            dirbridge.run_round(state, args, lambda idx: idx + 1)
        self.assertEqual(state['last_valid_buffer_list'], [1])
        self.assertEqual(state['last_invalid_buffer_list'], [0])
        self.assertTrue(torch.equal(state['net_glob'][1].running_mean, torch.full((2,), 8.)))
        self.assertEqual(state['arrival_count'], [1, 1])

    def test_bn_endpoint_dispatch_and_materialization_are_isolated(self):
        args = self._args(1, concurrency=1)
        state = self._state(1, batchnorm=True)
        dirbridge.init_state(state, args, lambda idx: 1)
        self.assertIsNone(state['buffer_endpoints'][0])
        first = state['inflight'][0]['buffer_endpoint']
        dirbridge._materialize_arrivals(state, [0])
        self.assertIs(state['buffer_endpoints'][0], first)
        previous_delta = state['delta'][0]
        previous_feature = state['client_features'][0]
        state['iterations'] = 4
        dirbridge._schedule_new_clients(state, args, [], 1, lambda idx: 1)
        self.assertIs(state['buffer_endpoints'][0], first)
        self.assertIs(state['delta'][0], previous_delta)
        self.assertIs(state['client_features'][0], previous_feature)
        self.assertEqual(state['stamp'][0], 0)
        second = state['inflight'][0]['buffer_endpoint']
        self.assertIsNot(second, first)
        dirbridge._materialize_arrivals(state, [0])
        self.assertIs(state['buffer_endpoints'][0], second)
        self.assertEqual(state['stamp'][0], 4)

    def test_full_warm_first_endpoints_transfer_into_flight(self):
        args = self._args(2, concurrency=2, e2_init_mode='full', e2_weight_source='full_count')
        state = self._state(2, batchnorm=True)
        dirbridge.init_state(state, args, lambda idx: idx + 1)
        self.assertEqual(state['buffer_endpoints'], [None, None])
        for record in state['inflight'].values():
            self.assertIsNotNone(record['buffer_endpoint'])
        with patch.object(dirbridge, '_schedule_new_clients'):
            dirbridge.run_round(state, args, lambda idx: idx + 1)
        arrived = state['last_buffer_list'][0]
        self.assertIsNotNone(state['buffer_endpoints'][arrived])
        self.assertEqual(int(state['net_glob'][1].num_batches_tracked), 1)

    def test_single_client_single_group_keeps_running(self):
        args = self._args(1, concurrency=1, dirbridge_num_groups=5)
        state = self._state(1)
        dirbridge.init_state(state, args, lambda idx: 1)
        for _ in range(3):
            dirbridge.run_round(state, args, lambda idx: 1)
        self.assertEqual(state['iterations'], 3)
        self.assertEqual(state['num_groups'], 1)
        self.assertEqual(state['group_counts'], [1])
        self.assertEqual(state['group_weights'], [1.])
        self.assertEqual(state['arrival_count'], [3])
        self.assertEqual(len(state['inflight']), 1)

    def test_saturated_concurrency_preserves_unfinished_flights(self):
        args = self._args(3, concurrency=3)
        state = self._state(3)
        dirbridge.init_state(state, args, lambda idx: idx + 1)
        for _ in range(4):
            pending = dict(state['inflight'])
            dirbridge.run_round(state, args, lambda idx: idx + 1)
            self.assertEqual(len(state['inflight']), 3)
            self.assertEqual(sum(cost > 0 for cost in state['cost']), 3)
            for idx, record in pending.items():
                if idx not in state['last_buffer_list']:
                    self.assertIs(state['inflight'][idx], record)
        self.assertEqual(state['iterations'], 4)
        self.assertEqual(sum(state['arrival_count']), 4)

    def test_none_server_lr_defaults_to_one_and_zero_stays_zero(self):
        initial = self._state(1)
        final = []
        for lr in (None, 1., 0.):
            state = copy.deepcopy(initial)
            args = self._args(1, concurrency=1, global_lr=lr)
            dirbridge.init_state(state, args, lambda idx: 1)
            with patch.object(dirbridge, '_schedule_new_clients'):
                dirbridge.run_round(state, args, lambda idx: 1)
            final.append(state['w_glob'])
        for key, value in initial['w_glob'].items():
            self.assertTrue(torch.equal(final[0][key], final[1][key]))
            self.assertTrue(torch.equal(final[2][key], value))
        self.assertTrue(any(not torch.equal(final[0][key], value) for key, value in initial['w_glob'].items()))

    def test_online_rejects_full_count_before_training(self):
        with patch.object(dirbridge, 'local_train_delta') as train:
            with self.assertRaises(ValueError):
                dirbridge.init_state(self._state(), self._args(e2_weight_source='full_count'), lambda idx: 1)
            train.assert_not_called()
        self.assertEqual(normalize_counts([0, 0]), [0.5, 0.5])
        self.assertEqual(coverage_bound(1, 4), 1.5)


if __name__ == '__main__':
    unittest.main()
