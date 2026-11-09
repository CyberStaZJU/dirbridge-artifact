import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from utils.fedscale_profile_coupling import build_profile_coupling
from utils.fedscale_trace import (
    FedScaleTraceSampler,
    _profile_duration,
    ensure_fedscale_trace_sampler,
)


class ProfileCouplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.profiles = {
            str(100 + index): {'computation': value, 'communication': 10000 / value}
            for index, value in enumerate((1, 2, 3, 100, 200, 300, 10000, 20000, 30000))
        }
        self.path = self._write(self.profiles)

    def _write(self, payload, filename='profiles.pkl'):
        path = Path(self.temp.name) / filename
        if path.suffix == '.json':
            path.write_text(json.dumps(payload))
        else:
            with path.open('wb') as handle:
                pickle.dump(payload, handle)
        return str(path)

    def _build(self, groups=None, **kwargs):
        if groups is None:
            groups = [0, 1, 2] * 4
        return build_profile_coupling(self.path, groups, 4, 2, **kwargs)

    def _args(self, **kwargs):
        args = SimpleNamespace(
            fedscale_client_profile_path=self.path,
            num_users=12,
            local_bs=4,
            local_period=2,
            seed=0,
            delay_seed=None,
            fedscale_profile_coupling='label_group',
            label_correlated_group_ids=[0, 1, 2] * 4,
        )
        vars(args).update(kwargs)
        return args

    def test_deterministic_custom_lloyd_and_seeded_label_permutation(self):
        result = self._build(seed=0)
        self.assertEqual(result, self._build(seed=0))
        self.assertEqual(result['kmeans'], 'custom_lloyd')
        self.assertEqual(result['features'], ['log_computation', 'log_communication'])
        self.assertEqual(result['profile_cluster_sizes'], {'0': 3, '1': 3, '2': 3})
        self.assertEqual(result['profile_cluster_order_fast_to_slow'], [1, 2, 0])
        expected = np.random.default_rng(104729).permutation(3)
        self.assertEqual(result['label_to_profile_class'], dict(enumerate(expected.tolist())))
        self.assertNotEqual(result['mapping'], self._build(seed=1)['mapping'])

    def test_filters_nonpositive_nonfinite_missing_and_malformed_profiles(self):
        invalid = {
            'missing': {'computation': 1},
            'zero': {'computation': 0, 'communication': 1},
            'negative': {'computation': 1, 'communication': -1},
            'nan': {'computation': float('nan'), 'communication': 1},
            'infinity': {'computation': 1, 'communication': float('inf')},
            'malformed': {'computation': 'bad', 'communication': 1},
            'not_a_record': None,
        }
        baseline = self._build()
        self.path = self._write(dict(self.profiles, **invalid), 'invalid.pkl')
        self.assertEqual(baseline, self._build())

    def test_supports_sampler_profile_container_and_canonical_ids(self):
        profiles = {int(key): value for key, value in self.profiles.items()}
        self.path = self._write({'client_profiles': profiles}, 'wrapped.json')
        self.assertEqual(len(self._build()['mapping']), 12)
        self.assertTrue(all(row['profile_id'] in self.profiles for row in self._build()['mapping']))

    def test_rejects_empty_invalid_and_too_few_profiles(self):
        for profiles in ({}, {'bad': None}, dict(list(self.profiles.items())[:2])):
            with self.subTest(profiles=profiles):
                self.path = self._write(profiles)
                with self.assertRaisesRegex(ValueError, 'three valid positive device profiles'):
                    self._build()

    def test_rejects_empty_kmeans_cluster_without_inventing_profiles(self):
        self.path = self._write({str(i): {'computation': 1, 'communication': 1} for i in range(3)})
        with self.assertRaisesRegex(ValueError, 'empty cluster'):
            self._build()

    def test_rejects_unknown_empty_or_missing_label_groups(self):
        for groups in ([], [3], [-1], [0.5], ['invalid'], [None]):
            with self.subTest(groups=groups):
                with self.assertRaisesRegex(ValueError, 'label group'):
                    build_profile_coupling(self.path, groups, 4, 2)
        with self.assertRaisesRegex(ValueError, 'label-derived'):
            build_profile_coupling(self.path, None, 4, 2)
        with self.assertRaisesRegex(ValueError, 'three device clusters'):
            self._build(num_groups=2)

    def test_assigns_profiles_in_service_order_and_records_reuse(self):
        result = self._build()
        self.assertEqual(result['profile_reuse_count'], 3)
        for group in range(3):
            rows = [row for row in result['mapping'] if row['label_group'] == group]
            self.assertEqual([row['duration'] for row in rows[:3]],
                             sorted(row['duration'] for row in rows[:3]))
            self.assertEqual(len({row['profile_id'] for row in rows[:3]}), 3)
            self.assertEqual(rows[0]['profile_id'], rows[3]['profile_id'])
            self.assertEqual([row['cycled'] for row in rows], [False, False, False, True])
            self.assertTrue(all(row['profile_class'] == result['label_to_profile_class'][group]
                                for row in rows))
        self.assertEqual([row['client_index'] for row in result['mapping']], list(range(12)))

    def test_service_units_and_default_symmetric_payload_match_sampler(self):
        result = self._build(upload_size=8, download_size=12, augmentation_factor=2)
        args = self._args(fedscale_upload_size_mb=8, fedscale_download_size_mb=12,
                          fedscale_augmentation_factor=2)
        for row in result['mapping']:
            record = self.profiles[row['profile_id']]
            expected = 2 * 4 * 2 * record['computation'] / 1000 + 20 / record['communication']
            self.assertAlmostEqual(row['duration'], expected)
            self.assertAlmostEqual(row['duration'], _profile_duration(record, args))
        symmetric = self._build(upload_size=7)
        explicit = self._build(upload_size=7, download_size=7)
        self.assertEqual(symmetric, explicit)

    def test_sampler_honors_delay_seed_parameter_overrides_and_time_scale(self):
        args = self._args(seed=99, delay_seed=0, fedscale_batch_size=8,
                          fedscale_local_steps=3, fedscale_augmentation_factor=1.5,
                          fedscale_upload_size_mb=4, fedscale_download_size_mb=None,
                          fedscale_time_scale=2)
        sampler = FedScaleTraceSampler(args)
        expected = build_profile_coupling(self.path, args.label_correlated_group_ids, 8, 3,
                                          4, None, 0, augmentation_factor=1.5)
        self.assertEqual(sampler.profile_coupling, expected)
        self.assertEqual(sampler.client_ids, [row['profile_id'] for row in expected['mapping']])
        for index, row in enumerate(expected['mapping']):
            self.assertAlmostEqual(sampler.sample_client_delay(index, 0), 2 * row['duration'])
        fallback = FedScaleTraceSampler(self._args(seed=0))
        self.assertEqual(fallback.profile_coupling, self._build(seed=0))

    def test_sampler_rejects_absent_or_mismatched_client_groups(self):
        for groups in (None, [], [0, 1, 2]):
            with self.subTest(groups=groups):
                with self.assertRaisesRegex(ValueError, 'one label group per client'):
                    FedScaleTraceSampler(self._args(label_correlated_group_ids=groups))

    def test_ensure_caches_sampler_and_records_coupling_metadata(self):
        args = self._args()
        state = {}
        sampler = ensure_fedscale_trace_sampler(args, state)
        self.assertIs(ensure_fedscale_trace_sampler(args, state), sampler)
        self.assertEqual(state['fedscale_trace_metadata']['fedscale_profile_coupling'],
                         sampler.profile_coupling)

    def test_disabled_coupling_preserves_default_mapping_delays_and_metadata(self):
        args = self._args(fedscale_profile_coupling='')
        expected_ids = np.random.default_rng(7919).choice(list(self.profiles), size=12,
                                                         replace=True).tolist()
        with patch('utils.fedscale_trace.build_profile_coupling',
                   side_effect=AssertionError('disabled coupling must not run')):
            sampler = FedScaleTraceSampler(args)
        self.assertEqual(sampler.client_ids, expected_ids)
        self.assertEqual(sampler.availability_client_ids, [None] * 12)
        self.assertEqual(set(sampler.metadata()), {
            'fedscale_trace_num_profiles', 'fedscale_trace_num_availability_clients',
            'fedscale_trace_client_ids', 'fedscale_trace_availability_client_ids',
            'fedscale_trace_wrap', 'fedscale_trace_time_scale',
        })
        for index, profile_id in enumerate(expected_ids):
            self.assertEqual(sampler.sample_client_delay(index, 0),
                             _profile_duration(self.profiles[profile_id], args))
        self.assertIsNone(sampler.profile_coupling)

    def test_coupling_preserves_independent_availability_assignment(self):
        availability = {'a': [[0, 100]], 'b': [[1, 100]]}
        path = self._write({'client_profiles': self.profiles, 'availability': availability})
        coupled = FedScaleTraceSampler(self._args(fedscale_client_profile_path=path))
        plain = FedScaleTraceSampler(self._args(fedscale_client_profile_path=path,
                                               fedscale_profile_coupling=''))
        self.assertEqual(coupled.availability_client_ids, ['b', 'a'] * 6)
        self.assertEqual(coupled.availability_client_ids, plain.availability_client_ids)


if __name__ == '__main__':
    unittest.main()
