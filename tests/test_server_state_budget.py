import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

from scripts.server_state_budget import (
    MIB,
    ROOT,
    BudgetConfig,
    calculate_budget,
    external_output_path,
    main,
)


class ServerStateBudgetTests(unittest.TestCase):
    def setUp(self):
        self.config = BudgetConfig()
        self.rows = calculate_budget(self.config)
        self.by_method = {row['method']: row for row in self.rows}

    def test_seven_method_table_matches_supplement(self):
        self.assertEqual(
            [row['table_mib'] for row in self.rows],
            ['469.3', '597.2', '515.7', '4735.5', '469.5', '4731.8', '737.3'],
        )
        self.assertEqual(len(self.by_method), 7)
        self.assertEqual(self.rows[0]['common_bytes'], 492_078_488)
        self.assertEqual(self.rows[0]['S'], 44_695_848)
        self.assertEqual(f"{self.rows[0]['common_bytes'] / MIB:.3f}", '469.283')
        self.assertEqual(f"{self.by_method['DirBridge']['total_bytes'] / MIB:.3f}", '737.316')
        self.assertEqual(f"{self.by_method['CA2FL']['total_bytes'] / MIB:.3f}", '4731.811')

    def test_exact_method_component_counts(self):
        expected = {
            'FedBuff': 0,
            'FADAS': 134_087_544,
            'MA-Light': 48_707_856,
            'FedASMU': 4_473_452_800,
            'CASA-inspired buffered variant': 223_424,
            'CA2FL': 4_469_584_800,
            'DirBridge': 281_052_986,
        }
        for method, byte_count in expected.items():
            with self.subTest(method=method):
                row = self.by_method[method]
                self.assertEqual(row['method_bytes'], byte_count)
                self.assertEqual(row['total_bytes'], row['common_bytes'] + byte_count)

    def test_ma_uses_two_distinct_fp64_histories_at_t500(self):
        row = self.by_method['MA-Light']
        self.assertEqual(row['T'], 500)
        components = self.components(row)
        self.assertEqual(components['history_t'], 2_000_000)
        self.assertEqual(components['history_t_plus_1'], 2_008_008)
        self.assertEqual(components['coefficients'], 4_000)
        changed = calculate_budget(replace(self.config, rounds=1))[2]
        self.assertEqual(changed['method_bytes'], 44_695_848 + 8 * (1 + 4 + 1))

    def test_dirbridge_explicit_plan_and_both_grouping_copies(self):
        row = self.by_method['DirBridge']
        components = self.components(row)
        self.assertEqual(components['explicit_sketch_plan'], 9 * 11_173_962)
        self.assertEqual(components['group_memories'], 4 * 44_695_848)
        self.assertEqual(components['client_features'], 819_200)
        self.assertEqual(components['clustering_feature_copy'], 819_200)
        self.assertEqual(components['centroids'], 32_768)
        self.assertEqual(components['centroid_copy'], 32_768)
        self.assertIn('9*P', row['method_formula'])
        self.assertEqual(sum(components.values()), row['method_bytes'])

    @staticmethod
    def components(row):
        return {name: int(value) for name, value in
                (item.split('=') for item in row['method_components_bytes'].split(';'))}

    def test_n_k_b_valid_boundaries_and_scaling(self):
        smallest = replace(self.config, clients=1, groups=1, buffer_size=1)
        row = calculate_budget(smallest)[-1]
        self.assertEqual(row['common_bytes'], 2 * (44_695_848 + 38_560))
        self.assertEqual(row['method_bytes'], 44_695_848 + 9 * 11_173_962 + 4 * 4 * 2048)
        largest = replace(self.config, groups=100, buffer_size=100)
        row = calculate_budget(largest)[-1]
        self.assertEqual(row['common_bytes'], 101 * (44_695_848 + 38_560))
        self.assertEqual(row['method_bytes'], 100 * 44_695_848 + 9 * 11_173_962 + 4 * 400 * 2048)

    def test_invalid_n_k_b_and_noninteger_counts_are_rejected(self):
        for changes in ({'clients': 0}, {'clients': -1}, {'groups': 0},
                        {'groups': 101}, {'buffer_size': 0}, {'buffer_size': 101},
                        {'clients': 3}, {'groups': 1.5}, {'buffer_size': True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)

    def test_other_dimensions_and_zero_optional_slots(self):
        for name in ('parameters', 'sketch_dim', 'rounds', 'casa_dim'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(self.config, **{name: 0})
        for name in ('bn_bytes', 'launch_snapshots', 'feature_records'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(self.config, **{name: -1})
        self.assertEqual(calculate_budget(replace(self.config, launch_snapshots=0))[3]['method_bytes'],
                         12_000)

    def test_default_cli_emits_only_csv_without_files(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main([])
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[-1]['table_mib'], '737.3')
        self.assertTrue(all(row['scope'] == 'analytical_populated_slots_transients_excluded'
                            for row in rows))
        self.assertTrue(all(row['note'] for row in rows))

    def test_output_requires_external_absolute_path_and_rejects_symlink_into_repo(self):
        for value in ('relative.csv', str(ROOT / 'budget.csv')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                external_output_path(value)
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / 'repo_link'
            link.symlink_to(ROOT, target_is_directory=True)
            with self.assertRaises(ValueError):
                external_output_path(str(link / 'budget.csv'))

    def test_output_is_exclusive_and_preserves_existing_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'budget.csv'
            output = io.StringIO()
            with redirect_stdout(output):
                main(['--output', str(path)])
            self.assertEqual(output.getvalue(), '')
            original = path.read_bytes()
            self.assertEqual(len(list(csv.DictReader(io.StringIO(original.decode('utf-8'))))), 7)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(['--output', str(path)])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_cli_config_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.csv'
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--clients', '0', '--output', str(path)])
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
