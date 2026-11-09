import csv
import json
from pathlib import Path
import statistics
import tempfile
import unittest

from ds.summarize_paper_runs import collect_runs, summarize_runs, main


class PaperSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def accuracy(self, path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(f'{value}\n' for value in values))

    def metrics(self, path, rounds=500, validation=False, bad=False, algo='FedBuff', seed=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ['round', 'bn_buffer_policy', 'simulated_wall_time', 'round_runtime_sec',
                  'regrouped', 'last_recluster_time_sec', 'validation_round', 'validation_accuracy',
                  'validation_loss_finite', 'validation_logits_finite', 'validation_predictions_valid',
                  'e2_eval_loss_finite', 'e2_eval_logits_finite', 'e2_eval_inputs_finite', 'e2_eval_predictions_valid']
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for i in range(1, rounds + 1):
                row = dict(round=i, bn_buffer_policy='endpoint_mean', simulated_wall_time=i * 2,
                           round_runtime_sec=1 if algo == 'FedBuff' else 2,
                           regrouped=int(algo == 'DirBridge' and i % 5 == 0),
                           last_recluster_time_sec=.02 if i % 5 == 0 else 0,
                           e2_eval_loss_finite=not bad, e2_eval_logits_finite=True,
                           e2_eval_inputs_finite=True, e2_eval_predictions_valid=True)
                if validation:
                    row.update(validation_round=i, validation_accuracy=80 + seed,
                               validation_loss_finite=True, validation_logits_finite=True,
                               validation_predictions_valid=True)
                writer.writerow(row)

    def table(self, seed=1, algo='FedBuff', table=4, metrics=True):
        name = f'cifar-{algo}-0.5-dir-skew-40-10-{seed}'
        path = self.root / 'test_acc' / f'table{table}' / 'cifar_alpha0.5' / algo / f'seed={seed}' / (name + '-test_acc.txt')
        self.accuracy(path, [40.] * 100 + [60. + seed] * 400)
        if metrics:
            self.metrics(self.root / 'system_metrics' / f'table{table}' / algo / f'seed={seed}' / (name + '-system_metrics.csv'), algo=algo, seed=seed)
        return path

    def revision(self, seed=1, algo='FedBuff', rounds=500, validation=True):
        run = self.root / 'revision' / 'tuned-validation' / f'uuid-{algo}-{seed}'
        path = run / 'acc' / f'cifar-{algo}-{seed}-test_acc.txt'
        self.accuracy(path, [10.] * rounds)
        params = dict(dataset='cifar', alpha=.5, algo=algo, seed=seed, total_rounds=rounds,
                      interval=1, bn_buffer_policy='endpoint_mean', validation_fraction=.1 if validation else 0,
                      output_dir=str(path.parent), system_metrics_log_dir=str(run / 'metrics'))
        (run / 'manifest.json').write_text(json.dumps(dict(suite='tuned-validation', resolved_parameters=params)))
        (run / 'completion.json').write_text(json.dumps(dict(status='completed', returncode=0)))
        self.metrics(run / 'metrics' / 'run-system_metrics.csv', rounds, validation, algo=algo, seed=seed)
        return path

    def test_real_table_layout_sample_sd_and_external_system_join(self):
        for seed in range(1, 6):
            for algo in ('FedBuff', 'DirBridge'):
                self.table(seed, algo)
        rows = summarize_runs(collect_runs(self.root))
        by_algo = {row['algo']: row for row in rows}
        self.assertEqual(by_algo['FedBuff']['tail_mean'], 63.)
        self.assertAlmostEqual(by_algo['FedBuff']['tail_std'], statistics.stdev([61, 62, 63, 64, 65]))
        self.assertEqual(by_algo['FedBuff']['threshold_round_mean'], 101)
        self.assertEqual(by_algo['FedBuff']['threshold_simulated_time_mean'], 202.)
        self.assertEqual(by_algo['DirBridge']['round_runtime_vs_fedbuff'], 2.)
        self.assertEqual(by_algo['DirBridge']['dirbridge_recluster_mean_sec'], .02)
        self.assertEqual(by_algo['FedBuff']['dataset'], 'cifar')
        self.assertEqual(by_algo['FedBuff']['alpha'], .5)

    def test_metrics_are_not_borrowed_from_another_table(self):
        self.table(table=1, metrics=False)
        self.table(table=4)
        runs = collect_runs(self.root)
        by_suite = {run['suite']: run for run in runs}
        self.assertIsNone(by_suite['table1']['metrics_path'])
        self.assertIsNotNone(by_suite['table4']['metrics_path'])
        rows = {row['suite']: row for row in summarize_runs(runs, expected_seeds=1)}
        self.assertIsNone(rows['table1']['round_runtime_mean'])
        self.assertEqual(rows['table4']['round_runtime_mean'], 1.)

    def test_accuracy_outside_percentage_range_rejected(self):
        path = self.table(metrics=False)
        for invalid in (-1., 101.):
            self.accuracy(path, [invalid] * 500)
            with self.assertRaisesRegex(ValueError, r'\[0, 100\]'):
                collect_runs(self.root)
        path = self.revision()
        self.accuracy(next(self.root.glob('test_acc/**/*-test_acc.txt')), [50.] * 500)
        metric = path.parent.parent / 'metrics' / 'run-system_metrics.csv'
        with metric.open(newline='') as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        rows[0]['validation_accuracy'] = '101'
        with metric.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, r'\[0, 100\]'):
            summarize_runs(collect_runs(self.root), channel='auto', expected_seeds=1)

    def test_systems_table_normalizes_recorded_legacy_baseline_methods(self):
        self.table(algo='FedBuff')
        self.table(algo='FedASMU')
        metric = next((self.root / 'system_metrics').rglob('*FedASMU*system_metrics.csv'))
        metric.write_text(metric.read_text().replace('endpoint_mean', 'legacy_delta'))
        rows = {row['algo']: row for row in summarize_runs(collect_runs(self.root), expected_seeds=1)}
        self.assertEqual(rows['FedASMU']['bn_buffer_policy'], 'legacy_delta')
        self.assertEqual(rows['FedASMU']['round_runtime_vs_fedbuff'], 2.)

    def test_revision_validation_channel_is_independent_and_process_recorded(self):
        for seed in range(1, 6):
            self.revision(seed)
        runs = collect_runs(self.root)
        self.assertEqual(summarize_runs(runs, channel='auto')[0]['tail_mean'], 83.)
        self.assertEqual(summarize_runs(runs, channel='test')[0]['tail_mean'], 10.)
        self.assertEqual(runs[0]['process_exit_status'], 'completed')

    def test_tuning_uses_manifest_150_rounds(self):
        self.revision(rounds=150)
        run = collect_runs(self.root)
        self.assertEqual(run[0]['rounds'], 150)
        self.assertEqual(summarize_runs(run, channel='validation', expected_seeds=1)[0]['tail_mean'], 81.)

    def test_short_nonfinite_duplicate_and_missing_seed_rejected(self):
        path = self.table(metrics=False)
        self.accuracy(path, [1.] * 499)
        with self.assertRaisesRegex(ValueError, 'expected 500'):
            collect_runs(self.root)
        self.accuracy(path, [1.] * 499 + [float('nan')])
        with self.assertRaisesRegex(ValueError, 'finite'):
            collect_runs(self.root)
        self.accuracy(path, [1.] * 500)
        with self.assertRaisesRegex(ValueError, 'Expected 5 seeds'):
            summarize_runs(collect_runs(self.root))
        self.accuracy(path.parent / 'duplicate-test_acc.txt', [1.] * 500)
        with self.assertRaisesRegex(ValueError, 'Duplicate seed'):
            summarize_runs(collect_runs(self.root), expected_seeds=1)

    def test_health_invalid_even_if_accuracy_finite(self):
        path = self.revision(validation=False)
        metric = path.parent.parent / 'metrics' / 'run-system_metrics.csv'
        self.metrics(metric, bad=True)
        with self.assertRaisesRegex(ValueError, 'invalid test'):
            summarize_runs(collect_runs(self.root), expected_seeds=1)

    def test_missing_validation_and_failed_process_rejected(self):
        path = self.revision(validation=False)
        with self.assertRaisesRegex(ValueError, 'test_acc cannot replace'):
            summarize_runs(collect_runs(self.root), channel='validation', expected_seeds=1)
        (path.parent.parent / 'completion.json').write_text('{"status":"failed","returncode":1}')
        with self.assertRaisesRegex(ValueError, 'did not complete'):
            collect_runs(self.root)

    def test_direction_full_rounds_and_no_threshold(self):
        original = self.table(algo='DirBridge', table=3, metrics=False)
        path = Path(str(original).replace('cifar_alpha0.5', 'cifar_alpha0.1_dirskew'))
        path.parent.mkdir(parents=True, exist_ok=True)
        original.rename(path)
        self.accuracy(path, [40.] * 500)
        direction = self.root / 'direction_skew' / 'cifar_alpha0.1' / 'seed=1' / path.name.replace('-test_acc.txt', '-direction_skew.csv')
        direction.parent.mkdir(parents=True)
        direction.write_text('round,buffer_phi,top2_share\n' + ''.join(f'{i},0.2,0.6\n' for i in range(1, 501)))
        result = summarize_runs(collect_runs(self.root, include_direction_skew=True), expected_seeds=1)[0]
        self.assertEqual(result['buffer_phi_mean'], .2)
        self.assertEqual(result['top2_share_mean'], .6)
        self.assertIsNone(result['threshold_simulated_time_mean'])
        self.assertEqual(result['runs'][0]['evaluation_health'], 'not_recorded')
        self.assertEqual(result['runs'][0]['process_exit_status'], 'not_recorded')

    def test_interval_and_duplicate_metric_rounds_rejected(self):
        path = self.revision()
        manifest = path.parent.parent / 'manifest.json'
        value = json.loads(manifest.read_text())
        value['resolved_parameters']['interval'] = 5
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'interval=1'):
            collect_runs(self.root)
        value['resolved_parameters']['interval'] = 1
        manifest.write_text(json.dumps(value))
        metric = path.parent.parent / 'metrics' / 'run-system_metrics.csv'
        metric.write_text(metric.read_text().replace('\n2,', '\n1,', 1))
        with self.assertRaisesRegex(ValueError, 'duplicate round'):
            summarize_runs(collect_runs(self.root), expected_seeds=1)

    def test_output_is_external_exclusive_and_empty_input_fails(self):
        with self.assertRaisesRegex(ValueError, 'No recognized'):
            collect_runs(self.root)
        self.revision()
        target = self.root / 'summary.json'
        self.assertEqual(main(['--root', str(self.root), '--expected-seeds', '1', '--output', str(target)]), 0)
        self.assertTrue(target.is_file())
        with self.assertRaises(SystemExit):
            main(['--root', str(self.root), '--expected-seeds', '1', '--output', str(target)])


if __name__ == '__main__':
    unittest.main()
