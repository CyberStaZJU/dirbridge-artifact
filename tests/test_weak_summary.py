import csv
import io
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from scripts.summarize_weak import ALGORITHMS, main, summarize


class WeakSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rounds = 10
        matrix = []
        for alpha in (.1, .5):
            for seed in range(1, 6):
                for algorithm in ALGORITHMS:
                    matrix.append(dict(alpha=alpha, seed=seed, algorithm=algorithm))
                    run = self.run_path(alpha, seed, algorithm)
                    run.mkdir(parents=True)
                    self.write_json(run / 'completion.json', dict(status='completed', rounds=self.rounds))
                    with (run / 'evaluation.csv').open('w', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=[
                            'round', 'test_accuracy', 'test_loss', 'simulated_time'])
                        writer.writeheader()
                        for number in range(1, self.rounds + 1):
                            writer.writerow(dict(round=number, test_accuracy=2 * seed + number,
                                                 test_loss=1 / number, simulated_time=number * 1.5))
                    records = []
                    for number in range(1, self.rounds + 1):
                        weak_age = min(seed, number - 1)
                        fifo_age = min(seed + 1, number - 1)
                        weak = [0] * 9 + [weak_age]
                        fifo = [0] * 8 + [fifo_age] * 2
                        records.append(dict(round=number, raw_costs=list(range(1, 21)),
                            stamps=[number - 1 - age for age in weak + fifo],
                            decision=dict(round_number=number, policy='weak',
                                          selected_phi=seed * .01 + number * .001,
                                          selected_ids=list(range(10))),
                            fifo_counterfactual=dict(round_number=number, policy='fifo',
                                          selected_phi=seed * .01 + number * .001 + .05,
                                          selected_ids=list(range(10, 20))),
                            next_state=dict(last_round=number), selected_staleness=weak,
                            fifo_counterfactual_staleness=fifo))
                    self.write_records(run / 'selection.jsonl', records)
        self.write_json(self.root / 'matrix.json', matrix)
        self.write_json(self.root / 'completion.json', dict(status='completed', runs=70))

    def run_path(self, alpha=.1, seed=1, algorithm='DirBridge'):
        return self.root / f'alpha={alpha:g}_seed={seed}' / algorithm

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    @staticmethod
    def write_records(path, records):
        path.write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')

    def evaluation_rows(self, run=None):
        with ((run or self.run_path()) / 'evaluation.csv').open(newline='') as handle:
            return list(csv.DictReader(handle))

    def write_evaluation(self, rows, run=None):
        with ((run or self.run_path()) / 'evaluation.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                'round', 'test_accuracy', 'test_loss', 'simulated_time'])
            writer.writeheader()
            writer.writerows(rows)

    def selection_rows(self):
        return [json.loads(line) for line in (self.run_path() / 'selection.jsonl').read_text().splitlines()]

    def test_full_matrix_tail_mean_and_sample_sd(self):
        result = summarize(self.root, self.rounds)
        self.assertEqual(result['runs'], 70)
        self.assertEqual(len(result['accuracy']), 14)
        self.assertEqual(len(result['per_seed_accuracy']), 70)
        for row in result['accuracy']:
            self.assertEqual(row['tail10_accuracy']['mean'], 11.5)
            self.assertAlmostEqual(row['tail10_accuracy']['sample_sd'], math.sqrt(10))
        self.assertEqual(result['tail_window'], 10)
        self.assertIn('not_independent_training', result['fifo_interpretation'])

    def test_round_first_age_percentiles_and_seed_sample_sd(self):
        result = summarize(self.root, self.rounds)
        self.assertEqual(len(result['dirbridge_same_state']), 4)
        for policy, cap_offset, mean_factor, p95_factor in (
                ('weak', 0, .1, .55), ('fifo', 1, .2, 1.)):
            caps = [sum(min(seed + cap_offset, number - 1) for number in range(1, 11)) / 10
                    for seed in range(1, 6)]
            row = next(row for row in result['dirbridge_same_state']
                       if row['alpha'] == .1 and row['policy'] == policy)
            for metric, factor in [('mean_age', mean_factor), ('p95_age', p95_factor), ('max_age', 1.)]:
                self.assertAlmostEqual(row[metric]['mean'], statistics.mean(caps) * factor)
                self.assertAlmostEqual(row[metric]['sample_sd'], statistics.stdev(caps) * factor)
            self.assertAlmostEqual(row['phi']['mean'], .0355 + (.05 if policy == 'fifo' else 0))
            self.assertAlmostEqual(row['phi']['sample_sd'], math.sqrt(.00025))

    def test_tail10_excludes_earlier_rounds(self):
        rows = self.evaluation_rows()
        for row in rows:
            row['round'] = str(int(row['round']) + 1)
        rows.insert(0, dict(round='1', test_accuracy='1000', test_loss='1', simulated_time='1'))
        self.write_evaluation(rows)
        with self.assertRaises(ValueError):
            summarize(self.root, self.rounds)
        from scripts.summarize_weak import _evaluation
        self.assertEqual(_evaluation(self.run_path() / 'evaluation.csv', 11), 7.5)

    def test_missing_seed_run_is_not_skipped(self):
        (self.run_path(seed=5) / 'completion.json').unlink()
        with self.assertRaises(FileNotFoundError):
            summarize(self.root, self.rounds)

    def test_matrix_duplicate_or_missing_cell_rejected(self):
        path = self.root / 'matrix.json'
        matrix = json.loads(path.read_text())
        matrix[-1] = matrix[0]
        self.write_json(path, matrix)
        with self.assertRaisesRegex(ValueError, '矩阵必须恰好'):
            summarize(self.root, self.rounds)

    def test_root_and_run_completion_required(self):
        self.write_json(self.root / 'completion.json', dict(status='running', runs=70))
        with self.assertRaisesRegex(ValueError, '根矩阵尚未完成'):
            summarize(self.root, self.rounds)
        self.write_json(self.root / 'completion.json', dict(status='completed', runs=70))
        self.write_json(self.run_path() / 'completion.json', dict(status='completed', rounds=9))
        with self.assertRaisesRegex(ValueError, 'completion.rounds'):
            summarize(self.root, self.rounds)

    def test_short_evaluation_and_out_of_order_round_rejected(self):
        rows = self.evaluation_rows()
        self.write_evaluation(rows[:-1])
        with self.assertRaisesRegex(ValueError, '需要 10 行评估'):
            summarize(self.root, self.rounds)
        rows[0], rows[1] = rows[1], rows[0]
        self.write_evaluation(rows)
        with self.assertRaisesRegex(ValueError, '有序'):
            summarize(self.root, self.rounds)

    def test_nonfinite_evaluation_metrics_rejected(self):
        rows = self.evaluation_rows()
        for column in ('test_accuracy', 'test_loss', 'simulated_time'):
            with self.subTest(column=column):
                changed = [dict(row) for row in rows]
                changed[0][column] = 'nan'
                self.write_evaluation(changed)
                with self.assertRaisesRegex(ValueError, '有限数值'):
                    summarize(self.root, self.rounds)

    def test_all_algorithms_selection_files_are_checked(self):
        run = self.run_path(algorithm='FedASMU')
        path = run / 'selection.jsonl'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
        with self.assertRaisesRegex(ValueError, '需要 10 行选择记录'):
            summarize(self.root, self.rounds)

    def test_invalid_phi_and_staleness_rejected(self):
        records = self.selection_rows()
        records[0]['decision']['selected_phi'] = float('nan')
        self.write_records(self.run_path() / 'selection.jsonl', records)
        with self.assertRaisesRegex(ValueError, '有限数值'):
            summarize(self.root, self.rounds)
        records[0]['decision']['selected_phi'] = .01
        records[0]['selected_staleness'][0] = 1
        self.write_records(self.run_path() / 'selection.jsonl', records)
        with self.assertRaisesRegex(ValueError, '时间戳不一致'):
            summarize(self.root, self.rounds)

    def test_default_is_500_and_relative_root_rejected(self):
        with self.assertRaisesRegex(ValueError, 'completion.rounds'):
            summarize(self.root)
        with self.assertRaisesRegex(ValueError, '绝对目录'):
            summarize(Path('.'), self.rounds)

    def test_cli_only_prints_json(self):
        before = sorted(str(path.relative_to(self.root)) for path in self.root.rglob('*'))
        output = io.StringIO()
        with redirect_stdout(output):
            main(['--root', str(self.root), '--expected-rounds', '10'])
        self.assertEqual(json.loads(output.getvalue())['runs'], 70)
        self.assertEqual(before, sorted(str(path.relative_to(self.root)) for path in self.root.rglob('*')))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(['--root', str(self.root)])
        self.assertEqual(caught.exception.code, 2)

    def test_native_selector_records_match_runner_schema(self):
        from scripts.summarize_weak import _selection
        from utils.direction_balanced_buffer import select_buffer
        groups = [0] * 10 + [1] * 10
        records = []
        for number in range(1, 11):
            costs = [number * 100 + client for client in range(20)]
            stamps = [number - 1] * 20
            decision, state = select_buffer(costs, stamps, groups, [.5, .5], number, 0, policy='weak')
            fifo, _ = select_buffer(costs, stamps, groups, [.5, .5], number, 0, policy='fifo')
            records.append(dict(round=number, raw_costs=costs, stamps=stamps,
                decision=decision, next_state=state, fifo_counterfactual=fifo,
                selected_staleness=[0] * 10, fifo_counterfactual_staleness=[0] * 10))
        self.write_records(self.run_path() / 'selection.jsonl', records)
        result = _selection(self.run_path() / 'selection.jsonl', self.rounds)
        self.assertEqual(result['fifo']['phi'], .5)
        self.assertLess(result['weak']['phi'], result['fifo']['phi'])
        self.assertEqual(result['weak']['p95_age'], 0)

    def test_external_symlink_is_not_followed(self):
        path = self.run_path() / 'evaluation.csv'
        path.unlink()
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / 'evaluation.csv'
            target.write_text('不可读取的外部数据')
            path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'root 之外'):
                summarize(self.root, self.rounds)


if __name__ == '__main__':
    unittest.main()
