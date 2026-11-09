import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from utils.options import args_parser


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('revision_runner', ROOT / 'scripts' / 'run_revision_experiments.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RevisionMatrixTests(unittest.TestCase):
    def test_paper_matrix_counts_and_unique_configurations(self):
        expected = {
            'tuning-ca2fl': 12, 'tuning-fadas': 32, 'tuned-validation': 80,
            'online-init': 15, 'profile-coupling': 70, 'group-sketch': 40,
            'beta-retention': 45, 'recluster-period': 45,
        }
        config = runner.load_config()
        self.assertEqual(set(config['suites']), set(expected))
        for suite, count in expected.items():
            runs = runner.expand_suite(suite)
            self.assertEqual(len(runs), count)
            self.assertEqual(config['suites'][suite]['expected_runs'], count)
            self.assertEqual(len({json.dumps(run, sort_keys=True) for run in runs}), count)

    def test_tuning_grids_and_heldout_split(self):
        ca2fl = runner.expand_suite('tuning-ca2fl')
        fadas = runner.expand_suite('tuning-fadas')
        self.assertEqual({(p['lr'], p['global_lr']) for p in ca2fl}, {
            (lr, glr) for lr in [0.003, 0.01, 0.03, 0.05] for glr in [0.1, 0.5, 1.0]
        })
        self.assertEqual({(p['lr'], p['global_lr'], p['fadas_delay_adaptive']) for p in fadas}, {
            (lr, glr, adaptive) for lr in [0.003, 0.01, 0.03, 0.1]
            for glr in [0.0001, 0.0003, 0.001, 0.003] for adaptive in [False, True]
        })
        for p in ca2fl + fadas:
            self.assertEqual((p['dataset'], p['alpha'], p['seed'], p['total_rounds']), ('cifar', 0.5, 100, 150))
            self.assertEqual((p['validation_fraction'], p['validation_seed']), (0.1, 20260922))
            self.assertEqual(p['bn_buffer_policy'], 'endpoint_mean')

    def test_tuned_effective_server_steps_are_explicit(self):
        for p in runner.expand_suite('tuned-validation'):
            expected = {'FedBuff': 1.0, 'DirBridge': 1.0, 'CA2FL': 0.5, 'FADAS': 0.0001}
            self.assertEqual(p['global_lr'], expected[p['algo']])
            self.assertEqual(p['lr'], 0.03)
            self.assertEqual(p['local_bs'], 100)
            self.assertEqual(p['local_period'], 10)
            self.assertEqual(p['total_rounds'], 500)
            self.assertEqual(p['validation_fraction'], 0.1)
            self.assertEqual(p['weight_decay'], 0.0001 if p['algo'] == 'FADAS' else 0)
            if p['algo'] == 'FADAS':
                self.assertTrue(p['fadas_delay_adaptive'])
                self.assertEqual(p['fadas_tau_c'], 4)

    def test_historical_online_profile_and_group_boundaries(self):
        online = runner.expand_suite('online-init')
        self.assertEqual({(p['e2_weight_source'], p['e2_init_mode']) for p in online}, {
            ('full_count', 'full'), ('arrival_freq', 'online'), ('unique_client', 'online')
        })
        profile = runner.expand_suite('profile-coupling')
        self.assertEqual({p['algo'] for p in profile}, {
            'DirBridge', 'CA2FL', 'FADAS', 'FedBuff', 'CASA', 'FedBuffMALight', 'FedASMU'
        })
        for p in profile:
            self.assertEqual((p['num_users'], p['concurrency'], p['buffer_size']), (1000, 400, 100))
            self.assertEqual(p['fedscale_profile_coupling'], 'label_group')
            if p['algo'] == 'FADAS':
                self.assertEqual(p['global_lr'], 0.0001)
                self.assertFalse(p.get('fadas_delay_adaptive', False))
                self.assertEqual(p['weight_decay'], 0)
        group = runner.expand_suite('group-sketch')
        self.assertEqual({(p['dirbridge_num_groups'], p['dirbridge_sketch_dim']) for p in group},
                         {(k, 2048) for k in [1, 2, 4, 7, 12, 16]} | {(7, 512), (7, 8192)})
        for p in online + profile + group:
            self.assertEqual(p['bn_buffer_policy'], 'legacy_delta')
            self.assertEqual(p['total_rounds'], 500)

    def test_beta_and_recluster_grids_are_explicit_new_reruns(self):
        beta = runner.expand_suite('beta-retention')
        period = runner.expand_suite('recluster-period')
        self.assertEqual({p['dirbridge_cache_beta'] for p in beta}, {0.5, 0.7, 0.9})
        self.assertEqual({p['dirbridge_recluster_interval'] for p in period}, {5, 10, 20})
        self.assertEqual({p['dirbridge_recluster_interval'] for p in beta}, {5})
        self.assertEqual({p['dirbridge_cache_beta'] for p in period}, {0.9})
        self.assertEqual({p['dataset'] for p in beta + period}, {'cifar', 'cifar100', 'tinyimagenet'})
        self.assertEqual(len({json.dumps(p, sort_keys=True) for p in beta + period}), 75)
        for p in beta + period:
            self.assertEqual((p['alpha'], p['lr'], p['local_bs'], p['local_period']), (0.5, 0.01, 100, 10))
            self.assertEqual((p['num_users'], p['concurrency'], p['buffer_size'], p['total_rounds']), (100, 40, 10, 500))
            self.assertEqual(p['bn_buffer_policy'], 'endpoint_mean')
        for suite in ['beta-retention', 'recluster-period']:
            spec = runner.load_config()['suites'][suite]
            self.assertEqual(spec['protocol_origin'], 'reconstructed')

    def test_shell_syntax_and_captured_commands_use_external_runtime(self):
        scripts = [ROOT / 'scripts' / ('run_table%d.sh' % index) for index in range(1, 7)]
        scripts += [ROOT / 'scripts' / ('run_%s.sh' % name) for name in [
            'quick_smoke', 'dirskew_main', 'fedscale_femnist', 'fedscale_gspeech'
        ]]
        capture_code = (
            "import json,os,sys; "
            "f=open(os.environ['CAPTURE_PATH'],'a'); "
            "f.write(json.dumps({'argv':sys.argv[1:],'cwd':os.getcwd()})+'\\n'); f.close()"
        )
        wrapper = (
            'capture() { "$CAPTURE_PYTHON" -c "$CAPTURE_CODE" "$@"; }; '
            'export -f capture; bash "$1"'
        )
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / 'state'
            explicit = Path(temporary) / 'explicit-output'
            capture = Path(temporary) / 'commands.jsonl'
            env = dict(os.environ, DIRBRIDGE_STATE_ROOT=str(state), PYTHON_BIN='capture',
                       CAPTURE_PYTHON=sys.executable, CAPTURE_CODE=capture_code,
                       CAPTURE_PATH=str(capture), SEEDS='1', OUT_ROOT=str(explicit), OUT_DIR=str(explicit))
            for script in scripts:
                text = script.read_text()
                self.assertNotIn('mild_label_correlated_hierarchical', text)
                self.assertNotIn('/home/', text)
                self.assertIn('PYTHON_BIN="${PYTHON_BIN:-python3}"', text)
                self.assertIn('"${ROOT_DIR}/main_fed.py"', text)
                subprocess.run(['bash', '-n', str(script)], check=True, capture_output=True)
                subprocess.run(['bash', '-c', wrapper, 'capture-wrapper', str(script)],
                               env=env, check=True, capture_output=True)
                first_capture = capture.read_text()
                repeated = subprocess.run(['bash', '-c', wrapper, 'capture-wrapper', str(script)],
                                          env=env, capture_output=True)
                if script.name.startswith('run_table'):
                    self.assertNotEqual(repeated.returncode, 0)
                    self.assertIn('输出已存在', repeated.stderr.decode())
                    self.assertEqual(capture.read_text(), first_capture)
                else:
                    self.assertEqual(repeated.returncode, 0)
            commands = [json.loads(line) for line in capture.read_text().splitlines()]
            self.assertEqual(len(commands), 123)
            for item in commands:
                self.assertEqual(item['cwd'], str(state.resolve()))
                self.assertEqual(item['argv'][0], str(ROOT / 'main_fed.py'))
                with mock.patch.object(sys, 'argv', item['argv']):
                    parsed = args_parser()
                self.assertIn(parsed.random_cost, ['dir-skew', 'fedscale_trace'])
                self.assertIn(explicit, Path(parsed.output_dir).parents)

    def test_all_commands_are_accepted_by_real_training_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            for suite in runner.load_config()['suites']:
                args = runner.build_parser().parse_args(['--suite', suite, '--state-root', temporary])
                for plan in runner.build_plan(args):
                    with mock.patch.object(sys, 'argv', ['main_fed.py'] + plan['command'][2:]):
                        parsed = args_parser()
                    self.assertEqual(parsed.global_lr, plan['resolved_parameters']['global_lr'])
                    self.assertEqual(parsed.seed, plan['resolved_parameters']['seed'])
                    self.assertEqual(parsed.bn_buffer_policy, plan['resolved_parameters']['bn_buffer_policy'])
                    heldout = parsed.validation_fraction > 0
                    self.assertEqual(plan['metric_split'], 'validation' if heldout else 'test')
                    self.assertEqual(plan['validation_sampling'], 'numpy_choice' if heldout else None)

    def test_seed_override_and_python_argument(self):
        args = runner.build_parser().parse_args([
            '--suite', 'online-init', '--seeds', '8,9', '--python', '/custom/python'
        ])
        plans = runner.build_plan(args)
        self.assertEqual(len(plans), 6)
        self.assertEqual({p['resolved_parameters']['seed'] for p in plans}, {8, 9})
        self.assertTrue(all(p['command'][0] == '/custom/python' for p in plans))
        for invalid in ['', '1,1', '-1', 'x']:
            with self.assertRaises(argparse.ArgumentTypeError):
                runner.parse_seeds(invalid)

    def test_outputs_are_external_unique_and_alias_is_documented(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = runner.build_parser().parse_args(['--suite', 'tuning-ca2fl', '--state-root', temporary])
            first = runner.build_plan(args)
            second = runner.build_plan(args)
            directories = [p['run_dir'] for p in first + second]
            self.assertEqual(len(set(directories)), 24)
            for plan in first:
                params = plan['resolved_parameters']
                self.assertEqual(params['random_cost'], 'dir-skew')
                self.assertEqual(plan['dir_skew_alias']['historical'], 'mild_label_correlated_hierarchical')
                self.assertEqual(Path(plan['cwd']), Path(temporary).resolve())
                for flag in ['output_dir', 'system_metrics_log_dir', 'direction_skew_log_dir']:
                    output = Path(params[flag])
                    self.assertIn(Path(temporary).resolve(), output.parents)
                    self.assertNotIn(ROOT, output.parents)
            self.assertEqual(list(Path(temporary).iterdir()), [])
        for invalid in [ROOT, ROOT / 'artifacts', ROOT.parent]:
            with self.assertRaises(ValueError):
                runner.external_state_root(invalid)

    def test_symlink_to_repository_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            link = Path(temporary) / 'linked-repo'
            link.symlink_to(ROOT, target_is_directory=True)
            with self.assertRaises(ValueError):
                runner.external_state_root(link / 'outputs')

    def test_nested_output_symlink_to_repository_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / 'revision').symlink_to(ROOT, target_is_directory=True)
            args = runner.build_parser().parse_args(['--suite', 'online-init', '--state-root', temporary])
            with self.assertRaises(ValueError):
                runner.build_plan(args)

    def test_manifest_is_exclusive_and_preserves_resolved_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = runner.build_parser().parse_args(['--suite', 'tuned-validation', '--state-root', temporary])
            plan = runner.build_plan(args)[0]
            directory = runner.write_manifest(plan, execute=True)
            original = (directory / 'manifest.json').read_text()
            self.assertEqual(json.loads(original)['command'], plan['command'])
            self.assertTrue(json.loads(original)['execution_requested'])
            self.assertIn('实际', json.loads(original)['notes'][0])
            with self.assertRaises(FileExistsError):
                runner.write_manifest(plan, execute=True)
            self.assertEqual((directory / 'manifest.json').read_text(), original)

    def test_dry_run_never_launches_training_or_writes_state(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(runner.subprocess, 'run') as launch:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = runner.main(['--suite', 'profile-coupling', '--state-root', temporary])
            self.assertEqual(result, 0)
            self.assertIn('DRY-RUN', output.getvalue())
            self.assertIn('70', output.getvalue())
            launch.assert_not_called()
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_execute_requires_profile_file_before_writing_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = runner.build_parser().parse_args([
                '--suite', 'profile-coupling', '--state-root', temporary, '--execute'
            ])
            with self.assertRaises(ValueError):
                runner.build_plan(args)
            args.fedscale_profile = str(Path(temporary) / 'absent')
            with self.assertRaises(ValueError):
                runner.build_plan(args)
            args.fedscale_profile = temporary
            with self.assertRaises(ValueError):
                runner.build_plan(args)
            profile = Path(temporary) / 'profile.pkl'
            profile.touch()
            args.fedscale_profile = str(profile)
            plan = runner.build_plan(args)[0]
            self.assertEqual(plan['resolved_parameters']['fedscale_client_profile_path'], str(profile.resolve()))

    def test_successful_execute_writes_completed_marker(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(runner.subprocess, 'run'):
            with contextlib.redirect_stdout(io.StringIO()):
                runner.main(['--suite', 'tuning-ca2fl', '--state-root', temporary, '--execute'])
            markers = list(Path(temporary).rglob('completion.json'))
            self.assertEqual(len(markers), 12)
            for marker in markers:
                self.assertEqual(json.loads(marker.read_text()), {'status': 'completed', 'returncode': 0})

    def test_failed_execute_records_exit_and_propagates_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            failure = subprocess.CalledProcessError(7, ['mock-python'])
            with mock.patch.object(runner.subprocess, 'run', side_effect=failure) as launch:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                    runner.main(['--suite', 'tuning-ca2fl', '--state-root', temporary, '--execute'])
            launch.assert_called_once()
            markers = list(Path(temporary).rglob('completion.json'))
            self.assertEqual(len(markers), 1)
            self.assertEqual(json.loads(markers[0].read_text()), {'status': 'failed', 'returncode': 7})

    def test_execute_saves_manifest_before_mocked_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            def inspect_launch(command, **kwargs):
                run_dir = Path(command[command.index('--output_dir') + 1]).parent
                self.assertTrue((run_dir / 'manifest.json').is_file())
                self.assertEqual(kwargs['cwd'], str(Path(temporary).resolve()))
                self.assertTrue(kwargs['check'])
            with mock.patch.object(runner.subprocess, 'run', side_effect=inspect_launch) as launch:
                with contextlib.redirect_stdout(io.StringIO()):
                    runner.main(['--suite', 'tuning-ca2fl', '--state-root', temporary, '--execute'])
            self.assertEqual(launch.call_count, 12)
            self.assertEqual(len(list(Path(temporary).rglob('manifest.json'))), 12)


if __name__ == '__main__':
    unittest.main()
