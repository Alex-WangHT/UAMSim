import json
from pathlib import Path
import sys
import tempfile
import unittest

from UAMSim.experiments import compare_runs, run_experiment, validate_config


class ExperimentTests(unittest.TestCase):
    def config(self, code):
        return dict(version=1, case_id='test-case', label='candidate', steps=[
            dict(action='check', name='probe', command=[sys.executable, '-c', code])])

    def test_collect_numeric_metric_without_ros_or_px4(self):
        config = self.config('print(\'{"error":0.125}\')')
        config['metrics'] = {'error_m': dict(check='probe', key='error')}
        with tempfile.TemporaryDirectory() as folder:
            result = run_experiment(config, Path(folder)/'run')
            self.assertTrue(result['passed'])
            self.assertEqual(result['metrics']['error_m'], .125)

    def test_failed_check_is_saved_and_not_reported_as_passed(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'run'
            with self.assertRaises(RuntimeError):
                run_experiment(self.config('raise SystemExit(3)'), output)
            result = json.loads((output/'summary.json').read_text())
            self.assertFalse(result['passed'])
            self.assertIn('failed (3)', result['error'])

    def test_timeout_cleans_up_started_service(self):
        config = self.config('import time; time.sleep(10)')
        config['steps'][0].update(action='start', ready_text='never', timeout=.2)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(TimeoutError):
                run_experiment(config, Path(folder)/'run')
            self.assertFalse(json.loads((Path(folder)/'run/summary.json').read_text())['passed'])

    def test_invalid_names_and_wait_references(self):
        config = self.config('pass')
        config['steps'][0]['name'] = '../outside'
        with self.assertRaises(ValueError):
            validate_config(config)
        config['steps'] = [dict(action='wait', name='missing')]
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_comparison_keeps_failures_and_rejects_different_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder)/name for name in ('a.json', 'b.json')]
            for i, path in enumerate(paths):
                path.write_text(json.dumps(dict(case_id='same', label=str(i), passed=not i, metrics={})))
            self.assertFalse(compare_runs(paths)['runs'][1]['passed'])
            paths[1].write_text(json.dumps(dict(case_id='different', label='1', passed=True, metrics={})))
            with self.assertRaises(ValueError):
                compare_runs(paths)
