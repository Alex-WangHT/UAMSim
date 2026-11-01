from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import UAMSim as ua
from UAMSim.evaluation import aggregate, replay_episode, run_episode
from UAMSim.tasks import FlightTask, TaskConfig
from test_controlled_simulation import runner


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.sim = runner()
        self.state = self.sim.state

    def test_success_requires_full_duration_and_continuous_hold(self):
        task = FlightTask(TaskConfig(duration=2, target_jitter=0), self.state, 0)
        state = replace(self.state, time=1.0)
        self.assertEqual(task.observe(state, 0.5)[0], "running")
        off_target = replace(state, time=1.5, base=ua.Pose([0.3, 0, 1.5], [1, 0, 0, 0]))
        task.observe(off_target, 0.5)
        self.assertEqual(task.observe(replace(state, time=2), 0.5)[0], "timeout")
        task = FlightTask(TaskConfig(duration=2, target_jitter=0), self.state, 0)
        for time in (0.5, 1., 1.5):
            self.assertEqual(task.observe(replace(self.state, time=time), 0.5)[0], "running")
        self.assertEqual(task.observe(replace(self.state, time=2), 0.5)[0], "succeeded")

    def test_collision_precedes_success(self):
        task = FlightTask(TaskConfig(duration=1, target_jitter=0), self.state, 0)
        status, reason, _ = task.observe(replace(self.state, time=1), 1, contact=True)
        self.assertEqual((status, reason), ("failed", "collision"))

    def test_failure_limits(self):
        task = FlightTask(TaskConfig(target_jitter=0), self.state, 0)
        for pose, reason in ((ua.Pose([0, 0, 0.1], [1, 0, 0, 0]), "minimum_height"),
                             (ua.Pose([2, 0, 1.5], [1, 0, 0, 0]), "position_divergence"),
                             (ua.Pose([0, 0, 1.5], [0, 1, 0, 0]), "excessive_tilt")):
            self.assertEqual(task.observe(replace(self.state, base=pose), 0.005)[1], reason)

    def test_reference_velocity_and_seeds(self):
        task = FlightTask(TaskConfig(name="tracking"), self.state, 7)
        twin = FlightTask(task.config, self.state, 7)
        other = FlightTask(task.config, self.state, 8)
        np.testing.assert_array_equal(task.center, twin.center)
        self.assertFalse(np.array_equal(task.center, other.center))
        numerical = (task.target(1.0 + 1e-5).position_world - task.target(1.0 - 1e-5).position_world) / 2e-5
        np.testing.assert_allclose(numerical, task.target(1).velocity_world, atol=1e-8)

    def test_arm_task_requires_arm(self):
        with self.assertRaises(ValueError):
            FlightTask(TaskConfig(name="arm_motion"), self.state, 0)


class EvaluationTests(unittest.TestCase):
    def test_real_episode_artifacts_and_repeatability(self):
        sim = runner(True)
        pid = ua.CascadedPID.from_robot(sim.robot)
        with tempfile.TemporaryDirectory() as folder:
            config = TaskConfig(target_jitter=0)
            results = [run_episode(sim, pid, config, output=Path(folder)/str(i)) for i in range(2)]
            self.assertEqual(results[0]["status"], "succeeded")
            self.assertEqual(results[0]["position_rmse_m"], results[1]["position_rmse_m"])
            self.assertEqual(results[0]["completed_steps"], 1200)
            trace = [json.loads(line) for line in (Path(folder)/"0"/"trace.jsonl").read_text().splitlines()]
            self.assertEqual(len(trace), 1200)
            self.assertEqual(trace[-1]["status"], "succeeded")
            errors = [row["metrics"]["position_error_m"] for row in trace]
            self.assertAlmostEqual(results[0]["position_rmse_m"], np.sqrt(np.mean(np.square(errors))))
            self.assertAlmostEqual(trace[0]["state"]["time"], sim.control_dt)
            manifest = json.loads((Path(folder)/"0"/"manifest.json").read_text())
            self.assertEqual(manifest["seed"], 0)
            self.assertIn("controller_gains", manifest)
            replay = replay_episode(sim, Path(folder)/"0")
            self.assertEqual(replay["verified_steps"], 1200)
            self.assertLessEqual(replay["max_absolute_state_error"], 1e-9)
            trace[-1]["state"]["base"]["position_world"][0] += 1
            (Path(folder)/"0"/"trace.jsonl").write_text("\n".join(json.dumps(row) for row in trace))
            with self.assertRaises(AssertionError):
                replay_episode(sim, Path(folder)/"0")
            with self.assertRaises(FileExistsError):
                run_episode(sim, pid, config, output=Path(folder)/"0")

    def test_controller_error_produces_failed_summary(self):
        class Broken:
            def reset(self, state):
                pass
            def compute(self, *args):
                raise RuntimeError("intentional failure")
        with tempfile.TemporaryDirectory() as folder:
            result = run_episode(runner(), Broken(), TaskConfig(), output=Path(folder)/"failure")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["completed_steps"], 0)
            self.assertIsNone(result["position_rmse_m"])
            self.assertIn("intentional failure", result["exception"])

    def test_nonintegral_duration_rejected(self):
        sim = runner()
        with self.assertRaises(ValueError):
            run_episode(sim, ua.CascadedPID.from_robot(sim.robot), TaskConfig(duration=1.001), output="unused")

    def test_real_timeout_is_not_success(self):
        sim = runner()
        with tempfile.TemporaryDirectory() as folder:
            result = run_episode(sim, ua.CascadedPID.from_robot(sim.robot),
                                 TaskConfig(duration=1, position_tolerance=1e-10), output=Path(folder)/"timeout")
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(result["completed_steps"], 200)

    def test_zero_thrust_fails_before_deadline(self):
        class NoThrust:
            def reset(self, state):
                pass
            def compute(self, *args):
                return ua.RobotCommand(ua.RotorCommand(np.zeros(6)))
        with tempfile.TemporaryDirectory() as folder:
            result = run_episode(runner(), NoThrust(), TaskConfig(), output=Path(folder)/"fall")
            self.assertEqual(result["status"], "failed")
            self.assertIn(result["reason"], ("position_divergence", "minimum_height", "collision"))
            self.assertLess(result["simulation_seconds"], 6)
            self.assertIsNone(result["exception"])

    def test_failure_counts_in_success_rate(self):
        rows = [dict(task="hover", status=status, position_rmse_m=value)
                for status, value in (("succeeded", .1), ("failed", None), ("timeout", .3))]
        summary = aggregate(rows)["hover"]
        self.assertAlmostEqual(summary["success_rate"], 1/3)
        self.assertEqual(summary["episodes_with_metrics"], 2)
        self.assertAlmostEqual(summary["mean_episode_position_rmse_m"], .2)
