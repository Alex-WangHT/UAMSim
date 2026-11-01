from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import UAMSim as ua
from UAMSim.disturbances import DisturbanceConfig, DisturbanceField, EndEffectorWrench, WindConfig
from UAMSim.evaluation import replay_episode, run_episode
from UAMSim.tasks import TaskConfig
from test_controlled_simulation import runner


class ForceTests(unittest.TestCase):
    def test_drag_sign_and_relative_velocity(self):
        sim = runner()
        field = DisturbanceField(sim.robot, DisturbanceConfig(wind=WindConfig(velocity_world=(1, 0, 0))))
        force = field.generalized_force()
        np.testing.assert_allclose(force[:3], [0.3, 0, 0], atol=1e-12)
        sim.env.data.qvel[:3] = [1, 0, 0]
        np.testing.assert_allclose(field.generalized_force(), 0, atol=1e-12)
        sim.env.data.qvel[:3] = [2, 0, 0]
        self.assertLess(field.generalized_force()[0], 0)

    def test_body_drag_axes_and_offset_moment(self):
        sim = runner()
        sim.env.data.qpos[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        cfg = DisturbanceConfig(wind=WindConfig(velocity_world=(1, 0, 0), drag_body=(1, 2, 3), point_body=(0, 0, .2)))
        field = DisturbanceField(sim.robot, cfg)
        force = field.generalized_force()
        np.testing.assert_allclose(force[:3], [2, 0, 0], atol=1e-12)
        rotation = sim.env.data.xmat[sim.robot.multirotor.body_id].reshape(3, 3)
        expected_torque = rotation.T @ np.cross([0, 0, .2], [2, 0, 0])
        np.testing.assert_allclose(force[3:6], expected_torque, atol=1e-12)

    def test_end_effector_force_transmits_to_base_and_joints(self):
        sim = runner(True)
        cfg = DisturbanceConfig(end_effector=EndEffectorWrench(force=(1, 0, 0), torque=(0, .1, 0), start=0, end=1))
        field = DisturbanceField(sim.robot, cfg)
        force = field.generalized_force()
        np.testing.assert_allclose(force[:3], [1, 0, 0], atol=1e-12)
        point = sim.env.data.site_xpos[sim.robot.manipulator.ee_site_id]
        base = sim.env.data.xpos[sim.robot.multirotor.body_id]
        np.testing.assert_allclose(force[3:6], np.cross(point-base, [1, 0, 0]) + [0, .1, 0], atol=1e-12)
        self.assertGreater(np.linalg.norm(force[6:]), 0)

    def test_wrench_frame_and_pulse_boundaries(self):
        sim = runner(True)
        cfg = DisturbanceConfig(end_effector=EndEffectorWrench(force=(1, 0, 0), frame="site", start=1, end=2))
        field = DisturbanceField(sim.robot, cfg)
        np.testing.assert_array_equal(field.generalized_force(), 0)
        sim.env.data.time = 1
        force = field.generalized_force()
        expected = sim.env.data.site_xmat[sim.robot.manipulator.ee_site_id].reshape(3, 3) @ [1, 0, 0]
        np.testing.assert_allclose(force[:3], expected, atol=1e-12)
        sim.env.data.time = 2
        np.testing.assert_array_equal(field.generalized_force(), 0)

    def test_gust_seed_and_reset(self):
        sim = runner()
        cfg = DisturbanceConfig(wind=WindConfig(gust_amplitude=(1, 1, 1)), seed=12)
        field = DisturbanceField(sim.robot, cfg)
        first = field.generalized_force()
        field.reset()
        np.testing.assert_array_equal(field.generalized_force(), first)
        other = DisturbanceField(sim.robot, DisturbanceConfig(wind=cfg.wind, seed=13))
        self.assertFalse(np.array_equal(first, other.generalized_force()))


class DisturbedSimulationTests(unittest.TestCase):
    def test_zero_disturbance_is_exact_baseline(self):
        trajectories = []
        for config in (None, DisturbanceConfig(wind=WindConfig(drag_body=(0, 0, 0)), end_effector=EndEffectorWrench())):
            sim = runner(True, disturbance=config)
            sim.reset(ua.CascadedPID.from_robot(sim.robot))
            trajectories.append([sim.step(ua.ControlTarget([.1, 0, 1.5])).base.position_world for _ in range(50)])
        np.testing.assert_array_equal(*trajectories)

    def test_other_applied_forces_survive_success_and_failure(self):
        sim = runner(disturbance=DisturbanceConfig(wind=WindConfig(velocity_world=(1, 0, 0))))
        original = np.arange(sim.env.model.nv) * .12345
        sim.env.data.qfrc_applied[:] = original
        sim.step_command(ua.RobotCommand(ua.RotorCommand(np.zeros(6))))
        np.testing.assert_array_equal(sim.env.data.qfrc_applied, original)
        with patch.object(sim.env, "step", side_effect=RuntimeError("failure")):
            with self.assertRaises(RuntimeError):
                sim.step_command(ua.RobotCommand(ua.RotorCommand(np.zeros(6))))
        np.testing.assert_array_equal(sim.env.data.qfrc_applied, original)

    def test_disturbed_episode_replay_and_reset(self):
        cfg = DisturbanceConfig(wind=WindConfig(velocity_world=(1, 0, 0), gust_amplitude=(.3, .2, 0)),
                                end_effector=EndEffectorWrench(force=(.5, 0, 0), start=.2, end=.7), seed=3)
        sim = runner(True, disturbance=cfg)
        pid = ua.CascadedPID.from_robot(sim.robot)
        with tempfile.TemporaryDirectory() as folder:
            results = [run_episode(sim, pid, TaskConfig(duration=1), output=Path(folder)/str(i)) for i in range(2)]
            self.assertEqual(results[0]["position_rmse_m"], results[1]["position_rmse_m"])
            self.assertEqual(replay_episode(sim, Path(folder)/"0")["max_absolute_state_error"], 0)
            sim.configure_disturbance(None)
            with self.assertRaises(ValueError):
                replay_episode(sim, Path(folder)/"0")

    def test_reconfigure_requires_reset(self):
        sim = runner()
        sim.configure_disturbance(DisturbanceConfig())
        with self.assertRaises(RuntimeError):
            sim.step_command(ua.RobotCommand())
        sim.reset()
        sim.step_command(ua.RobotCommand())

    def test_invalid_parameters(self):
        for make in (lambda: WindConfig(drag_body=(-1, 0, 0)), lambda: WindConfig(gust_period=0),
                     lambda: EndEffectorWrench(force=(np.nan, 0, 0)), lambda: EndEffectorWrench(start=2, end=1),
                     lambda: EndEffectorWrench(frame="body"), lambda: DisturbanceConfig(seed=-1)):
            with self.assertRaises(ValueError):
                make()
        with self.assertRaises(ValueError):
            runner(disturbance=DisturbanceConfig(end_effector=EndEffectorWrench()))
