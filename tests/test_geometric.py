from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

import UAMSim as ua
from UAMSim.controllers.geometric import rotation_matrix
from UAMSim.disturbances import DisturbanceConfig, EndEffectorWrench, WindConfig
from UAMSim.evaluation import replay_episode, run_episode
from UAMSim.tasks import FlightTask, TaskConfig
from test_controlled_simulation import runner


class GeometricTests(unittest.TestCase):
    def test_quaternion_sign_invariance(self):
        sim = runner()
        state = replace(sim.state, base=ua.Pose([0, 0, 1.5], [np.cos(.2), np.sin(.2), 0, 0]))
        alternate = replace(state, base=ua.Pose(state.base.position_world, -state.base.quaternion_wxyz))
        outputs = []
        for initial in (state, alternate):
            controller = ua.GeometricController.from_robot(sim.robot)
            controller.reset(initial)
            outputs.append(controller.compute(initial, ua.ControlTarget([0, 0, 1.5]), .005).rotors.thrust)
        np.testing.assert_array_equal(*outputs)

    def test_desired_rotation_valid_and_tilt_limited(self):
        sim = runner()
        controller = ua.GeometricController.from_robot(sim.robot)
        controller.reset(sim.state)
        command = controller.compute(sim.state, ua.ControlTarget([100, 100, 1.5], yaw=np.pi), .005)
        desired = controller.desired_rotation
        np.testing.assert_allclose(desired.T@desired, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(desired), 1)
        self.assertLessEqual(np.arccos(desired[2, 2]), controller.gains.tilt_limit + 1e-12)
        self.assertTrue(controller.saturated)
        self.assertTrue(np.isfinite(command.rotors.thrust).all())

    def test_rate_history_reset_and_discontinuous_target(self):
        sim = runner()
        controller = ua.GeometricController.from_robot(sim.robot)
        with self.assertRaises(RuntimeError):
            controller.compute(sim.state, ua.ControlTarget([0, 0, 1.5]), .005)
        controller.reset(sim.state)
        initial = controller.compute(sim.state, ua.ControlTarget([.1, 0, 1.5]), .005).rotors.thrust
        controller.compute(sim.state, ua.ControlTarget([0, 0, 2], yaw=1), .005)
        self.assertLessEqual(np.linalg.norm(controller.desired_rate), 4 + 1e-12)
        controller.reset(sim.state)
        repeated = controller.compute(sim.state, ua.ControlTarget([.1, 0, 1.5]), .005).rotors.thrust
        np.testing.assert_array_equal(initial, repeated)
        with self.assertRaises(ValueError):
            controller.compute(sim.state, ua.ControlTarget([0, 0, 1.5]), 0)

    def test_acceleration_reference_is_velocity_derivative(self):
        sim = runner()
        task = FlightTask(TaskConfig(name="tracking"), sim.state, 0)
        epsilon = 1e-5
        derivative = (task.target(1 + epsilon).velocity_world - task.target(1 - epsilon).velocity_world)/(2*epsilon)
        np.testing.assert_allclose(derivative, task.target(1).acceleration_world, atol=1e-8)
        np.testing.assert_array_equal(ua.ControlTarget([0, 0, 1]).acceleration_world, 0)
        with self.assertRaises(ValueError):
            ua.ControlTarget([0, 0, 1], acceleration_world=[np.nan, 0, 0])

    def test_rotation_matrix_known_yaw(self):
        rotation = rotation_matrix([np.sqrt(.5), 0, 0, np.sqrt(.5)])
        np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_tracking_improves_on_pid_and_replays_with_combined_disturbance(self):
        cfg = DisturbanceConfig(wind=WindConfig(velocity_world=(1, 0, 0), gust_amplitude=(.3, .2, 0)),
                                end_effector=EndEffectorWrench(force=(.5, 0, 0)))
        sim = runner(True, disturbance=cfg)
        results = []
        with tempfile.TemporaryDirectory() as folder:
            for cls in (ua.CascadedPID, ua.GeometricController):
                sim.reset()
                controller = cls.from_robot(sim.robot)
                results.append(run_episode(sim, controller, TaskConfig(name="tracking"),
                                           output=Path(folder)/cls.__name__))
            self.assertEqual(results[0]["status"], "timeout")
            self.assertEqual(results[1]["status"], "succeeded")
            self.assertLess(results[1]["position_rmse_m"], .85*results[0]["position_rmse_m"])
            self.assertLess(results[1]["velocity_rmse_m_s"], .6*results[0]["velocity_rmse_m_s"])
            self.assertEqual(replay_episode(sim, Path(folder)/"GeometricController")["max_absolute_state_error"], 0)
