"""Discoverable regression tests; run with unittest or pytest."""

from pathlib import Path
import threading
import unittest
import importlib
from unittest.mock import patch

import numpy as np

import UAMSim as ua
from UAMSim.utils.frame_sync import ControllerChannel


MODELS = Path(__file__).resolve().parents[1] / "examples" / "models"


class CountingDrone(ua.MultirotorController):
    def controller(self, feedback, values, calls):
        calls.append(feedback.timestamp)
        return values


class CountingArm(ua.ManipulatorController):
    def controller(self, feedback, values, calls):
        calls.append(feedback.timestamp)
        return values


def environment(with_arm=False):
    env = ua.Environment(str(MODELS / "environment.xml"))
    drone = ua.Multirotor(str(MODELS / "multirotor.xml"))
    arm = ua.Manipulator(str(MODELS / "Manipulator.xml")) if with_arm else None
    robot = env.attach_robot(drone, arm)
    return env, robot


class ControllerRegressionTests(unittest.TestCase):
    def test_single_evaluation_target_update_and_invalid_output(self):
        for arm in (False, True):
            with self.subTest(arm=arm):
                env, robot = environment(arm)
                component = robot.manipulator if arm else robot.multirotor
                channel = ControllerChannel()
                count = component.n_joints if arm else component.n_rotors
                calls = []
                cls = CountingArm if arm else CountingDrone
                ctrl = cls(component, channel, values=np.zeros(count), calls=calls)
                buffers = ctrl.servos if arm else ctrl.rotors
                ctrl.start()

                def exchange():
                    timeout = threading.Event()
                    timer = threading.Timer(2.0, timeout.set)
                    timer.start()
                    try:
                        channel.mailbox.publish(component.get_state())
                        self.assertTrue(channel.mailbox.wait_done(timeout), "controller did not acknowledge")
                        channel.flush()
                    finally:
                        timer.cancel()

                try:
                    exchange()
                    self.assertEqual(len(calls), 1)
                    targets = np.full(count, -0.1 if arm else 0.5)
                    ctrl.set_target(values=targets)
                    exchange()
                    self.assertEqual(len(calls), 2)
                    np.testing.assert_allclose([b.command for b in buffers], targets)
                    previous = env.data.ctrl.copy()
                    for bad in (np.full(count, np.nan), np.full(count, np.inf), np.zeros(count + 1)):
                        ctrl.set_target(values=bad)
                        exchange()
                        np.testing.assert_array_equal(env.data.ctrl, previous)
                        self.assertIsInstance(ctrl.last_error, ValueError)
                    self.assertEqual(ctrl.error_count, 3)
                    ctrl.set_target(values=targets)
                    exchange()
                    self.assertIsNone(ctrl.last_error)
                    self.assertEqual(len(calls), 6)
                finally:
                    channel.close()
                    ctrl.join(2)
                self.assertFalse(ctrl.is_alive())

    def test_channel_rejects_second_controller(self):
        channel = ControllerChannel()
        channel.attach([])
        with self.assertRaises(RuntimeError):
            channel.attach([])


class SimulationLifecycleTests(unittest.TestCase):
    def make_sim(self):
        return ua.MujocoSimulation(
            threading.Event(), ua.Environment(str(MODELS / "environment.xml")),
            ua.Multirotor(str(MODELS / "multirotor.xml")), use_viewer=False,
        )

    def test_startup_error_reaches_caller(self):
        sim = self.make_sim()
        with patch.object(sim, "_build", side_effect=ValueError("invalid model")):
            sim.start()
            with self.assertRaises(RuntimeError) as caught:
                sim.wait_ready(2)
            sim.join(2)
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertFalse(sim.is_alive())

    def test_missing_physics_engine_does_not_report_ready(self):
        sim = self.make_sim()
        module = importlib.import_module("UAMSim.simulation.mujoco_simulation")
        with patch.object(module, "mujoco", None):
            sim.start()
            with self.assertRaises(RuntimeError):
                sim.wait_ready(2)
            sim.join(2)
        self.assertFalse(sim.is_alive())
        self.assertIsNone(sim.uam)

    def test_stop_before_physics(self):
        sim = self.make_sim()
        sim.start()
        try:
            self.assertTrue(sim.wait_ready(2))
        finally:
            sim.stop()
            sim.join(2)
        self.assertFalse(sim.is_alive())
        self.assertEqual(sim.env.data.time, 0.0)
        self.assertIsNone(sim.error)

    def test_runtime_error_is_visible(self):
        sim = self.make_sim()
        with patch.object(sim, "_run_headless", side_effect=ValueError("step failed")):
            sim.start()
            self.assertTrue(sim.wait_ready(2))
            sim.start_physics()
            sim.join(2)
        self.assertFalse(sim.is_alive())
        self.assertIsInstance(sim.error, ValueError)
        self.assertTrue(sim._shutdown_event.is_set())

    def test_shutdown_during_control_wait_does_not_step(self):
        sim = self.make_sim()
        sim._build()
        sim.drone_channel.attach([])
        sim._shutdown_event.set()
        sim._frame()
        self.assertEqual(sim.env.data.time, 0.0)


class PhysicsBaselineTests(unittest.TestCase):
    def test_hover_climb_and_freefall_repeat_after_reset(self):
        env, robot = environment()
        results = []
        for _ in range(2):
            round_results = []
            for target in (1.5, 2.0, None):
                env.reset()
                trajectory = []
                for step in range(1500):
                    state = robot.multirotor.get_state()
                    thrust = 0.0 if target is None else (
                        robot.hover_thrust() + 4 * (target - state.dronePosition[2])
                        - 3 * state.droneVelocity[2]
                    )
                    robot.multirotor.set_actuator(np.full(robot.multirotor.n_rotors, thrust))
                    env.step()
                    trajectory.append(robot.multirotor.get_state().dronePosition.copy())
                trajectory = np.asarray(trajectory)
                self.assertTrue(np.isfinite(trajectory).all())
                round_results.append(trajectory)
            results.append(round_results)
        for first, second in zip(*results):
            np.testing.assert_allclose(first, second, atol=1e-12, rtol=0)
        self.assertLess(np.max(np.abs(results[0][0][:, 2] - 1.5)), 0.01)
        self.assertGreater(results[0][1][-1, 2], 1.9)
        self.assertLess(results[0][2][-1, 2], 0.2)


if __name__ == "__main__":
    unittest.main()
