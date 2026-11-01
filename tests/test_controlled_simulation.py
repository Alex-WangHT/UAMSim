import threading
import unittest

import numpy as np

import UAMSim as ua
from test_baseline import MODELS


def runner(arm=False, **kwargs):
    return ua.ControlledSimulation(
        ua.Environment(str(MODELS / "environment.xml")),
        ua.Multirotor(str(MODELS / "multirotor.xml")),
        ua.Manipulator(str(MODELS / "Manipulator.xml")) if arm else None,
        **kwargs,
    )


class ControlledSimulationTests(unittest.TestCase):
    def test_pid_climb_and_reset_repeatability_with_arm(self):
        sim = runner(True)
        controller = ua.CascadedPID.from_robot(sim.robot)
        target = ua.ControlTarget([0.2, 0.1, 2.0])
        runs = []
        for _ in range(2):
            sim.reset(controller)
            self.assertEqual(sim.frame, 0)
            self.assertEqual(sim.state.time, 0)
            self.assertIsNone(sim.last_command)
            trajectory = []
            for i in range(800):
                state = sim.step(target)
                trajectory.append(np.r_[state.base.position_world, state.base.quaternion_wxyz])
            runs.append(np.asarray(trajectory))
            self.assertAlmostEqual(state.time, 4.0)
            self.assertLess(np.linalg.norm(state.base.position_world - target.position_world), 0.15)
            self.assertLess(np.linalg.norm(state.angular_velocity_body), 0.1)
        np.testing.assert_allclose(*runs, atol=1e-12, rtol=0)
        sim.close()

    def test_control_period_and_joint_target(self):
        sim = runner(True, control_steps=10)
        sim.reset(ua.CascadedPID.from_robot(sim.robot))
        target = ua.ControlTarget([0, 0, 1.5], joints=ua.JointPositionCommand(sim.state.joints.names, [0, -0.5, 0]))
        for i in range(200):
            state = sim.step(target)
        self.assertAlmostEqual(state.time, 2.0)
        self.assertLess(abs(state.joints.position[1] + 0.5), 0.03)
        self.assertLess(np.linalg.norm(state.base.position_world - target.position_world), 0.15)

    def test_owner_and_closed_guards(self):
        sim = runner()
        errors = []
        def other_thread():
            try:
                sim.reset()
            except RuntimeError as error:
                errors.append(error)
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(2)
        self.assertEqual(len(errors), 1)
        sim.close()
        with self.assertRaises(RuntimeError):
            sim.reset()

    def test_invalid_controller_command_does_not_advance(self):
        class InvalidController:
            def reset(self, state):
                pass
            def compute(self, state, target, dt):
                return ua.RobotCommand(ua.RotorCommand([1000]*6))
        sim = runner()
        sim.reset(InvalidController())
        with self.assertRaises(ValueError):
            sim.step(ua.ControlTarget([0, 0, 1.5]))
        self.assertEqual(sim.env.data.time, 0)
        self.assertEqual(sim.frame, 0)
        np.testing.assert_array_equal(sim.env.data.ctrl, 0)

    def test_pacing_preserves_dynamics(self):
        trajectories = []
        for factor in (0, 1):
            sim = runner(real_time_factor=factor)
            sim.reset(ua.CascadedPID.from_robot(sim.robot))
            trajectories.append([sim.step(ua.ControlTarget([0, 0, 2])).base.position_world for _ in range(10)])
        np.testing.assert_array_equal(*trajectories)

    def test_saturation_and_reset(self):
        sim = runner()
        pid = ua.CascadedPID.from_robot(sim.robot)
        pid.reset(sim.state)
        output = pid.compute(sim.state, ua.ControlTarget([0, 0, 100]), 0.005)
        self.assertTrue(pid.saturated)
        self.assertTrue((output.rotors.thrust <= pid.limits[:, 1]).all())
        integral = pid.position_integral.copy()
        pid.compute(sim.state, ua.ControlTarget([0, 0, 100]), 0.005)
        np.testing.assert_array_equal(integral, pid.position_integral)
        pid.reset(sim.state)
        self.assertFalse(pid.saturated)
        np.testing.assert_array_equal(pid.position_integral, 0)

    def test_hover_thrust_uses_robot_mass_and_configured_gravity(self):
        sim = runner(True)
        sim.env.model.opt.gravity[:] = [0, 0, -3.0]
        sim.env.model.body_mass[0] = 1000  # unrelated world mass must not enter robot balance
        expected = sim.env.model.body_subtreemass[sim.robot.multirotor.body_id] * 3 / 6
        self.assertAlmostEqual(sim.robot.hover_thrust(), expected)

    def test_arm_motion_remains_bounded(self):
        sim = runner(True)
        sim.reset(ua.CascadedPID.from_robot(sim.robot))
        peak_error = 0.0
        for i in range(1200):
            angle = -0.5 * (1 - np.cos(2*np.pi*sim.state.time/3))
            target = ua.ControlTarget([0, 0, 1.5], joints=ua.JointPositionCommand(sim.state.joints.names, [0, angle, 0]))
            state = sim.step(target)
            peak_error = max(peak_error, np.linalg.norm(state.base.position_world-target.position_world))
        self.assertLess(peak_error, 0.25)
        self.assertTrue(np.isfinite(state.base.quaternion_wxyz).all())

    def test_invalid_configuration(self):
        for steps in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                runner(control_steps=steps)
        for factor in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                runner(real_time_factor=factor)


if __name__ == "__main__":
    unittest.main()
