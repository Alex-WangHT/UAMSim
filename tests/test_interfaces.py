"""State conventions and atomic command delivery against the actual model."""

import unittest

import numpy as np

import UAMSim as ua
from test_baseline import environment


class InterfaceTests(unittest.TestCase):
    def test_snapshot_frames_and_independence(self):
        env, robot = environment(True)
        root_joint = int(env.model.body_jntadr[robot.multirotor.body_id])
        qadr = int(env.model.jnt_qposadr[root_joint])
        vadr = int(env.model.jnt_dofadr[root_joint])
        env.data.qpos[qadr + 3:qadr + 7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        env.data.qvel[vadr:vadr + 6] = [1, 2, 3, 4, 5, 6]
        state = robot.read_state()
        np.testing.assert_allclose(state.linear_velocity_world, [1, 2, 3], atol=1e-12)
        np.testing.assert_allclose(state.angular_velocity_body, [4, 5, 6], atol=1e-12)
        np.testing.assert_allclose(state.base.quaternion_wxyz, [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
        self.assertIsNotNone(state.end_effector)
        old_position = state.base.position_world.copy()
        env.step()
        robot.read_state()
        np.testing.assert_array_equal(state.base.position_world, old_position)
        with self.assertRaises(ValueError):
            state.base.position_world[0] = 2

    def test_command_routes_by_name_and_preserves_other_group(self):
        env, robot = environment(True)
        names = tuple(reversed(robot.manipulator.joint_names))
        targets = [-0.1, -0.2, -0.3]
        robot.apply_command(ua.RobotCommand(
            ua.RotorCommand(np.full(6, 2.0)), ua.JointPositionCommand(names, targets),
        ))
        for name, target in zip(names, targets):
            self.assertEqual(env.data.ctrl[robot.manipulator.servos[name]], target)
        robot.apply_command(ua.RobotCommand(rotors=ua.RotorCommand(np.full(6, 1.0))))
        for name, target in zip(names, targets):
            self.assertEqual(env.data.ctrl[robot.manipulator.servos[name]], target)

    def test_invalid_combined_command_is_atomic(self):
        env, robot = environment(True)
        before = env.data.ctrl.copy()
        for joint_command in (
            ua.JointPositionCommand(("unknown",), [0]),
            ua.JointPositionCommand(tuple(robot.manipulator.joint_names), [100, 100, 100]),
        ):
            with self.assertRaises(ValueError):
                robot.apply_command(ua.RobotCommand(ua.RotorCommand(np.ones(6)), joint_command))
            np.testing.assert_array_equal(env.data.ctrl, before)

    def test_invalid_payloads(self):
        for value in ([np.nan], [np.inf], [[1, 2]]):
            with self.assertRaises(ValueError):
                ua.RotorCommand(value)
        with self.assertRaises(ValueError):
            ua.Pose([0, 0, 0], [0, 0, 0, 0])
        with self.assertRaises(ValueError):
            ua.JointPositionCommand(("a", "a"), [0, 0])

    def test_no_arm_snapshot_and_command_rejection(self):
        env, robot = environment()
        state = robot.read_state()
        self.assertEqual(state.joints.names, ())
        self.assertIsNone(state.end_effector)
        with self.assertRaises(ValueError):
            robot.apply_command(ua.RobotCommand(joints=ua.JointPositionCommand(("joint1",), [0])))

    def test_pose_matches_integrated_state_after_step(self):
        env, robot = environment()
        robot.apply_command(ua.RobotCommand(rotors=ua.RotorCommand(np.zeros(6))))
        env.step(20)
        state = robot.read_state()
        root_joint = int(env.model.body_jntadr[robot.multirotor.body_id])
        qadr = int(env.model.jnt_qposadr[root_joint])
        np.testing.assert_allclose(state.base.position_world, env.data.qpos[qadr:qadr + 3], atol=1e-12)
        self.assertAlmostEqual(state.time, 20 * env.model.opt.timestep)


if __name__ == "__main__":
    unittest.main()
