"""Simulation implementation of the same named-radian arm driver contract."""

import threading

from ..interfaces import JointPositionCommand, RobotCommand


class SimulationArmDriver:
    def __init__(self, robot):
        if robot.manipulator is None:
            raise ValueError('robot has no manipulator')
        self.robot = robot
        self.owner = threading.get_ident()

    def _check(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('simulation arm driver must run on the physics owner')

    def read_positions(self):
        self._check()
        arm = self.robot.manipulator
        return dict(zip(arm.joint_names, arm.get_joint_positions().tolist()))

    def write_positions(self, targets):
        self._check()
        self.robot.apply_command(RobotCommand(joints=JointPositionCommand(tuple(targets), tuple(targets.values()))))

    def stop(self):
        """Capture measured position as servo target; not an instantaneous physical stop."""
        self.write_positions(self.read_positions())
