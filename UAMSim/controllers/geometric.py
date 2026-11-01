"""SO(3) attitude feedback with trajectory acceleration and desired-rate feedforward.

Uses the rotation-error construction of Lee, Leok & McClamroch (2010),
https://arxiv.org/abs/1003.2005 . This bounded integral implementation uses
finite-difference desired rates; it is not the paper's full rigid-body law
or a whole-body aerial-manipulator controller.
"""

import numpy as np

from ..interfaces import RobotCommand, RotorCommand
from .cascaded_pid import CascadedPID


def rotation_matrix(quaternion):
    w, x, y, z = quaternion
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def vee(skew):
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


class GeometricController(CascadedPID):
    """Same robot validation/allocation as PID, with geometric attitude control.

    Derivative history and integral state are reset per episode. Attitude rate
    feedforward is bounded to 4 rad/s to handle discontinuous user targets.
    No disturbance truth or future task state is read by this controller.
    """

    def reset(self, state):
        super().reset(state)
        self._previous_desired_rotation = None
        self.desired_rotation = np.eye(3)
        self.desired_rate = np.zeros(3)

    def compute(self, state, target, dt):
        if not self._initialized:
            raise RuntimeError("reset controller before compute")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("control dt must be positive and finite")
        g = self.gains
        position_error = target.position_world - state.base.position_world
        velocity_error = target.velocity_world - state.linear_velocity_world
        if not self.saturated:
            self.position_integral = np.clip(self.position_integral + position_error*dt,
                                             -g.position_integral_limit, g.position_integral_limit)
        acceleration = (target.acceleration_world + g.position_p*position_error + g.position_d*velocity_error
                        + g.position_i*self.position_integral)
        desired_force = self.mass * (acceleration + [0, 0, self.gravity])
        # Keep the commanded thrust axis above the horizon and inside the configured cone.
        limited = bool(desired_force[2] < 0.1*self.mass*self.gravity)
        desired_force[2] = max(desired_force[2], 0.1*self.mass*self.gravity)
        horizontal = np.linalg.norm(desired_force[:2])
        horizontal_limit = desired_force[2] * np.tan(g.tilt_limit)
        if horizontal > horizontal_limit:
            desired_force[:2] *= horizontal_limit / horizontal
            limited = True
        z_desired = desired_force / np.linalg.norm(desired_force)
        heading = np.array([np.cos(target.yaw), np.sin(target.yaw), 0.0])
        y_desired = np.cross(z_desired, heading)
        y_desired /= np.linalg.norm(y_desired)
        desired = np.column_stack((np.cross(y_desired, z_desired), y_desired, z_desired))
        rotation = rotation_matrix(state.base.quaternion_wxyz)
        attitude_error = 0.5 * vee(desired.T @ rotation - rotation.T @ desired)
        desired_rate = np.zeros(3)
        if self._previous_desired_rotation is not None:
            relative = self._previous_desired_rotation.T @ desired
            desired_rate = vee(relative - relative.T) / (2*dt)
            norm = np.linalg.norm(desired_rate)
            if norm > 4:
                desired_rate *= 4/norm
        rate_error = state.angular_velocity_body - rotation.T @ desired @ desired_rate
        if not self.saturated:
            self.attitude_integral = np.clip(self.attitude_integral - attitude_error*dt,
                                             -g.attitude_integral_limit, g.attitude_integral_limit)
        torque = -g.attitude_p*attitude_error - g.attitude_d*rate_error + g.attitude_i*self.attitude_integral
        thrust = float(desired_force @ rotation[:, 2])
        wrench = np.r_[thrust, torque]
        raw = self.mixer @ wrench
        allocated = np.clip(raw, self.limits[:, 0], self.limits[:, 1])
        self.saturated = limited or bool(np.any(np.abs(raw-allocated) > 1e-10))
        self.allocation_residual = self.allocation @ allocated - wrench
        self.desired_rotation, self.desired_rate = desired, desired_rate
        self._previous_desired_rotation = desired.copy()
        return RobotCommand(RotorCommand(allocated), target.joints or self._initial_joints)
