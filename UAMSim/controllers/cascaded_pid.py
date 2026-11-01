"""Reusable near-hover PID for parallel, positive-Z rotors. No thread or mjData access."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..interfaces import ControlTarget, JointPositionCommand, RobotCommand, RobotState, RotorCommand, vector


class Controller(Protocol):
    def reset(self, state: RobotState) -> None: ...

    def compute(self, state: RobotState, target: ControlTarget, dt: float) -> RobotCommand: ...


@dataclass(frozen=True)
class PIDGains:
    position_p: tuple = (4.0, 4.0, 4.0)
    position_d: tuple = (3.0, 3.0, 3.0)
    position_i: tuple = (0.15, 0.15, 1.5)
    position_integral_limit: tuple = (1.0, 1.0, 1.0)
    attitude_p: tuple = (3.0, 3.0, 1.0)
    attitude_d: tuple = (0.30, 0.30, 0.10)
    attitude_i: tuple = (3.0, 3.0, 0.0)
    attitude_integral_limit: tuple = (0.6, 0.6, 0.0)
    tilt_limit: float = 0.3

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name == "tilt_limit":
                continue
            value = vector(getattr(self, name), 3, name)
            if (value < 0).any():
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if not np.isfinite(self.tilt_limit) or not 0 < self.tilt_limit < np.pi / 2:
            raise ValueError("tilt_limit must be between zero and pi/2")


class CascadedPID:
    """Position/yaw + joint-position controller for the existing UAM model.

    Position/attitude integral is frozen after allocation saturation. Joint
    commands pass through to the model's position actuators, not a CAN driver.
    """

    def __init__(self, mass, gravity, allocation, limits, gains=None):
        self.mass, self.gravity = float(mass), float(gravity)
        self.allocation = np.array(allocation, dtype=float, copy=True)
        self.limits = np.array(limits, dtype=float, copy=True)
        if not np.isfinite([self.mass, self.gravity]).all() or min(self.mass, self.gravity) <= 0:
            raise ValueError("mass and gravity must be positive")
        if self.allocation.ndim != 2 or self.allocation.shape[0] != 4:
            raise ValueError("allocation must have shape (4, n_rotors)")
        if not np.isfinite(self.allocation).all() or np.linalg.matrix_rank(self.allocation) != 4:
            raise ValueError("allocation must be finite and have rank four")
        if self.limits.shape != (self.allocation.shape[1], 2) or not np.isfinite(self.limits).all():
            raise ValueError("invalid rotor limits")
        if (self.limits[:, 0] > self.limits[:, 1]).any():
            raise ValueError("invalid rotor limit order")
        self.mixer = np.linalg.pinv(self.allocation)
        self.gains = gains if gains is not None else PIDGains()
        self._initial_joints = None
        self._initialized = False

    @classmethod
    def from_robot(cls, robot, gains=None):
        """Capture model parameters once, before entering the control loop."""
        robot.read_state()
        drone = robot.multirotor
        model, data = robot.model, robot.data
        positions, yaw_coeffs, limits = drone.get_rotor_geometry()
        rotation = data.xmat[drone.body_id].reshape(3, 3)
        for name in drone.rotor_names:
            actuator = drone.rotors[name]
            site = int(model.actuator_trnid[actuator, 0])
            site_rotation = rotation.T @ data.site_xmat[site].reshape(3, 3)
            force = site_rotation @ model.actuator_gear[actuator, :3]
            torque = site_rotation @ model.actuator_gear[actuator, 3:]
            if not np.allclose(force, [0, 0, 1]) or not np.allclose(torque, [0, 0, model.actuator_gear[actuator, 5]]):
                raise ValueError("CascadedPID requires parallel positive-Z rotor actuators")
            if not model.actuator_ctrllimited[actuator]:
                raise ValueError("CascadedPID requires finite rotor control limits")
        gravity = model.opt.gravity
        if not np.allclose(gravity[:2], 0) or gravity[2] >= 0:
            raise ValueError("CascadedPID requires a Z-up world with downward gravity")
        allocation = np.vstack((np.ones(drone.n_rotors), positions[:, 1], -positions[:, 0], yaw_coeffs))
        return cls(model.body_subtreemass[drone.body_id], -gravity[2], allocation, limits, gains)

    def reset(self, state):
        self.position_integral = np.zeros(3)
        self.attitude_integral = np.zeros(3)
        self.saturated = False
        self.allocation_residual = np.zeros(4)
        self._initial_joints = (
            JointPositionCommand(state.joints.names, state.joints.position) if state.joints.names else None
        )
        self._initialized = True

    def compute(self, state, target, dt):
        if not self._initialized:
            raise RuntimeError("reset controller before compute")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("control dt must be positive and finite")
        g = self.gains
        error = target.position_world - state.base.position_world
        if not self.saturated:
            self.position_integral = np.clip(
                self.position_integral + error * dt, -g.position_integral_limit, g.position_integral_limit,
            )
        acceleration = g.position_p * error + g.position_d * (target.velocity_world - state.linear_velocity_world)
        acceleration += g.position_i * self.position_integral
        w, x, y, z = state.base.quaternion_wxyz
        roll = np.arctan2(2 * (w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        desired_roll = np.clip((np.sin(yaw)*acceleration[0] - np.cos(yaw)*acceleration[1]) / self.gravity,
                               -g.tilt_limit, g.tilt_limit)
        desired_pitch = np.clip((np.cos(yaw)*acceleration[0] + np.sin(yaw)*acceleration[1]) / self.gravity,
                                -g.tilt_limit, g.tilt_limit)
        attitude_error = np.array([desired_roll - roll, desired_pitch - pitch, target.yaw - yaw])
        attitude_error = (attitude_error + np.pi) % (2*np.pi) - np.pi
        if not self.saturated:
            self.attitude_integral = np.clip(
                self.attitude_integral + attitude_error * dt, -g.attitude_integral_limit, g.attitude_integral_limit,
            )
        torque = g.attitude_p * attitude_error - g.attitude_d * state.angular_velocity_body
        torque += g.attitude_i * self.attitude_integral
        thrust = self.mass * (self.gravity + acceleration[2]) / max(np.cos(roll)*np.cos(pitch), 0.5)
        wrench = np.r_[thrust, torque]
        raw = self.mixer @ wrench
        allocated = np.clip(raw, self.limits[:, 0], self.limits[:, 1])
        self.saturated = bool(np.any(np.abs(raw - allocated) > 1e-10))
        self.allocation_residual = self.allocation @ allocated - wrench
        return RobotCommand(RotorCommand(allocated), target.joints or self._initial_joints)
