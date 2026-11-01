"""Flight benchmark references and explicit sampled termination criteria."""

from dataclasses import dataclass

import numpy as np

from .interfaces import ControlTarget, JointPositionCommand, vector


@dataclass(frozen=True)
class TaskConfig:
    name: str = "hover"
    duration: float = 6.0
    center: tuple = (0.0, 0.0, 1.5)
    target_jitter: float = 0.05
    amplitude: float = 0.15
    period: float = 3.0
    arm_joint_index: int = 1
    arm_amplitude: float = 0.5
    position_tolerance: float = 0.15
    velocity_tolerance: float = 0.25
    joint_tolerance: float = 0.1
    yaw_tolerance: float = 0.15
    hold_seconds: float = 1.0
    failure_position_error: float = 1.0
    failure_tilt: float = 1.0
    minimum_height: float = 0.3

    def __post_init__(self):
        if self.name not in ("hover", "tracking", "arm_motion"):
            raise ValueError("unknown task name")
        object.__setattr__(self, "center", tuple(vector(self.center, 3, "center")))
        for name in ("duration", "period", "position_tolerance", "velocity_tolerance", "joint_tolerance",
                     "yaw_tolerance", "hold_seconds", "failure_position_error", "failure_tilt"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("target_jitter", "amplitude", "arm_amplitude"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not np.isfinite(self.minimum_height) or self.hold_seconds > self.duration:
            raise ValueError("invalid height or hold duration")
        if isinstance(self.arm_joint_index, bool) or not isinstance(self.arm_joint_index, int) or self.arm_joint_index < 0:
            raise ValueError("arm_joint_index must be a nonnegative integer")
        if self.failure_position_error <= self.position_tolerance or self.failure_tilt > np.pi:
            raise ValueError("invalid failure thresholds")


class FlightTask:
    def __init__(self, config: TaskConfig, initial_state, seed: int):
        self.config = config
        rng = np.random.default_rng(seed)
        self.center = np.asarray(config.center) + rng.uniform(-config.target_jitter, config.target_jitter, 3)
        self.joint_names = initial_state.joints.names
        self.initial_joints = initial_state.joints.position.copy()
        if config.name == "arm_motion" and config.arm_joint_index >= len(self.joint_names):
            raise ValueError("arm_motion requires the selected arm joint")
        self.within_seconds = 0.0

    def target(self, time):
        cfg = self.config
        omega = 2 * np.pi / cfg.period
        position = self.center.copy()
        velocity = np.zeros(3)
        acceleration = np.zeros(3)
        joints = self.initial_joints.copy()
        if cfg.name == "tracking":
            position[:2] += cfg.amplitude * np.array([np.sin(omega*time), 0.5*np.sin(2*omega*time)])
            velocity[:2] = cfg.amplitude * omega * np.array([np.cos(omega*time), np.cos(2*omega*time)])
            acceleration[:2] = -cfg.amplitude * omega**2 * np.array([np.sin(omega*time), 2*np.sin(2*omega*time)])
        if cfg.name == "arm_motion":
            joints[cfg.arm_joint_index] -= cfg.arm_amplitude * (1 - np.cos(omega*time))
        joint_target = JointPositionCommand(self.joint_names, joints) if self.joint_names else None
        return ControlTarget(position, velocity_world=velocity, joints=joint_target, acceleration_world=acceleration)

    def observe(self, state, dt, contact=False):
        """Evaluate the post-step reference at the sample's simulation timestamp.

        Success requires the entire final hold window, and never ends early.
        Failure takes precedence over success/timeout at the deadline.
        """
        target = self.target(state.time)
        position_error = float(np.linalg.norm(state.base.position_world - target.position_world))
        velocity_error = float(np.linalg.norm(state.linear_velocity_world - target.velocity_world))
        w, x, y, z = state.base.quaternion_wxyz
        tilt = float(np.arccos(np.clip(1 - 2*(x*x + y*y), -1, 1)))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        yaw_error = float(abs((yaw - target.yaw + np.pi) % (2*np.pi) - np.pi))
        joint_error = float(np.max(np.abs(state.joints.position - target.joints.position))) if target.joints else 0.0
        metrics = dict(position_error_m=position_error, velocity_error_m_s=velocity_error,
                       tilt_rad=tilt, yaw_error_rad=yaw_error, joint_error_rad=joint_error)
        if not np.isfinite(list(metrics.values())).all():
            return "failed", "nonfinite_state", metrics
        cfg = self.config
        reason = None
        if contact:
            reason = "collision"
        elif state.base.position_world[2] < cfg.minimum_height:
            reason = "minimum_height"
        elif tilt > cfg.failure_tilt:
            reason = "excessive_tilt"
        elif position_error > cfg.failure_position_error:
            reason = "position_divergence"
        if reason:
            return "failed", reason, metrics
        within = (position_error <= cfg.position_tolerance and velocity_error <= cfg.velocity_tolerance
                  and yaw_error <= cfg.yaw_tolerance and joint_error <= cfg.joint_tolerance)
        self.within_seconds = self.within_seconds + dt if within else 0.0
        if state.time + 1e-9 >= cfg.duration:
            return (
                "succeeded" if self.within_seconds + 1e-9 >= cfg.hold_seconds else "timeout",
                "final_hold" if self.within_seconds + 1e-9 >= cfg.hold_seconds else "tracking_tolerance",
                metrics,
            )
        return "running", None, metrics
