"""Simulator-independent state and command contracts (SI units).

World axes are those of the configured model, not implicitly ENU or NED.
Quaternions use wxyz and rotate body vectors into world coordinates.
Transport adapters own ROS/PX4 frame conversion and CANopen device scaling.
"""

from dataclasses import dataclass

import numpy as np


def vector(value, size, name):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Pose:
    position_world: np.ndarray
    quaternion_wxyz: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "position_world", vector(self.position_world, 3, "position_world"))
        q = vector(self.quaternion_wxyz, 4, "quaternion_wxyz")
        if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-6, rtol=0):
            raise ValueError("quaternion must have unit norm")
        object.__setattr__(self, "quaternion_wxyz", q)


@dataclass(frozen=True)
class JointState:
    names: tuple[str, ...]
    position: np.ndarray  # rad
    velocity: np.ndarray  # rad/s

    def __post_init__(self):
        names = tuple(self.names)
        if len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("joint names must be unique nonempty strings")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "position", vector(self.position, len(names), "position"))
        object.__setattr__(self, "velocity", vector(self.velocity, len(names), "velocity"))


@dataclass(frozen=True)
class RobotState:
    time: float  # simulation seconds, reset with each episode
    physics_dt: float
    base: Pose
    linear_velocity_world: np.ndarray  # m/s, at base body origin
    angular_velocity_body: np.ndarray  # rad/s
    joints: JointState
    end_effector: Pose | None = None

    def __post_init__(self):
        if not np.isfinite(self.time) or self.time < 0:
            raise ValueError("time must be finite and nonnegative")
        if not np.isfinite(self.physics_dt) or self.physics_dt <= 0:
            raise ValueError("physics_dt must be finite and positive")
        object.__setattr__(self, "linear_velocity_world", vector(self.linear_velocity_world, 3, "linear velocity"))
        object.__setattr__(self, "angular_velocity_body", vector(self.angular_velocity_body, 3, "angular velocity"))


@dataclass(frozen=True)
class RotorCommand:
    thrust: np.ndarray  # N, in the robot's published rotor order

    def __post_init__(self):
        object.__setattr__(self, "thrust", vector(self.thrust, len(self.thrust), "thrust"))


@dataclass(frozen=True)
class JointPositionCommand:
    names: tuple[str, ...]
    position: np.ndarray  # rad; device encoder scaling belongs to the adapter

    def __post_init__(self):
        state = JointState(self.names, self.position, np.zeros(len(self.names)))
        object.__setattr__(self, "names", state.names)
        object.__setattr__(self, "position", state.position)


@dataclass(frozen=True)
class RobotCommand:
    rotors: RotorCommand | None = None
    joints: JointPositionCommand | None = None


@dataclass(frozen=True)
class ControlTarget:
    position_world: np.ndarray
    yaw: float = 0.0
    velocity_world: np.ndarray | None = None
    joints: JointPositionCommand | None = None
    acceleration_world: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "position_world", vector(self.position_world, 3, "target position"))
        velocity = np.zeros(3) if self.velocity_world is None else self.velocity_world
        object.__setattr__(self, "velocity_world", vector(velocity, 3, "target velocity"))
        acceleration = np.zeros(3) if self.acceleration_world is None else self.acceleration_world
        object.__setattr__(self, "acceleration_world", vector(acceleration, 3, "target acceleration"))
        if not np.isfinite(self.yaw):
            raise ValueError("target yaw must be finite")
