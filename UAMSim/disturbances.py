"""Explicit world wind and external end-effector wrench models for MuJoCo.

Linear drag coefficients are model parameters, not identified aerodynamics.
External wrenches are applied at their physical points using mj_applyFT.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from .interfaces import vector


@dataclass(frozen=True)
class WindConfig:
    velocity_world: tuple = (0.0, 0.0, 0.0)  # m/s
    gust_amplitude: tuple = (0.0, 0.0, 0.0)  # m/s, sinusoidal, independently seeded phases
    gust_period: float = 2.0  # s
    drag_body: tuple = (0.3, 0.3, 0.3)  # N/(m/s), principal body axes
    point_body: tuple = (0.0, 0.0, 0.0)  # m, force application point relative to base origin

    def __post_init__(self):
        for name in ("velocity_world", "gust_amplitude", "drag_body", "point_body"):
            value = vector(getattr(self, name), 3, name)
            if name in ("gust_amplitude", "drag_body") and (value < 0).any():
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, tuple(value))
        if not np.isfinite(self.gust_period) or self.gust_period <= 0:
            raise ValueError("gust_period must be positive and finite")


@dataclass(frozen=True)
class EndEffectorWrench:
    force: tuple = (0.0, 0.0, 0.0)  # N
    torque: tuple = (0.0, 0.0, 0.0)  # N m
    frame: str = "world"  # world or site (end-effector axes)
    start: float = 1.0
    end: float = 3.0

    def __post_init__(self):
        for name in ("force", "torque"):
            object.__setattr__(self, name, tuple(vector(getattr(self, name), 3, name)))
        if self.frame not in ("world", "site"):
            raise ValueError("wrench frame must be world or site")
        if not np.isfinite([self.start, self.end]).all() or self.start < 0 or self.end <= self.start:
            raise ValueError("wrench requires 0 <= start < end, both finite")


@dataclass(frozen=True)
class DisturbanceConfig:
    wind: WindConfig | None = None
    end_effector: EndEffectorWrench | None = None
    seed: int = 0

    def __post_init__(self):
        if self.wind is not None and not isinstance(self.wind, WindConfig):
            raise TypeError("wind must be WindConfig")
        if self.end_effector is not None and not isinstance(self.end_effector, EndEffectorWrench):
            raise TypeError("end_effector must be EndEffectorWrench")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    @classmethod
    def from_dict(cls, config):
        return cls(wind=WindConfig(**config["wind"]) if config.get("wind") else None,
                   end_effector=EndEffectorWrench(**config["end_effector"]) if config.get("end_effector") else None,
                   seed=config.get("seed", 0))


class DisturbanceField:
    def __init__(self, robot, config):
        if not isinstance(config, DisturbanceConfig):
            raise TypeError("expected DisturbanceConfig")
        if config.end_effector is not None and robot.manipulator is None:
            raise ValueError("end-effector wrench requires a manipulator")
        self.robot, self.config = robot, config
        self.enabled = bool(
            (config.wind is not None and np.any(config.wind.drag_body))
            or (config.end_effector is not None and (np.any(config.end_effector.force) or np.any(config.end_effector.torque)))
        )
        self.reset()

    def reset(self):
        self.phases = np.random.default_rng(self.config.seed).uniform(0, 2*np.pi, 3)
        self.last_sample = None

    def generalized_force(self):
        """Compute additive generalized forces; never overwrite caller forces."""
        robot = self.robot
        model, data = robot.model, robot.data
        applied = np.zeros(model.nv)
        if not self.enabled:
            return applied
        mujoco.mj_forward(model, data)
        sample = dict(time=float(data.time), wind_velocity_world=np.zeros(3), wind_force_world=np.zeros(3),
                      wind_point_world=None, ee_force_world=np.zeros(3), ee_torque_world=np.zeros(3), ee_point_world=None)
        wind = self.config.wind
        if wind is not None and np.any(wind.drag_body):
            body = robot.multirotor.body_id
            rotation = data.xmat[body].reshape(3, 3)
            offset = rotation @ wind.point_body
            point = data.xpos[body] + offset
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            point_velocity = velocity[3:] + np.cross(velocity[:3], offset)
            wind_velocity = np.asarray(wind.velocity_world) + np.asarray(wind.gust_amplitude) * np.sin(
                2*np.pi*data.time/wind.gust_period + self.phases)
            force = rotation @ (np.asarray(wind.drag_body) * (rotation.T @ (wind_velocity - point_velocity)))
            mujoco.mj_applyFT(model, data, force, np.zeros(3), point, body, applied)
            sample.update(wind_velocity_world=wind_velocity, wind_force_world=force, wind_point_world=point.copy())
        wrench = self.config.end_effector
        if wrench is not None and wrench.start <= data.time < wrench.end:
            site = robot.manipulator.ee_site_id
            rotation = data.site_xmat[site].reshape(3, 3) if wrench.frame == "site" else np.eye(3)
            force, torque = rotation @ wrench.force, rotation @ wrench.torque
            point = data.site_xpos[site]
            mujoco.mj_applyFT(model, data, force, torque, point, int(model.site_bodyid[site]), applied)
            sample.update(ee_force_world=force, ee_torque_world=torque, ee_point_world=point.copy())
        self.last_sample = sample
        if not np.isfinite(applied).all():
            raise ValueError("disturbance produced non-finite force")
        return applied
