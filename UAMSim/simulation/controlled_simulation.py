"""Single-owner synchronous execution, optionally paced for real-time use."""

import threading
import time

import numpy as np

from ..controllers.cascaded_pid import Controller
from ..interfaces import ControlTarget, RobotCommand
from ..disturbances import DisturbanceField
from .environment import Environment


class ControlledSimulation:
    """Own an uncompiled environment and advance exactly control_steps per call.

    Unlike MujocoSimulation this runner creates no threads. Construct, reset,
    and step it from one thread; external transports must use command buffers.
    The controller only sees copied RobotState and never accesses mjData.
    """

    def __init__(self, environment, multirotor, manipulator=None, *, control_steps=5, real_time_factor=0.0,
                 disturbance=None):
        if not isinstance(environment, Environment) or environment.model is not None:
            raise ValueError("ControlledSimulation requires an uncompiled Environment")
        if isinstance(control_steps, bool) or not isinstance(control_steps, int) or control_steps < 1:
            raise ValueError("control_steps must be a positive integer")
        if not np.isfinite(real_time_factor) or real_time_factor < 0:
            raise ValueError("real_time_factor must be finite and nonnegative; zero disables pacing")
        self._owner = threading.get_ident()
        self.env = environment
        self.robot = environment.attach_robot(multirotor, manipulator)
        self.control_steps = control_steps
        self.control_dt = float(environment.model.opt.timestep) * control_steps
        self.real_time_factor = real_time_factor
        self.controller: Controller | None = None
        self.last_command = None
        self.last_control_seconds = 0.0
        self.frame = 0
        self.state = self.robot.read_state()
        self.closed = False
        self._needs_reset = False
        self.disturbance = DisturbanceField(self.robot, disturbance) if disturbance is not None else None

    def configure_disturbance(self, config):
        """Configure the next episode; reset is required before further stepping."""
        self._check_owner()
        self.disturbance = DisturbanceField(self.robot, config) if config is not None else None
        self._needs_reset = True

    def _check_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("simulation must be accessed by its owner thread")
        if self.closed:
            raise RuntimeError("simulation is closed")

    def reset(self, controller=None):
        self._check_owner()
        if controller is not None:
            self.controller = controller
        self.env.reset()
        if self.disturbance is not None:
            self.disturbance.reset()
        self._needs_reset = False
        self.state = self.robot.read_state()
        self.last_command = None
        self.last_control_seconds = 0.0
        self.frame = 0
        if self.controller is not None:
            self.controller.reset(self.state)
        return self.state

    def step(self, target: ControlTarget):
        self._check_owner()
        if self.controller is None:
            raise RuntimeError("reset with a controller before stepping")
        started = time.perf_counter()
        command = self.controller.compute(self.state, target, self.control_dt)
        self.last_control_seconds = time.perf_counter() - started
        return self._advance(command, started)

    def step_command(self, command: RobotCommand):
        """Advance using an external actuator command (replay or future SITL adapter)."""
        self._check_owner()
        self.last_control_seconds = 0.0
        return self._advance(command, time.perf_counter())

    def _advance(self, command, started):
        if self._needs_reset:
            raise RuntimeError("reset after configuring disturbances")
        self.robot.apply_command(command)
        if self.disturbance is None or not self.disturbance.enabled:
            self.env.step(self.control_steps)
        else:
            for _ in range(self.control_steps):
                applied = self.disturbance.generalized_force()
                original_forces = self.env.data.qfrc_applied.copy()
                self.env.data.qfrc_applied[:] += applied
                try:
                    self.env.step()
                finally:
                    self.env.data.qfrc_applied[:] = original_forces
        self.state = self.robot.read_state()
        self.last_command = command
        self.frame += 1
        if self.real_time_factor:
            remaining = self.control_dt / self.real_time_factor - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
        return self.state

    def read_imu(self):
        """Sample current IMU including instantaneous disturbances on the owner thread.

        A normal mj_forward after a step recomputes acceleration with temporary
        disturbances removed. Restore caller forces even if sampling fails.
        """
        import mujoco

        self._check_owner()
        original = self.env.data.qfrc_applied.copy()
        try:
            if self.disturbance is not None and self.disturbance.enabled:
                self.env.data.qfrc_applied[:] += self.disturbance.generalized_force()
            mujoco.mj_forward(self.env.model, self.env.data)
            return self.robot.multirotor.get_imu()
        finally:
            self.env.data.qfrc_applied[:] = original

    def close(self):
        self._check_owner()
        self.closed = True
