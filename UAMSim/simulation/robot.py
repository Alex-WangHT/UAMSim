"""已挂载机器人的视图句柄。

``Robot`` 由 ``Environment.attach_robot()`` 返回（也存于 ``Environment.robots``），
把绑定到场景共享模型上的 ``Multirotor`` / ``Manipulator`` 组件视图成组携带，
并提供整机便捷量（如悬停推力）。spec 组合与编译全部由 ``Environment`` 完成，
本类不含任何组合/编译逻辑。
"""

from __future__ import annotations

import numpy as np

from ..interfaces import JointState, Pose, RobotCommand, RobotState

from .manipulator import Manipulator
from .multirotor import Multirotor


class Robot:
    """已挂载机器人的视图句柄（``Environment.attach_robot()`` 的返回类型）。

    属性：
    - ``multirotor`` / ``manipulator``：绑定到场景共享模型的组件视图
      （纯多旋翼构型下 ``manipulator`` 为 ``None``）
    - ``prefix``：本机器人在场景中的名称前缀（如 ``uam/``）
    """

    def __init__(
        self,
        multirotor: Multirotor,
        manipulator: Manipulator | None,
        prefix: str,
    ):
        self.multirotor = multirotor
        self.manipulator = manipulator
        self.prefix = prefix

    @property
    def has_manipulator(self) -> bool:
        """当前构型是否包含机械臂。"""
        return self.manipulator is not None

    @property
    def model(self):
        """组件绑定的场景共享 MjModel（未绑定时为 None）。"""
        return self.multirotor.model

    @property
    def data(self):
        """组件绑定的场景共享 MjData（未绑定时为 None）。"""
        return self.multirotor.data

    def hover_thrust(self) -> float:
        """整机悬停所需的单旋翼推力 [N]（总质量 × g / 旋翼数）。"""
        if self.model is None or self.data is None:
            raise RuntimeError(
                "Robot 尚未绑定编译产物：请经 Environment.attach_robot() 挂载编译后再调用"
            )
        total_mass = float(self.model.body_subtreemass[self.multirotor.body_id])
        return total_mass * float(np.linalg.norm(self.model.opt.gravity)) / self.multirotor.n_rotors

    def read_state(self) -> RobotState:
        """Copy one coherent snapshot; call only from the physics owner thread.

        Forward dynamics refreshes derived poses/velocities after mj_step.
        This does not advance simulation time.
        """
        import mujoco

        model, data = self.model, self.data
        mujoco.mj_forward(model, data)
        body = self.multirotor.body_id
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        rotation = data.xmat[body].reshape(3, 3)
        joints = JointState((), np.empty(0), np.empty(0))
        ee = None
        if self.manipulator is not None:
            arm = self.manipulator
            joints = JointState(tuple(arm.joint_names), arm.get_joint_positions(), arm.get_joint_velocities())
            position, matrix = arm.get_ee_pose()
            quaternion = np.zeros(4)
            mujoco.mju_mat2Quat(quaternion, matrix.reshape(9))
            ee = Pose(position, quaternion)
        return RobotState(
            float(data.time), float(model.opt.timestep), Pose(data.xpos[body], data.xquat[body]),
            velocity[3:], rotation.T @ velocity[:3], joints, ee,
        )

    def apply_command(self, command: RobotCommand) -> None:
        """Validate the complete command before writing any actuator.

        Unspecified groups hold their previous command. Reject out-of-range
        commands instead of silently clipping. Only the physics owner may call.
        Joint names must match the model; ordering may differ.
        """
        if not isinstance(command, RobotCommand):
            raise TypeError("expected RobotCommand")
        ids, values = [], []
        if command.rotors is not None:
            if len(command.rotors.thrust) != self.multirotor.n_rotors:
                raise ValueError("rotor count does not match robot")
            ids.extend(self.multirotor.rotors[n] for n in self.multirotor.rotor_names)
            values.extend(command.rotors.thrust)
        if command.joints is not None:
            arm = self.manipulator
            if arm is None or set(command.joints.names) != set(arm.joint_names):
                raise ValueError("joint names do not match robot")
            ids.extend(arm.servos[n] for n in command.joints.names)
            values.extend(command.joints.position)
        if not ids:
            return
        values = np.asarray(values)
        limits = self.model.actuator_ctrlrange[ids]
        limited = self.model.actuator_ctrllimited[ids].astype(bool)
        if not np.isfinite(values).all() or np.any(limited & ((values < limits[:, 0]) | (values > limits[:, 1]))):
            raise ValueError("actuator command is non-finite or outside model limits")
        self.data.ctrl[ids] = values
