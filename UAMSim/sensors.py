"""Synchronous camera and planar lidar samples, acquired by the physics owner.

Camera depth is optical-axis distance in metres; lidar rays are instantaneous,
counterclockwise around local +Z, with +X forward. No noise or rolling shutter.
"""

from dataclasses import dataclass
import threading

import mujoco
import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    time: float
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    position_world: np.ndarray
    quaternion_wxyz: np.ndarray  # optical frame: x right, y down, z forward


@dataclass(frozen=True)
class LidarFrame:
    time: float
    ranges: np.ndarray
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    position_world: np.ndarray
    quaternion_wxyz: np.ndarray


class CameraSensor:
    def __init__(self, model, camera, *, width=160, height=120):
        self.model = model
        self.camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if self.camera < 0:
            raise ValueError(f'unknown camera: {camera}')
        if type(width) is not int or type(height) is not int or not (1 <= width <= 1920 and 1 <= height <= 1080):
            raise ValueError('invalid camera resolution')
        self.width, self.height = width, height
        self.owner = threading.get_ident()
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        fy = .5*height/np.tan(np.radians(model.cam_fovy[self.camera])*.5)
        self.intrinsic = np.array([[fy, 0., (width-1)*.5], [0., fy, (height-1)*.5], [0., 0., 1.]])

    def _check_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('camera must be sampled/closed on its creating thread')

    def sample(self, data):
        self._check_owner()
        mujoco.mj_forward(self.model, data)
        self.renderer.update_scene(data, camera=self.camera)
        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        try:
            depth = self.renderer.render().astype(np.float32, copy=True)
        finally:
            self.renderer.disable_depth_rendering()
        far = self.model.vis.map.zfar*self.model.stat.extent
        depth[depth >= far*.999] = np.inf
        optical_rotation = data.cam_xmat[self.camera].reshape(3, 3)@np.diag([1., -1., -1.])
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, optical_rotation.reshape(9))
        return CameraFrame(float(data.time), rgb, depth, self.intrinsic.copy(),
                           data.cam_xpos[self.camera].copy(), quat)

    def close(self):
        self._check_owner()
        self.renderer.close()


class PlanarLidar:
    def __init__(self, model, site, *, rays=181, angle_min=-np.pi, angle_max=np.pi,
                 range_min=.05, range_max=20., exclude_body=-1, geom_groups=None):
        self.model = model
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site < 0:
            raise ValueError(f'unknown lidar site: {site}')
        if type(rays) is not int or not 2 <= rays <= 4096:
            raise ValueError('lidar ray count must be 2..4096')
        if not np.isfinite([angle_min, angle_max, range_min, range_max]).all() or not (
                angle_min < angle_max and 0 <= range_min < range_max):
            raise ValueError('invalid lidar angle/range limits')
        self.angle_min = float(angle_min)
        self.angle_increment = float((angle_max-angle_min)/(rays-1))
        self.range_min, self.range_max = float(range_min), float(range_max)
        angles = np.linspace(angle_min, angle_max, rays)
        self.directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(rays)))
        self.exclude_body = exclude_body
        self.groups = None if geom_groups is None else np.asarray(geom_groups, dtype=np.uint8)
        if self.groups is not None and self.groups.shape != (6,):
            raise ValueError('expected six geometry group flags')
        self.owner = threading.get_ident()

    def sample(self, data):
        if threading.get_ident() != self.owner:
            raise RuntimeError('lidar must be sampled on the physics owner thread')
        mujoco.mj_forward(self.model, data)
        origin = data.site_xpos[self.site].copy()
        rotation = data.site_xmat[self.site].reshape(3, 3)
        directions = self.directions@rotation.T
        distances = np.empty(len(directions), dtype=np.float32)
        hit = np.empty(1, dtype=np.int32)
        for i, direction in enumerate(directions):
            distance = mujoco.mj_ray(self.model, data, origin, direction, self.groups, True, self.exclude_body, hit)
            distances[i] = distance if self.range_min <= distance <= self.range_max else np.inf
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rotation.reshape(9))
        return LidarFrame(float(data.time), distances, self.angle_min, self.angle_increment,
                          self.range_min, self.range_max, origin, quat)


def add_example_sensors(multirotor):
    """Add front-facing camera and lidar above the rotor plane before compilation."""
    if multirotor.model is not None:
        raise ValueError('add sensors before compiling the robot')
    body = multirotor.spec.body(multirotor.BODY_NAME)
    body.add_camera(name='front_camera', pos=[.12, 0., .10], xyaxes=[0, -1, 0, 0, 0, 1], fovy=60.)
    body.add_site(name='lidar', pos=[0., 0., .12], size=[.005], rgba=[0, 1, 0, 0])
