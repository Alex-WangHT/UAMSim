"""PX4 MAVLink simulator boundary. World NWU, body FLU; PX4 NED/FRD.

Optional pymavlink is imported only when the TCP transport is constructed.
The physics owner calls every method; no transport thread touches mjData.
"""

import math
import socket
import time

import numpy as np

from .interfaces import RotorCommand


def to_px4(vector):
    """NWU -> NED or FLU -> FRD (same proper rotation)."""
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("expected a finite vector of length three")
    return value * np.array([1., -1., -1.])


def px4_quaternion(wxyz):
    """Convert FLU-to-NWU orientation to FRD-to-NED, scalar first."""
    q = np.asarray(wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1):
        raise ValueError("expected a unit quaternion")
    return q * np.array([1., 1., -1., -1.])


def rotor_command(controls, mode, maximum):
    """Linear normalized thrust model; PX4 THR_MDL_FAC must be zero."""
    limits = np.asarray(maximum, dtype=float)
    values = np.asarray(controls, dtype=float)
    if limits.ndim != 1 or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("invalid motor limits")
    if values.ndim != 1 or len(values) < len(limits):
        raise ValueError("missing motor outputs")
    values = values[:len(limits)]
    if not np.isfinite(values).all() or np.any(values < -1e-6) or np.any(values > 1+1e-6):
        raise ValueError("PX4 motor outputs must be normalized to [0,1]")
    return RotorCommand(np.clip(values, 0, 1) * limits if mode & 128 else np.zeros_like(limits))


class SimSensors:
    """Native IMU, local GPS and atmosphere; optional seeded white sensor noise.

    Noise avoids perfectly constant samples being classified as stuck by PX4.
    Std dev: accel .002 m/s^2, gyro .0002 rad/s, mag .0005 G, pressure .002 hPa.
    GPS is ideal. These are example values, not a calibrated hardware noise model.
    """

    def __init__(self, latitude=47.397742, longitude=8.545594, altitude=488., *, noise=False, seed=0):
        self.latitude, self.longitude, self.altitude = latitude, longitude, altitude
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def sample(self, state, imu):
        north, west, up = state.base.position_world
        altitude = self.altitude + up
        if not -1000 < altitude < 20000:
            raise ValueError("altitude outside atmospheric model")
        lat = self.latitude + math.degrees(north / 6378137.)
        lon = self.longitude - math.degrees(west / (6378137.*math.cos(math.radians(self.latitude))))
        velocity = to_px4(state.linear_velocity_world)
        course = math.degrees(math.atan2(velocity[1], velocity[0])) % 360
        result = dict(time_usec=round(state.time*1e6)+1,
                    accel=to_px4(imu['accel']), gyro=to_px4(imu['gyro']), mag=to_px4(imu['mag']),
                    pressure=1013.25*(1-altitude/44330.)**5.255, altitude=altitude,
                    latitude=round(lat*1e7), longitude=round(lon*1e7), velocity=velocity,
                    speed=float(np.linalg.norm(velocity[:2])), course=round(course*100) % 36000)
        if self.noise:
            for name, std in [('accel', .002), ('gyro', .0002), ('mag', .0005)]:
                result[name] += self.rng.normal(0., std, 3)
            result['pressure'] += self.rng.normal(0., .002)
        return result


class MavlinkSimulator:
    """One PX4 TCP connection; actuator waits are bounded, EOF is an error."""

    def __init__(self, connection):
        from pymavlink.dialects.v20 import common
        self.connection = connection
        self.mav = common.MAVLink(self, srcSystem=1, srcComponent=51)
        self.mav.robust_parsing = True
        self.pending = []
        self.received = 0

    def write(self, data):
        self.connection.sendall(data)

    def send_sensors(self, sample, *, gps=False):
        s = sample
        self.mav.hil_sensor_send(s['time_usec'], *s['accel'], *s['gyro'], *s['mag'],
                                 s['pressure'], 0., s['altitude'], 15., 0x1FFF)
        if gps:
            v = np.rint(s['velocity']*100).astype(int)
            self.mav.hil_gps_send(s['time_usec'], 3, s['latitude'], s['longitude'],
                                 round(s['altitude']*1000), 50, 80, round(s['speed']*100),
                                 *v, s['course'], 12)

    def receive_actuators(self, timeout):
        deadline = time.monotonic()+timeout
        while True:
            while self.pending:
                message = self.pending.pop(0)
                if message.get_type() == 'HIL_ACTUATOR_CONTROLS':
                    self.received += 1
                    return message
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                return None
            self.connection.settimeout(remaining)
            try:
                chunk = self.connection.recv(65536)
            except socket.timeout:
                return None
            if not chunk:
                raise ConnectionError('PX4 disconnected')
            self.pending.extend(self.mav.parse_buffer(chunk) or [])
