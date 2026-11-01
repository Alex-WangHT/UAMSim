import importlib.util
from pathlib import Path
import socket
import unittest

import mujoco
import numpy as np

import UAMSim as ua
from UAMSim.disturbances import DisturbanceConfig, WindConfig
from UAMSim.sitl import MavlinkSimulator, SimSensors, px4_quaternion, rotor_command, to_px4
from test_controlled_simulation import runner


class SensorTests(unittest.TestCase):
    def test_frame_conversion_and_quaternion(self):
        np.testing.assert_array_equal(to_px4([1, 2, 3]), [1, -2, -3])
        q = np.array([.5, .5, .5, .5])
        a, b = np.empty(9), np.empty(9)
        mujoco.mju_quat2Mat(a, q)
        mujoco.mju_quat2Mat(b, px4_quaternion(q))
        c = np.diag([1., -1., -1.])
        np.testing.assert_allclose(b.reshape(3, 3), c@a.reshape(3, 3)@c)

    def test_specific_force_hover_and_freefall(self):
        sim = runner()
        np.testing.assert_allclose(sim.read_imu()['accel'], 0, atol=1e-10)
        sim.robot.apply_command(ua.RobotCommand(ua.RotorCommand(np.full(6, sim.robot.hover_thrust()))))
        np.testing.assert_allclose(to_px4(sim.read_imu()['accel']), [0, 0, -9.81], atol=1e-10)

    def test_imu_contains_wind_and_preserves_caller_force(self):
        sim = runner(disturbance=DisturbanceConfig(wind=WindConfig(velocity_world=(1, 0, 0))))
        sim.env.data.qfrc_applied[0] = .2
        original = sim.env.data.qfrc_applied.copy()
        mass = sim.env.model.body_subtreemass[sim.robot.multirotor.body_id]
        self.assertAlmostEqual(sim.read_imu()['accel'][0], .5/mass, places=10)
        np.testing.assert_array_equal(sim.env.data.qfrc_applied, original)

    def test_gps_direction_and_time(self):
        sim = runner()
        sensors = SimSensors()
        initial = sensors.sample(sim.state, sim.read_imu())
        sim.env.data.qpos[:3] += [10, 10, 10]
        sim.env.data.qvel[:3] = [1, 2, 3]
        sim.env.data.time = 1.25
        sample = sensors.sample(sim.robot.read_state(), sim.read_imu())
        self.assertGreater(sample['latitude'], initial['latitude'])
        self.assertLess(sample['longitude'], initial['longitude'])
        self.assertLess(sample['pressure'], initial['pressure'])
        self.assertEqual(sample['time_usec'], 1250001)
        np.testing.assert_array_equal(sample['velocity'], [1, -2, -3])

    def test_noise_reproducible_and_not_stuck(self):
        sim = runner()
        a, b = SimSensors(noise=True), SimSensors(noise=True)
        x = a.sample(sim.state, sim.read_imu())
        y = b.sample(sim.state, sim.read_imu())
        np.testing.assert_array_equal(x['mag'], y['mag'])
        z = a.sample(sim.state, sim.read_imu())
        self.assertFalse(np.array_equal(x['mag'], z['mag']))
        self.assertNotEqual(x['pressure'], z['pressure'])

    def test_rotor_order_limits_and_disarm(self):
        values = np.arange(6)/6
        np.testing.assert_allclose(rotor_command(values, 128, [12]*6).thrust, values*12)
        np.testing.assert_array_equal(rotor_command(values, 0, [12]*6).thrust, 0)
        for invalid in ([float('nan')]*6, [-.1]*6, [1.1]*6, [1]*5):
            with self.assertRaises(ValueError):
                rotor_command(invalid, 128, [12]*6)



@unittest.skipUnless(importlib.util.find_spec('pymavlink'), 'optional SITL dependency')
class WireTests(unittest.TestCase):
    def test_mavlink_sensor_encoding_actuator_decode_and_eof(self):
        from pymavlink.dialects.v20 import common
        left, right = socket.socketpair()
        with left, right:
            link = MavlinkSimulator(left)
            parser = common.MAVLink(None)
            sim = runner()
            link.send_sensors(SimSensors().sample(sim.state, sim.read_imu()), gps=True)
            right.settimeout(1)
            messages = []
            while len(messages) < 2:
                messages += parser.parse_buffer(right.recv(4096)) or []
            self.assertEqual([m.get_type() for m in messages], ['HIL_SENSOR', 'HIL_GPS'])
            self.assertEqual(messages[0].fields_updated, 0x1FFF)
            packet = common.MAVLink_hil_actuator_controls_message(4, [.2]*16, 128, 1).pack(parser)
            right.sendall(packet[:5])
            self.assertIsNone(link.receive_actuators(.01))
            right.sendall(packet[5:])
            result = link.receive_actuators(1)
            self.assertEqual(result.mode, 128)
            self.assertAlmostEqual(result.controls[5], .2)
            right.shutdown(socket.SHUT_RDWR)
            with self.assertRaises(ConnectionError):
                link.receive_actuators(1)
