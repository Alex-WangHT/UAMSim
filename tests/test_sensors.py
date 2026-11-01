import threading
import unittest

import mujoco
import numpy as np

from UAMSim.sensors import CameraSensor, PlanarLidar
from UAMSim.transport.protocol import Decoder, encode
from UAMSim.transport.sensor_protocol import SENSOR_PACKET_LIMIT, pack_frame, unpack_frame


def scene():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <worldbody>
        <geom name="wall" type="box" pos="2 0 1" size=".1 3 3" rgba="1 0 0 1"/>
        <camera name="camera" pos="0 0 1" xyaxes="0 -1 0 0 0 1" fovy="60"/>
        <site name="lidar" pos="0 0 1" size=".01" rgba="0 0 0 0"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class SensorTests(unittest.TestCase):
    def test_lidar_metric_distance_no_return_and_rotation(self):
        model, data = scene()
        lidar = PlanarLidar(model, 'lidar', rays=3, angle_min=-np.pi/2, angle_max=np.pi/2)
        frame = lidar.sample(data)
        np.testing.assert_allclose(frame.ranges[1], 1.9, atol=1e-6)
        self.assertTrue(np.isinf(frame.ranges[[0, 2]]).all())
        model.site_quat[0] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        mujoco.mj_setConst(model, data)  # static-world transform changed in the model
        rotated = lidar.sample(data)
        self.assertAlmostEqual(rotated.ranges[0], 1.9, places=5)
        self.assertTrue(np.isinf(rotated.ranges[1]))

    def test_camera_depth_is_optical_distance_not_ray_length(self):
        model, data = scene()
        camera = CameraSensor(model, 'camera', width=64, height=48)
        try:
            frame = camera.sample(data)
            self.assertEqual(frame.rgb.shape, (48, 64, 3))
            np.testing.assert_allclose(frame.depth, 1.9, atol=1e-4)
            self.assertGreater(float(frame.rgb[:, :, 0].mean()), float(frame.rgb[:, :, 1].mean()))
            self.assertAlmostEqual(frame.intrinsic[0, 0], 24/np.tan(np.pi/6))
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, frame.quaternion_wxyz)
            np.testing.assert_allclose(rotation.reshape(3, 3)[:, 2], [1, 0, 0], atol=1e-12)
        finally:
            camera.close()

    def test_packet_binary_roundtrip_fragmentation_and_size_validation(self):
        model, data = scene()
        camera = CameraSensor(model, 'camera', width=160, height=120)
        try:
            frame = camera.sample(data)
            scan = PlanarLidar(model, 'lidar', rays=3).sample(data)
            packet = pack_frame(frame, scan)
            with self.assertRaises(ValueError):
                encode(packet)  # arm command channel deliberately retains 64 KiB limit
            encoded = encode(packet, SENSOR_PACKET_LIMIT)
            decoder = Decoder(SENSOR_PACKET_LIMIT)
            packets = []
            for offset in range(0, len(encoded), 10007):
                packets.extend(decoder.feed(encoded[offset:offset+10007]))
            self.assertEqual(len(packets), 1)
            raw = unpack_frame(packets[0])
            np.testing.assert_array_equal(np.frombuffer(raw['depth'], dtype='<f4').reshape(frame.depth.shape), frame.depth)
            np.testing.assert_array_equal(np.frombuffer(raw['ranges'], dtype='<f4'), scan.ranges)
            packets[0]['camera']['width'] += 1
            with self.assertRaises(ValueError):
                unpack_frame(packets[0])
        finally:
            camera.close()

    def test_sampling_owner_enforced(self):
        model, data = scene()
        lidar = PlanarLidar(model, 'lidar')
        errors = []
        def other_thread():
            try:
                lidar.sample(data)
            except RuntimeError as error:
                errors.append(str(error))
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
