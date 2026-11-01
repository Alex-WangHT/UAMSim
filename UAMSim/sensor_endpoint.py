"""Optional camera/lidar stream for an existing owner-thread simulation."""

import socket

from .sensors import CameraSensor, PlanarLidar
from UAMSim.transport.sensor_protocol import SENSOR_PACKET_LIMIT, pack_frame
from UAMSim.transport.transport import Peer


class SensorEndpoint:
    def __init__(self, robot, host='127.0.0.1', port=9212):
        self.robot = robot
        self.camera = CameraSensor(robot.model, robot.prefix+'front_camera')
        self.lidar = PlanarLidar(robot.model, robot.prefix+'lidar', exclude_body=robot.multirotor.body_id)
        self.peer = None
        self.frames = 0
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.listener.bind((host, port))
            self.listener.listen(1)
            self.listener.setblocking(False)
        except Exception:
            self.listener.close()
            self.camera.close()
            raise

    def publish(self):
        if self.peer is None:
            try:
                connection, _ = self.listener.accept()
                self.peer = Peer(connection, max_packet=SENSOR_PACKET_LIMIT)
            except BlockingIOError:
                return
        if not self.peer.stop_event.is_set():
            self.peer.send(pack_frame(self.camera.sample(self.robot.data), self.lidar.sample(self.robot.data)))
            self.frames += 1

    def close(self):
        self.listener.close()
        if self.peer is not None:
            self.peer.close()
        self.camera.close()
