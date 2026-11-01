"""Independent arm TCP channel serviced by the shared physics owner."""

import socket
import time
import uuid

from .interfaces import JointPositionCommand
from UAMSim.transport.protocol import validate_joint_command
from UAMSim.transport.transport import Peer
from .simulation.arm_driver import SimulationArmDriver


class ArmEndpoint:
    def __init__(self, robot, host='127.0.0.1', port=9211, timeout=.5):
        if robot.manipulator is None:
            raise ValueError('arm endpoint requires a manipulator')
        self.robot = robot
        self.driver = SimulationArmDriver(robot)
        self.timeout = timeout
        self.epoch = uuid.uuid4().hex
        self.sequence = self.accepted = self.rejected = 0
        self.last_received = float('-inf')
        self.holding = True
        state = robot.read_state()
        self.target = JointPositionCommand(state.joints.names, state.joints.position)
        self.peer = None
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((host, port))
        self.listener.listen(1)
        self.listener.setblocking(False)

    def poll(self, state):
        if self.peer is None:
            try:
                connection, _ = self.listener.accept()
                self.peer = Peer(connection)
            except BlockingIOError:
                return self.target
        incoming = self.peer.receive()
        if incoming is not None:
            packet, received = incoming
            try:
                sequence = validate_joint_command(packet, self.epoch, state.joints.names, self.sequence)
                if time.monotonic()-received > self.timeout:
                    raise ValueError('expired arm target')
                candidate = JointPositionCommand(packet['names'], packet['position'])
                self.driver.write_positions(dict(zip(candidate.names, candidate.position.tolist())))
                self.target, self.sequence, self.last_received = candidate, sequence, received
                self.accepted += 1
                self.holding = False
            except (ValueError, TypeError):
                self.rejected += 1
        if not self.holding and (self.peer.stop_event.is_set() or time.monotonic()-self.last_received > self.timeout):
            self.target = JointPositionCommand(state.joints.names, state.joints.position)
            self.holding = True
        return self.target

    def publish(self, state, frame):
        if self.peer is not None and not self.peer.stop_event.is_set():
            self.peer.send(dict(version=1, type='state', epoch=self.epoch, frame=frame, time=state.time,
                                joints=dict(names=list(state.joints.names), position=state.joints.position.tolist(),
                                            velocity=state.joints.velocity.tolist()), holding=self.holding))

    def close(self):
        self.listener.close()
        if self.peer is not None:
            self.peer.close()
