"""Standalone arm protocol server with a geometric reference controller."""

import argparse
import json
from pathlib import Path
import socket
import time
import uuid

import numpy as np

import UAMSim as ua
from UAMSim.transport.protocol import validate_joint_command
from UAMSim.transport.transport import Peer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9211)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--wait-seconds", type=float, default=60)
    args = parser.parse_args()
    if not np.isfinite([args.duration, args.wait_seconds]).all() or min(args.duration, args.wait_seconds) <= 0:
        parser.error("durations must be positive and finite")
    models = Path(__file__).with_name("models")
    sim = ua.ControlledSimulation(ua.Environment(str(models/"environment.xml")),
                                 ua.Multirotor(str(models/"multirotor.xml")),
                                 ua.Manipulator(str(models/"Manipulator.xml")), real_time_factor=1)
    sim.reset(ua.GeometricController.from_robot(sim.robot))
    steps = round(args.duration / sim.control_dt)
    if not np.isclose(steps * sim.control_dt, args.duration, rtol=0, atol=1e-9):
        parser.error("duration must be an integer multiple of the control period")
    epoch = uuid.uuid4().hex
    last_sequence, accepted, rejected = 0, 0, 0
    last_received = float("-inf")
    holding = True
    joint_target = ua.JointPositionCommand(sim.state.joints.names, sim.state.joints.position)
    peer = None
    try:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.host, args.port))
            listener.listen(1)
            listener.settimeout(args.wait_seconds)
            print(f"Listening on {args.host}:{args.port}", flush=True)
            connection, _ = listener.accept()
        peer = Peer(connection)
        for _ in range(steps):
            incoming = peer.receive()
            if incoming is not None:
                packet, received_at = incoming
                try:
                    sequence = validate_joint_command(packet, epoch, sim.state.joints.names, last_sequence)
                    if time.monotonic() - received_at > .5:
                        raise ValueError("command expired")
                    candidate = ua.JointPositionCommand(packet["names"], packet["position"])
                    # Validate ranges atomically using the same model contract before accepting the target.
                    sim.robot.apply_command(ua.RobotCommand(joints=candidate))
                    joint_target, last_sequence, last_received = candidate, sequence, received_at
                    accepted += 1
                    holding = False
                except (TypeError, ValueError):
                    rejected += 1
            if not holding and (peer.stop_event.is_set() or time.monotonic()-last_received > .5):
                joint_target = ua.JointPositionCommand(sim.state.joints.names, sim.state.joints.position)
                holding = True
            state = sim.step(ua.ControlTarget([0, 0, 1.5], joints=joint_target))
            if sim.frame % 4 == 0 and not peer.stop_event.is_set():
                peer.send(dict(version=1, type="state", epoch=epoch, frame=sim.frame, time=state.time,
                               joints=dict(names=list(state.joints.names), position=state.joints.position.tolist(),
                                           velocity=state.joints.velocity.tolist()), holding=holding))
        print(json.dumps(dict(accepted=accepted, rejected=rejected, holding=holding,
                              time=sim.state.time, joint_positions=sim.state.joints.position.tolist())), flush=True)
    finally:
        if peer is not None:
            peer.close()
        sim.close()


if __name__ == "__main__":
    main()
