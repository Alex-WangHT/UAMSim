"""MuJoCo HIL sensor / actuator TCP server for PX4 v1.16, with optional arm/perception."""

import argparse
import json
from pathlib import Path
import socket
import time

import numpy as np
import mujoco

import UAMSim as ua
from UAMSim.sitl import MavlinkSimulator, SimSensors, rotor_command
from UAMSim.arm_endpoint import ArmEndpoint
from UAMSim.sensors import add_example_sensors
from UAMSim.sensor_endpoint import SensorEndpoint
from UAMSim.recording import StateRecorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=4560)
    parser.add_argument('--duration', type=float, default=120)
    parser.add_argument('--wait-seconds', type=float, default=120)
    parser.add_argument('--trace', type=Path, default=Path('logs/px4-trace.jsonl'))
    parser.add_argument('--arm', action='store_true', help='attach arm + landing legs; controller must match model')
    parser.add_argument('--sensors', action='store_true', help='enable camera/lidar TCP stream on 9212')
    parser.add_argument('--record', type=Path, help='new directory for model and timestamped state recording')
    args = parser.parse_args()
    if not np.isfinite([args.duration, args.wait_seconds]).all() or min(args.duration, args.wait_seconds) <= 0:
        parser.error('durations must be finite and positive')
    models = Path(__file__).with_name('models')
    multirotor = ua.Multirotor(str(models/'multirotor.xml'))
    arm = ua.Manipulator(str(models/'Manipulator.xml')) if args.arm else None
    if args.arm:
        # Simple simulation landing gear keeps the arm clear of the floor.
        body = multirotor.spec.body(multirotor.BODY_NAME)
        for x in (-.23, .23):
            for y in (-.20, .20):
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[x, y, 0, x, y, -.45],
                              size=[.008], mass=.005, rgba=[.2, .2, .2, 1])
    if args.sensors:
        add_example_sensors(multirotor)
    sim = ua.ControlledSimulation(ua.Environment(str(models/'environment.xml')), multirotor, arm, control_steps=4)
    sim.env.model.qpos0[2] = .458 if args.arm else .04
    sim.env.model.opt.magnetic[:] = [.215, -.007, -.427]  # Gauss, NWU
    sim.reset()
    maximum = sim.robot.multirotor.get_rotor_geometry()[2][:, 1]
    sensors = SimSensors(noise=True, seed=0)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    steps = round(args.duration/sim.control_dt)
    connected_at = None
    armed_samples = 0
    arm_endpoint = sensor_endpoint = None
    recorder = None
    completed = False
    try:
        if args.record is not None:
            recorder = StateRecorder(sim.env.model, args.record)
            recorder.capture(sim.env.data)
        if args.arm:
            arm_endpoint = ArmEndpoint(sim.robot, args.host)
        if args.sensors:
            sensor_endpoint = SensorEndpoint(sim.robot, args.host)
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.host, args.port))
            listener.listen(1)
            listener.settimeout(args.wait_seconds)
            print(f'Waiting for PX4 on {args.host}:{args.port}', flush=True)
            connection, address = listener.accept()
        with connection, args.trace.open('w', encoding='utf-8') as trace:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            link = MavlinkSimulator(connection)
            print(f'PX4 connected: {address}', flush=True)
            connected_at = time.monotonic()
            locked = False
            for frame in range(steps):
                started = time.monotonic()
                link.send_sensors(sensors.sample(sim.state, sim.read_imu()), gps=frame % 25 == 0)
                message = link.receive_actuators(2. if locked else .02)
                if message is None:
                    if locked or time.monotonic()-connected_at > 15:
                        raise TimeoutError('PX4 actuator stream timed out; physics stopped')
                    command = ua.RotorCommand(np.zeros(len(maximum)))
                else:
                    locked = bool(message.flags & 1)
                    command = rotor_command(message.controls, message.mode, maximum)
                    armed_samples += int(bool(message.mode & 128))
                joints = arm_endpoint.poll(sim.state) if arm_endpoint is not None else None
                state = sim.step_command(ua.RobotCommand(rotors=command, joints=joints))
                if recorder is not None:
                    recorder.capture(sim.env.data)
                if arm_endpoint is not None and frame % 5 == 0:
                    arm_endpoint.publish(state, frame)
                if sensor_endpoint is not None and frame % 25 == 0:
                    sensor_endpoint.publish()
                if frame % 25 == 0:
                    trace.write(json.dumps(dict(time=state.time, position=state.base.position_world.tolist(),
                                               quaternion=state.base.quaternion_wxyz.tolist(),
                                               velocity=state.linear_velocity_world.tolist(),
                                               thrust=command.thrust.tolist(), locked=locked,
                                               joints=state.joints.position.tolist()))+'\n')
                    trace.flush()
                if frame % 2500 == 0:
                    print(json.dumps(dict(time=round(state.time, 3), position=state.base.position_world.tolist(),
                                          messages=link.received, armed_samples=armed_samples)), flush=True)
                # Wall-clock pacing only; sensor timestamps remain simulation time.
                remaining = sim.control_dt-(time.monotonic()-started)
                if remaining > 0:
                    time.sleep(remaining)
            print(json.dumps(dict(completed=True, time=sim.state.time, messages=link.received,
                                  armed_samples=armed_samples,
                                  arm_accepted=arm_endpoint.accepted if arm_endpoint else 0,
                                  sensor_frames=sensor_endpoint.frames if sensor_endpoint else 0)), flush=True)
            completed = True
    finally:
        if recorder is not None:
            recorder.close(completed=completed)
        if arm_endpoint is not None:
            arm_endpoint.close()
        if sensor_endpoint is not None:
            sensor_endpoint.close()
        sim.close()


if __name__ == '__main__':
    main()
