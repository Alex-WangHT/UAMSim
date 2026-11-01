"""Owner-thread RGB/depth/lidar acquisition, streamed to an independent ROS bridge."""

import argparse
import json
from pathlib import Path
import socket

import UAMSim as ua
from UAMSim.sensors import CameraSensor, PlanarLidar, add_example_sensors
from UAMSim.transport.sensor_protocol import SENSOR_PACKET_LIMIT, pack_frame
from UAMSim.transport.transport import Peer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9212)
    parser.add_argument('--duration', type=int, default=15)
    args = parser.parse_args()
    if args.duration < 1:
        parser.error('duration must be positive')
    models = Path(__file__).with_name('models')
    multirotor = ua.Multirotor(str(models/'multirotor.xml'))
    add_example_sensors(multirotor)
    sim = ua.ControlledSimulation(ua.Environment(str(models/'environment.xml')), multirotor, real_time_factor=1)
    sim.env.model.qpos0[2] = .8
    sim.reset(ua.GeometricController.from_robot(sim.robot))
    camera = CameraSensor(sim.env.model, 'uam/front_camera')
    lidar = PlanarLidar(sim.env.model, 'uam/lidar', exclude_body=sim.robot.multirotor.body_id)
    peer, frames = None, 0
    try:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.host, args.port))
            listener.listen(1)
            listener.settimeout(60)
            print(f'Sensor server listening on {args.host}:{args.port}', flush=True)
            connection, _ = listener.accept()
        peer = Peer(connection, max_packet=SENSOR_PACKET_LIMIT)
        for _ in range(round(args.duration/sim.control_dt)):
            sim.step(ua.ControlTarget([0., 0., .8]))
            if sim.frame % 20 == 0:
                peer.send(pack_frame(camera.sample(sim.env.data), lidar.sample(sim.env.data)))
                frames += 1
        print(json.dumps(dict(time=sim.state.time, sensor_frames=frames, transport_error=peer.error)), flush=True)
    finally:
        if peer is not None:
            peer.close()
        camera.close()
        sim.close()


if __name__ == '__main__':
    main()
