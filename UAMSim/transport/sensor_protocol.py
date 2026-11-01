"""Bounded sensor snapshots. Binary arrays are little-endian base64, not JSON NaN."""

import base64
import math

SENSOR_PACKET_LIMIT = 8*1024*1024


def pack_frame(camera, lidar):
    def blob(array, dtype):
        return base64.b64encode(array.astype(dtype).tobytes()).decode('ascii')

    return dict(version=1, type='sensors', time=camera.time,
                camera=dict(width=camera.rgb.shape[1], height=camera.rgb.shape[0],
                            rgb=blob(camera.rgb, 'u1'), depth=blob(camera.depth, '<f4'),
                            intrinsic=camera.intrinsic.reshape(-1).tolist(),
                            position=camera.position_world.tolist(), quaternion=camera.quaternion_wxyz.tolist()),
                lidar=dict(ranges=blob(lidar.ranges, '<f4'), count=len(lidar.ranges),
                           angle_min=lidar.angle_min, angle_increment=lidar.angle_increment,
                           range_min=lidar.range_min, range_max=lidar.range_max,
                           position=lidar.position_world.tolist(), quaternion=lidar.quaternion_wxyz.tolist()))


def unpack_frame(packet):
    """Validate before allocating ROS messages; keep byte arrays for zero type conversion."""
    if packet.get('type') != 'sensors' or type(packet.get('time')) not in (int, float) or not math.isfinite(packet['time']) or packet['time'] < 0:
        raise ValueError('invalid sensor timestamp/type')
    camera, lidar = packet['camera'], packet['lidar']
    width, height, count = camera['width'], camera['height'], lidar['count']
    if any(type(v) is not int for v in (width, height, count)) or not (1 <= width <= 1920 and 1 <= height <= 1080 and 2 <= count <= 4096):
        raise ValueError('invalid sensor dimensions')
    decoded = {}
    for name, encoded, size in [('rgb', camera['rgb'], width*height*3),
                              ('depth', camera['depth'], width*height*4), ('ranges', lidar['ranges'], count*4)]:
        value = base64.b64decode(encoded, validate=True)
        if len(value) != size:
            raise ValueError('sensor payload length mismatch')
        decoded[name] = value
    for config in (camera, lidar):
        for key, count in [('position', 3), ('quaternion', 4)]:
            values = config[key]
            if len(values) != count or not all(math.isfinite(v) for v in values):
                raise ValueError('invalid sensor pose')
        if abs(sum(v*v for v in config['quaternion'])-1) > 1e-4:
            raise ValueError('invalid sensor orientation')
    if len(camera['intrinsic']) != 9 or not all(math.isfinite(v) for v in camera['intrinsic']):
        raise ValueError('invalid intrinsic matrix')
    if not all(math.isfinite(lidar[k]) for k in ('angle_min', 'angle_increment', 'range_min', 'range_max')) or not (
            lidar['angle_increment'] > 0 and 0 <= lidar['range_min'] < lidar['range_max']):
        raise ValueError('invalid scan limits')
    return decoded
