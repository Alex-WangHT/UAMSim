"""Versioned newline JSON messages for the independent arm/state link."""

import json
import math

MAX_PACKET = 65536


def encode(packet, max_packet=MAX_PACKET):
    payload = json.dumps(packet, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    if len(payload) > max_packet:
        raise ValueError("packet exceeds size limit")
    return payload


class Decoder:
    def __init__(self, max_packet=MAX_PACKET):
        self.buffer = bytearray()
        self.max_packet = max_packet

    def feed(self, payload):
        self.buffer.extend(payload)
        packets = []
        while b"\n" in self.buffer:
            end = self.buffer.index(b"\n")
            if end + 1 > self.max_packet:
                raise ValueError("packet exceeds size limit")
            raw = bytes(self.buffer[:end])
            del self.buffer[:end+1]
            packet = json.loads(raw, parse_constant=self._invalid_constant)
            if not isinstance(packet, dict) or type(packet.get("version")) is not int or packet["version"] != 1:
                raise ValueError("unsupported packet version")
            packets.append(packet)
        if len(self.buffer) >= self.max_packet:
            raise ValueError("unterminated oversized packet")
        return packets

    @staticmethod
    def _invalid_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")


def validate_joint_command(packet, epoch, names, last_sequence):
    if packet.get("type") != "joint_command" or packet.get("epoch") != epoch:
        raise ValueError("wrong command type or episode")
    sequence = packet.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= last_sequence:
        raise ValueError("stale or invalid command sequence")
    given_names, position = packet.get("names"), packet.get("position")
    if not isinstance(given_names, list) or not isinstance(position, list):
        raise ValueError("names and position must be arrays")
    if len(given_names) != len(names) or len(position) != len(names):
        raise ValueError("wrong command length")
    if any(not isinstance(name, str) for name in given_names) or set(given_names) != set(names):
        raise ValueError("wrong joint names")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in position):
        raise ValueError("joint positions must be finite numbers")
    return sequence


def validate_state(packet):
    if packet.get("type") != "state" or not isinstance(packet.get("epoch"), str) or not packet["epoch"]:
        raise ValueError("invalid state type or epoch")
    timestamp = packet.get("time")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("invalid simulation timestamp")
    joints = packet.get("joints")
    if not isinstance(joints, dict):
        raise ValueError("missing joint state")
    names = joints.get("names")
    if (not isinstance(names, list) or len(names) > 64 or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("invalid joint names")
    for field in ("position", "velocity"):
        values = joints.get(field)
        if not isinstance(values, list) or len(values) != len(names):
            raise ValueError("invalid joint state length")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("non-finite joint state")
