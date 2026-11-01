import socket
import time
import unittest

from UAMSim.transport.protocol import Decoder, MAX_PACKET, encode, validate_joint_command, validate_state
from UAMSim.transport.transport import Peer


class BridgeTests(unittest.TestCase):
    def test_fragmented_and_coalesced_packets(self):
        decoder = Decoder()
        first = dict(version=1, type="state", text="机械臂")
        payload = encode(first)
        self.assertEqual(decoder.feed(payload[:8]), [])
        self.assertEqual(decoder.feed(payload[8:]+payload), [first, first])

    def test_size_version_and_nonfinite_rejected(self):
        for payload in (b"x"*MAX_PACKET, b'{"version":1,"x":NaN}\n', b'{"version":true}\n', b'[]\n'):
            with self.assertRaises(ValueError):
                Decoder().feed(payload)
        with self.assertRaises(ValueError):
            encode(dict(version=1, value=float("nan")))

    def test_episode_sequence_and_joint_validation(self):
        good = dict(version=1, type="joint_command", epoch="a", sequence=2, names=["j2", "j1"], position=[.1, .2])
        self.assertEqual(validate_joint_command(good, "a", ("j1", "j2"), 1), 2)
        for update in (dict(epoch="b"), dict(sequence=1), dict(sequence=True), dict(names=["j1", "j1"]),
                       dict(position=[float("nan"), 0]), dict(position=[True, 0]), dict(type="state")):
            with self.assertRaises(ValueError):
                validate_joint_command(dict(good, **update), "a", ("j1", "j2"), 1)

    def test_bidirectional_tcp_and_peer_close(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            client = socket.create_connection(listener.getsockname())
            server, _ = listener.accept()
        peers = [Peer(client), Peer(server)]
        try:
            for index, peer in enumerate(peers):
                peer.send(dict(version=1, type="test", value=index))
            for index, peer in enumerate(peers):
                deadline = time.monotonic()+2
                received = None
                while received is None and time.monotonic() < deadline:
                    received = peer.receive()
                    time.sleep(.01)
                self.assertIsNotNone(received)
                self.assertEqual(received[0]["value"], 1-index)
            peers[0].close()
            peers[1].thread.join(2)
            self.assertTrue(peers[1].stop_event.is_set())
        finally:
            for peer in peers:
                peer.close()

    def test_state_validation(self):
        good = dict(version=1, type="state", epoch="test", time=1.,
                    joints=dict(names=["j"], position=[.1], velocity=[0.]))
        validate_state(good)
        for update in (dict(time=-1), dict(epoch=""), dict(joints=dict(names=["j"], position=[], velocity=[0]))):
            with self.assertRaises(ValueError):
                validate_state(dict(good, **update))
