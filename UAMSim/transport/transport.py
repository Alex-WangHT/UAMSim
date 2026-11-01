"""One TCP peer, bounded latest-value queues; no simulator access in I/O threads."""

import queue
import select
import socket
import threading
import time

from .protocol import Decoder, encode, MAX_PACKET


class Peer:
    def __init__(self, connection, *, max_packet=MAX_PACKET):
        self.connection = connection
        self.max_packet = max_packet
        connection.settimeout(0.25)
        self.incoming = queue.Queue(maxsize=1)
        self.outgoing = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="UAMBridgePeer", daemon=True)
        self.thread.start()

    @staticmethod
    def _replace(target, item):
        try:
            target.put_nowait(item)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            target.put_nowait(item)

    def send(self, packet):
        self._replace(self.outgoing, encode(packet, self.max_packet))

    def receive(self):
        try:
            return self.incoming.get_nowait()
        except queue.Empty:
            return None

    def _run(self):
        decoder = Decoder(self.max_packet)
        try:
            while not self.stop_event.is_set():
                readable, _, _ = select.select([self.connection], [], [], .01)
                if readable:
                    payload = self.connection.recv(16384)
                    if not payload:
                        break
                    for packet in decoder.feed(payload):
                        self._replace(self.incoming, (packet, time.monotonic()))
                try:
                    outgoing = self.outgoing.get_nowait()
                except queue.Empty:
                    continue
                self.connection.sendall(outgoing)
        except (OSError, ValueError) as exc:
            self.error = str(exc)
        finally:
            self.stop_event.set()
            self.connection.close()

    def close(self):
        self.stop_event.set()
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.thread.join(2)
        self.connection.close()
