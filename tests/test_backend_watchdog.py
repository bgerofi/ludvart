"""The backend's client-liveness watchdog and the client's keepalive pings.

End-of-stream normally tells the backend its client is gone. Over a multiplexed
SSH connection it may never arrive -- sshd keeps the pipe's write end open after
a channel is dropped -- and the backend then sits in read() forever with a
litellm gateway still running. Pings let it tell a dead client from an idle one.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_backend_watchdog.py
"""

import io
import time

from ludvart.protocol import FrameChannel, MsgType, encode_frame, message
from ludvart.server import watch_client


def _channel(saw_ping: bool, silent_for: float) -> FrameChannel:
    channel = FrameChannel(io.BytesIO(), io.BytesIO())
    channel.saw_ping = saw_ping
    channel.last_recv = time.monotonic() - silent_for
    return channel


def _wait(flag: list, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if flag:
            return True
        time.sleep(0.01)
    return False


def test_a_silent_client_shuts_the_backend_down():
    codes, cleaned = [], []
    channel = _channel(saw_ping=True, silent_for=10.0)

    stop = watch_client(
        channel,
        lambda: cleaned.append(1),
        timeout=0.05,
        exit_process=lambda code: codes.append(code),
    )
    try:
        assert _wait(codes), "the backend kept serving a client that had gone away"
        assert cleaned, "the gateway was left running"
    finally:
        stop.set()
    print("a silent client shuts the backend down: OK")


def test_the_gateway_goes_before_the_process_does():
    """Exiting first would strand the litellm proxy, which is the whole point."""
    order = []
    channel = _channel(saw_ping=True, silent_for=10.0)

    stop = watch_client(
        channel,
        lambda: order.append("cleanup"),
        timeout=0.05,
        exit_process=lambda code: order.append("exit"),
    )
    try:
        assert _wait(order)
        time.sleep(0.1)
        assert order[:2] == ["cleanup", "exit"], order
    finally:
        stop.set()
    print("the gateway goes before the process does: OK")


def test_the_watchdog_fires_only_once():
    """It runs to completion rather than looping on an already-dead client."""
    codes = []
    channel = _channel(saw_ping=True, silent_for=10.0)

    stop = watch_client(
        channel, lambda: None, timeout=0.02, exit_process=lambda c: codes.append(c)
    )
    try:
        assert _wait(codes)
        time.sleep(0.3)
        assert len(codes) == 1, codes
    finally:
        stop.set()
    print("the watchdog fires only once: OK")


def test_an_idle_but_live_client_is_left_alone():
    """A user who walks away for an hour must not lose their session."""
    codes = []
    channel = _channel(saw_ping=True, silent_for=0.0)

    stop = watch_client(
        channel, lambda: None, timeout=5.0, exit_process=lambda c: codes.append(c)
    )
    try:
        time.sleep(0.3)
        assert not codes, "an idle session was killed"
    finally:
        stop.set()
    print("an idle but live client is left alone: OK")


def test_a_client_that_never_pings_is_never_judged():
    """An older client sends no pings; it must be served exactly as before."""
    codes = []
    channel = _channel(saw_ping=False, silent_for=10.0)

    stop = watch_client(
        channel, lambda: None, timeout=0.05, exit_process=lambda c: codes.append(c)
    )
    try:
        time.sleep(0.3)
        assert not codes, "a client with no keepalive support was cut off"
    finally:
        stop.set()
    print("a client that never pings is never judged: OK")


def test_a_ping_keeps_the_backend_serving():
    """The whole loop: a ping arrives, is swallowed, and resets the clock."""
    codes = []
    reader = io.BytesIO(encode_frame(message(MsgType.PING)))
    channel = FrameChannel(reader, io.BytesIO())
    channel.last_recv = time.monotonic() - 10.0

    stop = watch_client(
        channel, lambda: None, timeout=1.0, exit_process=lambda c: codes.append(c)
    )
    try:
        assert channel.recv() is None  # the ping is consumed, then EOF
        assert channel.saw_ping is True
        time.sleep(0.3)
        assert not codes, "a ping did not count as the client checking in"
    finally:
        stop.set()
    print("a ping keeps the backend serving: OK")


def test_the_client_only_pings_a_backend_that_asked_for_it():
    """An older backend treats an unexpected frame as a protocol error."""
    from ludvart.backend_client import BackendReconnector

    class _Transport:
        def __init__(self, hello):
            self.channel = FrameChannel(io.BytesIO(encode_frame(hello)), io.BytesIO())
            self.started = False

        def start_keepalive(self):
            self.started = True

    old = _Transport(message(MsgType.HELLO, active_label="x"))
    BackendReconnector(lambda: old).connect()
    assert old.started is False

    new = _Transport(message(MsgType.HELLO, active_label="x", keepalive=True))
    BackendReconnector(lambda: new).connect()
    assert new.started is True
    print("the client only pings a backend that asked for it: OK")


def main():
    test_a_silent_client_shuts_the_backend_down()
    test_the_gateway_goes_before_the_process_does()
    test_the_watchdog_fires_only_once()
    test_an_idle_but_live_client_is_left_alone()
    test_a_client_that_never_pings_is_never_judged()
    test_a_ping_keeps_the_backend_serving()
    test_the_client_only_pings_a_backend_that_asked_for_it()
    print("\nALL backend watchdog tests passed.")


if __name__ == "__main__":
    main()
