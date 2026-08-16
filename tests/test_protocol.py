"""Unit tests for the framed client<->backend protocol.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_protocol.py
"""

import io
import os
import threading

from ludvart.protocol import (
    DEFAULT_MAX_FRAME,
    FrameChannel,
    MsgType,
    ProtocolError,
    encode_frame,
    message,
    msg_type,
    read_frame,
    require,
    write_frame,
)


def test_frame_roundtrip():
    buf = io.BytesIO()
    write_frame(buf, {"type": "hello", "n": 1})
    buf.seek(0)
    got = read_frame(buf)
    assert got == {"type": "hello", "n": 1}, got
    # Nothing left -> clean EOF returns None.
    assert read_frame(buf) is None
    print("frame roundtrip + clean EOF: OK")


def test_multiple_frames_in_stream():
    buf = io.BytesIO()
    write_frame(buf, {"type": "a"})
    write_frame(buf, {"type": "b", "x": [1, 2, 3]})
    buf.seek(0)
    assert read_frame(buf) == {"type": "a"}
    assert read_frame(buf) == {"type": "b", "x": [1, 2, 3]}
    assert read_frame(buf) is None
    print("multiple frames decode in order: OK")


def test_frame_length_prefix_is_four_bytes():
    frame = encode_frame({"type": "x"})
    # 4-byte big-endian length header followed by the JSON body.
    body = b'{"type":"x"}'
    assert frame == len(body).to_bytes(4, "big") + body, frame
    print("frame uses a 4-byte big-endian length prefix: OK")


def test_truncated_body_raises():
    frame = encode_frame({"type": "x", "data": "hello"})
    truncated = frame[:-3]  # drop part of the body
    buf = io.BytesIO(truncated)
    try:
        read_frame(buf)
    except ProtocolError as exc:
        assert "truncated" in str(exc) or "EOF" in str(exc), exc
    else:
        raise AssertionError("expected ProtocolError on a truncated frame")
    print("truncated frame raises ProtocolError: OK")


def test_truncated_header_raises():
    buf = io.BytesIO(b"\x00\x00")  # only 2 of the 4 header bytes
    try:
        read_frame(buf)
    except ProtocolError:
        pass
    else:
        raise AssertionError("expected ProtocolError on a truncated header")
    print("truncated header raises ProtocolError: OK")


def test_zero_length_frame_rejected():
    buf = io.BytesIO((0).to_bytes(4, "big"))
    try:
        read_frame(buf)
    except ProtocolError as exc:
        assert "empty" in str(exc) or "zero" in str(exc), exc
    else:
        raise AssertionError("expected ProtocolError on a zero-length frame")
    print("zero-length frame rejected: OK")


def test_oversize_frame_rejected_on_read():
    # A header claiming a length beyond the cap must be rejected before any body
    # read is attempted.
    buf = io.BytesIO((DEFAULT_MAX_FRAME + 1).to_bytes(4, "big"))
    try:
        read_frame(buf)
    except ProtocolError as exc:
        assert "exceeds" in str(exc) or "limit" in str(exc), exc
    else:
        raise AssertionError("expected ProtocolError on an over-long frame")
    print("over-long frame length rejected on read: OK")


def test_oversize_frame_rejected_on_write():
    try:
        encode_frame({"type": "x", "blob": "y"}, max_frame=4)
    except ProtocolError as exc:
        assert "too large" in str(exc), exc
    else:
        raise AssertionError("expected ProtocolError encoding past max_frame")
    print("over-long frame rejected on encode: OK")


def test_invalid_json_payload_raises():
    body = b"not json"
    buf = io.BytesIO(len(body).to_bytes(4, "big") + body)
    try:
        read_frame(buf)
    except ProtocolError:
        pass
    else:
        raise AssertionError("expected ProtocolError on invalid JSON")
    print("invalid JSON payload raises ProtocolError: OK")


def test_non_object_payload_raises():
    body = b"[1,2,3]"  # valid JSON, but not an object
    buf = io.BytesIO(len(body).to_bytes(4, "big") + body)
    try:
        read_frame(buf)
    except ProtocolError as exc:
        assert "not a JSON object" in str(exc), exc
    else:
        raise AssertionError("expected ProtocolError on a non-object payload")
    print("non-object payload raises ProtocolError: OK")


def test_message_builder_and_type():
    msg = message(MsgType.SUBMIT, text="hi", snapshot="SCREEN")
    assert msg == {"type": "submit", "text": "hi", "snapshot": "SCREEN"}
    assert msg_type(msg) == "submit"
    print("message() builds a typed dict and msg_type() reads it: OK")


def test_message_type_reserved():
    try:
        message(MsgType.SUBMIT, type="oops")
    except ValueError:
        pass
    else:
        raise AssertionError("passing 'type' in fields should raise")
    print("message() rejects a 'type' field in kwargs: OK")


def test_msg_type_missing_raises():
    try:
        msg_type({"no": "type"})
    except ProtocolError:
        pass
    else:
        raise AssertionError("expected ProtocolError when 'type' is missing")
    print("msg_type() raises when 'type' is absent: OK")


def test_require_fields():
    require({"type": "submit", "text": "x", "snapshot": "s"}, "text", "snapshot")
    try:
        require({"type": "submit", "text": "x"}, "text", "snapshot")
    except ProtocolError as exc:
        assert "snapshot" in str(exc), exc
    else:
        raise AssertionError("expected ProtocolError for a missing field")
    print("require() validates present/missing fields: OK")


def test_frame_channel_over_os_pipe():
    # A real OS pipe exercises short reads and buffering the way a subprocess
    # stdio pipe would.
    a_r, a_w = os.pipe()  # backend -> client
    b_r, b_w = os.pipe()  # client -> backend
    client = FrameChannel(os.fdopen(a_r, "rb"), os.fdopen(b_w, "wb"))
    backend = FrameChannel(os.fdopen(b_r, "rb"), os.fdopen(a_w, "wb"))

    received = []

    def backend_echo():
        while True:
            msg = backend.recv()
            if msg is None:
                break
            backend.send(message("echo", got=msg))

    t = threading.Thread(target=backend_echo, daemon=True)
    t.start()

    client.send(message(MsgType.SUBMIT, text="ping"))
    reply = client.recv()
    assert reply == {"type": "echo", "got": {"type": "submit", "text": "ping"}}, reply

    client.close()  # closes writer -> backend sees EOF and stops
    t.join(timeout=2)
    assert not t.is_alive()
    backend.close()
    print("FrameChannel round-trips over an OS pipe and closes cleanly: OK")


def test_frame_channel_send_is_thread_safe():
    # Two threads sending concurrently must not interleave bytes; the reader must
    # decode both frames intact.
    r_fd, w_fd = os.pipe()
    writer = FrameChannel(io.BytesIO(), os.fdopen(w_fd, "wb"))
    reader_stream = os.fdopen(r_fd, "rb")

    def sender(n):
        for i in range(50):
            writer.send(message("m", who=n, i=i))

    threads = [threading.Thread(target=sender, args=(k,)) for k in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    writer.close()

    count = 0
    while True:
        msg = read_frame(reader_stream)
        if msg is None:
            break
        assert msg["type"] == "m" and "who" in msg and "i" in msg, msg
        count += 1
    reader_stream.close()
    assert count == 4 * 50, count
    print("concurrent FrameChannel sends do not interleave: OK")


def test_recv_swallows_keepalives():
    """A ping may land while a read site awaits one specific reply.

    Those sites reject anything unexpected, so the keepalive must never reach
    them -- but it still has to count as the peer being heard from.
    """
    import time as _time

    from ludvart.protocol import FrameChannel, MsgType, encode_frame

    payload = (
        encode_frame(message(MsgType.PING))
        + encode_frame(message(MsgType.PING))
        + encode_frame(message(MsgType.REPLY, text="hi"))
    )
    channel = FrameChannel(io.BytesIO(payload), io.BytesIO())
    channel.last_recv = _time.monotonic() - 100

    got = channel.recv()

    assert got["type"] == MsgType.REPLY, got
    assert channel.saw_ping is True
    assert _time.monotonic() - channel.last_recv < 5
    assert channel.recv() is None
    print("recv swallows keepalives: OK")


def test_a_channel_without_pings_is_not_marked_alive():
    """The backend arms its watchdog off this flag; a silent peer must not."""
    from ludvart.protocol import FrameChannel, encode_frame

    channel = FrameChannel(io.BytesIO(encode_frame(message("reply"))), io.BytesIO())

    assert channel.recv()["type"] == "reply"
    assert channel.saw_ping is False
    print("a channel without pings is not marked alive: OK")


def main():
    test_frame_roundtrip()
    test_multiple_frames_in_stream()
    test_frame_length_prefix_is_four_bytes()
    test_truncated_body_raises()
    test_truncated_header_raises()
    test_zero_length_frame_rejected()
    test_oversize_frame_rejected_on_read()
    test_oversize_frame_rejected_on_write()
    test_invalid_json_payload_raises()
    test_non_object_payload_raises()
    test_message_builder_and_type()
    test_message_type_reserved()
    test_msg_type_missing_raises()
    test_require_fields()
    test_frame_channel_over_os_pipe()
    test_frame_channel_send_is_thread_safe()
    test_recv_swallows_keepalives()
    test_a_channel_without_pings_is_not_marked_alive()
    print("\nALL protocol tests passed.")


if __name__ == "__main__":
    main()
