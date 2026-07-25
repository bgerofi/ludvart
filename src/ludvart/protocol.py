"""Framed message protocol for the client <-> backend split.

ludvart can run its agent loop (the "backend") in a separate process from the
terminal/PTY half (the "client"), talking over a single duplex byte stream --
either a locally forked subprocess's stdin/stdout, or an ``ssh`` process's
stdin/stdout to a remote host. This module defines how messages are framed on
that stream and the vocabulary of message types.

Framing
-------
The stream is a raw byte pipe, so message boundaries are restored with a simple
length-prefixed frame: a 4-byte big-endian unsigned length followed by exactly
that many bytes of UTF-8 JSON payload. To send, write the length then the body;
to receive, read 4 bytes to learn the length, then read exactly that many bytes.

Only protocol frames ever travel on the stream (a process's stdout). Human logs
and diagnostics must go to stderr so they can never corrupt the framed channel.

Every message is a JSON object carrying a ``"type"`` field (see :class:`MsgType`)
plus type-specific fields.
"""

from __future__ import annotations

import json
import struct
import threading
from typing import Any, BinaryIO

#: 4-byte big-endian unsigned length prefix on every frame.
_HEADER = struct.Struct(">I")

#: Reject absurd frame lengths up front so a corrupt/ hostile stream cannot make
#: the reader attempt a multi-gigabyte allocation. Snapshots of a large screen
#: plus scrollback stay comfortably under this.
DEFAULT_MAX_FRAME = 64 * 1024 * 1024  # 64 MiB


class ProtocolError(Exception):
    """Raised on a malformed frame or an out-of-range frame length."""


class MsgType:
    """The vocabulary of message types exchanged over the channel.

    Direction is a convention, not enforced by the codec:

    * ``C->B`` client to backend, ``B->C`` backend to client.
    """

    #: B->C: first message from the backend, carrying its protocol/app version.
    HELLO = "hello"
    #: C->B: bind to a session ("new" or a session id) with the terminal size.
    ATTACH = "attach"
    #: B->C: acknowledgement of ATTACH with the current transcript/panel state.
    ATTACHED = "attached"
    #: C->B: submit a user question, carrying the ask-time screen snapshot.
    SUBMIT = "submit"
    #: C->B: run an internal slash command (e.g. "/sessions list").
    COMMAND = "command"
    #: C->B: answer an approval prompt with "y" / "n" / "a".
    APPROVAL = "approval"
    #: C->B: submit a steering instruction / cancel the in-flight turn.
    STEER = "steer"
    CANCEL = "cancel"
    #: B->C: run a client-side (terminal) tool; expects a TOOL_RESULT reply.
    TOOL_INVOKE = "tool_invoke"
    #: C->B: the result of a client-side tool call, keyed by call_id.
    TOOL_RESULT = "tool_result"
    #: B->C: a value-returning call on the client's terminal host (snapshot,
    #: terminal tool), keyed by call_id; the client answers with RESPONSE.
    REQUEST = "request"
    #: C->B: the result of a REQUEST, keyed by the same call_id.
    RESPONSE = "response"
    #: C->B: a value-returning call on the backend, made *while* the client is
    #: serving a REQUEST (the backend is blocked waiting, so the channel is
    #: free). Used for work that needs the backend's model but the client's
    #: screen, e.g. the settle detector's out-of-band status check.
    BACKEND_REQUEST = "backend_request"
    #: B->C: the result of a BACKEND_REQUEST, keyed by the same call_id.
    BACKEND_RESPONSE = "backend_response"
    #: B->C: the final assistant reply that ends one submitted turn.
    REPLY = "reply"
    #: B->C: structured panel-state update for the client to render.
    PANEL_UPDATE = "panel_update"
    #: B->C: ask the user something (approval / steer / confirm) on the panel.
    PROMPT = "prompt"
    #: B->C: a diagnostic line (shown in the panel or a log, never fatal).
    LOG = "log"
    #: Either direction: clean shutdown request/acknowledgement.
    BYE = "bye"
    #: Either direction: a non-fatal error report tied (optionally) to a call.
    ERROR = "error"


def message(type_: str, **fields: Any) -> dict[str, Any]:
    """Build a protocol message dict with a ``type`` and extra ``fields``.

    ``type`` is reserved; passing it in ``fields`` is a programming error.
    """
    if "type" in fields:
        raise ValueError("'type' must be passed positionally, not in fields")
    msg = {"type": type_}
    msg.update(fields)
    return msg


def msg_type(obj: dict[str, Any]) -> str:
    """Return a message's ``type``, or raise :class:`ProtocolError` if missing."""
    try:
        value = obj["type"]
    except (KeyError, TypeError) as exc:
        raise ProtocolError(f"message has no 'type': {obj!r}") from exc
    if not isinstance(value, str):
        raise ProtocolError(f"message 'type' is not a string: {value!r}")
    return value


def require(obj: dict[str, Any], *fields: str) -> None:
    """Validate that ``obj`` contains every name in ``fields``.

    Raises :class:`ProtocolError` naming the first missing field, so a malformed
    peer message fails loudly instead of surfacing as a later ``KeyError``.
    """
    for name in fields:
        if name not in obj:
            raise ProtocolError(
                f"message {msg_type(obj)!r} missing required field {name!r}"
            )


def encode_frame(obj: dict[str, Any], *, max_frame: int = DEFAULT_MAX_FRAME) -> bytes:
    """Serialise ``obj`` to a single length-prefixed frame."""
    payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    if len(payload) > max_frame:
        raise ProtocolError(
            f"frame too large: {len(payload)} bytes > limit {max_frame}"
        )
    return _HEADER.pack(len(payload)) + payload


def write_frame(
    stream: BinaryIO, obj: dict[str, Any], *, max_frame: int = DEFAULT_MAX_FRAME
) -> None:
    """Write one frame to ``stream`` and flush it."""
    stream.write(encode_frame(obj, max_frame=max_frame))
    stream.flush()


def read_frame(
    stream: BinaryIO, *, max_frame: int = DEFAULT_MAX_FRAME
) -> dict[str, Any] | None:
    """Read one frame from ``stream``.

    Returns the decoded message, or ``None`` on a clean end-of-stream at a frame
    boundary (the peer closed the connection). Raises :class:`ProtocolError` on a
    truncated frame, an over-long frame, or invalid JSON.
    """
    header = _read_exactly(stream, _HEADER.size)
    if header is None:
        return None  # clean EOF between frames
    (length,) = _HEADER.unpack(header)
    if length == 0:
        raise ProtocolError("received an empty (zero-length) frame")
    if length > max_frame:
        raise ProtocolError(f"frame length {length} exceeds limit {max_frame}")
    body = _read_exactly(stream, length)
    if body is None:
        raise ProtocolError("stream ended mid-frame (truncated body)")
    try:
        obj = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid frame payload: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError(f"frame payload is not a JSON object: {obj!r}")
    return obj


def _read_exactly(stream: BinaryIO, n: int) -> bytes | None:
    """Read exactly ``n`` bytes from ``stream``.

    Returns the bytes, or ``None`` if the stream is at a clean EOF before any of
    the ``n`` bytes arrive. Raises :class:`ProtocolError` if EOF happens after a
    partial read (a truncated frame). Loops because a pipe/socket read can return
    fewer bytes than requested.
    """
    if n == 0:
        return b""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if not buf:
                return None  # nothing read yet -> clean EOF
            raise ProtocolError(
                f"unexpected EOF: read {len(buf)} of {n} bytes"
            )
        buf += chunk
    return bytes(buf)


class FrameChannel:
    """A framed message channel over a duplex byte stream.

    Wraps a binary ``reader`` (peer -> us) and ``writer`` (us -> peer). Sends are
    serialised with a lock so multiple threads can emit frames without
    interleaving bytes; reads are expected to be driven by a single consumer.
    """

    def __init__(
        self,
        reader: BinaryIO,
        writer: BinaryIO,
        *,
        max_frame: int = DEFAULT_MAX_FRAME,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._max_frame = max_frame
        self._write_lock = threading.Lock()
        self._closed = False

    def send(self, obj: dict[str, Any]) -> None:
        """Serialise and write one message frame (thread-safe)."""
        with self._write_lock:
            write_frame(self._writer, obj, max_frame=self._max_frame)

    def recv(self) -> dict[str, Any] | None:
        """Read one message frame, or ``None`` at a clean end-of-stream."""
        return read_frame(self._reader, max_frame=self._max_frame)

    def close(self) -> None:
        """Close both underlying streams, ignoring errors (idempotent)."""
        if self._closed:
            return
        self._closed = True
        for stream in (self._writer, self._reader):
            try:
                stream.close()
            except Exception:
                pass

    def __enter__(self) -> "FrameChannel":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
