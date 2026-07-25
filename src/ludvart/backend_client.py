"""Client-side driver for a remote (or forked) backend agent loop.

:class:`BackendClient` submits a user question over a
:class:`~ludvart.protocol.FrameChannel`, then services the backend's
``REQUEST`` frames (snapshot / terminal tool) using the local
:class:`~ludvart.terminal_host.TerminalHost`, applies ``PANEL_UPDATE``
notifications, and returns the final ``REPLY`` text. This is the counterpart to
:class:`ludvart.remote_host.RemoteTerminalHost`.

When a :class:`BackendReconnector` is supplied, a dropped connection (e.g. a
flaky SSH link) is detected, the backend process is respawned and its last saved
session reloaded -- with progress reported on the panel -- and the failed
turn/command is retried.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .protocol import FrameChannel, MsgType, message, msg_type
from .terminal_host import TerminalHost


def read_hello(channel: FrameChannel, on_log: Callable[[str], None]) -> dict:
    """Stream startup ``LOG`` frames to ``on_log`` and return the ``HELLO`` dict.

    Before HELLO the backend reports the gateway launch and each model's
    verification as LOG frames. Raises :class:`ConnectionError` if the stream
    ends before HELLO arrives.
    """
    while True:
        msg = channel.recv()
        if msg is None:
            raise ConnectionError("backend closed before handshake")
        kind = msg_type(msg)
        if kind == MsgType.LOG:
            on_log(msg.get("text", ""))
            continue
        if kind == MsgType.HELLO:
            return msg
        # Ignore any other pre-session frames.


class BackendReconnector:
    """Owns the backend transport and respawns it when the connection drops.

    ``spawn`` is a zero-arg factory returning a fresh transport (with ``channel``
    and ``close()``) -- a local fork or an SSH process. On reconnect the backend
    is a brand-new process, so its last *saved* session is reloaded to restore
    the conversation up to the last completed turn (the backend persists after
    each turn). Progress is reported through the ``notify`` callback so it shows
    on the panel like narration.
    """

    _MAX_SPAWN_ATTEMPTS = 3
    _MAX_BACKOFF = 8.0

    def __init__(
        self, spawn: Callable[[], Any], *, on_log: Callable[[str], None] | None = None
    ) -> None:
        self._spawn = spawn
        self._on_log = on_log or (lambda _m: None)
        self._transport = None
        self.label: str | None = None
        self.session_id: str | None = None
        self.verified: bool = True
        self.verify_error: str | None = None

    def connect(self, on_log: Callable[[str], None] | None = None) -> dict:
        """Spawn the backend and complete the HELLO handshake; return HELLO."""
        log = on_log or self._on_log
        self._transport = self._spawn()
        hello = read_hello(self._transport.channel, log)
        self.label = hello.get("active_label") or "backend"
        self.session_id = hello.get("session_id")
        self.verified = bool(hello.get("verified"))
        self.verify_error = hello.get("verify_error")
        return hello

    @property
    def channel(self) -> FrameChannel | None:
        return self._transport.channel if self._transport is not None else None

    def reconnect(
        self, notify: Callable[[str], None], host: TerminalHost
    ) -> FrameChannel:
        """Respawn the backend, restore the session, and return a live channel.

        Retries the spawn with backoff. Raises :class:`ConnectionError` if the
        backend cannot be brought back.
        """
        notify("backend connection lost; reconnecting...")
        # The session to restore is the one we were on before the drop; the
        # respawned backend's own HELLO carries a fresh, empty session id, so
        # capture the target now (connect() overwrites self.session_id).
        target_session = self.session_id
        old = self._transport
        self._transport = None
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        last_exc: Exception | None = None
        for attempt in range(1, self._MAX_SPAWN_ATTEMPTS + 1):
            try:
                self.connect(on_log=notify)
                break
            except Exception as exc:  # noqa: BLE001 - reported and retried
                last_exc = exc
                notify(
                    f"reconnect attempt {attempt}/{self._MAX_SPAWN_ATTEMPTS} "
                    f"failed: {exc}"
                )
                time.sleep(min(2.0 ** attempt, self._MAX_BACKOFF))
        else:
            raise ConnectionError(f"could not reconnect to backend: {last_exc}")
        notify(f"reconnected to backend ({self.label})")
        self._restore_session(target_session, host, notify)
        return self.channel

    def _restore_session(
        self, target: str | None, host: TerminalHost, notify: Callable[[str], None]
    ) -> None:
        """Reload the last saved session so the conversation survives a respawn."""
        if not target:
            return
        notify(f"restoring session {target}...")
        BackendClient(self.channel).command(f"sessions load {target}", host)
        # We are now on the restored session again.
        self.session_id = target

    def close(self) -> None:
        if self._transport is not None:
            try:
                self._transport.close()
            except Exception:
                pass
            self._transport = None


class BackendClient:
    """Runs one question/command against a backend, servicing its requests.

    With a ``reconnector`` a dropped connection is recovered transparently: the
    backend is respawned, its session restored, and the operation retried once.
    """

    def __init__(
        self,
        channel: FrameChannel,
        *,
        reconnector: "BackendReconnector | None" = None,
    ) -> None:
        self._channel = channel
        self._reconnector = reconnector

    def ask(self, question: str, snapshot: str, host: TerminalHost) -> str:
        """Submit ``question`` (with ``snapshot``) and return the reply text."""
        def attempt() -> str:
            self._channel.send(
                message(MsgType.SUBMIT, text=question, snapshot=snapshot)
            )
            return self._pump(host)

        return self._run(attempt, host)

    def command(self, line: str, host: TerminalHost) -> None:
        """Forward a slash command and render its output (``line`` has no '/')."""
        def attempt() -> str:
            self._channel.send(message(MsgType.COMMAND, command=line))
            return self._pump(host)

        self._run(attempt, host)

    def _run(self, attempt: Callable[[], str], host: TerminalHost) -> str:
        """Run ``attempt``; on a dropped connection, reconnect once and retry."""
        try:
            return attempt()
        except (ConnectionError, OSError):
            if self._reconnector is None:
                raise
            host.set_activity("Reconnecting")
            self._channel = self._reconnector.reconnect(
                notify=host.add_info, host=host
            )
            return attempt()

    def _pump(self, host: TerminalHost) -> str:
        """Service backend frames until the turn/command ends with a ``REPLY``."""
        while True:
            msg = self._channel.recv()
            if msg is None:
                raise ConnectionError("backend disconnected during a turn")
            kind = msg_type(msg)
            if kind == MsgType.REPLY:
                return msg.get("text", "")
            self._handle(msg, host)

    def _handle(self, msg: dict, host: TerminalHost) -> None:
        kind = msg_type(msg)
        if kind == MsgType.REQUEST:
            result = self._serve_request(msg, host)
            self._channel.send(
                message(
                    MsgType.RESPONSE, call_id=msg.get("call_id"), result=result
                )
            )
        elif kind == MsgType.PANEL_UPDATE:
            self._apply_panel_update(msg, host)
        elif kind == MsgType.LOG:
            host.add_info(msg.get("text", ""))
        # Unknown message kinds are ignored so the protocol can grow.

    @staticmethod
    def _serve_request(msg: dict, host: TerminalHost) -> str:
        method = msg.get("method")
        params = msg.get("params") or {}
        if method == "snapshot":
            return host.snapshot()
        if method == "tool":
            return host.run_terminal_tool(
                params.get("name", ""), params.get("args") or {}
            )
        return f"[ludvart] unknown host request: {method!r}"

    def _apply_panel_update(self, msg: dict, host: TerminalHost) -> None:
        kind = msg.get("kind")
        if kind == "interim":
            host.narrate(msg.get("text", ""))
        elif kind == "activity":
            host.set_activity(msg.get("label", ""))
        elif kind == "info":
            host.add_info(msg.get("text", ""))
        elif kind == "system":
            host.add_system(msg.get("text", ""))
        elif kind == "model":
            host.set_model(msg.get("label", ""))
        elif kind == "transcript":
            host.set_transcript(msg.get("messages") or [])
        elif kind == "session":
            # Track the backend's current session so a reconnect can restore it.
            if self._reconnector is not None:
                self._reconnector.session_id = msg.get("session_id")
