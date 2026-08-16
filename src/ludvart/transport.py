"""Process transports for the client <-> backend split.

A :class:`Transport` gives the client a :class:`~ludvart.protocol.FrameChannel`
to a backend process, plus lifecycle management. Two concrete transports spawn
that process differently but expose the identical channel:

* :func:`local_backend` -- fork ``python -m ludvart serve`` on this host and talk
  to its stdin/stdout. Used when no ``--backend`` is given, so the client and
  backend always speak the same protocol regardless of where the backend runs.
* :func:`ssh_backend` -- run the backend on a remote host reachable by key-based
  SSH: ``ssh HOST 'cd FOLDER && .venv/bin/python -m ludvart serve'``. SSH itself
  provides authentication, encryption, and the duplex stdin/stdout pipe.

The backend must emit protocol frames only on stdout; its stderr is captured
separately (a log file or an inherited stream) so diagnostics never corrupt the
channel.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from typing import IO, Sequence

from .protocol import FrameChannel, MsgType, message

#: Seconds to wait for the backend to exit after we close its stdin (EOF) before
#: escalating to terminate(), then to kill().
_STOP_GRACE = 5.0
_TERM_GRACE = 2.0

#: How often the client tells the backend it is still there. Comfortably below
#: the backend's own patience (``server.CLIENT_TIMEOUT``) so a slow link or a
#: busy client can miss several in a row without being declared dead.
PING_INTERVAL = 30.0


def parse_backend_spec(spec: str) -> tuple[str, str]:
    """Parse a ``--backend`` spec of the form ``host:folder``.

    ``host`` is anything SSH accepts (``name`` or ``user@name``); ``folder`` is
    the path to the ludvart checkout on that host (with its ``.venv``). Splitting
    on the first colon lets the folder contain further colons. Raises
    :class:`ValueError` if either half is empty.
    """
    if ":" not in spec:
        raise ValueError(
            f"invalid --backend {spec!r}; expected 'host:folder'"
        )
    host, folder = spec.split(":", 1)
    host = host.strip()
    folder = folder.strip()
    if not host or not folder:
        raise ValueError(
            f"invalid --backend {spec!r}; both host and folder are required"
        )
    return host, folder


class Transport:
    """A live backend process and the framed channel to it.

    Not usually constructed directly; use :func:`local_backend` or
    :func:`ssh_backend`. ``close()`` shuts the backend down cleanly and is safe
    to call more than once; the object is also a context manager.
    """

    def __init__(self, proc: subprocess.Popen) -> None:
        if proc.stdin is None or proc.stdout is None:
            raise ValueError("backend process needs piped stdin and stdout")
        self._proc = proc
        self.channel = FrameChannel(proc.stdout, proc.stdin)
        self._closed = False
        self._stop_ping = threading.Event()
        self._pinger: threading.Thread | None = None

    def start_keepalive(self, interval: float = PING_INTERVAL) -> None:
        """Ping the backend periodically so it can tell we are still alive.

        Only worth starting against a backend that advertises support: an older
        one treats the unexpected frame as a protocol error. Idempotent.
        """
        if self._stop_ping.is_set() or self._pinger is not None:
            return

        def loop() -> None:
            while not self._stop_ping.wait(interval):
                try:
                    self.channel.send(message(MsgType.PING))
                except Exception:
                    return  # the channel is gone; close() handles the rest

        self._pinger = threading.Thread(
            target=loop, name="ludvart-backend-keepalive", daemon=True
        )
        self._pinger.start()

    @property
    def pid(self) -> int:
        return self._proc.pid

    def poll(self) -> int | None:
        """Return the backend's exit code, or ``None`` while it is running."""
        return self._proc.poll()

    def close(self) -> None:
        """Shut the backend down and reap it (idempotent).

        Closes stdin so the backend sees EOF and exits its read loop, then waits;
        if it does not exit it is terminated, then killed. The process is always
        waited on so it never lingers as a zombie.
        """
        if self._closed:
            return
        self._closed = True
        self._stop_ping.set()
        proc = self._proc
        # Signal EOF on the backend's stdin without closing our reader yet, so a
        # backend that flushes a final BYE is not cut off mid-write.
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=_STOP_GRACE)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=_TERM_GRACE)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=_TERM_GRACE)
                except subprocess.TimeoutExpired:
                    pass
        # Drop the channel's file objects (close is idempotent and ignores errors).
        self.channel.close()

    def __enter__(self) -> "Transport":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def spawn_transport(
    argv: Sequence[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    stderr: IO | int | None = None,
) -> Transport:
    """Spawn ``argv`` with piped stdin/stdout and wrap it in a :class:`Transport`.

    ``stderr`` defaults to inheriting the parent's stderr; pass an open file (or
    ``subprocess.DEVNULL``) to redirect the backend's diagnostics so they do not
    land on a rendered terminal.
    """
    proc = subprocess.Popen(
        list(argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr,
        cwd=cwd,
        env=env,
        bufsize=0,
    )
    return Transport(proc)


def local_backend_argv(python: str | None = None) -> list[str]:
    """Argv that runs the backend in this same environment."""
    return [python or sys.executable, "-m", "ludvart", "serve"]


def local_backend(
    *,
    python: str | None = None,
    env: dict[str, str] | None = None,
    stderr: IO | int | None = None,
) -> Transport:
    """Fork a local ``python -m ludvart serve`` backend and connect to it."""
    return spawn_transport(
        local_backend_argv(python), env=env, stderr=stderr
    )


def ssh_backend_argv(
    host: str, folder: str, *, remote_env: dict[str, str] | None = None
) -> list[str]:
    """Argv that runs the backend on ``host`` from the checkout at ``folder``.

    Uses the checkout's own virtualenv so the remote side matches the repo the
    user pointed at. ``-T`` disables PTY allocation (we want a clean binary
    stdio pipe, not a terminal), and ``-o BatchMode=yes`` fails fast instead of
    prompting when key-based auth is not set up.

    SSH does not forward the local environment; ``remote_env`` optionally sets
    variables for the backend via ``env NAME=VALUE ...`` (each value is
    shell-quoted).
    """
    cmd = ".venv/bin/python -m ludvart serve"
    if remote_env:
        assigns = " ".join(
            f"{name}={_sh_quote(str(value))}" for name, value in remote_env.items()
        )
        cmd = f"env {assigns} {cmd}"
    remote = f"cd {_sh_quote(folder)} && exec {cmd}"
    return [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        host,
        remote,
    ]


def ssh_backend(
    host: str,
    folder: str,
    *,
    remote_env: dict[str, str] | None = None,
    stderr: IO | int | None = None,
) -> Transport:
    """Run the backend on a remote host over SSH and connect to it."""
    return spawn_transport(
        ssh_backend_argv(host, folder, remote_env=remote_env), stderr=stderr
    )


def _sh_quote(value: str) -> str:
    """Single-quote ``value`` for safe embedding in the remote shell command."""
    import shlex

    return shlex.quote(value)
