"""Shared helpers for the ``main()``-style end-to-end scripts.

The e2e scripts fork a real ``ludvart`` over a PTY. The agent loop always runs
in a backend process, and by default ``ludvart`` forks one over a plain
stdin/stdout pipe -- no SSH, no network, no assumption that localhost is
reachable. That pipe carries exactly the same framed protocol an SSH backend
would, so the scripts need no flag at all for the common case.

Set ``LUDVART_E2E_BACKEND=host:folder`` to run the same scripts against an SSH
backend instead; the only difference is how the backend is spawned.

:class:`Checks` is how these scripts report a verdict. They are run by pytest
(via the collector in ``conftest.py``), which only sees a failure if ``main()``
raises -- printing ``[FAIL]`` and returning is reported as a pass.
"""

import os

E2E_BACKEND_ENV = "LUDVART_E2E_BACKEND"

DEFAULT_COMMAND = ["bash", "--norc", "-i"]


def e2e_backend() -> str:
    """An explicit backend spec for this run, or '' to use ludvart's default."""
    return os.environ.get(E2E_BACKEND_ENV, "").strip()


def ludvart_argv(*flags: str, command=None) -> list[str]:
    """Argv for forking ``ludvart``, honouring an explicit e2e backend spec.

    ``flags`` are ludvart's own options; ``command`` (default an interactive
    bash) is what ludvart should run. ``--no-llm`` wins over the backend spec:
    that relay-only mode has no agent loop to place anywhere.
    """
    argv = ["ludvart"]
    spec = e2e_backend()
    if spec and "--no-llm" not in flags:
        argv += ["--backend", spec]
    argv += [*flags, "--", *(DEFAULT_COMMAND if command is None else command)]
    return argv


class E2EFailure(AssertionError):
    """Raised by :meth:`Checks.report` when at least one check failed."""


class Checks:
    """Named boolean results for one e2e script, with a failing verdict.

    Usage::

        checks = Checks()
        checks.add("panel opened", "ludvart>" in text)
        ...
        checks.report()   # prints the table, raises if anything failed

    Every check is printed either way, so a failing run still shows which parts
    worked -- the transcript above it is usually what you need to diagnose.
    """

    def __init__(self) -> None:
        self._results: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: object, detail: str = "") -> bool:
        """Record a check. ``detail`` is shown only when it fails."""
        ok = bool(ok)
        self._results.append((name, ok, detail))
        return ok

    def report(self) -> None:
        """Print every check; raise :class:`E2EFailure` if any failed."""
        print("===== CHECKS =====")
        failed = []
        for name, ok, detail in self._results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
            if not ok:
                failed.append(f"{name}{f' ({detail})' if detail else ''}")
        if not self._results:
            raise E2EFailure("the script recorded no checks at all")
        if failed:
            raise E2EFailure(
                f"{len(failed)}/{len(self._results)} checks failed: "
                + "; ".join(failed)
            )


def tail(text: str, limit: int = 3000) -> str:
    """The last ``limit`` characters of a transcript, for failure diagnosis."""
    return text[-limit:]


#: Substring of the panel's inject_input confirmation prompt.
APPROVAL_PROMPT = "Approve terminal input"


class Approver:
    """Answers ludvart's inject_input approval prompt during a scripted run.

    Before ludvart types anything into the terminal it asks the user to confirm
    (``[y]es / [n]o / [a]lways``). A script that never answers leaves the tool
    call -- and the whole turn -- blocked until it times out, which looks like a
    slow model rather than an unanswered prompt. Feeding this whatever is read
    from the PTY answers "always" the first time the prompt appears.

    :attr:`approved` doubles as evidence that a tool call actually reached the
    terminal, which is a useful mechanical check on its own.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buf = ""
        self.approved = False

    def feed(self, data: bytes) -> None:
        if self.approved or not data:
            return
        # The prompt can straddle two reads, so keep a small overlap.
        self._buf = (self._buf + data.decode("utf-8", "replace"))[-4096:]
        if APPROVAL_PROMPT in self._buf:
            os.write(self._fd, b"a")  # [a]lways: later tool calls run unattended
            self.approved = True


# -- reading the PTY ---------------------------------------------------------
#
# Waiting a fixed number of seconds for something to appear is both slow (the
# script always pays the worst case) and unreliable (a slower-than-usual start
# silently drops the keystrokes that follow, which shows up as a bogus product
# failure). ``wait_for`` polls a condition instead and returns as soon as it
# holds, so the scripts are quicker *and* deterministic.


def read_available(fd: int, sink, seconds: float, approver: "Approver | None" = None):
    """Feed ``sink`` whatever arrives on ``fd`` for up to ``seconds``.

    ``sink`` takes bytes: a ``pyte`` stream's ``feed``, or a ``bytearray``'s
    ``extend``. Returns ``False`` once the child has closed the PTY.
    """
    import errno
    import select
    import time as _time

    end = _time.monotonic() + seconds
    while _time.monotonic() < end:
        r, _, _ = select.select([fd], [], [], 0.05)
        if fd not in r:
            continue
        try:
            data = os.read(fd, 65536)
        except OSError as exc:
            if exc.errno == errno.EIO:  # child exited
                return False
            raise
        if not data:
            return False
        sink(data)
        if approver is not None:
            approver.feed(data)
    return True


def wait_for(fd, sink, ready, timeout: float, approver=None, settle: float = 0.0):
    """Pump ``fd`` into ``sink`` until ``ready()`` or ``timeout``; report success.

    ``settle`` keeps reading for that long after ``ready()`` first holds, for
    conditions that become true mid-redraw.
    """
    import time as _time

    end = _time.monotonic() + timeout
    while _time.monotonic() < end:
        if not read_available(fd, sink, 0.2, approver):
            break
        if ready():
            if settle:
                read_available(fd, sink, settle, approver)
            return True
    return bool(ready())


def screen_text(screen) -> str:
    """A ``pyte`` screen as plain text."""
    return "\n".join(screen.display)


#: Shown by ludvart's startup banner once the backend model is live. Waiting for
#: this is what makes keystrokes land reliably: before it, ludvart is still
#: verifying the model and the terminal echoes them instead.
STARTUP_DONE = "backend model"


def wait_until_started(fd, sink, screen, timeout: float = 90.0) -> bool:
    """Block until ludvart has finished starting up and the shell is ready.

    The banner is transient -- ludvart redraws over it as soon as the child
    shell paints its prompt -- so this latches the marker out of the byte
    stream rather than polling the rendered screen. A warm start can otherwise
    come and go entirely between two polls, which looks like a failed startup.
    """
    seen = []
    tailbuf = [""]

    def latching_sink(data: bytes) -> None:
        sink(data)
        if not seen:
            # The marker can straddle two reads, so keep a small overlap.
            tailbuf[0] = (tailbuf[0] + data.decode("utf-8", "replace"))[-4096:]
            if STARTUP_DONE in tailbuf[0]:
                seen.append(True)

    return wait_for(fd, latching_sink, lambda: bool(seen), timeout, settle=0.5)
