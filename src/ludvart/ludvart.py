"""Transparent PTY relay.

Spawns a child command in a pseudo-terminal and shuttles bytes between the real
terminal and the child. Output is passed through verbatim (so any program --
including full-screen ncurses apps and nested ssh/tmux sessions -- behaves
exactly as if ludvart were not there) while also being fed into a ``pyte`` screen
model that maintains a live 2D view of the terminal. That screen model is the
foundation the AI overlay/agent will later read from.
"""

from __future__ import annotations

import base64
import contextlib
import errno
import fcntl
import os
import pty
import re
import select
import shutil
import signal
import struct
import sys
import termios
import threading
import time
import tty
from typing import TYPE_CHECKING, Sequence

import pyte

from .overlay import ScrollbackViewer
from .panel import AiPanel
from .render import Compositor, render_row
from .screen import LudvartScreen
from .terminal_host import TerminalHost
from .helper_src import (
    LUDVART_HELPER_MD5,
    LUDVART_HELPER_SOURCE,
    LUDVART_HELPER_VERSION,
    helper_install_command,
    helper_probe_command,
)
from .session import (
    SLASH_COMMAND_HELP,
    complete_slash,
)
from .models import PROVIDER_MENU, SERVICE_PROMPT

# ludvart commands are entered with a prefix key (like screen/tmux) followed by a
# command letter. A single-byte control character is used as the prefix so no
# terminal emulator remaps it and it survives SSH and nested screen/tmux.
#
# Default prefix: Ctrl-G (0x07). Commands:
#   <prefix> s          open the scrollback viewer
#   <prefix> a          open the AI panel (same as the summon key)
#   <prefix> o          send a literal summon byte (Ctrl-O) to the child
#   <prefix> <prefix>   send a literal prefix byte to the child
DEFAULT_PREFIX = b"\x07"  # Ctrl-G

#: Slash commands whose state lives with the agent loop (the conversation, the
#: model registry, the sessions, the MCP servers), so the client forwards them
#: to the backend instead of handling them itself. Everything else -- helper
#: installation, perf timings, approval -- is genuinely client-side.
_BACKEND_COMMANDS = frozenset(
    {"model", "sessions", "compact", "mcp_refresh", "mcp_login", "mcp_auth"}
)

# In addition to the prefix commands, a single dedicated "summon" key opens the
# AI panel in one keystroke. Ctrl-O (0x0F) is used because screen (Ctrl-A) and
# tmux (Ctrl-B) leave it alone, so it works even when ludvart runs inside them.
# To send a literal Ctrl-O to the child, use ``<prefix> o``.
DEFAULT_SUMMON = b"\x0f"  # Ctrl-O

# Bracketed paste: while the AI panel is open we enable it so the terminal wraps
# pasted text (incl. mouse/middle-click paste) in these markers. That lets us
# insert a paste verbatim without its embedded newlines submitting the prompt.
_PASTE_ON = b"\x1b[?2004h"
_PASTE_OFF = b"\x1b[?2004l"
_PASTE_START = b"\x1b[200~"
_PASTE_END = b"\x1b[201~"


def _get_winsize(fd: int) -> tuple[int, int]:
    """Return (rows, cols) for the terminal on ``fd``.

    Falls back to a sane default if the size cannot be queried or is reported as
    zero (e.g. an unsized PTY), so callers never receive a 0 dimension.
    """
    try:
        packed = fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\x00" * 8)
        rows, cols, _, _ = struct.unpack("HHHH", packed)
        if rows and cols:
            return rows, cols
    except OSError:
        pass
    size = shutil.get_terminal_size(fallback=(80, 24))
    rows = size.lines if size.lines > 0 else 24
    cols = size.columns if size.columns > 0 else 80
    return rows, cols


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    """Apply the window size ``(rows, cols)`` to the PTY on ``fd``."""
    packed = struct.pack("HHHH", rows, cols, 0, 0)
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)
    except OSError:
        pass


class TerminalLiveness:
    """Notice a terminal that has gone away without ever hanging up.

    A lost terminal normally shows up as EOF on stdin, but sshd can keep the pty
    open after the network underneath it drops. Nothing then arrives and nothing
    ever closes, so the client idles forever holding a backend -- and that
    backend's gateway -- open. After a long silence this asks the terminal to
    report its cursor position, a question every terminal answers and none of
    them displays. Only an unanswered probe counts against the session, and
    several in a row are needed before it is given up on, so a slow link is
    never mistaken for a dead one.

    Asking only while the human is silent is also what keeps the answer ours: a
    full-screen app can ask for the cursor position too and its reply travels
    the same path, but it only asks in response to input we have just seen.
    """

    #: Device Status Report -- "where is the cursor?".
    QUERY = b"\x1b[6n"

    #: The terminal's answer, ``ESC[<row>;<col>R``.
    _REPLY = re.compile(rb"\x1b\[\d+;\d+R")

    def __init__(
        self,
        idle: float,
        reply_wait: float,
        max_misses: int,
        clock=time.monotonic,  # noqa: ANN001 - injected for tests
    ) -> None:
        self.idle = idle
        self.reply_wait = reply_wait
        self.max_misses = max_misses
        self._clock = clock
        self._quiet_since = clock()
        self._probed_at: float | None = None
        self._misses = 0

    def touch(self) -> None:
        """Record a sign of life from the far side of the terminal."""
        self._quiet_since = self._clock()
        self._probed_at = None
        self._misses = 0

    def answered(self, data: bytes) -> bytes:
        """Record ``data`` as life and take this probe's answer out of it.

        The answer replies to a question the human never asked, so it must not
        reach the child; anything typed alongside it is passed on untouched.
        """
        probed = self._probed_at is not None
        self.touch()
        if probed:
            data = self._REPLY.sub(b"", data, count=1)
        return data

    def timeout(self) -> float:
        """Seconds to wait for input before the next liveness decision."""
        now = self._clock()
        if self._probed_at is not None:
            return max(0.0, self._probed_at + self.reply_wait - now)
        return max(0.0, self._quiet_since + self.idle - now)

    def check(self) -> str:
        """Return what the relay should do now: ``""``, ``probe`` or ``dead``."""
        now = self._clock()
        if self._probed_at is not None:
            if now - self._probed_at < self.reply_wait:
                return ""
            self._misses += 1
            self._probed_at = None
            if self._misses >= self.max_misses:
                return "dead"
        elif now - self._quiet_since < self.idle:
            return ""
        self._probed_at = now
        return "probe"


class _AskCancelled(Exception):
    """Raised inside the agent loop to unwind a user-cancelled LLM request."""


class _ClientTerminalHost(TerminalHost):
    """Adapts a running :class:`Ludvart` client to the :class:`TerminalHost` API.

    Used only in backend (split) mode: the backend calls these methods over the
    wire and this adapter maps them onto the live terminal -- the screen
    snapshot, the terminal tools (with their approval gate), and the panel.
    """

    def __init__(self, app: "Ludvart") -> None:
        self._app = app

    def snapshot(self) -> str:
        return self._app.snapshot_text()

    def run_terminal_tool(self, name: str, args: dict) -> str:
        with self._app._perf_timer(f"tool:{name}"):
            if name == "inject_input":
                return self._app._tool_inject_input(args)
            if name == "capture_screen_history":
                return self._app._tool_capture_screen_history(args)
            return f"[ludvart] unknown terminal tool: {name}"

    def narrate(self, text: str) -> None:
        panel = self._app._panel
        # The backend has started streaming this step's answer, so stop showing
        # the elapsed counter (same as a local streaming turn).
        self._app._mark_wait_streaming()
        if panel is not None:
            panel.interim = text

    def set_activity(self, label: str) -> None:
        # Each new activity starts a fresh waiting phase. The elapsed seconds are
        # timed here on the client (wall clock, so they include link latency);
        # the main loop's _refresh_wait tick advances the counter.
        self._app._begin_wait(label)

    def set_context_pct(self, pct: float | None) -> None:
        self._app._panel_context_pct = pct
        panel = self._app._panel
        if panel is not None:
            panel.context_pct = pct

    def add_summary(self, text: str) -> None:
        panel = self._app._panel
        if panel is not None:
            panel.add_summary(text)

    def add_info(self, text: str) -> None:
        panel = self._app._panel
        if panel is not None:
            panel.add_info(text)

    def add_system(self, text: str) -> None:
        panel = self._app._panel
        if panel is not None:
            panel.add_system(text)

    def add_system_row(self, text: str) -> None:
        panel = self._app._panel
        if panel is not None:
            panel.add_system_row(text)

    def set_model(self, label: str) -> None:
        self._app._backend_label = label
        panel = self._app._panel
        if panel is not None:
            panel.provider = label

    def set_transcript(self, messages: list) -> None:
        msgs = [tuple(m) for m in messages]
        self._app._panel_messages = msgs
        panel = self._app._panel
        if panel is not None:
            panel.restore(msgs)


class Ludvart:
    """A transparent PTY relay around a single child command.

    Parameters
    ----------
    command:
        The argv of the command to spawn (e.g. ``["bash"]`` or
        ``["ssh", "host"]``).
    prefix:
        The single-byte prefix key that introduces a ludvart command. Defaults to
        Ctrl-G. Pressing it twice sends a literal prefix byte to the child.
    summon:
        The single-byte key that opens the AI panel in one keystroke. Defaults
        to Ctrl-O, which screen/tmux leave alone. Use ``<prefix> o`` to send a
        literal summon byte to the child.
    llm:
        An optional, already-verified LLM client. When ``None``, ludvart runs as a
        plain relay with AI features disabled.
    """

    #: How many bytes to read from a fd at a time.
    READ_SIZE = 65536

    #: Completion detection polls the screen model this often (seconds). The
    #: main split loop feeds the PTY on its own thread, so this only reads.
    SETTLE_POLL = 0.05

    #: A waiting activity (a tool run or the next model response) only shows its
    #: live elapsed-seconds counter once it has been waiting at least this long,
    #: so quick operations do not flash a distracting "0s".
    ACTIVITY_ELAPSED_HINT = 2.0

    #: How long the screen must stay unchanged before the (patient) quiescence
    #: fallback considers a prompt-less context settled. Kept large so a normal
    #: command's brief silence never pre-empts the fast prompt-return path.
    SETTLE_QUIET_WINDOW = 1.30

    #: Absolute cap (seconds) on how long to wait for injected input to settle
    #: in the normal (shell/REPL) case.
    SETTLE_MAX_WAIT = 20.0

    #: Budget (seconds) set aside for one out-of-band status check. The check is
    #: an LLM round-trip and cannot be interrupted once sent, so the cap above is
    #: only honoured if we refuse to start a check we have no room for.
    SETTLE_CHECK_RESERVE = 6.0

    #: Bytes per write, and the pause between writes, when feeding the pty a
    #: large block. An unpaced burst outruns the tty's input buffer and the
    #: dropped characters are silent.
    INJECT_CHUNK_BYTES = 256
    INJECT_CHUNK_PAUSE = 0.01

    #: Cap (seconds) on waiting for the helper install to report its result.
    HELPER_INIT_MAX_WAIT = 90.0

    #: Cap (seconds) on the checksum probe that runs before the install. Short:
    #: it is one small command, and an unanswered probe just means we transfer
    #: the payload as before.
    HELPER_PROBE_MAX_WAIT = 20.0

    #: Terminal liveness (see :class:`TerminalLiveness`). After LIVENESS_IDLE
    #: seconds without a keystroke the terminal is asked whether it is still
    #: there; a live one answers well within LIVENESS_REPLY_WAIT, and only
    #: LIVENESS_MAX_MISSES unanswered probes in a row end the session.
    LIVENESS_IDLE = 300.0
    LIVENESS_REPLY_WAIT = 10.0
    LIVENESS_MAX_MISSES = 3

    #: How long the child is given to act on the hangup before it is killed.
    HANGUP_GRACE = 5.0

    #: The install result line. ``status`` is constrained to real words so the
    #: echoed command template (which contains ``status=%s``) is never mistaken
    #: for the result.
    _HELPER_INIT_RE = re.compile(
        r"LUDVART_HELPER_INIT status=(installed|current) version=(\S+) "
        r"ok=([01]) reason=(\w+)"
    )

    #: The probe result line. Matching a real digest (or ``-``) rather than the
    #: template's ``%s`` keeps the echoed command from answering itself.
    _HELPER_PROBE_RE = re.compile(r"LUDVART_HELPER_HAVE md5=([0-9a-f]{32}|-)")

    #: A full-screen (alternate-buffer) app -- vim, less, htop, screen, tmux --
    #: has no learnable shell prompt and may repaint a status line/clock forever,
    #: so the prompt-return fast path never fires and the quiescence fallback can
    #: burn the whole SETTLE_MAX_WAIT. For these we treat a much shorter unchanged
    #: window as "settled" and cap the total wait low, so injecting a keystroke
    #: (e.g. a screen "Ctrl-a n") returns promptly instead of appearing to hang.
    SETTLE_TUI_QUIET_WINDOW = 0.15
    SETTLE_TUI_MAX_WAIT = 1.5

    def __init__(
        self,
        command: Sequence[str],
        prefix: bytes = DEFAULT_PREFIX,
        summon: bytes = DEFAULT_SUMMON,
        backend_channel: "object | None" = None,
        backend_label: str | None = None,
        backend_reconnector: "object | None" = None,
        backend_needs_setup: bool = False,
    ) -> None:
        self.command = list(command)
        self.prefix = prefix
        self.summon = summon
        # When a backend channel is supplied, the agent loop runs in a separate
        # process: asks are routed over the wire and this process only provides
        # the terminal (see :meth:`_ask_via_backend`).
        self._backend_client = None
        if backend_channel is not None:
            from .backend_client import BackendClient

            self._backend_client = BackendClient(
                backend_channel, reconnector=backend_reconnector
            )
        # Label shown for the backend's active model (updated by HELLO and by
        # a backend-side /model use).
        self._backend_label = backend_label
        # The backend has an empty model registry and cannot prompt for one
        # itself, so the panel starts the guided registration on first open.
        self._backend_needs_setup = bool(backend_needs_setup)
        # Guided ``/model add`` input flow state (None unless collecting fields).
        self._model_add: dict | None = None
        self._child_pid: int = -1
        self._master_fd: int = -1
        self._stdin_fd = sys.stdin.fileno()
        self._stdout_fd = sys.stdout.fileno()
        self._old_term_attrs: list | None = None
        self._resized = False
        # True after the prefix key was pressed, while waiting for the command
        # letter (the next byte selects the ludvart command).
        self._awaiting_command = False
        # AI panel state. ``_panel`` is non-None only while the split is open;
        # ``_panel_messages`` keeps the transcript alive across toggles.
        # ``_panel_context_pct`` preserves the last context usage badge across
        # toggles so re-opening the panel keeps showing it until the next turn.
        self._panel: AiPanel | None = None
        self._panel_closing = False
        self._panel_messages: list[tuple[str, str]] = []
        self._panel_context_pct: float | None = None
        # Unsent input line preserved across panel toggles, so text typed but
        # not yet submitted survives closing and re-opening the panel.
        self._panel_draft = ""
        self._panel_draft_cursor = 0
        # True while the panel is showing the "cancel in-flight request and
        # close?" confirmation; the next keystroke answers it (y/n).
        self._confirm_close = False
        # Steering is a second in-flight request state: it collects a new
        # instruction, then cancels and replaces the active agent turn.
        self._steer_input = False
        self._steer_saved_draft = ""
        self._steer_saved_cursor = 0
        self._steer_pending: str | None = None
        self._steer_user_echo: str | None = None
        self._ask_root_question = ""
        # Prompt learned before the first chunk of a command line that is being
        # typed across several injections (see _injection_prompt_prefix).
        self._partial_line_prompt: str | None = None
        # Approval gate for LLM-triggered inject_input calls.
        self._inject_approval_all = False
        self._inject_approval_pending = False
        self._inject_approval_event = threading.Event()
        self._inject_approval_decision: bool | None = None
        # Live progress for the current waiting phase (a tool run or the wait
        # for the next model response). ``_wait_since`` is the monotonic start;
        # ``_wait_streaming`` suppresses the elapsed counter once the model has
        # begun streaming text (the narration itself is then the progress).
        self._wait_since: float | None = None
        self._wait_streaming = False
        # Bracketed-paste accumulator for the panel input (paste bursts may span
        # several stdin reads and can embed newlines).
        self._panel_pasting = False
        self._panel_pastebuf = bytearray()
        self._compositor: Compositor | None = None
        # Panel height in rows. 0 means "not yet sized": the panel defaults to
        # half the screen height the first time it opens (see _open_panel). A
        # user resize sets a concrete height that then persists across opens.
        self._panel_height = 0
        # Height to restore when PageDown undoes a PageUp "half screen" resize.
        self._panel_height_prev = 0
        self._phys_rows = 0
        self._phys_cols = 0
        self._ai_ask = None
        # The running conversation, kept in a provider-neutral log (see the
        # neutral-log schema in llm.py). Each user turn embeds the terminal
        # screen snapshot taken at ask time (the panel transcript only keeps the
        # visible question/answer text, so this is a separate buffer). The
        # provider-native context is rebuilt from this log at every request, so
        # the same conversation can be continued by any model.
        # Background LLM request while the panel spinner animates.
        self._ask_thread: threading.Thread | None = None
        self._ask_result = ""
        self._ask_done = threading.Event()
        # True only while a user/model LLM ask is running. Background actions
        # share the worker plumbing but do not need a close-confirmation prompt.
        self._llm_request_in_flight = False
        # Set to request that the in-flight background ask abandon itself: the
        # agent loop checks it between steps (and while streaming) so a closed
        # panel does not keep issuing requests or firing tool calls.
        self._ask_cancel = threading.Event()
        # How a finished background job is delivered into the panel: LLM replies
        # go through ``_deliver_reply`` (persisted), deterministic actions (e.g.
        # ``/init_helpers``) through ``_deliver_system`` (ephemeral). Set when a
        # job starts; ``_finish_ask`` falls back to a reply if unset.
        self._deliver = None
        # Internal profiling: per operation-type list of durations (seconds).
        # Populated by ``_perf_timer`` around backend asks and client-side tool
        # calls, and reported by ``/perf summary`` / ``/perf dump``.
        self._perf: dict[str, list[float]] = {}
        # Private per-run scratch directory for files saved by ``fetch_url``.
        # Created lazily on first fetch (see ``_fetch_tmp_dir``) as a 0700
        # directory owned by this user, and removed wholesale when ``run``
        # exits (including on unhandled exceptions and Ctrl-C). We do NOT sweep
        # a shared prefix at startup: /tmp is multi-user and files there may
        # belong to other users or concurrent ludvart sessions.

        rows, cols = _get_winsize(self._stdout_fd)
        # pyte keeps a live model of what the child has drawn on screen, plus a
        # scrollback of normal-buffer output that scrolled off the top.
        self.screen = LudvartScreen(cols, rows)
        self.stream = pyte.ByteStream(self.screen)
        # GNU screen / tmux "set window title" sequences (ESC k <text> ST) are
        # not understood by pyte, which then prints the title text into the
        # model -- so our snapshots show garbage like the title glued in front
        # of the prompt. We strip these from the copy fed to the pyte model
        # only; the verbatim passthrough to the real terminal is untouched, so
        # the actual screen/tmux tab title still updates correctly. This buffer
        # holds a partial sequence split across reads.
        self._title_carry = b""

        # Optional raw-output capture for diagnosing display glitches. When
        # ``LUDVART_CAPTURE`` names a path, every byte read from the child (plus
        # markers for events ludvart injects, such as the resize on panel open) is
        # appended there verbatim so the exact escape sequences can be replayed.
        self._capture_fd: int | None = None
        cap = os.environ.get("LUDVART_CAPTURE")
        if cap:
            try:
                self._capture_fd = os.open(
                    cap, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
                )
            except OSError:
                self._capture_fd = None

    # -- lifecycle -----------------------------------------------------------

    def run(self) -> int:
        """Spawn the child and relay until it exits. Returns its exit status."""
        self._child_pid, self._master_fd = pty.fork()
        if self._child_pid == 0:
            # Child process: exec the target command. On success this never
            # returns; the child's stdio is already wired to the PTY slave.
            try:
                os.execvp(self.command[0], self.command)
            except OSError as exc:
                sys.stderr.write(f"ludvart: cannot run {self.command[0]!r}: {exc}\n")
                os._exit(127)

        # Parent process.
        rows, cols = _get_winsize(self._stdout_fd)
        _set_winsize(self._master_fd, rows, cols)

        self._install_raw_mode()
        self._install_winch_handler()
        try:
            return self._loop()
        finally:
            self._restore_term()
            if self._capture_fd is not None:
                os.close(self._capture_fd)
                self._capture_fd = None

    # -- screen inspection (for the AI layer) --------------------------------

    def snapshot_text(self, trim_trailing_blank_lines: bool = True) -> str:
        """Return the visible screen as plain text.

        This is what the user currently sees, rendered by the ``pyte`` screen
        model -- correct even for full-screen ncurses apps. It is the natural
        input to hand to an LLM.

        Parameters
        ----------
        trim_trailing_blank_lines:
            If true, drop empty rows at the bottom so short screens don't come
            with a block of blank padding.
        """
        lines = list(self.screen.display)
        if trim_trailing_blank_lines:
            while lines and not lines[-1].strip():
                lines.pop()
        return "\n".join(lines)

    def scrollback_text(self) -> str:
        """Return logical output that scrolled off the top (oldest first).

        This is normal-buffer scrollback only; output produced by full-screen
        alternate-buffer apps (vim/htop/less) is intentionally excluded.
        """
        return "\n".join(self.screen.scrollback_lines())

    def snapshot(self, include_scrollback: bool = False) -> dict:
        """Return a structured snapshot of the current screen state.

        Includes the plain-text view, the terminal size, the cursor position,
        and whether a full-screen (alternate-buffer) app is active -- enough
        context for an agent to reason about the screen and decide where input
        would go.

        Parameters
        ----------
        include_scrollback:
            If true, also include the normal-buffer scrollback text under the
            ``"scrollback"`` key.
        """
        snap = {
            "rows": self.screen.lines,
            "cols": self.screen.columns,
            "cursor": {"row": self.screen.cursor.y, "col": self.screen.cursor.x},
            "alt_screen": self.screen.in_alt_screen,
            "text": self.snapshot_text(),
        }
        if include_scrollback:
            snap["scrollback"] = self.scrollback_text()
        return snap

    # -- terminal setup ------------------------------------------------------

    def _install_raw_mode(self) -> None:
        """Put the real terminal into raw mode so keys pass through untouched."""
        if not os.isatty(self._stdin_fd):
            return
        self._old_term_attrs = termios.tcgetattr(self._stdin_fd)
        tty.setraw(self._stdin_fd)

    def _restore_term(self) -> None:
        """Restore the terminal attributes saved by :meth:`_install_raw_mode`."""
        if self._old_term_attrs is not None:
            termios.tcsetattr(
                self._stdin_fd, termios.TCSAFLUSH, self._old_term_attrs
            )
            self._old_term_attrs = None

    def _install_winch_handler(self) -> None:
        """Propagate real-terminal resizes to the child PTY and pyte screen."""

        def _handler(signum, frame):  # noqa: ANN001 - signal handler signature
            self._resized = True

        signal.signal(signal.SIGWINCH, _handler)

    def _handle_resize(self) -> None:
        rows, cols = _get_winsize(self._stdout_fd)
        _set_winsize(self._master_fd, rows, cols)
        self.screen.resize(rows, cols)
        self._resized = False

    # -- main loop -----------------------------------------------------------

    def _loop(self) -> int:
        """Shuttle bytes between stdin and the PTY master until EOF/child exit."""
        master = self._master_fd
        stdin = self._stdin_fd
        liveness = TerminalLiveness(
            self.LIVENESS_IDLE, self.LIVENESS_REPLY_WAIT, self.LIVENESS_MAX_MISSES
        )

        # Nothing can be asked of a backend with an empty registry, so collect a
        # model right away rather than waiting for the user to find the summon
        # key. Closing the panel drops straight into the normal relay.
        if self._backend_needs_setup:
            self._open_panel()
            liveness.touch()

        # Set when the terminal, not the child, is what ended the session.
        lost = False
        while True:
            if self._resized:
                self._handle_resize()

            try:
                readable, _, _ = select.select(
                    [master, stdin], [], [], liveness.timeout()
                )
            except InterruptedError:
                # Interrupted by SIGWINCH (or similar); loop to handle it.
                continue

            if master in readable:
                data = self._read(master)
                if data is None:  # child closed the PTY -> it has exited
                    break
                if data:
                    # Feed the screen model, then pass through verbatim.
                    self._feed_model(data)
                    self._write_all(self._stdout_fd, data)

            if stdin in readable:
                data = self._read(stdin)
                if data is None:  # the terminal hung up -> the human is gone
                    lost = True
                    break
                if data:
                    data = liveness.answered(data)
                    if data:
                        self._handle_input(data)
                        # The panel runs its own loop, so this returns long
                        # after the keystroke that opened it.
                        liveness.touch()

            verdict = liveness.check()
            if verdict == "dead":
                lost = True
                break
            if verdict == "probe":
                try:
                    self._write_all(self._stdout_fd, TerminalLiveness.QUERY)
                except OSError:
                    # Nowhere left to write: the terminal is already gone.
                    lost = True
                    break

        if lost:
            self._hang_up_child()
        return self._reap_child()

    def _hang_up_child(self) -> None:
        """Hang the child up once the terminal it was talking to is gone.

        The child holds its own session with the PTY as its controlling
        terminal, so nothing has told it the human left. Without this,
        :meth:`_reap_child` would wait forever on a shell that is itself waiting
        for input which can never arrive.
        """
        try:
            pgid = os.getpgid(self._child_pid)
        except OSError:
            return
        with contextlib.suppress(OSError):
            os.killpg(pgid, signal.SIGHUP)
        deadline = time.monotonic() + self.HANGUP_GRACE
        while time.monotonic() < deadline:
            try:
                if os.waitpid(self._child_pid, os.WNOHANG)[0]:
                    return
            except OSError:
                return
            time.sleep(self.SETTLE_POLL)
        with contextlib.suppress(OSError):
            os.killpg(pgid, signal.SIGKILL)

    # -- input handling / prefix commands -----------------------------------

    def _handle_input(self, data: bytes) -> None:
        """Forward human input to the child, intercepting ludvart prefix commands.

        Input is processed one byte at a time so the prefix key and its
        following command letter are recognized even when they arrive in the
        same read or split across reads. The prefix key itself is never
        forwarded unless pressed twice (``<prefix> <prefix>`` sends a literal).
        """
        for i in range(len(data)):
            byte = data[i : i + 1]
            if self._awaiting_command:
                self._awaiting_command = False
                self._run_prefix_command(byte)
            elif byte == self.prefix:
                # Enter command mode; the next byte selects the command.
                self._awaiting_command = True
            elif byte == self.summon:
                # Single-key summon: open the AI panel immediately.
                self._open_panel()
            else:
                self._write_all(self._master_fd, byte)

    def _run_prefix_command(self, byte: bytes) -> None:
        """Handle the command byte following the prefix key."""
        if byte == self.prefix:
            # Doubled prefix -> send a literal prefix byte to the child.
            self._write_all(self._master_fd, self.prefix)
        elif byte in (b"o", b"O"):
            # Send a literal summon byte (Ctrl-O) to the child.
            self._write_all(self._master_fd, self.summon)
        elif byte in (b"s", b"S"):
            self._open_scrollback_viewer()
        elif byte in (b"a", b"A"):
            self._open_panel()
        # Unknown command letters are ignored (not forwarded), matching the
        # screen/tmux convention of swallowing unrecognized prefix commands.

    def _open_scrollback_viewer(self) -> None:
        """Pause passthrough and show the scrollback overlay."""
        rows, cols = _get_winsize(self._stdout_fd)
        lines = self.screen.full_text(include_scrollback=True)
        viewer = ScrollbackViewer(self._stdout_fd, self._stdin_fd, rows, cols)
        viewer.show(lines)

    def _ai_ask_callback(self):
        """Return the ``ask`` callable and a short provider label.

        The agent loop always runs in the backend process, so the only two
        cases are "we have a backend" and "no provider is configured".
        """
        if self._backend_client is not None:
            return self._ask_via_backend, (self._backend_label or "backend")

        def ask(_question: str) -> str:
            return (
                "No LLM provider is configured. Set the "
                "{OPENAI,ANTHROPIC,GOOGLE,CUSTOM}_API_URL/_API_KEY/_MODEL "
                "environment variables and restart ludvart."
            )

        return ask, "no LLM"

    def _ask_via_backend(self, question: str) -> str:
        """Run one ask through the backend, serving its terminal requests locally.

        The backend owns the conversation and the LLM; this process captures the
        ask-time snapshot, then answers the backend's snapshot/terminal-tool
        requests (via a client-side host adapter) and renders its narration.
        """
        host = _ClientTerminalHost(self)
        snapshot = self.snapshot_text()
        try:
            with self._perf_timer("backend_ask"):
                return self._backend_client.ask(question, snapshot, host=host)
        finally:
            # Stop the elapsed-seconds clock the backend's activity frames started.
            self._end_wait()

    def _forward_command_to_backend(self, command_line: str, payload=None) -> None:
        """Forward a slash command (without its '/') to the backend on a worker.

        Runs on the panel's background action thread because a backend
        ``/model use|add`` builds and verifies a model (and may launch a
        gateway), which can block. ``payload`` carries structured data (e.g. a
        new registration for ``model add``). Result lines are streamed back via
        the host adapter.
        """
        host = _ClientTerminalHost(self)

        def worker() -> str:
            try:
                self._backend_client.command(command_line, host, payload=payload)
            finally:
                self._end_wait()
            return ""

        self._start_action(worker, activity="Working")

    # -- AI panel (bottom split) --------------------------------------------

    def _open_panel(self) -> None:
        """Open the AI panel as a bottom split and run it until it is closed.

        The application is resized to the region above the panel (it just sees a
        smaller terminal, via SIGWINCH) and ludvart switches from passthrough to
        compositing: the child draws into the pyte model, which ludvart renders
        onto the top region while owning the panel rows below.
        """
        rows, cols = _get_winsize(self._stdout_fd)
        if rows < 5 or cols < 10:
            return  # too small to usefully split
        self._phys_rows, self._phys_cols = rows, cols
        # Default the panel to half the screen height on first open; a height the
        # user has chosen (via resize) persists and is kept on later opens.
        if self._panel_height <= 0:
            self._panel_height = max(3, rows // 2)
        height = max(3, min(self._panel_height, rows - 2))
        self._panel_height = height
        if self._panel_height_prev <= 0:
            self._panel_height_prev = height

        ask, provider = self._ai_ask_callback()
        self._ai_ask = ask
        self._panel = AiPanel(cols, height, provider)
        self._panel.restore(self._panel_messages)
        self._panel.context_pct = self._panel_context_pct
        # Restore any unsent input line typed before the last toggle.
        self._restore_panel_draft()
        self._panel_closing = False
        self._confirm_close = False
        self._panel_pasting = False
        self._panel_pastebuf = bytearray()

        self._apply_split_size()
        self._compositor = Compositor(rows, cols)
        self._write_all(
            self._stdout_fd, b"\x1b[?25h" + _PASTE_ON + self._compositor.clear()
        )
        self._render_split()
        self._maybe_start_backend_setup()
        try:
            self._split_loop()
        finally:
            self._leave_split()

    def _maybe_start_backend_setup(self) -> None:
        """Run the guided registration when the backend has no model yet.

        The backend's stdin/stdout carry the protocol, so it cannot ask for a
        model itself; it advertises ``needs_setup`` in its HELLO and the panel
        collects the fields here. The registration is then sent over the wire and
        stored on the backend, which is what makes this work identically for a
        forked and an SSH backend.
        """
        if not self._backend_needs_setup:
            return
        self._backend_needs_setup = False
        panel = self._panel
        if panel is None:
            return
        panel.add_system("No model is registered on the backend yet.")
        self._model_add_start()

    def _apply_split_size(self) -> None:
        """Resize the model and child PTY to the region above the panel."""
        app_rows = max(1, self._phys_rows - self._panel_height)
        self.screen.resize(app_rows, self._phys_cols)
        _set_winsize(self._master_fd, app_rows, self._phys_cols)
        self._capture(marker=b"resize %dx%d" % (app_rows, self._phys_cols))

    def _split_loop(self) -> None:
        master = self._master_fd
        stdin = self._stdin_fd
        while not self._panel_closing:
            if self._resized:
                self._handle_split_resize()
            # While waiting on the LLM, wake up periodically to advance the
            # spinner animation.
            timeout = 0.12 if (self._panel and self._panel.thinking) else None
            try:
                readable, _, _ = select.select([master, stdin], [], [], timeout)
            except InterruptedError:
                continue
            if master in readable:
                data = self._read(master)
                if data is None:  # child exited
                    self._panel_closing = True
                    break
                if data:
                    self._feed_model(data)
                    self._render_split()
            if stdin in readable:
                data = self._read(stdin)
                if data:
                    self._panel_input(data)
                    if not self._panel_closing:
                        self._render_split()
            if self._panel is not None and self._panel.thinking:
                if self._ask_done.is_set():
                    self._finish_ask()
                else:
                    self._panel.tick += 1
                    self._refresh_wait()
                    self._render_split()

    def _handle_split_resize(self) -> None:
        """Re-lay-out the split after the real terminal changed size."""
        self._resized = False
        rows, cols = _get_winsize(self._stdout_fd)
        self._phys_rows, self._phys_cols = rows, cols
        self._panel_height = max(3, min(self._panel_height, rows - 2))
        self._panel.height = self._panel_height
        self._panel.set_cols(cols)
        self._apply_split_size()
        self._compositor = Compositor(rows, cols)
        self._write_all(self._stdout_fd, self._compositor.clear())
        self._render_split()

    def _resize_panel(self, delta: int) -> None:
        """Grow (delta>0) or shrink (delta<0) the panel by ``delta`` rows."""
        self._set_panel_height(self._panel_height + delta)

    def _panel_half(self) -> None:
        """Resize the panel to half the overall screen height (PageUp).

        Remembers the current height first so PageDown can restore it. A second
        PageUp while already at half is a no-op.
        """
        half = max(3, self._phys_rows // 2)
        if self._panel_height != half:
            self._panel_height_prev = self._panel_height
            self._set_panel_height(half)

    def _panel_restore_height(self) -> None:
        """Restore the height remembered before the last PageUp (PageDown)."""
        self._set_panel_height(self._panel_height_prev)

    def _set_panel_height(self, height: int) -> None:
        """Set the panel height (clamped) and repaint the split."""
        height = max(3, min(height, self._phys_rows - 2))
        if height == self._panel_height:
            return
        self._panel_height = height
        self._panel.height = height
        self._apply_split_size()
        self._compositor = Compositor(self._phys_rows, self._phys_cols)
        self._write_all(self._stdout_fd, self._compositor.clear())
        self._render_split()

    def _render_split(self) -> None:
        """Composite the app region (from the model) and the panel to screen."""
        comp = self._compositor
        panel = self._panel
        if comp is None or panel is None:
            return
        cols = self._phys_cols
        app_rows = self.screen.lines
        out = bytearray()
        for y in range(app_rows):
            out += comp.row_update(y, render_row(self.screen, y, cols))
        for i, payload in enumerate(panel.render(panel.height, cols)):
            out += comp.row_update(app_rows + i, payload)
        cur_row, cur_col = panel.cursor_rowcol()
        out += b"\x1b[%d;%dH" % (app_rows + cur_row + 1, cur_col)
        self._write_all(self._stdout_fd, out)

    def _leave_split(self) -> None:
        """Tear down the split: resize the app back and restore the screen."""
        rows, cols = self._phys_rows, self._phys_cols
        if self._panel is not None:
            self._panel_messages = self._panel.messages  # keep for next toggle
            self._panel_context_pct = self._panel.context_pct
            # Preserve the unsent input line so it survives the toggle.
            self._save_panel_draft()
        self._confirm_close = False
        self.screen.resize(rows, cols)
        _set_winsize(self._master_fd, rows, cols)
        self._compositor = None
        self._panel = None
        self._panel_pasting = False
        self._panel_pastebuf = bytearray()
        # Repaint the full-size app from the model; the child's own SIGWINCH
        # redraw will then flow in via passthrough and stay consistent.
        out = bytearray(_PASTE_OFF + b"\x1b[?25h\x1b[2J")
        for y in range(rows):
            out += b"\x1b[%d;1H" % (y + 1) + render_row(self.screen, y, cols)
        out += b"\x1b[%d;%dH" % (self.screen.cursor.y + 1, self.screen.cursor.x + 1)
        self._write_all(self._stdout_fd, out)

    # -- panel input ---------------------------------------------------------

    def _panel_input(self, data: bytes) -> None:
        """Route a stdin read to the panel, extracting bracketed pastes first."""
        if self._inject_approval_pending:
            self._handle_inject_approval(data)
            return
        if self._steer_input:
            self._handle_steer_input(data)
            return
        if self._confirm_close:
            self._handle_confirm_close(data)
            return
        if self._panel_pasting:
            self._panel_pastebuf += data
            self._drain_paste()
            return
        start = data.find(_PASTE_START)
        if start != -1:
            before = data[:start]
            if before:
                self._panel_dispatch(before)
            self._panel_pasting = True
            self._panel_pastebuf = bytearray(data[start + len(_PASTE_START) :])
            self._drain_paste()
            return
        self._panel_dispatch(data)

    def _drain_paste(self) -> None:
        """Consume the paste buffer up to the end marker, if it has arrived."""
        end = self._panel_pastebuf.find(_PASTE_END)
        if end == -1:
            return  # marker not here yet; keep accumulating across reads
        pasted = bytes(self._panel_pastebuf[:end])
        rest = bytes(self._panel_pastebuf[end + len(_PASTE_END) :])
        self._panel_pasting = False
        self._panel_pastebuf = bytearray()
        self._apply_paste(pasted)
        if rest:
            self._panel_input(rest)

    def _apply_paste(self, pasted: bytes) -> None:
        """Insert pasted bytes into the input, newlines and all."""
        panel = self._panel
        if panel is None:
            return
        text = pasted.decode("utf-8", "replace")
        # Newlines are kept so a pasted snippet stays the shape it was written
        # in; bracketed paste is what makes that safe, since the terminal tells
        # us these bytes are content rather than an Enter the user pressed.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = "".join(ch if ch >= " " or ch == "\n" else " " for ch in text)
        if cleaned:
            panel.editor.insert(cleaned)
            panel.scroll = 0

    def _panel_dispatch(self, data: bytes) -> None:
        """Route a non-paste stdin read (commands, summon/prefix, keys)."""
        if self._awaiting_command:
            self._awaiting_command = False
            self._panel_command(data)
            return
        if data == self.summon:
            self._request_toggle_close()  # summon key toggles the panel closed
            return
        if data == self.prefix:
            self._awaiting_command = True
            return
        if data[:1] == self.prefix and len(data) > 1:
            self._panel_command(data[1:])
            return
        self._panel_key(data)

    def _request_toggle_close(self) -> None:
        """Toggle the panel closed, confirming first if an LLM request is running.

        With an in-flight LLM ask, closing would abandon it, so the user is asked
        to confirm; the next keystroke is routed to :meth:`_handle_confirm_close`.
        Deterministic background actions (no LLM) close without prompting.
        """
        panel = self._panel
        if panel is None:
            return
        if self._llm_request_in_flight:
            if not self._confirm_close:
                self._confirm_close = True
                panel.confirm_prompt = (
                    "LLM request in progress: (a)bort & close  (c)ontinue  "
                    "(s)teer"
                )
            return
        self._panel_closing = True

    def _handle_confirm_close(self, data: bytes) -> None:
        """Answer the in-flight request close prompt.

        ``a`` cancels the request and closes the panel; ``c`` (or Esc / Ctrl-C)
        keeps it open; ``s`` collects a steering instruction. Any other key
        leaves the prompt pending.
        """
        if data in (b"a", b"A"):
            self._confirm_close = False
            if self._panel is not None:
                self._panel.confirm_prompt = ""
            self._cancel_ask()
            self._panel_closing = True
        elif data in (b"c", b"C", b"\x1b", b"\x03"):
            self._confirm_close = False
            if self._panel is not None:
                self._panel.confirm_prompt = ""
        elif data in (b"s", b"S"):
            self._enter_steer_input()

    def _enter_steer_input(self) -> None:
        """Replace the close prompt with an editable steering input line."""
        panel = self._panel
        if panel is None:
            return
        self._confirm_close = False
        panel.confirm_prompt = ""
        self._steer_saved_draft = panel.editor.text
        self._steer_saved_cursor = panel.editor.cursor
        panel.editor.set_text("")
        panel.steer_prompt = "Steer request: "
        self._steer_input = True

    def _exit_steer_input(self, *, restore_draft: bool) -> None:
        """Leave steering mode and optionally restore the prior input draft."""
        panel = self._panel
        self._steer_input = False
        if panel is not None:
            panel.steer_prompt = ""
            if restore_draft:
                panel.editor.set_text(self._steer_saved_draft)
                panel.editor.cursor = min(
                    self._steer_saved_cursor, len(self._steer_saved_draft)
                )
        self._steer_saved_draft = ""
        self._steer_saved_cursor = 0

    def _handle_steer_input(self, data: bytes) -> None:
        """Edit, submit, or abandon the steering instruction."""
        panel = self._panel
        if panel is None:
            return
        if data in (b"\r", b"\n"):
            self._submit_steer()
        elif data in (b"\x1b", b"\x03"):
            self._exit_steer_input(restore_draft=True)
        elif data in (b"\x7f", b"\x08"):
            panel.editor.backspace()
            panel.scroll = 0
        elif data in (b"\x1b[C", b"\x1bOC"):
            panel.editor.right()
        elif data in (b"\x1b[D", b"\x1bOD"):
            panel.editor.left()
        else:
            text = data.decode("utf-8", "replace")
            cleaned = "".join(ch for ch in text if ch >= " ")
            if cleaned:
                panel.editor.insert(cleaned)
                panel.scroll = 0

    def _submit_steer(self) -> None:
        """Queue a steered replacement after the current worker unwinds."""
        panel = self._panel
        if panel is None:
            return
        steer_text = panel.take_input().strip()
        if not steer_text:
            self._exit_steer_input(restore_draft=True)
            return
        self._steer_pending = self._compose_steer_question(steer_text)
        self._steer_user_echo = steer_text
        self._exit_steer_input(restore_draft=True)
        # Keep the spinner active so _split_loop continues polling _ask_done.
        self._ask_cancel.set()
        if self._backend_client is not None:
            self._backend_client.cancel()
        self._end_wait()
        panel.interim = ""
        panel.activity = "Steering"

    @staticmethod
    def _compose_steer_question(steer: str) -> str:
        """Wrap a steering instruction as the user turn that replaces the ask.

        The interrupted turn keeps whatever it finished, so the model reads its
        own partial narration and tool results directly above this and needs no
        reconstruction of them.
        """
        return (
            "The user interrupted you with a new instruction. Continue from the "
            "work above and the current terminal state, and do not repeat what "
            "is already done. Where this conflicts with the earlier request, "
            "follow this.\n"
            f"<steeringInstruction>\n{steer.strip()}\n</steeringInstruction>"
        )

    def _cancel_ask(self) -> None:
        """Abandon the in-flight LLM request and drop its eventual result.

        The worker runs on a daemon thread that cannot be force-killed, so this
        sets a cancellation flag the agent loop checks between steps (and while
        streaming) to unwind promptly without issuing further requests or firing
        more tool calls. Its result is ignored (see :meth:`_finish_ask`).
        """
        self._ask_cancel.set()
        self._llm_request_in_flight = False
        self._resolve_inject_approval(False)
        self._end_wait()
        panel = self._panel
        if panel is not None:
            panel.thinking = False
            panel.interim = ""

    def _inject_approval_preview(self, text: str) -> str:
        """Return a readable preview for an inject_input approval request."""
        helper_run = re.search(
            r"(?:^|[;&|]\s*)\S*ludvart_helper\s+run\s+--b64\s+(\S+)", text
        )
        if helper_run:
            try:
                return base64.b64decode(helper_run.group(1), validate=True).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                pass
        return text

    def _inject_approval_prompt(self, text: str) -> str:
        """Prompt line for an inject_input approval request.

        Keep the typed payload visible but compact enough for a single-line
        input row by escaping line breaks and clipping long text.
        """
        shown = self._inject_approval_preview(text)
        shown = shown.replace("\r", r"\r").replace("\n", r"\n")
        if len(shown) > 160:
            shown = shown[:157] + "..."
        return (
            f'WARNING: Approve terminal input: "{shown}"? [y]es / [n]o / [a]lways'
        )


    def _resolve_inject_approval(self, approved: bool) -> None:
        """Finish a pending inject_input approval request and unblock the tool."""
        if not self._inject_approval_pending:
            return
        self._inject_approval_pending = False
        self._inject_approval_decision = approved
        if self._panel is not None:
            self._panel.confirm_prompt = ""
        self._inject_approval_event.set()

    def _handle_inject_approval(self, data: bytes) -> None:
        """Handle y/n/a answers for a pending inject_input approval prompt."""
        if data in (b"y", b"Y"):
            self._resolve_inject_approval(True)
        elif data in (b"a", b"A"):
            self._inject_approval_all = True
            self._resolve_inject_approval(True)
        elif data in (b"n", b"N", b"\x1b", b"\x03"):
            self._resolve_inject_approval(False)

    def _await_inject_approval(self, text: str) -> bool:
        """Block the inject_input tool call until the user answers y/n/a."""
        if self._inject_approval_all:
            return True
        panel = self._panel
        if panel is None:
            return False
        self._inject_approval_decision = None
        self._inject_approval_event = threading.Event()
        self._inject_approval_pending = True
        panel.confirm_prompt = self._inject_approval_prompt(text)
        while True:
            if self._inject_approval_event.wait(0.05):
                return bool(self._inject_approval_decision)
            if self._ask_cancel.is_set():
                self._resolve_inject_approval(False)
                return False

    def _begin_wait(self, label: str) -> None:
        """Start a waiting phase (a tool run or the next model response).

        Sets the spinner label and starts the elapsed-time clock that
        :meth:`_refresh_wait` reads so the user can see the wait progressing.
        """
        self._wait_since = time.monotonic()
        self._wait_streaming = False
        panel = self._panel
        if panel is not None:
            panel.activity = label
            panel.activity_elapsed = None

    def _end_wait(self) -> None:
        """Clear the waiting phase so the elapsed counter stops and hides."""
        self._wait_since = None
        self._wait_streaming = False
        if self._panel is not None:
            self._panel.activity_elapsed = None

    def _mark_wait_streaming(self) -> None:
        """Note that the model has begun streaming, hiding the elapsed counter."""
        self._wait_streaming = True
        if self._panel is not None:
            self._panel.activity_elapsed = None

    def _refresh_wait(self) -> None:
        """Update the spinner's live elapsed-seconds counter (main-loop tick)."""
        panel = self._panel
        if panel is None or not panel.thinking or self._wait_since is None:
            return
        if self._wait_streaming:
            return
        elapsed = time.monotonic() - self._wait_since
        panel.activity_elapsed = elapsed if elapsed >= self.ACTIVITY_ELAPSED_HINT else None

    def _save_panel_draft(self) -> None:
        """Remember the unsent input line so it survives a panel toggle."""
        if self._panel is not None:
            self._panel_draft = self._panel.editor.text
            self._panel_draft_cursor = self._panel.editor.cursor

    def _restore_panel_draft(self) -> None:
        """Reapply the input line preserved by :meth:`_save_panel_draft`."""
        if self._panel is not None:
            self._panel.editor.text = self._panel_draft
            self._panel.editor.cursor = min(
                self._panel_draft_cursor, len(self._panel_draft)
            )

    def _panel_command(self, key: bytes) -> None:
        """Handle a prefix command while the panel is open."""
        if key in (b"a", b"A"):
            self._request_toggle_close()  # toggle closed
        elif key in (b"\x1b[A", b"\x1bOA"):  # Up -> grow panel
            self._resize_panel(1)
        elif key in (b"\x1b[B", b"\x1bOB"):  # Down -> shrink panel
            self._resize_panel(-1)
        elif key == b"\x1b[5~":  # PageUp -> half the screen height
            self._panel_half()
        elif key == b"\x1b[6~":  # PageDown -> restore previous height
            self._panel_restore_height()

    def _panel_key(self, key: bytes) -> None:
        """Handle a normal keystroke while the panel is focused."""
        panel = self._panel
        editor = panel.editor
        if key in (b"\r", b"\n"):
            self._panel_submit()
        elif key in (b"\x1b\r", b"\x1b\n"):  # Alt-Enter -> newline, not submit
            editor.insert("\n")
            panel.scroll = 0
        elif key in (b"\x7f", b"\x08"):  # Backspace
            editor.backspace()
            panel.scroll = 0
        elif key == b"\x1b":  # bare Esc closes
            self._request_toggle_close()
        elif key in (b"\x1b[C", b"\x1bOC"):  # Right
            editor.right()
        elif key in (b"\x1b[D", b"\x1bOD"):  # Left
            editor.left()
        elif key in (b"\x1b[A", b"\x1bOA"):  # Up -> within the input, else scroll
            if not editor.up() and not editor.mark:
                panel.scroll_up(1)
        elif key in (b"\x1b[B", b"\x1bOB"):  # Down -> within the input, else scroll
            if not editor.down() and not editor.mark:
                panel.scroll_down(1)
        elif key in (b"\x00", b"\x1b[32;2u"):  # Ctrl-Space -> set/clear the mark
            editor.toggle_mark()
        # Shift-arrows extend a selection, as they do in every GUI editor. A
        # shifted Up/Down at the edge of the input does not fall through to
        # scrolling the transcript: the user is selecting, not navigating.
        elif key in (b"\x1b[1;2C", b"\x1b[c"):  # Shift-Right
            editor.right(select=True)
        elif key in (b"\x1b[1;2D", b"\x1b[d"):  # Shift-Left
            editor.left(select=True)
        elif key in (b"\x1b[1;2A", b"\x1b[a"):  # Shift-Up
            editor.up(select=True)
        elif key in (b"\x1b[1;2B", b"\x1b[b"):  # Shift-Down
            editor.down(select=True)
        elif key in (b"\x1b[1;2H", b"\x1b[1;2~", b"\x1b[7$"):  # Shift-Home
            editor.home(select=True)
        elif key in (b"\x1b[1;2F", b"\x1b[4;2~", b"\x1b[8$"):  # Shift-End
            editor.end(select=True)
        elif key in (b"\x1b[H", b"\x1bOH", b"\x1b[1~", b"\x1b[7~"):  # Home
            editor.home()
        elif key in (b"\x1b[F", b"\x1bOF", b"\x1b[4~", b"\x1b[8~"):  # End
            editor.end()
        elif key == b"\x1b[3~":  # Delete (forward)
            editor.delete()
            panel.scroll = 0
        elif key == b"\x1b[5~":  # PageUp
            panel.scroll_up(max(1, panel.height - 2))
        elif key == b"\x1b[6~":  # PageDown
            panel.scroll_down(max(1, panel.height - 2))
        elif key == b"\x01":  # Ctrl-A -> line start
            editor.home()
        elif key == b"\x05":  # Ctrl-E -> line end
            editor.end()
        elif key == b"\x15":  # Ctrl-U -> kill to start
            editor.kill_to_start()
            panel.scroll = 0
        elif key == b"\x0b":  # Ctrl-K -> kill to end
            editor.kill_to_end()
            panel.scroll = 0
        elif key == b"\x17":  # Ctrl-W -> delete word back
            editor.delete_word_back()
            panel.scroll = 0
        elif key == b"\t":  # Tab -> complete an internal slash command
            self._complete_input()
        elif key[:1] == b"\x1b":
            return  # ignore other escape sequences
        else:
            try:
                text = key.decode("utf-8")
            except UnicodeDecodeError:
                return
            text = "".join(ch for ch in text if ch >= " ")
            if text:
                editor.insert(text)
                panel.scroll = 0

    def _panel_submit(self) -> None:
        """Send the typed question to the LLM on a background thread.

        The request runs off the render loop so the spinner keeps animating and
        the application region keeps updating while we wait for the reply.
        """
        panel = self._panel
        if panel.thinking:
            return
        question = panel.take_input()
        # A guided ``/model add`` flow consumes plain input lines until done.
        if self._model_add is not None:
            self._feed_model_add(question)
            return
        if not question:
            return
        if question.startswith("/"):
            self._handle_slash_command(question)
            return
        self._start_ask(question, user_echo=question)

    # -- internal commands ---------------------------------------------------

    def _complete_input(self) -> None:
        """Tab-complete the current input if it is an internal slash command."""
        panel = self._panel
        if panel is None:
            return
        completed = complete_slash(panel.editor.text)
        if completed is not None and completed != panel.editor.text:
            panel.editor.set_text(completed)
            panel.scroll = 0

    def _handle_slash_command(self, line: str) -> None:
        """Run an internal (``/``-prefixed) command; never sent to the LLM.

        The command echo and its output are shown as ephemeral "system" lines
        that are not persisted to the saved conversation.
        """
        panel = self._panel
        if panel is None:
            return
        panel.add_system(f"> {line}")
        parts = line[1:].split()
        cmd = parts[0] if parts else ""
        args = parts[1:]
        # The conversation, the model registry, the sessions and the MCP servers
        # all live on the backend, so those commands are forwarded there.
        # ``/model add`` is the exception: its guided prompts run locally (on the
        # panel) and only the finished registration is sent over to be verified.
        if cmd in _BACKEND_COMMANDS:
            if self._backend_client is None:
                panel.add_system(f"/{cmd} needs an agent backend (not in --no-llm).")
            elif cmd == "model" and (args[0] if args else "list") == "add":
                self._model_add_start()
            else:
                self._forward_command_to_backend(line[1:])
            self._render_split()
            return
        if cmd == "init_helpers":
            self._cmd_init_helpers()
        elif cmd == "perf":
            self._cmd_perf(args)
        elif cmd == "revoke_approval":
            self._cmd_revoke_approval()
        elif cmd == "help":
            self._cmd_help()
        else:
            panel.add_system(f"Unknown command: /{cmd or ''}")
        self._render_split()

    def _cmd_help(self) -> None:
        """Handle ``/help``: list the internal panel commands and what they do."""
        panel = self._panel
        if panel is None:
            return
        panel.add_system("Internal panel commands (not sent to the LLM):")
        width = max(len(usage) for usage, _ in SLASH_COMMAND_HELP)
        for usage, desc in SLASH_COMMAND_HELP:
            panel.add_system(f"  {usage.ljust(width)}  {desc}")

    def _cmd_perf(self, args: list[str]) -> None:
        """Handle ``/perf [summary|dump]``: report internal operation timings.

        Timings are collected for every major operation -- each LLM request
        (``llm_request``) and each tool call (``tool:<name>``) -- and kept as a
        per-type list of durations. ``summary`` reports min/avg/max per type;
        ``dump`` prints the raw records. Defaults to ``summary``.
        """
        panel = self._panel
        if panel is None:
            return
        sub = args[0] if args else "summary"
        if sub == "summary":
            self._perf_summary()
        elif sub == "dump":
            self._perf_dump()
        else:
            panel.add_system("Usage: /perf [summary|dump]")

    def _perf_summary(self) -> None:
        """Report min/avg/max duration (seconds) per recorded operation type."""
        panel = self._panel
        if panel is None:
            return
        if not self._perf:
            panel.add_system("No performance records yet.")
            return
        panel.add_system("Performance summary (durations in seconds):")
        name_w = max(max(len(op) for op in self._perf), len("operation"))
        panel.add_system(
            f"  {'operation'.ljust(name_w)}  {'n':>4}  "
            f"{'min':>8}  {'avg':>8}  {'max':>8}"
        )
        for op in sorted(self._perf):
            samples = self._perf[op]
            n = len(samples)
            lo = min(samples)
            hi = max(samples)
            avg = sum(samples) / n
            panel.add_system(
                f"  {op.ljust(name_w)}  {n:>4}  "
                f"{lo:>8.3f}  {avg:>8.3f}  {hi:>8.3f}"
            )

    def _perf_dump(self) -> None:
        """Dump the raw per-operation timing records (seconds) into the panel."""
        panel = self._panel
        if panel is None:
            return
        if not self._perf:
            panel.add_system("No performance records yet.")
            return
        panel.add_system("Performance records (raw, durations in seconds):")
        for op in sorted(self._perf):
            samples = self._perf[op]
            joined = ", ".join(f"{d:.3f}" for d in samples)
            panel.add_system(f"  {op} ({len(samples)}): {joined}")

    def _cmd_init_helpers(self) -> None:
        """Handle ``/init_helpers``: install or repair ~/.ludvart/bin/ludvart_helper.

        This is deterministic and does NOT involve the LLM. The harness ships the
        canonical helper source and injects one self-contained shell command that
        compares the on-disk md5 to the pinned golden md5 (without executing the
        existing file) and rewrites it from an embedded base64 payload only when
        it is missing, outdated, or modified. The command relies solely on the
        foreground host's own python3/HOME, so it also works over ssh.

        A short checksum probe runs first so an up-to-date host is spared the
        payload, which can only get there by being typed at its shell.
        """
        panel = self._panel
        if panel is None:
            return

        def worker() -> str:
            if self._helper_is_current():
                return (
                    f"ludvart_helper is already up to date "
                    f"(v{LUDVART_HELPER_VERSION}, checksum verified)."
                )
            command = helper_install_command()
            self._write_paced(self._master_fd, command.replace("\n", "\r").encode("utf-8") + b"\r")
            return self._parse_helper_init(self._wait_for_helper_init())

        self._start_action(
            worker,
            info=f"Installing/verifying ludvart_helper v{LUDVART_HELPER_VERSION}\u2026",
            activity="Installing ludvart_helper",
        )

    def _helper_is_current(self) -> bool:
        """Ask the foreground host whether its helper already matches the golden copy.

        An unanswered or unreadable probe reports False, so a host we cannot
        question still gets the install (which does its own md5 check).
        """
        self._write_paced(
            self._master_fd, helper_probe_command().encode("utf-8") + b"\r"
        )
        found = self._HELPER_PROBE_RE.search(
            self._wait_for_marker(self._HELPER_PROBE_RE, self.HELPER_PROBE_MAX_WAIT)
        )
        return found is not None and found.group(1) == LUDVART_HELPER_MD5

    def _write_paced(self, fd: int, data: bytes) -> None:
        """Feed the pty in small bursts so its input buffer can keep up.

        A tty drops characters when a write outruns it, which corrupts the
        install payload silently.
        """
        for i in range(0, len(data), self.INJECT_CHUNK_BYTES):
            self._write_all(fd, data[i:i + self.INJECT_CHUNK_BYTES])
            time.sleep(self.INJECT_CHUNK_PAUSE)

    def _wait_for_helper_init(self) -> str:
        """Poll the screen for the install result line."""
        return self._wait_for_marker(self._HELPER_INIT_RE, self.HELPER_INIT_MAX_WAIT)

    def _wait_for_marker(self, pattern: "re.Pattern[str]", timeout: float) -> str:
        """Poll the screen until ``pattern`` appears, then return the snapshot.

        The install is many short commands, so the shell prompt returns between
        them and the generic settle heuristic would call it done after the first.
        """
        deadline = time.time() + timeout
        text = self._safe_snapshot() or ""
        while time.time() < deadline:
            if pattern.search(text):
                break
            time.sleep(self.SETTLE_POLL)
            text = self._safe_snapshot() or text
        return text


    def _cmd_revoke_approval(self) -> None:
        """Handle ``/revoke_approval``: clear a prior 'approve everything' choice.

        After this, every future inject_input tool call prompts for approval
        again (y/n/a), even if the user had previously chosen 'a'.
        """
        panel = self._panel
        if panel is None:
            return
        if self._inject_approval_all:
            self._inject_approval_all = False
            panel.add_system(
                "inject_input approval revoked; future injections will ask again."
            )
        else:
            panel.add_system(
                "inject_input approval is not currently granted; nothing to revoke."
            )


    @staticmethod
    def _parse_helper_init(snapshot: str) -> str:
        """Turn the helper install command's output line into a status message.

        Looks for the ``LUDVART_HELPER_INIT status=... version=... ok=... reason=...``
        line the injected command prints. The ``status`` value is constrained to
        real words so the echoed command template (which contains ``status=%s``)
        is not mistaken for the result.
        """
        m = Ludvart._HELPER_INIT_RE.search(snapshot)
        if m is None:
            return (
                "Could not confirm ludvart_helper install -- no result seen. Make "
                "sure the foreground is an interactive shell, then run "
                "/init_helpers again."
            )
        status, ver, ok, reason = m.groups()
        if ok != "1":
            got = re.search(r"bytes=(\d+) got=(\w+)", snapshot)
            detail = ""
            if got:
                detail = (
                    f" The copy that arrived was {got.group(1)} bytes "
                    f"(md5 {got.group(2)[:8]}), expected "
                    f"{len(LUDVART_HELPER_SOURCE)} bytes "
                    f"(md5 {LUDVART_HELPER_MD5[:8]}) -- the install command was "
                    "corrupted on its way through the terminal. Try again."
                )
            return (
                f"ludvart_helper install FAILED (reason={reason}); the file on disk "
                "does not match the expected checksum." + detail
            )
        if status == "current":
            return f"ludvart_helper is already up to date (v{ver}, checksum verified)."
        if reason == "missing":
            return f"ludvart_helper v{ver} installed (was not present)."
        return (
            f"ludvart_helper v{ver} reinstalled "
            "(previous copy was outdated or modified)."
        )




    # -- /model: multi-model registry ---------------------------------------






    def _model_add_start(self) -> None:
        """Begin the guided ``/model add`` flow (fields typed in the panel)."""
        panel = self._panel
        if panel is None:
            return
        if self._backend_client is None:
            return
        panel.add_system("Add a model (type 'cancel' at any prompt to abort).")
        panel.add_system(SERVICE_PROMPT)
        self._model_add = {"step": "service", "data": {}}
        self._render_split()

    def _model_add_show_provider_menu(self) -> None:
        """Show the endpoint-type menu and advance to the provider step."""
        panel = self._panel
        if panel is None:
            return
        panel.add_system("Select the API endpoint type:")
        for i, (_name, menu_label, _url) in enumerate(PROVIDER_MENU, 1):
            panel.add_system(f"  {i}) {menu_label}")
        panel.add_system(f"Choice [1-{len(PROVIDER_MENU)}]:")
        self._model_add["step"] = "provider"

    def _show_copilot_choices(self, choices: list) -> None:
        """Render the Copilot model pick-list (or a slug prompt if empty)."""
        panel = self._panel
        if panel is None:
            return
        if choices:
            panel.add_system("Models available to your GitHub Copilot account:")
            for i, slug in enumerate(choices, 1):
                panel.add_system(f"  {i}) {slug}")
            panel.add_system(f"Choice [1-{len(choices)}] or type a model slug:")
        else:
            panel.add_system("Copilot model slug (e.g. gpt-4o, claude-opus-4.8):")

    def _feed_model_add(self, line: str) -> None:
        """Advance the guided ``/model add`` flow with one typed answer."""
        panel = self._panel
        if panel is None or self._model_add is None:
            return
        answer = line.strip()
        if answer.lower() == "cancel":
            self._model_add = None
            panel.masked = False
            panel.add_system("Model add cancelled.")
            self._render_split()
            return

        step = self._model_add["step"]
        data = self._model_add["data"]
        if step == "service":
            data["service"] = answer
            self._model_add_show_provider_menu()
        elif step == "provider":
            self._model_add_provider(answer)
        elif step == "url":
            if not answer and not data.get("default_url"):
                panel.add_system("An endpoint URL is required (or 'cancel').")
            else:
                data["api_url"] = answer or data["default_url"]
                self._model_add["step"] = "key"
                panel.masked = True
                panel.add_system("API key (input hidden):")
        elif step == "key":
            if not answer:
                panel.add_system("An API key is required (or 'cancel').")
            else:
                data["api_key"] = answer
                panel.masked = False
                self._model_add["step"] = "model"
                panel.add_system("Model name (e.g. gpt-4o, claude-..., gemini-...):")
        elif step == "model":
            if not answer:
                panel.add_system("A model name is required (or 'cancel').")
            else:
                data["model"] = answer
                self._finish_model_add(
                    {
                        "provider": data["provider"],
                        "service": data.get("service", ""),
                        "api_url": data.get("api_url", ""),
                        "api_key": data.get("api_key", ""),
                        "model": answer,
                        "context_window": 0,
                        "active": False,
                    }
                )
                return
        elif step == "copilot_model":
            choices = data.get("copilot_choices") or []
            slug = answer
            if answer.isdigit() and choices:
                n = int(answer)
                if not (1 <= n <= len(choices)):
                    panel.add_system(
                        f"Please enter a number 1-{len(choices)} or a model slug."
                    )
                    self._render_split()
                    return
                slug = choices[n - 1]
            if not slug:
                panel.add_system("A model slug is required (or 'cancel').")
            else:
                self._finish_model_add(
                    {
                        "provider": "copilot",
                        "service": data.get("service", ""),
                        "api_url": "",
                        "api_key": "",
                        "model": slug,
                        "context_window": 0,
                        "active": False,
                    }
                )
                return
        self._render_split()

    def _model_add_provider(self, answer: str) -> None:
        """Handle the provider-selection step of ``/model add``."""
        panel = self._panel
        if panel is None:
            return
        choice = answer.lower()
        picked = None
        if choice.isdigit() and 1 <= int(choice) <= len(PROVIDER_MENU):
            picked = PROVIDER_MENU[int(choice) - 1]
        else:
            picked = next((p for p in PROVIDER_MENU if p[0] == choice), None)
        if picked is None:
            panel.add_system(f"Please enter a number 1-{len(PROVIDER_MENU)}.")
            return
        provider, _menu_label, default_url = picked
        data = self._model_add["data"]
        data["provider"] = provider
        if provider == "copilot":
            if self._backend_client is not None:
                # The backend host owns Copilot authorization and the gateway,
                # so ask it for the subscription's model list to pick from --
                # same experience as local mode.
                self._model_add["step"] = "copilot_model"
                host = _ClientTerminalHost(self)
                try:
                    reply = self._backend_client.request(
                        "model copilot-models", host
                    )
                except (ConnectionError, OSError):
                    reply = {}
                choices = list(reply.get("copilot_models") or [])
                data["copilot_choices"] = choices
                self._show_copilot_choices(choices)
                return
            from .gateway import (
                copilot_authenticated,
                list_copilot_models,
                litellm_available,
            )

            if not (litellm_available() and copilot_authenticated()):
                self._model_add = None
                panel.add_system(
                    "GitHub Copilot isn't authorized here. Run `ludvart` in a "
                    "terminal and add the Copilot model via the setup wizard first."
                )
                return
            self._model_add["step"] = "copilot_model"
            choices = list_copilot_models()
            data["copilot_choices"] = choices
            self._show_copilot_choices(choices)
            return
        data["default_url"] = default_url
        self._model_add["step"] = "url"
        prompt = (
            f"Endpoint URL [{default_url}]:" if default_url else "Endpoint URL:"
        )
        panel.add_system(prompt)

    def _finish_model_add(self, reg: dict) -> None:
        """Verify and register the collected model on the panel spinner."""
        panel = self._panel
        self._model_add = None
        if panel is not None:
            panel.masked = False
        if self._backend_client is not None:
            # The registry lives on the backend: send the finished registration
            # over the wire to be verified and stored there.
            self._forward_command_to_backend("model add", payload=reg)


    def _start_ask(
        self,
        question: str,
        *,
        user_echo: str | None = None,
        info: str | None = None,
        root_question: str | None = None,
    ) -> None:
        """Kick off an agent turn on a background thread.

        ``user_echo`` is shown as the user's line in the transcript (typed
        questions); ``info`` shows a dim status note (auto-initiated turns). The
        ``question`` is what the model actually receives.
        """
        panel = self._panel
        if panel is None or panel.thinking:
            return
        if info:
            panel.add_info(info)
        if user_echo:
            panel.add_user(user_echo)
        panel.thinking = True
        panel.activity = "Thinking"
        panel.tick = 0
        self._deliver = self._deliver_reply
        self._llm_request_in_flight = True
        self._ask_root_question = (
            root_question if root_question is not None else question
        )
        self._render_split()  # show the question and the spinner immediately

        ask = self._ai_ask
        self._ask_cancel = threading.Event()
        self._ask_done = threading.Event()

        def worker() -> None:
            try:
                result = ask(question)
            except _AskCancelled:
                result = ""  # user abandoned the request; result is dropped
            except Exception as exc:  # surfaced to the user, never crashes ludvart
                result = f"[ludvart] request failed: {exc}"
            self._ask_result = result
            self._ask_done.set()

        self._ask_thread = threading.Thread(target=worker, daemon=True)
        self._ask_thread.start()

    def _start_action(self, worker, *, info: str | None = None,
                      activity: str = "Working") -> None:
        """Run a deterministic background job (no LLM) with the panel spinner.

        ``worker`` runs on a daemon thread and returns a status string that is
        shown as an ephemeral "system" line (not persisted, not part of the LLM
        conversation). Used for harness-driven actions such as ``/init_helpers``.
        """
        panel = self._panel
        if panel is None or panel.thinking:
            return
        if info:
            panel.add_info(info)
        panel.thinking = True
        panel.activity = activity
        panel.tick = 0
        self._deliver = self._deliver_system
        self._render_split()

        self._ask_done = threading.Event()

        def run() -> None:
            try:
                result = worker()
            except Exception as exc:  # surfaced to the user, never crashes ludvart
                result = f"[ludvart] action failed: {exc}"
            self._ask_result = result
            self._ask_done.set()

        self._ask_thread = threading.Thread(target=run, daemon=True)
        self._ask_thread.start()

    def _deliver_reply(self, result: str) -> None:
        """Deliver a completed reply into the panel (the backend persists it)."""
        panel = self._panel
        if panel is None:
            return
        panel.add_reply(result)

    def _deliver_system(self, result: str) -> None:
        """Deliver a deterministic action's status as an ephemeral system line."""
        panel = self._panel
        if panel is None or not result:
            return
        panel.add_system(result)




    def _finish_ask(self) -> None:
        """Deliver the completed background result into the panel."""
        if self._ask_thread is not None:
            self._ask_thread.join(timeout=1)
            self._ask_thread = None
        panel = self._panel
        if panel is None:
            return
        self._end_wait()
        # The interrupted worker has now completed its history rollback, so it
        # is safe to start the queued replacement turn.
        if self._steer_pending is not None:
            pending, echo = self._steer_pending, self._steer_user_echo
            self._steer_pending = None
            self._steer_user_echo = None
            self._confirm_close = False
            panel.confirm_prompt = ""
            panel.thinking = False
            panel.interim = ""
            # A turn that answered before it noticed the interruption is a
            # finished turn, and the backend has already persisted that answer.
            # Dropping it here would leave the panel showing a conversation the
            # model does not have.
            if self._ask_result:
                self._deliver_reply(self._ask_result)
                self._ask_result = ""
            self._start_ask(
                pending,
                user_echo=echo,
                root_question=self._ask_root_question,
            )
            self._render_split()
            return
        # A request that finished on its own resolves any pending close prompt.
        self._confirm_close = False
        panel.confirm_prompt = ""
        self._llm_request_in_flight = False
        # A natural completion while the user is entering a steer instruction
        # wins; discard the half-typed steer text and restore the input draft.
        if self._steer_input:
            self._exit_steer_input(restore_draft=True)
        # If it was cancelled, drop the result silently (the panel is closing).
        if self._ask_cancel.is_set():
            panel.thinking = False
            panel.interim = ""
            return
        panel.thinking = False
        panel.interim = ""
        deliver = self._deliver or self._deliver_reply
        deliver(self._ask_result)
        self._render_split()


    #: Breadcrumb that replaces the screen snapshot of superseded user turns in


    def _perf_add(self, op: str, seconds: float) -> None:
        """Record one timing sample (seconds) for operation type ``op``."""
        self._perf.setdefault(op, []).append(seconds)

    @contextlib.contextmanager
    def _perf_timer(self, op: str):
        """Time the wrapped block and record it under operation type ``op``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self._perf_add(op, time.perf_counter() - start)

    def _tool_inject_input(self, args: dict) -> str:
        """Inject keystrokes into the child PTY (ludvart performs the tool call).

        Control keys cannot survive as raw bytes in the model's JSON tool
        arguments, so ``text`` is decoded for backslash escapes by default
        (``\\xHH``, ``\\cX``, ``\\e``, ``\\t``, ``\\r``, ``\\n``, ``\\\\``) --
        letting the model page down in vim with ``\\x06`` etc. Pass
        ``interpret_escapes=false`` to send the text verbatim.

        After injecting, the command's output is not available immediately and we
        cannot know when it finishes. ludvart learns the prompt from the cursor
        line captured just before injection and watches the screen model: when
        that prompt returns (any shell/REPL), or output goes quiet, the input is
        settled. Only an ambiguous quiet screen with no recognizable prompt
        falls back to a one-off out-of-band LLM ``status check`` (never part of
        the conversation history). The tool result then returns the up-to-date
        screen snapshot so the main conversation continues with what the
        injected input actually produced.
        """
        text = args.get("text", "")
        if not isinstance(text, str):
            return "[ludvart] inject_input: 'text' must be a string."
        if not self._await_inject_approval(text):
            return "[ludvart] inject_input declined by user approval gate."
        if args.get("interpret_escapes", True):
            data = self._decode_escapes(text)
        else:
            data = text.encode("utf-8", "replace")
        if args.get("submit"):
            data += b"\r"
        if not data:
            return "[ludvart] inject_input: nothing to inject (empty 'text')."
        prompt_prefix = self._injection_prompt_prefix(data.endswith((b"\r", b"\n")))
        try:
            self._write_all(self._master_fd, data)
        except OSError as exc:
            return f"[ludvart] inject_input failed: {exc}"
        snapshot = self._wait_for_injection_to_settle(text, prompt_prefix)
        return (
            f"Injected {len(data)} byte(s) into the terminal. The input was sent "
            "to the foreground program and its output has settled. This is the "
            "terminal screen now:\n"
            "<screenContext>\n"
            f"{snapshot}\n"
            "</screenContext>"
        )

    @staticmethod
    def _decode_escapes(text: str) -> bytes:
        """Decode C-style backslash escapes in ``text`` into raw bytes.

        Supports ``\\n \\r \\t \\e \\a \\b \\f \\v \\\\ \\' \\"``, ``\\xHH`` (1-2
        hex digits), ``\\ooo`` (1-3 octal digits), and ``\\cX`` (control key, e.g.
        ``\\cf`` -> Ctrl-F). Unknown escapes and a trailing backslash are kept
        literally. Non-escaped characters are encoded as UTF-8.
        """
        simple = {
            "n": 0x0A, "r": 0x0D, "t": 0x09, "e": 0x1B, "a": 0x07,
            "b": 0x08, "f": 0x0C, "v": 0x0B, "\\": 0x5C, "'": 0x27, '"': 0x22,
        }
        out = bytearray()
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch != "\\":
                out += ch.encode("utf-8", "replace")
                i += 1
                continue
            if i + 1 >= n:
                out += b"\\"  # trailing backslash kept literal
                break
            nxt = text[i + 1]
            if nxt in simple:
                out.append(simple[nxt])
                i += 2
            elif nxt in "xX":
                digits = ""
                j = i + 2
                while j < n and len(digits) < 2 and text[j] in "0123456789abcdefABCDEF":
                    digits += text[j]
                    j += 1
                if digits:
                    out.append(int(digits, 16))
                    i = j
                else:
                    out += b"\\x"  # malformed -> keep literal
                    i += 2
            elif nxt in "cC":
                if i + 2 < n:
                    out.append(ord(text[i + 2].upper()) ^ 0x40)
                    i += 3
                else:
                    out += b"\\c"
                    i += 2
            elif nxt in "01234567":
                digits = ""
                j = i + 1
                while j < n and len(digits) < 3 and text[j] in "01234567":
                    digits += text[j]
                    j += 1
                out.append(int(digits, 8) & 0xFF)
                i = j
            else:
                out += ("\\" + nxt).encode("utf-8", "replace")
                i += 2
        return bytes(out)

    def _current_prompt_prefix(self) -> str:
        """Learn the current prompt: the cursor line up to the cursor column.

        Captured just before injecting, this is exactly the prompt string the
        shell/REPL is showing (nothing has been typed yet), with no dependence
        on hardcoded ``$``/``#`` markers -- so it generalizes across shells and
        interactive programs. Returns ``""`` if it cannot be read.
        """
        try:
            row = self.screen.display[self.screen.cursor.y]
            return row[: self.screen.cursor.x]
        except Exception:
            return ""

    def _injection_prompt_prefix(self, submits: bool) -> str:
        """The prompt to watch for once this injection has been executed.

        A command line too long for one call is typed by several injections, and
        only the last of them submits. By the time that one arrives the cursor
        line is the prompt plus a half-typed command, so a prefix learned then
        could never match the bare prompt that comes back when the command
        finishes -- leaving the cheap fast path dead for exactly the longest
        commands, which are also the ones worth not waiting on. Hold on to the
        prefix learned before the first chunk and drop it once a line has been
        submitted.

        A sequence abandoned half-way (the user hits Ctrl-C) leaves a prefix that
        is simply the earlier prompt, which normally still matches; when it does
        not, the settle wait falls back to quiescence exactly as it did before.
        """
        if self._partial_line_prompt is None:
            self._partial_line_prompt = self._current_prompt_prefix()
        prefix = self._partial_line_prompt
        if submits:
            self._partial_line_prompt = None
        return prefix

    def _prompt_returned(self, prompt_prefix: str) -> bool:
        """True when the learned prompt is back with nothing typed after it."""
        plen = len(prompt_prefix)
        if not plen:
            return False
        try:
            if self.screen.cursor.x != plen:
                return False
            return self.screen.display[self.screen.cursor.y][:plen] == prompt_prefix
        except Exception:
            return False

    def _wait_for_injection_to_settle(
        self, injected: str, prompt_prefix: str = ""
    ) -> str:
        """Poll the screen model until the injected input looks finished.

        Fast path: as soon as the learned prompt returns (a shell/REPL is ready
        for the next command), we are done -- no LLM call. Fallback: if the
        screen instead goes quiet with no recognizable prompt, confirm once with
        the out-of-band LLM status check, backing off (widening the quiet
        window) if it is actually still running. The main split loop feeds the
        PTY into the model on its own thread while this worker-thread method
        sleeps between polls, so each snapshot reflects the latest output.
        """
        # A full-screen (alternate-buffer) app -- screen/tmux/vim/less/htop --
        # has no learnable shell prompt, so the prompt-return fast path can never
        # fire, and its status line/clock can keep repainting so the quiescence
        # fallback (and the LLM check) would burn the whole timeout. Detect that
        # case up front and use a short quiet window with a low overall cap, so an
        # injected keystroke (e.g. a screen "Ctrl-a n") returns promptly instead
        # of appearing to hang. Re-checked each poll because the app may enter or
        # leave the alternate buffer as a result of the injected input.
        tui = bool(getattr(self.screen, "in_alt_screen", False))
        quiet_window = (
            self.SETTLE_TUI_QUIET_WINDOW if tui else self.SETTLE_QUIET_WINDOW
        )
        max_wait = self.SETTLE_TUI_MAX_WAIT if tui else self.SETTLE_MAX_WAIT
        deadline = time.time() + max_wait
        last_text = self._safe_snapshot() or ""
        # The screen exactly as it was just before the input was injected. Passed
        # to the LLM status check so it can compare before -> after and judge
        # whether the injection actually took effect (not just whether the screen
        # looks idle right now).
        before_text = last_text
        last_change = time.time()
        changed_once = False
        while time.time() < deadline:
            time.sleep(self.SETTLE_POLL)
            text = self._safe_snapshot()
            if text is None:
                continue  # transient read during a concurrent feed; retry
            now = time.time()
            if text != last_text:
                last_text = text
                last_change = now
                changed_once = True
            # A full-screen app entered/left since the last poll -> re-derive the
            # timing so we do not wait a shell-length window on a TUI (or vice
            # versa), and shrink the deadline when switching into TUI mode.
            now_tui = bool(getattr(self.screen, "in_alt_screen", False))
            if now_tui != tui:
                tui = now_tui
                quiet_window = (
                    self.SETTLE_TUI_QUIET_WINDOW
                    if tui
                    else self.SETTLE_QUIET_WINDOW
                )
                if tui:
                    deadline = min(
                        deadline, now + self.SETTLE_TUI_MAX_WAIT
                    )
            # Fast path: the learned prompt is back -> command finished. Only
            # meaningful outside a full-screen app (a TUI has no shell prompt).
            if changed_once and not tui and self._prompt_returned(prompt_prefix):
                return text
            # Quiescence fallback. In a TUI we trust a short unchanged window
            # directly (no shell prompt to match, no LLM round-trip). Otherwise we
            # are patient and confirm once with the LLM so we do not misjudge a
            # pause in a long-running command.
            if changed_once and (now - last_change) >= quiet_window:
                if tui or self._backend_client is None:
                    return text
                # An LLM round-trip cannot be called back once sent, so starting
                # one with no room left is what turns the cap into a suggestion.
                # Out of budget means out of time: report what is on screen.
                if now + self.SETTLE_CHECK_RESERVE > deadline:
                    return text
                if self._injection_finished(injected, text, before_text):
                    return text
                last_change = now  # really still running; back off
                quiet_window = min(quiet_window * 2, 2.0)
        return last_text

    def _safe_snapshot(self) -> str | None:
        """Snapshot the screen, returning ``None`` on a transient read error."""
        try:
            return self.snapshot_text()
        except Exception:
            return None

    def _injection_finished(
        self, injected: str, screen_text: str, before_text: str = ""
    ) -> bool:
        """Out-of-band status check: did the injected input take effect / finish?

        The LLM is shown three things: the screen exactly BEFORE the input was
        injected, the injected input itself, and the screen AFTER. Comparing
        before -> after lets it judge whether the injection actually landed and
        completed, rather than only guessing from whether the current screen
        looks idle (which is ambiguous for a full-screen app that always looks
        "busy", or a command whose output happens to resemble a prompt).

        This is a standalone model call that is deliberately NOT added to the
        conversation history -- it only decides whether to keep waiting. The
        model lives on the backend, so it is borrowed for one round-trip (this
        runs while the backend is blocked serving ``inject_input``). On any
        error (or with no backend) it reports finished so the tool never hangs.
        """
        if self._backend_client is None:
            return True
        system = {
            "role": "system",
            "content": (
                "You monitor a terminal. Some keystrokes/command were just "
                "injected into it. You are given the screen BEFORE the "
                "injection, the injected input, and the screen AFTER. By "
                "comparing before to after, decide whether that input has "
                "FINISHED taking effect (the change it triggered is complete and "
                "the terminal is now idle -- a shell prompt waits for the next "
                "command, or a full-screen app has finished redrawing and is "
                "waiting for input) or is STILL RUNNING (output is still being "
                "produced, a long-running command has not returned, the screen "
                "is mid-redraw, or the injected input has not visibly taken "
                "effect yet). Reply with exactly one word: DONE or RUNNING."
            ),
        }
        user = {
            "role": "user",
            "content": (
                f"Injected input (repr): {injected!r}\n\n"
                "Terminal screen BEFORE the injection:\n"
                "--- BEGIN BEFORE ---\n"
                f"{before_text}\n"
                "--- END BEFORE ---\n\n"
                "Terminal screen AFTER (current):\n"
                "--- BEGIN AFTER ---\n"
                f"{screen_text}\n"
                "--- END AFTER ---\n\n"
                "Comparing before to after, has the injected input finished "
                "taking effect? Answer DONE or RUNNING."
            ),
        }
        try:
            reply = self._backend_client.backend_request(
                "complete",
                {"messages": [system, user], "max_tokens": 8, "max_retries": 0},
            )
        except Exception:
            return True  # never hang the tool on a status-check failure
        if not isinstance(reply, str):
            return True
        verdict = reply.strip().upper()
        return "RUNNING" not in verdict

    def _tool_capture_screen_history(self, args: dict) -> str:
        """Return a slice of the scrollback history for the model.

        The history is the full logical output (everything that scrolled off
        the top, followed by the current viewport). ``offset`` is a line count
        from the current position (the end of the buffer) and is expected to be
        negative to look upward; ``length`` is how many lines to return.
        """
        try:
            offset = int(args.get("offset"))
            length = int(args.get("length"))
        except (TypeError, ValueError):
            return (
                "[ludvart] capture_screen_history: 'offset' and 'length' must be "
                "integers."
            )
        if length <= 0:
            return (
                "[ludvart] capture_screen_history: 'length' must be a positive "
                "integer."
            )
        # Read the full logical history; retry briefly in case the main thread
        # is mutating the screen model concurrently.
        full: list[str] | None = None
        for _ in range(3):
            try:
                full = self.screen.full_text(include_scrollback=True)
                break
            except Exception:
                time.sleep(0.02)
        if full is None:
            return (
                "[ludvart] capture_screen_history: could not read the screen "
                "history, please try again."
            )
        total = len(full)
        start = max(0, min(total + offset, total))
        end = max(start, min(total, start + length))
        lines = full[start:end]
        if not lines:
            return (
                "[ludvart] capture_screen_history: the requested range is empty "
                f"(offset={offset}, length={length}). The history currently has "
                f"{total} line(s); use a negative offset no smaller than "
                f"-{total}."
            )
        body = "\n".join(lines)
        return (
            f"Screen history: {len(lines)} line(s) starting {total - start} "
            f"line(s) above the current position ({total} line(s) available in "
            "total):\n"
            "<screenHistory>\n"
            f"{body}\n"
            "</screenHistory>"
        )

    # Screen/tmux "set window title" sequences: ESC k <text> (ST | BEL).
    # ST is ESC \ or the single-byte 0x9c; some emitters use BEL (0x07).
    _TITLE_SEQ = re.compile(rb"\x1bk[^\x1b\x07\x9c]*(?:\x1b\\|\x07|\x9c)")

    def _feed_model(self, data: bytes) -> None:
        """Feed child output to the pyte model, stripping screen/tmux title
        sequences that pyte does not understand (it would otherwise print the
        title text into the model, corrupting our snapshots). The verbatim
        passthrough to the real terminal is unaffected, so the actual tab
        title still updates."""
        buf = self._title_carry + data
        self._title_carry = b""
        buf = self._TITLE_SEQ.sub(b"", buf)
        # Hold back an unterminated title sequence (ESC k with no ST/BEL yet)
        # so its partial payload never reaches the model; feed it once the
        # terminator arrives in a later read. Cap the carry so a malformed
        # stream cannot grow it without bound.
        idx = buf.rfind(b"\x1bk")
        if idx != -1 and not re.search(rb"\x1b\\|\x07|\x9c", buf[idx:]):
            if len(buf) - idx <= 4096:
                self._title_carry = buf[idx:]
                buf = buf[:idx]
        if buf:
            self.stream.feed(buf)

    def _read(self, fd: int) -> bytes | None:
        """Read from ``fd``. Return ``None`` on EOF/child-gone, else bytes."""
        try:
            data = os.read(fd, self.READ_SIZE)
        except OSError as exc:
            # On Linux, reading the master after the child exits raises EIO.
            if exc.errno == errno.EIO:
                return None
            if exc.errno == errno.EAGAIN:
                return b""
            raise
        if not data:
            return None
        if fd == self._master_fd:
            self._capture(data)
        return data

    def _write_all(self, fd: int, data: bytes) -> None:
        """Write all of ``data`` to ``fd``, handling short writes."""
        while data:
            try:
                n = os.write(fd, data)
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    continue
                raise
            data = data[n:]

    def _capture(self, data: bytes = b"", marker: bytes | None = None) -> None:
        """Append raw child output (or an event ``marker``) to the capture file.

        No-op unless ``LUDVART_CAPTURE`` was set. Markers are wrapped so they are
        visibly distinct from real child bytes when the file is inspected.
        """
        if self._capture_fd is None:
            return
        try:
            if marker is not None:
                os.write(self._capture_fd, b"\n<<ludvart:" + marker + b">>\n")
            else:
                os.write(self._capture_fd, data)
        except OSError:
            pass

    def _reap_child(self) -> int:
        """Wait for the child and translate its status into an exit code."""
        try:
            _, status = os.waitpid(self._child_pid, 0)
        except OSError:
            return 0
        if os.WIFEXITED(status):
            return os.WEXITSTATUS(status)
        if os.WIFSIGNALED(status):
            return 128 + os.WTERMSIG(status)
        return 0
