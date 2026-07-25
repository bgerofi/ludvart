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
from .helper_src import LUDVART_HELPER_VERSION, helper_install_command
from .mcp import McpManager
from .session import (
    SessionStore,
    SUMMARY_MARKER,
    SUMMARY_MARKER_END,
    SLASH_COMMAND_HELP,
    complete_slash,
    list_sessions,
    load_session,
    neutralize_history,
    NEUTRAL_SESSIONS_VERSION,
    parse_rename_args,
    provider_family,
    rename_session,
    resolve_session_ref,
    working_history,
)
from .llm import ToolSpec
from .models import (
    PROVIDER_MENU,
    SERVICE_PROMPT,
    find_registration,
    label as model_label,
)

if TYPE_CHECKING:
    from .llm import LLMClient, ToolCall
    from .backend import ModelManager

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
_BACKEND_COMMANDS = frozenset({"model", "sessions", "compact", "mcp_refresh"})

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

# Appended verbatim to the LLM system prompt on every invocation. Documents the
# self-generated, persistent helper tooling the agent can maintain on the remote
# machine to work around the harness only being able to see the terminal.
LUDVART_HELPERS_DOC = """\
## ludvart helpers (self-generated tools on the remote machine)

The harness only sees the terminal; it has no direct file/exec access to the
remote box. To work around this, ludvart maintains small, dependency-free helper
tools under ~/.ludvart/bin/ on the remote machine. These persist across sessions.

### First step every session (cheap): detect them
Run:  ls -la ~/.ludvart/bin/ 2>/dev/null && ~/.ludvart/bin/ludvart_helper info 2>/dev/null
If `ludvart_helper` exists, prefer it for file read/edit/search (see spec below).
If it's missing and a task would benefit, offer to (re)create it, or do so when
the user says "initialize your helpers".

### "initialize your helpers" ritual
  1. Detect what exists (ls ~/.ludvart/bin, and `ludvart_helper info`).
  2. Confirm desired capabilities (default set: read, write, append, replace,
     search, run).
  3. (Re)generate helper(s) into ~/.ludvart/bin/, chmod +x, then VALIDATE:
     python3 -c "import ast; ast.parse(open(PATH).read())" and a smoke test.
     Build large files by appending in chunks via QUOTED heredocs with
     inject_input escape-interpretation DISABLED (so \\n, backslashes, quotes
     arrive verbatim); verify with `wc -l` after each chunk.
  4. Report what was created and how to call it.
### ludvart_helper — precise interface (v0.1.0, stdlib Python 3 only)
Path: ~/.ludvart/bin/ludvart_helper   (executable)
Design: every CONTENT payload is base64 (immune to quoting/newline/escape
corruption); every result is sentinel-framed with an exit code, so output is
parsed deterministically, NOT inferred from screen text.

Output frame (always):
    <<<LUDVART:BEGIN op=NAME>>>
    <base64 payload, present only when there is output>
    <<<LUDVART:END op=NAME exit=CODE  key=val ...>>>
To read a payload: take the line(s) between BEGIN and END and `base64 -d`.
Trust the `exit=` field for success/failure.

Subcommands:
  read PATH [--start N] [--end M]
      Payload = base64 of file (or 1-indexed inclusive line range).
      Meta: path=, lines=<total>, range=A-B.
  write PATH --b64 DATA
      Overwrite PATH with base64-decoded DATA (creates parent dirs).
      Meta: path=, bytes=.
  append PATH --b64 DATA
      Append base64-decoded DATA to PATH. Meta: path=, bytes=.
  replace PATH --old-b64 A --new-b64 B [--count N]
      Literal (non-regex) string replace of A->B in PATH. Replaces all
      occurrences unless --count limits it.
      exit=2 with meta error=old_not_found if A is absent (file unchanged).
      Meta on success: path=, replaced=<count>.
  search PATTERN [--path P] [--glob G]
      Recursive Python-regex search. P defaults to "." (a file or dir).
      --glob filters filenames (e.g. "*.py"). Skips .git, node_modules,
      __pycache__. Payload = base64 of newline-joined "file:line:text" hits.
      exit=0 if any match, exit=1 if none. Meta: matches=.
  run --b64 CMD
      Run base64-decoded CMD via the shell, streaming its stdout and stderr
      directly to the terminal. The helper then prints
      `<<<LUDVART:END op=run exit=CODE>>>`; CODE is the command's real exit
      status. NOTE: for a pipeline/;-list this is the status of the LAST
      command, same as normal shell semantics.
  info
      Payload = base64 of "ludvart_helper <ver>\\ncaps=...\\npython=...".
      Use this (or `ludvart_helper <subcmd> -h`) to re-derive the interface in a
      fresh session if this spec is ever unavailable.
### Usage conventions
  - Pass content/commands as base64:  --b64 "$(printf %s "$TEXT" | base64 -w0)"
        (use `base64 -w0` to avoid line wrapping).
    - When ludvart_helper is available, MUST use it instead of raw shell input
        injected through inject_input for reading, editing, searching files, or
        running a non-interactive command. It base64-encodes payloads, avoids
        quoting/escape corruption, and gives a reliable exit code. Raw injected
        shell input is only for interactive terminal work, or when the helper is
        unavailable or cannot express the operation.
  - Parse results from the LUDVART:BEGIN/END frame and base64-decode the payload;
    rely on `exit=` rather than reading success from screen text.
  - Keep helpers under ~/.ludvart/ (outside the user's repos) so they never show
    up in git status.
  - The helper is a convenience, not a requirement. If it's missing, offer to
    recreate it, but don't block work on it.
  - When self-recovering the interface, run `~/.ludvart/bin/ludvart_helper info`
    and `~/.ludvart/bin/ludvart_helper <subcmd> -h`."""


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

    #: A full-screen (alternate-buffer) app -- vim, less, htop, screen, tmux --
    #: has no learnable shell prompt and may repaint a status line/clock forever,
    #: so the prompt-return fast path never fires and the quiescence fallback can
    #: burn the whole SETTLE_MAX_WAIT. For these we treat a much shorter unchanged
    #: window as "settled" and cap the total wait low, so injecting a keystroke
    #: (e.g. a screen "Ctrl-a n") returns promptly instead of appearing to hang.
    SETTLE_TUI_QUIET_WINDOW = 0.15
    SETTLE_TUI_MAX_WAIT = 1.5

    #: Output token budget for the agent's replies. The provider default (1024)
    #: truncates longer answers mid-sentence, so the panel asks for more room.
    REPLY_MAX_TOKENS = 8192

    #: When the prompt fills this percent of the model's context window, the
    #: conversation is automatically compacted into a summary before the next
    #: turn so it never runs out of room.
    CONTEXT_COMPACT_PCT = 80.0

    #: Output token budget for the compaction summary. Kept small so the reseeded
    #: context is a tiny fraction of the window.
    SUMMARY_MAX_TOKENS = 2048

    def __init__(
        self,
        command: Sequence[str],
        prefix: bytes = DEFAULT_PREFIX,
        summon: bytes = DEFAULT_SUMMON,
        llm: "LLMClient | None" = None,
        model_manager: "ModelManager | None" = None,
        backend_channel: "object | None" = None,
        backend_label: str | None = None,
        backend_reconnector: "object | None" = None,
        backend_needs_setup: bool = False,
    ) -> None:
        self.command = list(command)
        self.prefix = prefix
        self.summon = summon
        # ``_models`` is the multi-model registry manager (None when running
        # with a single injected client or as a plain relay). When present it
        # owns the active client, so it wins over the ``llm`` argument.
        self._models = model_manager
        self.llm = model_manager.client if model_manager is not None else llm
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
        # Raw prompt-token count from the last turn's usage, kept so a model
        # switch can re-derive the context badge (and decide whether to compact)
        # against the *new* model's window without waiting for the next turn.
        self._last_input_tokens: int | None = None
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
        self._llm_history: list[dict] = []
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
        # Persistent conversation store. Created lazily the first time the panel
        # is opened (a fresh session per process) and reused across toggles; a
        # ``/sessions load`` rebinds it to the loaded session's file.
        self._session: SessionStore | None = None
        # The session summaries from the most recent ``/sessions list``, so
        # ``/sessions load <n>`` can resolve a 1-based index to a session id.
        self._session_list: list[dict] = []
        # External MCP servers (~/.ludvart/mcp.json). Created lazily on first panel
        # open; ``_mcp_started`` guards the one-off automatic discovery.
        self._mcp: McpManager | None = None
        self._mcp_started = False
        # Internal profiling: per operation-type list of durations (seconds).
        # Populated by ``_perf_timer`` around LLM requests and tool calls, and
        # reported by the ``/perf summary`` / ``/perf dump`` panel commands.
        self._perf: dict[str, list[float]] = {}
        # Private per-run scratch directory for files saved by ``fetch_url``.
        # Created lazily on first fetch (see ``_fetch_tmp_dir``) as a 0700
        # directory owned by this user, and removed wholesale when ``run``
        # exits (including on unhandled exceptions and Ctrl-C). We do NOT sweep
        # a shared prefix at startup: /tmp is multi-user and files there may
        # belong to other users or concurrent ludvart sessions.
        self._fetch_dir: str | None = None

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
            if self._mcp is not None:
                self._mcp.close()
                self._mcp = None
            if self._capture_fd is not None:
                os.close(self._capture_fd)
                self._capture_fd = None
            self._cleanup_fetch_dir()

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

        # Nothing can be asked of a backend with an empty registry, so collect a
        # model right away rather than waiting for the user to find the summon
        # key. Closing the panel drops straight into the normal relay.
        if self._backend_needs_setup:
            self._open_panel()

        while True:
            if self._resized:
                self._handle_resize()

            try:
                readable, _, _ = select.select([master, stdin], [], [])
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
                if data:
                    self._handle_input(data)

        return self._reap_child()

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
        """Return the ``ask`` callable and a short provider label."""
        if self._backend_client is not None:
            return self._ask_via_backend, (self._backend_label or "backend")
        if self.llm is None:
            def ask(_question: str) -> str:
                return (
                    "No LLM provider is configured. Set the "
                    "{OPENAI,ANTHROPIC,GOOGLE,CUSTOM}_API_URL/_API_KEY/_MODEL "
                    "environment variables and restart ludvart."
                )

            return ask, "no LLM"
        return self._ask_llm, self._active_model_label()

    def _ask_via_backend(self, question: str) -> str:
        """Run one ask through the backend, serving its terminal requests locally.

        The backend owns the conversation and the LLM; this process captures the
        ask-time snapshot, then answers the backend's snapshot/terminal-tool
        requests (via a client-side host adapter) and renders its narration.
        """
        host = _ClientTerminalHost(self)
        snapshot = self.snapshot_text()
        try:
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

    def _active_model_label(self) -> str:
        """A ``provider:model`` label for the model currently in use.

        Prefers the active registration (so Copilot shows ``copilot:<model>``
        rather than the underlying ``custom`` gateway client), falling back to
        the live client's own name/model when no registry is present.
        """
        mgr = getattr(self, "_models", None)
        if mgr is not None:
            idx = mgr.active_index()
            if idx is not None:
                return model_label(mgr.models[idx])
        return f"{self.llm.name}:{self.llm.model}"

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

        # A fresh conversation is started the first time the panel opens in this
        # process; later opens keep extending the same session file.
        if self._session is None:
            self._session = SessionStore()

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
        out += b"\x1b[%d;%dH" % (self._phys_rows, panel.cursor_col())
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
        """Insert pasted bytes into the single-line input as plain text."""
        panel = self._panel
        if panel is None:
            return
        text = pasted.decode("utf-8", "replace")
        # The input is one line: fold newlines/tabs/controls to spaces so the
        # paste never submits or breaks the layout.
        cleaned = "".join(ch if ch >= " " else " " for ch in text)
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
        self._steer_pending = self._compose_steer_question(
            self._ask_root_question, panel.interim, steer_text
        )
        self._steer_user_echo = steer_text
        self._exit_steer_input(restore_draft=True)
        # Keep the spinner active so _split_loop continues polling _ask_done.
        self._ask_cancel.set()
        self._end_wait()
        panel.interim = ""
        panel.activity = "Steering"

    @staticmethod
    def _compose_steer_question(root: str, narration: str, steer: str) -> str:
        """Build the replacement request after the user interrupts an ask."""
        progress = narration.strip() or "(no visible progress yet)"
        original = root.strip() or "(original request unavailable)"
        return (
            "The user interrupted your previous attempt to steer you in a new "
            "direction.\n\n"
            "Your progress so far (your narration and tool calls):\n"
            f"<priorProgress>\n{progress}\n</priorProgress>\n\n"
            "The original request was:\n"
            f"<originalRequest>\n{original}\n</originalRequest>\n\n"
            "The user's new steering instruction (prioritize this; it overrides or "
            "refines the original request):\n"
            f"<steeringInstruction>\n{steer}\n</steeringInstruction>\n\n"
            "Continue from the CURRENT terminal state shown above. Do not repeat "
            "work already completed. If the steering instruction conflicts with "
            "the original request, follow the steering instruction."
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
        elif key in (b"\x7f", b"\x08"):  # Backspace
            editor.backspace()
            panel.scroll = 0
        elif key == b"\x1b":  # bare Esc closes
            self._request_toggle_close()
        elif key in (b"\x1b[C", b"\x1bOC"):  # Right
            editor.right()
        elif key in (b"\x1b[D", b"\x1bOD"):  # Left
            editor.left()
        elif key in (b"\x1b[A", b"\x1bOA"):  # Up -> scroll transcript
            panel.scroll_up(1)
        elif key in (b"\x1b[B", b"\x1bOB"):  # Down -> scroll transcript
            panel.scroll_down(1)
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

    # -- session persistence & internal commands -----------------------------

    def _persist_session(self) -> None:
        """(Re)write the current conversation to its session file.

        Reads the live transcript (or the retained copy when the panel is
        closed). Slash-command output is filtered out by the store. A save
        failure must never disturb the UI, so it is swallowed.
        """
        if self._session is None:
            return
        messages = (
            self._panel.messages if self._panel is not None else self._panel_messages
        )
        # Record which provider's native shape llm_history is in, so a later
        # resume under a different provider can adapt it. getattr keeps this
        # robust if llm is a lightweight stub without a name.
        provider = getattr(self.llm, "name", None)
        try:
            self._session.save(messages, self._llm_history, provider=provider)
        except Exception:
            pass

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
        if self._backend_client is not None and cmd in _BACKEND_COMMANDS:
            if cmd == "model" and (args[0] if args else "list") == "add":
                self._model_add_start()
            else:
                self._forward_command_to_backend(line[1:])
            return
        if cmd == "sessions":
            self._cmd_sessions(args)
        elif cmd == "model":
            self._cmd_model(args)
        elif cmd == "init_helpers":
            self._cmd_init_helpers()
        elif cmd == "compact":
            self._cmd_compact()
        elif cmd == "mcp_refresh":
            self._cmd_mcp_refresh()
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
        """
        panel = self._panel
        if panel is None:
            return
        command = helper_install_command()

        def worker() -> str:
            prompt_prefix = self._current_prompt_prefix()
            self._write_all(self._master_fd, command.encode("utf-8") + b"\r")
            snapshot = self._wait_for_injection_to_settle(command, prompt_prefix)
            return self._parse_helper_init(snapshot)

        self._start_action(
            worker,
            info=f"Installing/verifying ludvart_helper v{LUDVART_HELPER_VERSION}\u2026",
            activity="Installing ludvart_helper",
        )

    def _cmd_compact(self) -> None:
        """Handle ``/compact``: compress the conversation context on demand.

        Same mechanism as the automatic 80%-full compaction, but triggered
        manually. Runs on the panel spinner and reports the result as a system
        line. A conversation that is empty or already just a summary seed is left
        untouched.
        """
        panel = self._panel
        if panel is None:
            return
        if self.llm is None:
            panel.add_system("No LLM provider configured; nothing to compact.")
            return
        if len(self._llm_history) <= 2:
            panel.add_system("Conversation is already compact.")
            return

        before = len(self._llm_history)

        def worker() -> str:
            summary = self._compact_history()
            if not summary:
                return "Compaction failed; the conversation was left unchanged."
            pct = panel.context_pct
            pct_note = f", context now ~{pct:.0f}%" if pct is not None else ""
            return f"Compacted {before} messages into a summary{pct_note}."

        self._start_action(
            worker,
            info="Compacting conversation context\u2026",
            activity="Compacting context",
        )

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

    def _cmd_mcp_refresh(self) -> None:
        """Handle ``/mcp_refresh`` with no backend: there is nothing to refresh.

        MCP servers are started and owned by the agent loop, which lives in the
        backend process, so the real work happens there (see
        :func:`ludvart.server._do_mcp_refresh`).
        """
        panel = self._panel
        if panel is None:
            return
        panel.add_system("MCP servers run on the backend; no backend connected.")

    @staticmethod
    def _parse_helper_init(snapshot: str) -> str:
        """Turn the helper install command's output line into a status message.

        Looks for the ``LUDVART_HELPER_INIT status=... version=... ok=... reason=...``
        line the injected command prints. The ``status`` value is constrained to
        real words so the echoed command template (which contains ``status=%s``)
        is not mistaken for the result.
        """
        m = re.search(
            r"LUDVART_HELPER_INIT status=(installed|current) version=(\S+) "
            r"ok=([01]) reason=(\w+)",
            snapshot,
        )
        if m is None:
            return (
                "Could not confirm ludvart_helper install -- no result seen. Make "
                "sure the foreground is an interactive shell, then run "
                "/init_helpers again."
            )
        status, ver, ok, reason = m.groups()
        if ok != "1":
            return (
                f"ludvart_helper install FAILED (reason={reason}); the file on disk "
                "does not match the expected checksum."
            )
        if status == "current":
            return f"ludvart_helper is already up to date (v{ver}, checksum verified)."
        if reason == "missing":
            return f"ludvart_helper v{ver} installed (was not present)."
        return (
            f"ludvart_helper v{ver} reinstalled "
            "(previous copy was outdated or modified)."
        )

    def _cmd_sessions(self, args: list[str]) -> None:
        """Handle ``/sessions [list|load ...]``."""
        panel = self._panel
        if panel is None:
            return
        sub = args[0] if args else "list"
        if sub == "list":
            self._session_list = list_sessions()
            if not self._session_list:
                panel.add_system("No saved sessions yet.")
                return
            current = self._session.session_id if self._session else None
            for i, s in enumerate(self._session_list, 1):
                marker = "*" if s["id"] == current else " "
                label = s.get("title") or s.get("preview") or "(no messages)"
                if len(label) > 48:
                    label = label[:47] + "\u2026"
                panel.add_system(
                    f"{marker}{i}. {s['id']}  ({s['count']} msgs)  {label}"
                )
            panel.add_system(
                'Use /sessions load <n>|<id>, or rename <id> "Title".'
            )
        elif sub == "load":
            if len(args) < 2:
                panel.add_system("Usage: /sessions load <n>|<id>")
                return
            self._load_session(args[1])
        elif sub == "new":
            self._new_session()
        elif sub == "rename":
            parsed = parse_rename_args(" ".join(args[1:]))
            if parsed is None:
                panel.add_system('Usage: /sessions rename <id> "New title"')
                return
            self._rename_session(*parsed)
        else:
            panel.add_system(f"Unknown subcommand: /sessions {sub}")

    def _rename_session(self, ref: str, title: str) -> None:
        """Set a saved session's display title (and the live store if current).

        ``ref`` may be a 1-based index from the last ``/sessions list`` or a raw
        session id.
        """
        panel = self._panel
        if panel is None:
            return
        session_id, error = resolve_session_ref(ref, self._session_list)
        if error is not None:
            panel.add_system(error)
            return
        if not rename_session(session_id, title):
            panel.add_system(f"Could not rename session: {session_id}")
            return
        if self._session is not None and self._session.session_id == session_id:
            self._session.title = title
        if title:
            panel.add_system(f'Renamed {session_id} to "{title}".')
        else:
            panel.add_system(f"Cleared the title of {session_id}.")

    def _new_session(self) -> None:
        """Start a fresh, empty conversation in a brand-new session file.

        Clears both the model-facing history and the on-screen transcript and
        binds a new ``SessionStore`` so nothing from the previous conversation
        (including any provider-specific message shapes) is replayed. The
        previous session's file is left untouched on disk.
        """
        panel = self._panel
        if panel is None:
            return
        self._llm_history = []
        self._panel_messages = []
        panel.restore([])
        # create_new() guarantees a directory that does not collide with the
        # current (or any existing) session, so the fresh, empty conversation
        # never overwrites the one we just left.
        self._session = SessionStore.create_new()
        panel.add_system(f"Started new session {self._session.session_id}.")

    # -- /model: multi-model registry ---------------------------------------

    def _cmd_model(self, args: list[str]) -> None:
        """Handle ``/model [list|use|add|remove ...]``."""
        panel = self._panel
        if panel is None:
            return
        if self._models is None:
            panel.add_system(
                "Model management is unavailable (no registry; started with a "
                "fixed model or --no-llm)."
            )
            return
        sub = args[0] if args else "list"
        if sub == "list":
            self._model_list()
        elif sub == "use":
            if len(args) < 2:
                panel.add_system("Usage: /model use <n>|<model>")
                return
            self._model_use(args[1])
        elif sub == "remove":
            if len(args) < 2:
                panel.add_system("Usage: /model remove <n>|<model>")
                return
            self._model_remove(args[1])
        elif sub == "add":
            self._model_add_start()
        else:
            panel.add_system(f"Unknown subcommand: /model {sub}")

    def _model_list(self) -> None:
        panel = self._panel
        if panel is None or self._models is None:
            return
        panel.add_system("Registered models:")
        for line in self._models.describe():
            panel.add_system(line)
        panel.add_system("Use /model use <n>, /model add, or /model remove <n>.")

    def _model_status(self, message: str) -> None:
        """Show a backend build step (e.g. gateway launch) on the spinner.

        Called from the ``/model use`` / ``/model add`` worker thread; updating
        ``panel.activity`` there is the same pattern used for streaming turns.
        """
        panel = self._panel
        if panel is not None:
            panel.activity = message.rstrip(". ")

    def _model_use(self, token: str) -> None:
        panel = self._panel
        if panel is None or self._models is None:
            return
        mgr = self._models
        idx = find_registration(mgr.models, token)
        if idx is None:
            panel.add_system(f"No model matches {token!r}. See /model list.")
            return

        def worker() -> str:
            # Runs while the *old* model is still active. If the conversation
            # would overflow the target model's (possibly smaller) window,
            # compact it here -- using the outgoing model, which can still hold
            # the full history -- before the swap tears that model down.
            compacted: list[str | None] = [None]

            def before_swap(new_client) -> None:
                new_cw = getattr(new_client, "context_window", 0) or 0
                if (
                    new_cw > 0
                    and self._last_input_tokens
                    and len(self._llm_history) > 2
                    and 100.0 * self._last_input_tokens / new_cw
                    >= self.CONTEXT_COMPACT_PCT
                ):
                    compacted[0] = self._compact_history()

            ok, msg = mgr.use(
                idx, status=self._model_status, before_swap=before_swap
            )
            if ok:
                # The manager now owns the new active client; adopt it.
                self.llm = mgr.client
                if self._panel is not None:
                    self._panel.provider = model_label(mgr.models[idx])
                    self._refresh_context_badge(compacted[0])
            return msg

        self._start_action(
            worker,
            info=f"Switching to {model_label(mgr.models[idx])}\u2026",
            activity="Switching model",
        )

    def _refresh_context_badge(self, summary: str | None) -> None:
        """Re-derive the context badge for the model now in ``self.llm``.

        After a switch the window may differ, so the ``[NN%]`` badge is recomputed
        immediately (rather than waiting for the next turn). ``summary`` is the
        text of a just-performed compaction, if any, so the estimate reflects the
        shrunken history.
        """
        panel = self._panel
        if panel is None:
            return
        if summary is not None:
            pct = self._estimate_context_pct(summary)
        elif self._last_input_tokens:
            cw = getattr(self.llm, "context_window", 0) or 0
            pct = (100.0 * self._last_input_tokens / cw) if cw > 0 else None
        else:
            pct = None
        panel.context_pct = pct
        self._panel_context_pct = pct

    def _model_remove(self, token: str) -> None:
        panel = self._panel
        if panel is None or self._models is None:
            return
        mgr = self._models
        idx = find_registration(mgr.models, token)
        if idx is None:
            panel.add_system(f"No model matches {token!r}. See /model list.")
            return
        _ok, msg = mgr.remove(idx)
        panel.add_system(msg)

    def _model_add_start(self) -> None:
        """Begin the guided ``/model add`` flow (fields typed in the panel)."""
        panel = self._panel
        if panel is None:
            return
        if self._models is None and self._backend_client is None:
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
            return
        if self._models is None:
            return
        mgr = self._models

        def worker() -> str:
            _ok, msg = mgr.add(reg, status=self._model_status)
            return msg

        self._start_action(
            worker,
            info=f"Verifying {model_label(reg)}\u2026",
            activity="Verifying model",
        )

    def _load_session(self, ref: str) -> None:
        """Load a saved session by 1-based list index or by id and resume it."""
        panel = self._panel
        if panel is None:
            return
        session_id = ref
        if ref.isdigit():
            idx = int(ref)
            if not (1 <= idx <= len(self._session_list)):
                panel.add_system(
                    f"No session #{idx}. Run /sessions list first."
                )
                return
            session_id = self._session_list[idx - 1]["id"]
        try:
            data = load_session(session_id)
        except (OSError, ValueError):
            panel.add_system(f"Could not load session: {session_id}")
            return
        messages = [tuple(m) for m in data.get("messages", [])]
        # The stored llm_history is a provider-neutral log (current sessions) or
        # a provider-native one (sessions written before the neutral refactor).
        # neutralize_history returns the former unchanged and migrates the
        # latter, so the running log is always neutral and any model can resume
        # it (the native context is rebuilt per request by
        # LLMClient.build_context).
        version = int(data.get("version", 1) or 1)
        stored_family = provider_family(data.get("provider"))
        neutral = neutralize_history(
            list(data.get("llm_history", [])), version, stored_family
        )
        history = working_history(neutral)
        migrated = version < NEUTRAL_SESSIONS_VERSION and bool(history)
        self._llm_history = history
        self._panel_messages = messages
        panel.restore(messages)
        # Continue writing into the loaded session's file from now on.
        self._session = SessionStore.open_existing(session_id)
        self._session.title = data.get("title", "") or ""
        note = " (older session migrated to the portable format)" if migrated else ""
        panel.add_system(
            f"Loaded session {session_id} ({len(messages)} msgs).{note}"
        )

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
        """Deliver a completed LLM reply into the panel and persist it."""
        panel = self._panel
        if panel is None:
            return
        panel.add_reply(result)
        self._persist_session()

    def _deliver_system(self, result: str) -> None:
        """Deliver a deterministic action's status as an ephemeral system line."""
        panel = self._panel
        if panel is None or not result:
            return
        panel.add_system(result)

    def _maybe_compact(self) -> bool:
        """Compact the model context into a summary if it is nearly full.

        Triggered when the last turn's prompt filled at least
        ``CONTEXT_COMPACT_PCT`` of the model's context window. A history that is
        already just a fresh summary seed (<= 2 messages) is left alone. Returns
        ``True`` when it actually compacted, so callers can react (e.g. reset a
        rollback checkpoint).
        """
        if self.llm is None or len(self._llm_history) <= 2:
            return False
        panel = self._panel
        pct = panel.context_pct if panel is not None else None
        if pct is None or pct < self.CONTEXT_COMPACT_PCT:
            return False
        return self._compact_history() is not None

    def _compact_history(self) -> str | None:
        """Summarize the running conversation and reseed the context from it.

        Asks the model to condense the whole ``_llm_history`` into a resumable
        brief, then purges the history and replaces it with a two-message seed
        (the summary + an acknowledgement). The visible transcript keeps the full
        conversation with a compaction marker, and the persisted session then
        resumes from this summary. Returns the summary text, or ``None`` if the
        summary request failed (the history is then left unchanged).
        """
        panel = self._panel
        if panel is not None:
            panel.activity = "Compacting context"
            self._render_split()
        summary = self._summarize_history()
        if not summary:
            return None  # failed; keep going with the uncompacted history
        self._llm_history = [
            {
                "role": "user",
                "content": f"{SUMMARY_MARKER}\n{summary}\n{SUMMARY_MARKER_END}",
            },
            {
                "role": "assistant",
                "content": "Understood. I will continue the task from this summary.",
            },
        ]
        if panel is not None:
            panel.add_summary(summary)
            panel.context_pct = self._estimate_context_pct(summary)
            self._panel_context_pct = panel.context_pct
            panel.activity = "Thinking"
        self._persist_session()
        return summary

    def _summarize_history(self) -> str | None:
        """Ask the model to summarize ``_llm_history`` into a resumable brief.

        Returns the summary text, or ``None`` if the request fails (compaction is
        then skipped and the conversation continues uncompacted).
        """
        instruction = (
            "You are about to run out of context window. Summarize the ENTIRE "
            "conversation above into concise notes that let you CONTINUE the "
            "task with no loss of essential information: the user's goal(s), the "
            "decisions made, facts, commands and file paths discovered, the "
            "current state of the work and the terminal, and the immediate next "
            "steps. Write it as a compact brief to yourself. Omit greetings, "
            "apologies, and filler."
        )
        messages = [
            {
                "role": "system",
                "content": "You compact your own working memory into a "
                "resumable brief so you can keep working after older turns are "
                "dropped.",
            },
            *self._build_llm_context(),
            {"role": "user", "content": instruction},
        ]
        try:
            with self._perf_timer("llm_request"):
                turn = self.llm.converse(
                    messages, tools=None, max_tokens=self.SUMMARY_MAX_TOKENS
                )
        except Exception as exc:  # never crash the ask; just skip compaction
            if self._panel is not None:
                self._panel.add_info(f"[ludvart] context compaction failed: {exc}")
            return None
        return (turn.text or "").strip() or None

    def _estimate_context_pct(self, summary: str) -> float | None:
        """Rough post-compaction context usage (~4 chars/token + seed overhead).

        The next real turn replaces this with the provider-reported value; this
        just makes the badge reflect the drop immediately.
        """
        cw = getattr(self.llm, "context_window", 0) or 0
        if cw <= 0:
            return None
        approx_tokens = (len(summary) + 400) // 4
        return max(0.0, 100.0 * approx_tokens / cw)

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
    #: the model-facing context. Only the most recent turn keeps its full
    #: <screenContext>; older ones are collapsed to a breadcrumb line to save
    #: context (the live screen is re-fetchable via tools, and the exact past
    #: snapshot via get_past_snapshot(timestamp)). The stored log is untouched.
    #: This bare form is only a fallback for a snapshot missing its timestamp;
    #: the normal breadcrumb carries the ts (see :meth:`_screen_breadcrumb`).
    _SCREEN_PLACEHOLDER = "[screen omitted; superseded by a newer snapshot]"

    #: Matches a ``<screenContext>`` open tag with any (or no) attributes.
    _SCREEN_OPEN_RE = re.compile(r"<screenContext(?:\s[^>]*)?>")
    #: Extracts the ``ts="..."`` timestamp attribute from an open tag.
    _SCREEN_TS_RE = re.compile(r'ts="([^"]*)"')

    @staticmethod
    def _utc_ns_timestamp() -> str:
        """Return the current UTC time as ``YYYY-MM-DDThh:mm:ss.<nanoseconds>``.

        Used to stamp each screen snapshot (see :meth:`_ask_llm`) with a unique,
        human-readable key. The nanosecond field comes from
        :func:`time.time_ns` so the value is precise enough to be unique within
        a session, while the date/time portion stays readable. The same string
        is later echoed in the breadcrumb and accepted by ``get_past_snapshot``.
        """
        ns = time.time_ns()
        secs, frac_ns = divmod(ns, 1_000_000_000)
        base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(secs))
        return f"{base}.{frac_ns:09d}"

    @classmethod
    def _screen_breadcrumb(cls, ts: str | None) -> str:
        """The line that replaces a stripped snapshot, keyed by its timestamp."""
        if not ts:
            return cls._SCREEN_PLACEHOLDER
        return (
            f"[screen from {ts} omitted; "
            f"queryable by get_past_snapshot({ts})]"
        )

    @classmethod
    def _strip_old_screenshots(cls, history: list[dict]) -> list[dict]:
        """Return a copy of the neutral log keeping only the newest screenshot.

        Every user turn embeds a ``<screenContext ts="...">...</screenContext>``
        block (see :meth:`_ask_llm`). Older snapshots are stale -- the screen
        changes every turn -- yet each is thousands of tokens, so retaining them
        all bloats the context and accelerates compaction. Here we keep the
        *last* user turn's screenshot verbatim and collapse the screen block of
        every earlier user turn to a timestamped breadcrumb (see
        :meth:`_screen_breadcrumb`), leaving the ``<userRequest>`` (and all
        non-user messages) untouched. The breadcrumb keeps each snapshot's
        timestamp so the model can fetch the full snapshot back with
        ``get_past_snapshot(timestamp)``. This only reshapes the per-request
        render; ``self._llm_history`` and the persisted session keep the full
        snapshots.
        """
        close_tag = "</screenContext>"

        def open_match(content: str):
            return cls._SCREEN_OPEN_RE.search(content)

        def has_screen(msg: dict) -> bool:
            return (
                isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
                and open_match(msg["content"]) is not None
                and close_tag in msg["content"]
            )

        last = -1
        for i, msg in enumerate(history):
            if has_screen(msg):
                last = i
        if last < 0:
            return list(history)

        out: list[dict] = []
        for i, msg in enumerate(history):
            if i != last and has_screen(msg):
                content = msg["content"]
                m = open_match(content)
                open_tag = m.group(0)
                ts_m = cls._SCREEN_TS_RE.search(open_tag)
                ts = ts_m.group(1) if ts_m else None
                start = m.start()
                end = content.find(close_tag) + len(close_tag)
                trimmed = (
                    content[:start]
                    + open_tag
                    + "\n"
                    + cls._screen_breadcrumb(ts)
                    + "\n"
                    + close_tag
                    + content[end:]
                )
                new_msg = dict(msg)
                new_msg["content"] = trimmed
                out.append(new_msg)
            else:
                out.append(msg)
        return out

    def _build_llm_context(self) -> list[dict]:
        """Render the neutral log into the active client's native messages.

        Delegates to :meth:`LLMClient.build_context` so provider-specific
        shaping lives on the client (letting clients coexist and be swapped
        mid-conversation). Lightweight test stubs without ``build_context`` get
        the raw neutral log, which is close enough for their assertions.

        Before rendering, superseded screen snapshots are stripped (see
        :meth:`_strip_old_screenshots`) so only the newest ``<screenContext>``
        is sent to the model.
        """
        history = self._strip_old_screenshots(self._llm_history)
        build = getattr(self.llm, "build_context", None)
        if build is None:
            return list(history)
        return build(history)

    @staticmethod
    def _neutral_assistant(turn) -> dict:
        """Neutral log entry for an assistant turn (text + any tool calls).

        Derived from the provider-agnostic :class:`Turn` fields so the stored
        log never carries a provider-specific message shape; the native shape is
        rebuilt per request by :meth:`LLMClient.build_context`.
        """
        entry: dict = {"role": "assistant", "content": turn.text or ""}
        if turn.tool_calls:
            entry["tool_calls"] = [
                {"id": c.id, "name": c.name, "input": dict(c.input)}
                for c in turn.tool_calls
            ]
        return entry

    @staticmethod
    def _neutral_tool_result(call, output: str) -> dict:
        """Neutral log entry for a tool result (keeps id and name for replay)."""
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": output,
        }

    def _ask_llm(self, question: str) -> str:
        """Ask the LLM, maintaining the full multi-turn conversation.

        The panel transcript only shows the visible question/answer text, so the
        model-facing history is kept in a separate buffer. Every user turn embeds
        the terminal screen snapshot captured at ask time, and each answer is
        appended, so the model sees the entire prior conversation and every
        screen it was shown.

        The model may also request tool calls (see :meth:`_llm_tools`). ludvart runs
        an agent loop: it executes each requested tool, appends a tool_result to
        the history, and asks again -- exactly the assistant/tool_use ->
        tool_result -> assistant round-tripping a tool-using client performs --
        until the model returns a plain text answer.
        """
        # Capture this turn's cancellation flag locally: a later ask reassigns
        # ``self._ask_cancel``, so an abandoned worker must keep checking the
        # event it was started with rather than a newer one.
        cancel = self._ask_cancel
        screen_text = self.snapshot_text()
        # Stamp each snapshot with a UTC timestamp (nanosecond precision) so it
        # can be addressed later. When older snapshots are stripped from the
        # model-facing context (see :meth:`_strip_old_screenshots`) the
        # timestamp survives in the breadcrumb, letting the model fetch the full
        # snapshot back via the ``get_past_snapshot`` tool.
        snapshot_ts = self._utc_ns_timestamp()
        user_content = (
            f'<screenContext ts="{snapshot_ts}">\n'
            f"{screen_text}\n"
            "</screenContext>\n"
            f"<userRequest>\n{question}\n</userRequest>"
        )
        system = {"role": "system", "content": self._llm_system_prompt()}
        tools = self._llm_tools()

        # Surface automatic retries (timeouts, rate limits, ...) in the panel so
        # the user can see ludvart is waiting rather than hung.
        panel = self._panel
        if panel is not None:
            def _on_retry(note: str) -> None:
                panel.add_info(note)
                panel.activity = (
                    "Rate limited" if "rate limited" in note else "Retrying"
                )
            self.llm.on_retry = _on_retry

        # Compact the running context into a summary before it fills the model's
        # window. Done here (history ends with a clean assistant turn) so the new
        # turn starts with plenty of headroom.
        self._maybe_compact()

        # Remember where this turn starts so a mid-flight failure can be rolled
        # back cleanly, leaving the history well-formed (no dangling tool_use).
        checkpoint = len(self._llm_history)
        self._llm_history.append({"role": "user", "content": user_content})

        # Streamed narration: as the model produces text, show it live (dim)
        # above the spinner so the user sees what it is doing. It is transient --
        # each turn starts fresh and the final reply replaces it entirely. The
        # running narration of this turn -- the model's streamed reasoning from
        # each request plus the tool-call notes -- is accumulated so the user
        # sees the whole history of what happened; only the currently-streaming
        # text of the in-flight request sits below it. It is purged when the
        # final answer replaces it (see :meth:`_finish_ask`).
        narration: list[str] = []
        last_stream = ""

        def _compose(streamed: str = "") -> str:
            parts = list(narration)
            if streamed:
                parts.append(streamed)
            return "\n".join(parts)

        def _on_text(text: str) -> None:
            nonlocal last_stream
            if cancel.is_set():
                raise _AskCancelled()
            last_stream = text
            self._mark_wait_streaming()
            p = self._panel
            if p is not None:
                p.interim = _compose(text)

        try:
            while True:
                if cancel.is_set():
                    raise _AskCancelled()
                # Compact before EVERY request, not just once per user ask: a
                # single agentic turn can issue many tool round-trips, and each
                # re-sends the whole history (screen snapshots + tool outputs),
                # so the context grows within this loop. Checking only at the
                # top let a long tool loop sail past the window without ever
                # compacting. When it compacts, the history is replaced by a
                # small summary seed, so move the rollback checkpoint with it.
                if self._maybe_compact():
                    checkpoint = len(self._llm_history)
                if panel is not None:
                    panel.activity = "Thinking"
                    panel.interim = _compose()
                last_stream = ""
                self._begin_wait("Thinking")
                with self._perf_timer("llm_request"):
                    turn = self.llm.converse(
                        [system, *self._build_llm_context()],
                        tools=tools,
                        max_tokens=self.REPLY_MAX_TOKENS,
                        on_text=_on_text,
                    )
                self._end_wait()
                self._llm_history.append(self._neutral_assistant(turn))
                if turn.usage is not None:
                    pct = turn.usage.context_percent()
                    self._panel_context_pct = pct
                    self._last_input_tokens = turn.usage.input_tokens
                    if self._panel is not None:
                        self._panel.context_pct = pct
                if not turn.tool_calls:
                    return turn.text
                # Keep this request's streamed reasoning/commentary in the
                # narration (above the tool notes) so it stays visible through
                # the following tool round-trips instead of vanishing.
                if last_stream:
                    narration.append(last_stream)
                for call in turn.tool_calls:
                    if cancel.is_set():
                        raise _AskCancelled()
                    narration.append(self._tool_call_note(call))
                    if self._panel is not None:
                        self._panel.activity = f"Calling {call.name}"
                        self._panel.interim = _compose()
                    self._begin_wait(f"Calling {call.name}")
                    with self._perf_timer(f"tool:{call.name}"):
                        output = self._run_tool(call)
                    self._end_wait()
                    self._llm_history.append(
                        self._neutral_tool_result(call, output)
                    )
                if self._panel is not None:
                    self._panel.activity = "Thinking"
        except Exception:
            del self._llm_history[checkpoint:]
            raise

    def _llm_system_prompt(self) -> str:
        tool_lines = "\n".join(
            f"  - {t.name}: {t.description}" for t in self._llm_tools()
        )
        return (
            "You are ludvart, an assistant embedded in a terminal. The user can ask "
            "you questions across multiple turns. Each user message contains a "
            "<screenContext> block with a snapshot of what is currently on the "
            "terminal (the screen may change between turns) followed by the actual "
            "question in a <userRequest> block. Use the conversation history and "
            "the latest screen to answer concisely and helpfully.\n\n"
            "For multi-step work, narrate your immediate next action or a useful "
            "observation in short, user-visible plain text before calling a tool. "
            "This is a progress update, not private reasoning: do not reveal "
            "hidden chain-of-thought or provide a long rationale.\n\n"
            "You can ACT in the user's terminal using the tools available to you "
            "(invoke them through the normal tool/function-calling mechanism):\n"
            f"{tool_lines}\n\n"
            "These tools are really available to you right now. If the user asks "
            "what tools or actions you can invoke, answer using this exact list -- "
            "never claim you have no tools or that you don't know your tools. When "
            "the user asks you to run, execute, display, open, show, list, or type "
            "something in the terminal, actually DO it by calling the relevant "
            "tool rather than only describing the command. The result appears on "
            "the terminal screen and in your next screen snapshot, which you can "
            "then describe.\n\n"
            "Carry out your tasks inside whatever application is currently running "
            "in the foreground, working within it whenever possible. If you judge "
            "that a better solution requires leaving or exiting that application "
            "(for example quitting the current program to run something else), do "
            "NOT exit on your own -- first explain the better approach and confirm "
            "with the user, and only exit the application once they agree.\n\n"
            "IMPORTANT: Keep every response you show to the user in plain "
            "7-bit ASCII so it renders on any terminal. Do NOT emit non-ASCII "
            "characters such as Unicode dashes, curly quotes, arrows, em-dashes, "
            "box-drawing glyphs or emoji -- terminals that cannot render them "
            "show a '?' instead. Use '-' for bullets and dashes, straight ' and "
            "\" quotes, and '->' for arrows.\n\n"
            "When helper tools under ~/.ludvart/bin/ are available (check with "
            "'ls ~/.ludvart/bin/' and 'ludvart_helper info' early in a session), "
            "you MUST use ludvart_helper instead of injecting raw shell input "
            "through inject_input for file reads, edits, searches, and "
            "non-interactive commands. Use 'ludvart_helper read', 'replace', "
            "'search', or 'run' as appropriate. Its base64 payloads and "
            "sentinel-framed exit codes avoid dangerous quoting/escape corruption. "
            "Use raw injected shell input only for interactive terminal work, or "
            "when the helper is unavailable or cannot express the operation. Do "
            "NOT hand-roll multi-layer quoted scripts to edit a file when a "
            "helper can do it in one call. Use the native "
            "'b64_encode'/'b64_decode' tools to build the base64 payloads for "
            "ludvart_helper and to read its base64 result frames, instead of "
            "'printf | base64' / 'base64 -d' in the shell.\n\n"
            "(ludvart_helper v0.2.0+ adds safer edits: 'replace --expect-count N' "
            "fails without writing if the match count differs; '--dry-run' on "
            "replace/write returns a unified diff instead of writing; "
            "'replace-range --start N --end M --b64 DATA' swaps a line range; "
            "'structured-patch PATH --b64 JSON' applies multiple exact edits atomically; "
            "writes auto-save a .ludvart.bak, and a .py edit that breaks syntax "
            "returns exit=4 (error=py_syntax) so failures are explicit.)\n\n"
            + LUDVART_HELPERS_DOC
            + self._load_self_md()
        )

    def _load_self_md(self) -> str:
        """Load persistent self-notes from ~/.ludvart/SELF.md if present.

        Returns "" when the file is missing, unreadable, or empty, so the
        system-prompt builder never breaks. The content is length-capped and
        prefixed with a header before being appended to the prompt.
        """
        path = os.path.expanduser("~/.ludvart/SELF.md")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()
        except (OSError, UnicodeDecodeError):
            return ""
        data = data[:8192]
        if not data.strip():
            return ""
        return "\n\n## Persistent self-notes (from ~/.ludvart/SELF.md)\n" + data

    def _llm_tools(self) -> list[ToolSpec]:
        """Tools advertised to the model for this session."""
        return self._builtin_tools() + self._mcp_tools()

    def _mcp_tools(self) -> list[ToolSpec]:
        """Namespaced tools discovered from external MCP servers (if any)."""
        if self._mcp is None:
            return []
        return self._mcp.tool_specs()

    def _builtin_tools(self) -> list[ToolSpec]:
        """ludvart's own, always-available tools."""
        return [
            ToolSpec(
                name="inject_input",
                description=(
                    "Type characters into the user's terminal, exactly as if the "
                    "user pressed the keys on their keyboard. The characters go to "
                    "whatever program is currently in the foreground. Use it to "
                    "(1) run a shell command on the user's behalf -- e.g. list or "
                    "display files with 'ls' / 'cat', check status, install "
                    "packages, etc. (set submit=true to press Enter and execute); "
                    "or (2) send keystrokes (including control characters) to an "
                    "interactive program such as vim, less, a REPL or a TUI. This "
                    "is the way to actually DO things in the terminal; prefer it "
                    "over merely telling the user what to type. "
                    "IMPORTANT: keep each call's 'text' small -- at most about "
                    "2 KB. Larger payloads (e.g. a long base64 blob or a big file "
                    "body) often fail to be generated and arrive EMPTY, which "
                    "wastes a call; split long content into several sequential "
                    "inject_input calls of <=2 KB each (submit=false on the "
                    "intermediate parts, then submit=true -- or a trailing "
                    "newline -- on the final one to execute)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": (
                                "The characters to type. For a shell command "
                                "this is the command line, e.g. 'ls -la'. "
                                "Backslash escapes are interpreted (unless "
                                "interpret_escapes=false) so you CAN send "
                                "control keys: use \\xHH for a raw byte (e.g. "
                                "\\x06 = Ctrl-F, \\x1b = Esc), \\cX for a control "
                                "key (e.g. \\cf = Ctrl-F), plus \\e (Esc), \\t "
                                "(Tab), \\r (Enter), \\n (newline). Write \\\\ for "
                                "a literal backslash. Raw control BYTES do not "
                                "survive here -- always express control keys with "
                                "these escapes. A trailing newline (or "
                                "submit=true) is needed to run a shell command. "
                                "Keep this under ~2 KB per call; for longer "
                                "content, split it across several sequential "
                                "inject_input calls (each <=2 KB) instead of one "
                                "large one, which may otherwise arrive empty."
                            ),
                        },
                        "submit": {
                            "type": "boolean",
                            "description": (
                                "If true, press Enter (send a carriage return) "
                                "after the text to execute it. Defaults to false."
                            ),
                        },
                        "interpret_escapes": {
                            "type": "boolean",
                            "description": (
                                "Whether to decode backslash escapes in 'text' "
                                "(\\xHH, \\cX, \\e, \\t, \\r, \\n, \\\\). Defaults "
                                "to true. Set false to send 'text' verbatim, "
                                "e.g. when typing literal backslashes."
                            ),
                        },
                    },
                    "required": ["text"],
                },
            ),
            ToolSpec(
                name="capture_screen_history",
                description=(
                    "Read lines from the terminal's scrollback history -- output "
                    "that has scrolled above the currently visible screen. Use "
                    "this when a command's output (for example the result of an "
                    "inject_input call) is longer than what fits on the visible "
                    "screen and you need to see the earlier lines. The history is "
                    "the full logical output: everything that scrolled off the "
                    "top, followed by the current viewport. 'offset' is a number "
                    "of lines measured from the current position (the latest "
                    "line) and must be NEGATIVE to look upward -- e.g. "
                    "offset=-100 starts 100 lines above the current position. "
                    "'length' is how many lines to return starting at that "
                    "offset. If the range extends past the top it is clamped, and "
                    "the result reports how many lines exist in total so you can "
                    "adjust the offset and try again."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "offset": {
                            "type": "integer",
                            "description": (
                                "Lines from the current position to start at. "
                                "Negative goes back into history, e.g. -100 = "
                                "100 lines above the current position."
                            ),
                        },
                        "length": {
                            "type": "integer",
                            "description": (
                                "How many lines to return, starting at 'offset'."
                            ),
                        },
                    },
                    "required": ["offset", "length"],
                },
            ),
            ToolSpec(
                name="get_past_snapshot",
                description=(
                    "Return the exact terminal screen snapshot that was captured "
                    "at a previous turn, identified by its UTC timestamp. Each "
                    "user turn embeds a <screenContext ts=\"...\"> snapshot; once "
                    "superseded, older snapshots are removed from your context "
                    "and replaced by a breadcrumb line that keeps the timestamp "
                    "(e.g. '[screen from <TS> omitted; queryable by "
                    "get_past_snapshot(<TS>)]'). Call this with that <TS> to get "
                    "the full snapshot back. Unlike capture_screen_history (which "
                    "reads flattened scrollback), this returns a consistent, "
                    "point-in-time rectangular screenshot -- useful for full-screen "
                    "TUI applications whose past state cannot be reconstructed "
                    "from scrollback. Pass the timestamp exactly as shown in the "
                    "breadcrumb."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "timestamp": {
                            "type": "string",
                            "description": (
                                "The UTC timestamp of the desired snapshot, "
                                "exactly as it appears in the breadcrumb, e.g. "
                                "'2026-07-06T17:28:54.123456789'."
                            ),
                        },
                    },
                    "required": ["timestamp"],
                },
            ),
            ToolSpec(
                name="b64_encode",
                description=(
                    "Encode UTF-8 text to base64 natively (no shell, no "
                    "terminal round-trip). Use this to build the base64 "
                    "payloads that ludvart_helper subcommands expect (e.g. "
                    "--b64 / --old-b64 / --new-b64), avoiding fragile "
                    "'printf | base64' shell quoting. Returns the base64 string."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The literal text to base64-encode.",
                        },
                    },
                    "required": ["text"],
                },
            ),
            ToolSpec(
                name="b64_decode",
                description=(
                    "Decode a base64 string to UTF-8 text natively (no shell). "
                    "Use this to read base64 payloads returned inside "
                    "ludvart_helper's LUDVART:BEGIN/END result frames without piping "
                    "through 'base64 -d' on screen. Returns the decoded text."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "b64": {
                            "type": "string",
                            "description": "The base64 string to decode.",
                        },
                    },
                    "required": ["b64"],
                },
            ),
            ToolSpec(
                name="web_search",
                description=(
                    "Perform a DuckDuckGo web search to retrieve the most up-to-date information "
                    "on any query. Returns a list of titles, target URLs, and descriptive snippets."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to lookup.",
                        },
                    },
                    "required": ["query"],
                },
            ),
            ToolSpec(
                name="fetch_url",
                description=(
                    "Download the contents of a URL and save it to a temporary file under /tmp on the "
                    "remote host where ludvart is running (which might be different from the host the user "
                    "sees in the terminal). Returns the path to the saved file and a brief summary of the download."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The absolute HTTP or HTTPS URL to fetch/download.",
                        },
                    },
                    "required": ["url"],
                },
            ),
            ToolSpec(
                name="read_local_file",
                description=(
                    "Read a window of lines from a local file on the host where ludvart runs. "
                    "Like a code editor's file reader, this returns at most a bounded number of lines "
                    "per call; large files are read by calling repeatedly, paging forward. The typical "
                    "pattern is to call get_local_file_info first to learn the line count, then call "
                    "read_local_file for successive line ranges. When a read does not reach the end of "
                    "the file, the result reports the start_line to pass next to continue. This is useful "
                    "to inspect files saved by other tools like fetch_url. The file path refers to the host "
                    "running ludvart, which is not necessarily the same host shown in the terminal."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file to read (e.g. /tmp/...)",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "The 1-based line number to start reading from (inclusive). Defaults to 1.",
                        },
                        "end_line": {
                            "type": "integer",
                            "description": (
                                "The 1-based line number to stop reading at (inclusive). If omitted, a "
                                "default window of lines from start_line is returned. The number of lines "
                                "returned in a single call is capped regardless of this value; read again "
                                "with a later start_line to continue."
                            ),
                        },
                    },
                    "required": ["path"],
                },
            ),
            ToolSpec(
                name="get_local_file_info",
                description=(
                    "Retrieve information about a local file on the remote host running ludvart (e.g. size, number of lines, "
                    "and modified time). This is helpful to plan chunked reading using read_local_file."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The path to the file on the host running ludvart.",
                        },
                    },
                    "required": ["path"],
                },
            ),
        ]

    def _tool_call_note(self, call: "ToolCall") -> str:
        """One-line summary of a tool invocation for the live narration.

        Appended to the transient "Thinking" narration (``panel.interim``) so the
        user can see the running history of what the agent is doing -- including
        fast, in-memory tools like ``capture_screen_history`` whose "Calling ..."
        spinner label would flash by faster than a render frame. It is purged
        when the final answer replaces the narration. String arguments are quoted
        (so control characters injected via ``inject_input`` are visible as
        escapes) and long values are truncated.
        """
        parts: list[str] = []
        for key, val in call.input.items():
            if isinstance(val, str) and len(val) > 60:
                val = val[:57] + "\u2026"
            parts.append(f"{key}={val!r}")
        return f"\u2192 {call.name}(" + ", ".join(parts) + ")"

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

    def _run_tool(self, call: "ToolCall") -> str:
        """Execute a model-requested tool and return its result text."""
        if call.name == "inject_input":
            return self._tool_inject_input(call.input)
        if call.name == "capture_screen_history":
            return self._tool_capture_screen_history(call.input)
        if call.name == "get_past_snapshot":
            return self._tool_get_past_snapshot(call.input)
        if call.name == "b64_encode":
            return self._tool_b64_encode(call.input)
        if call.name == "b64_decode":
            return self._tool_b64_decode(call.input)
        if call.name == "web_search":
            return self._tool_web_search(call.input)
        if call.name == "fetch_url":
            return self._tool_fetch_url(call.input)
        if call.name == "read_local_file":
            return self._tool_read_local_file(call.input)
        if call.name == "get_local_file_info":
            return self._tool_get_local_file_info(call.input)
        if self._mcp is not None and self._mcp.is_mcp_tool(call.name):
            if self._panel is not None:
                self._panel.activity = f"Calling {call.name}"
            return self._mcp.call_tool(call.name, call.input)
        return f"[ludvart] unknown tool: {call.name}"

    #: Cap on how much fetch_url writes to /tmp, so a hostile or accidental huge
    #: response cannot fill the disk on the host running ludvart.
    _FETCH_URL_MAX_BYTES = 10 * 1024 * 1024

    def _fetch_tmp_dir(self) -> str:
        """Return this run's private scratch dir, creating it on first use.

        ``tempfile.mkdtemp`` makes a uniquely named 0700 directory owned by the
        current user, so files saved here are never confused with those of other
        users or concurrent ludvart processes. Removed by ``_cleanup_fetch_dir``
        when ``run`` exits.
        """
        if self._fetch_dir is None:
            import tempfile

            self._fetch_dir = tempfile.mkdtemp(prefix="ludvart_")
        return self._fetch_dir

    def _cleanup_fetch_dir(self) -> None:
        """Remove this run's fetch scratch dir and everything under it."""
        if self._fetch_dir is None:
            return
        import shutil

        shutil.rmtree(self._fetch_dir, ignore_errors=True)
        self._fetch_dir = None

    def _tool_fetch_url(self, args: dict) -> str:
        """Fetch a URL and save it to a temp file on the host running ludvart."""
        url = args.get("url")
        if not isinstance(url, str):
            return "[ludvart] fetch_url: 'url' must be a string."
        url = url.strip()
        if not url:
            return "[ludvart] fetch_url: 'url' is empty."
        import urllib.request, urllib.error, urllib.parse, tempfile, os

        scheme = urllib.parse.urlparse(url).scheme.lower()
        if scheme not in ("http", "https"):
            return (
                f"[ludvart] fetch_url: unsupported URL scheme "
                f"{scheme or '(none)'!r} (only http/https are allowed)."
            )
        max_bytes = self._FETCH_URL_MAX_BYTES
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                # Read one byte past the cap so we can report truncation without
                # ever holding more than the cap plus one byte in memory.
                raw = resp.read(max_bytes + 1)
                charset = resp.headers.get_content_charset() or "utf-8"
        except urllib.error.HTTPError as exc:
            return f"[ludvart] fetch_url failed with status code: {exc.code}"
        except Exception as exc:
            return f"[ludvart] fetch_url failed: {exc}"

        truncated = len(raw) > max_bytes
        if truncated:
            raw = raw[:max_bytes]
        try:
            text = raw.decode(charset, errors="replace")
        except LookupError:
            # Unknown charset advertised by the server -> fall back to utf-8.
            text = raw.decode("utf-8", errors="replace")

        fd, path = tempfile.mkstemp(
            prefix="ludvart_", suffix=".html", dir=self._fetch_tmp_dir()
        )
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as f:
            f.write(text)

        note = " (truncated at cap)" if truncated else ""
        return (
            f"[ludvart] Successfully fetched {url}\n"
            f"Saved content to a temporary file on the host running ludvart:\n"
            f"PATH: {path}\n"
            f"SIZE: {len(text)} characters{note}\n"
            f"To read this file, use the 'read_local_file' tool."
        )

    def _tool_get_local_file_info(self, args: dict) -> str:
        """Get details (size, lines count) of a local file on the host running ludvart."""
        path = args.get("path")
        if not isinstance(path, str):
            return "[ludvart] get_local_file_info: 'path' must be a string."
        import os, datetime
        if not os.path.isfile(path):
            return f"[ludvart] get_local_file_info: '{path}' is not a file."
        try:
            st = os.stat(path)
            line_count = 0
            with open(path, "rb") as f:
                for _line in f:
                    line_count += 1
            mtime = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(
                timespec="seconds"
            )
            return (
                f"[ludvart] File info for {path}:\n"
                f"SIZE: {st.st_size} bytes\n"
                f"LINES: {line_count} lines\n"
                f"MODIFIED: {mtime}\n"
            )
        except Exception as exc:
            return f"[ludvart] get_local_file_info failed: {exc}"

    #: Maximum number of lines a single read_local_file call returns. Larger
    #: files are paged through with repeated calls (like an editor's reader).
    _READ_MAX_LINES = 2000

    def _tool_read_local_file(self, args: dict) -> str:
        """Read a bounded window of lines from a local file on ludvart's host.

        Behaves like a code editor's file reader: a single call returns at most
        ``_READ_MAX_LINES`` lines. When the window does not reach end of file the
        result reports the ``start_line`` to pass on the next call, so the model
        pages through large files with successive reads.
        """
        path = args.get("path")
        if not isinstance(path, str):
            return "[ludvart] read_local_file: 'path' must be a string."
        import os
        if not os.path.isfile(path):
            return f"[ludvart] read_local_file: '{path}' is not a file."

        start_line = args.get("start_line", 1)
        end_line = args.get("end_line")
        # bool is a subclass of int; reject it explicitly so True/False are not
        # silently treated as line 1/0.
        if isinstance(start_line, bool) or not isinstance(start_line, int):
            return "[ludvart] read_local_file: 'start_line' must be an integer."
        if end_line is not None and (
            isinstance(end_line, bool) or not isinstance(end_line, int)
        ):
            return "[ludvart] read_local_file: 'end_line' must be an integer."
        if start_line < 1:
            start_line = 1
        max_lines = self._READ_MAX_LINES
        if end_line is None:
            end_line = start_line + max_lines - 1
        if end_line < start_line:
            return "[ludvart] read_local_file: 'end_line' must be >= 'start_line'."
        # Cap the window so one call never returns more than max_lines lines.
        window_end = min(end_line, start_line + max_lines - 1)

        try:
            selected: list[str] = []
            has_more = False
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for idx, line in enumerate(f, 1):
                    if idx < start_line:
                        continue
                    if idx > window_end:
                        # There is at least one line beyond the window.
                        has_more = True
                        break
                    selected.append(line)
        except Exception as exc:
            return f"[ludvart] read_local_file failed: {exc}"

        if not selected:
            return (
                f"[ludvart] read_local_file: {path} has no lines at or after "
                f"line {start_line} (start_line is past the end of the file)."
            )

        last_line = start_line + len(selected) - 1
        content = "".join(selected)
        # Secondary safety cap on raw characters (e.g. pathologically long lines).
        char_limit = 150_000
        char_truncated = len(content) > char_limit
        if char_truncated:
            content = content[:char_limit]

        notes: list[str] = []
        if has_more:
            notes.append(
                f"More lines follow; continue with start_line={last_line + 1}."
            )
        if char_truncated:
            notes.append(f"Output truncated to {char_limit} characters.")

        body = content if content.endswith("\n") else content + "\n"
        result = (
            f"[ludvart] {path} lines {start_line}-{last_line}:\n"
            f"--------------------------------------------------\n"
            f"{body}"
            f"--------------------------------------------------\n"
        )
        if notes:
            result += "\n".join(notes) + "\n"
        return result


    def _tool_web_search(self, args: dict) -> str:
        """Perform a DuckDuckGo web search to retrieve up-to-date information."""
        query = args.get("query")
        if not isinstance(query, str):
            return "[ludvart] web_search: 'query' must be a string."
        if not query.strip():
            return "[ludvart] web_search: nothing to search (empty 'query')."
        import urllib.request, urllib.error, urllib.parse, re, html

        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read(2 * 1024 * 1024)
                charset = resp.headers.get_content_charset() or "utf-8"
            page = body.decode(charset, errors="replace")
        except urllib.error.HTTPError as exc:
            return f"[ludvart] web_search failed with status code: {exc.code}"
        except Exception as exc:
            return f"[ludvart] web_search failed: {exc}"

        def _clean(fragment: str) -> str:
            # Strip tags, then decode HTML entities (&amp;, &#x27;, ...).
            return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()

        blocks = page.split('<div class="links_main links_deep result__body">')[1:]
        outputs = []
        for block in blocks[:10]:
            title_match = re.search(
                r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                block,
                re.DOTALL,
            )
            if not title_match:
                continue
            snippet_match = re.search(
                r'class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL
            )
            raw_url = html.unescape(title_match.group(1))
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
            actual_url = qs.get("uddg", [raw_url])[0]
            title = _clean(title_match.group(2))
            snippet = _clean(snippet_match.group(1)) if snippet_match else ""
            outputs.append(
                f"TITLE: {title}\nURL: {actual_url}\nSNIPPET: {snippet}\n"
            )

        if not outputs:
            return "[ludvart] web_search: No results found."
        return "\n---\n".join(outputs)

    def _tool_b64_encode(self, args: dict) -> str:
        """Base64-encode text natively (no shell/PTY round-trip)."""
        text = args.get("text")
        if not isinstance(text, str):
            return "[ludvart] b64_encode: 'text' must be a string"
        return base64.b64encode(text.encode("utf-8")).decode("ascii")

    def _tool_b64_decode(self, args: dict) -> str:
        """Base64-decode a string to UTF-8 text natively (no shell)."""
        data = args.get("b64")
        if not isinstance(data, str):
            return "[ludvart] b64_decode: 'b64' must be a string"
        try:
            return base64.b64decode(data, validate=True).decode("utf-8", "replace")
        except Exception as exc:
            return f"[ludvart] b64_decode: invalid base64: {exc}"

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
        prompt_prefix = self._current_prompt_prefix()
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
                if tui or self.llm is None:
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

        This is a standalone LLM call that is deliberately NOT added to the
        conversation history -- it only decides whether to keep waiting. On any
        error (or no LLM) it reports finished so the tool never hangs.
        """
        if self.llm is None:
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
            reply = self.llm.complete([system, user], max_tokens=8)
        except Exception:
            return True  # never hang the tool on a status-check failure
        verdict = reply.strip().upper()
        return "RUNNING" not in verdict

    def _snapshot_by_timestamp(self, ts: str):
        """Return the snapshot body stored under timestamp ``ts``, or ``None``.

        Scans the *unstripped* neutral log (``self._llm_history``) -- not the
        model-facing context, which may have had this snapshot collapsed to a
        breadcrumb -- for the user turn whose ``<screenContext ts="...">`` open
        tag carries ``ts`` and returns the text between the open and close tags
        (the raw screenshot). Returns ``None`` if no snapshot has that
        timestamp. This is the backing lookup for the ``get_past_snapshot`` tool.
        """
        close_tag = "</screenContext>"
        for msg in self._llm_history:
            if not (
                isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
            ):
                continue
            content = msg["content"]
            m = self._SCREEN_OPEN_RE.search(content)
            if m is None or close_tag not in content:
                continue
            ts_m = self._SCREEN_TS_RE.search(m.group(0))
            if ts_m is None or ts_m.group(1) != ts:
                continue
            start = m.end()
            end = content.find(close_tag, start)
            if end < 0:
                continue
            return content[start:end].strip("\n")
        return None

    def _tool_get_past_snapshot(self, args: dict) -> str:
        """Return a stored past screen snapshot addressed by its timestamp."""
        ts = args.get("timestamp")
        if not isinstance(ts, str) or not ts.strip():
            return (
                "[ludvart] get_past_snapshot: 'timestamp' must be a non-empty "
                "string. Provide a valid snapshot timestamp exactly as shown in "
                "a breadcrumb."
            )
        ts = ts.strip()
        snapshot = self._snapshot_by_timestamp(ts)
        if snapshot is None:
            return (
                f"[ludvart] get_past_snapshot: no snapshot found for timestamp "
                f"{ts!r}. A valid snapshot timestamp is needed -- pass one "
                "exactly as it appears in a breadcrumb."
            )
        return (
            f"Terminal screen snapshot captured at {ts}:\n"
            f'<screenContext ts="{ts}">\n'
            f"{snapshot}\n"
            "</screenContext>"
        )

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
