"""The bottom AI panel: a resizable, scrollable chat pane.

The panel owns only its own state (conversation transcript, the question being
typed, scroll offset, height). It renders itself to a list of drawable row
payloads; the compositor in :mod:`ludvart` places those on the physical screen
below the (resized) application region. The panel never touches the child's
screen model.
"""

from __future__ import annotations

from .lineedit import LineEditor
from .overlay import _wrap

_RESET = b"\x1b[0m"
_EOL = b"\x1b[K"
_REVERSE = b"\x1b[7m"
_CYAN = b"\x1b[36m"
_DIM = b"\x1b[2m"
_BOLD = b"\x1b[1m"

_PROMPT = "ludvart> "

#: Most rows the input block may take before it scrolls internally. The panel is
#: shared with the transcript, so a long paste must not squeeze it out entirely.
INPUT_MAX_ROWS = 8

# Animated ellipsis frames: dots grow then shrink after the "thinking" text.
_THINK_FRAMES = ("", ".", "..", "...", "..", ".")


def _clip(text: str, width: int) -> str:
    """Fit ``text`` on one row, marking a cut with a trailing ellipsis."""
    width = max(1, width)
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."[: min(3, width)]


class AiPanel:
    """State and rendering for the bottom AI interaction panel."""

    def __init__(self, cols: int, height: int, provider: str = "") -> None:
        self.cols = max(1, cols)
        self.height = height
        self.provider = provider
        self.editor = LineEditor()
        self.thinking = False
        # The verb shown by the animated indicator while ``thinking`` is True.
        # Defaults to "Thinking"; set to e.g. "Calling inject_input" during a
        # tool call so the user can see what the agent is doing.
        self.activity = "Thinking"
        self.tick = 0  # advances while thinking, drives the spinner animation
        self.scroll = 0  # rows scrolled up from the bottom of the transcript
        self._messages: list[tuple[str, str]] = []
        # Live, transient narration streamed from the model during a turn. Shown
        # dim just above the spinner while ``thinking`` and cleared once the turn
        # completes (the final reply replaces it). Never persisted or part of the
        # saved transcript.
        self.interim = ""
        # Percent of the context window used by the last request (None = unknown).
        self.context_pct: float | None = None
        # When True the input line is rendered masked (e.g. while typing an API
        # key during the guided /model add flow). The stored text is untouched.
        self.masked = False
        # When set, the bottom input line is replaced by this confirmation
        # question (e.g. "cancel request and toggle panel? (y/n)") until the
        # user answers it. The typed input buffer is left untouched.
        self.confirm_prompt = ""
        # When set, the bottom input line accepts a steering instruction using
        # this prompt instead of the normal ludvart prompt.
        self.steer_prompt = ""
        # Seconds the current activity has been waiting, appended to the spinner
        # label (e.g. "Thinking (openai) - 8s") so a long, silent wait for a tool
        # or the next model response is visibly progressing. None hides it.
        self.activity_elapsed: float | None = None

    # -- state mutation ------------------------------------------------------

    def set_cols(self, cols: int) -> None:
        self.cols = max(1, cols)

    @property
    def messages(self) -> list[tuple[str, str]]:
        return self._messages

    def restore(self, messages: list[tuple[str, str]]) -> None:
        """Repopulate the transcript (e.g. after re-opening the panel)."""
        self._messages = list(messages)
        self.scroll = 0

    def add_user(self, text: str) -> None:
        self._messages.append(("you", text))
        self.scroll = 0

    def add_reply(self, text: str) -> None:
        self._messages.append(("ludvart", text))
        self.scroll = 0

    def add_info(self, text: str) -> None:
        self._messages.append(("info", text))
        self.scroll = 0

    def add_summary(self, text: str) -> None:
        """Mark a context-compaction point in the transcript.

        The transcript keeps the full human-readable conversation, but the
        model-facing context is purged and reseeded from this summary. The line
        is persisted so a reloaded session shows where compaction happened.
        """
        self._messages.append(("summary", text))
        self.scroll = 0

    def add_system(self, text: str) -> None:
        """Add an ephemeral in-panel note (slash-command echo/output).

        System messages are shown like info lines but are never persisted to the
        saved conversation nor sent to the LLM.
        """
        self._messages.append(("system", text))
        self.scroll = 0

    def add_system_row(self, text: str) -> None:
        """Add a system line that is clipped to the width instead of wrapped.

        For tabular output such as ``/sessions list``, where one item per row is
        what makes the list scannable and the tail of a long line is the least
        interesting part of it.
        """
        self._messages.append(("row", text))
        self.scroll = 0

    def type_text(self, text: str) -> None:
        self.editor.insert(text)
        self.scroll = 0

    def backspace(self) -> None:
        self.editor.backspace()

    def take_input(self) -> str:
        return self.editor.take()

    def scroll_up(self, n: int) -> None:
        self.scroll += n

    def scroll_down(self, n: int) -> None:
        self.scroll = max(0, self.scroll - n)

    # -- rendering -----------------------------------------------------------

    def _prompt_prefix(self) -> str:
        """A dim badge shown before the prompt, e.g. "[45%] " for context use.

        Empty string when the context usage is unknown.
        """
        if self.context_pct is None:
            return ""
        return f"[{self.context_pct:.0f}%] "

    def _input_view(
        self, prompt: str = _PROMPT, badge: bool = True
    ) -> tuple[str, int]:
        """Return the visible slice of the input and the 1-based cursor column.

        The input is a single line that scrolls horizontally so the cursor is
        always visible even when the text is wider than the panel.
        """
        prefix_w = len(self._prompt_prefix()) if badge else 0
        avail = max(1, self.cols - len(prompt) - prefix_w)
        text = self.editor.text
        cur = self.editor.cursor
        if self.masked:
            # Render the key as asterisks; same length so the cursor math holds.
            text = "*" * len(text)
        if len(text) <= avail:
            start = 0
        else:
            start = max(0, min(cur - avail + 1, len(text) - avail))
            if cur < start:
                start = cur
        visible = text[start : start + avail]
        col = min(self.cols, prefix_w + len(prompt) + (cur - start) + 1)
        return visible, col

    def cursor_col(self) -> int:
        """1-based column of the input cursor on the panel's input row."""
        if self.confirm_prompt:
            return min(self.cols, len(self.confirm_prompt) + 1)
        if self.steer_prompt:
            return self._input_view(prompt=self.steer_prompt, badge=False)[1]
        if self.masked:
            return self._input_view()[1]
        return self._input_block()[2]

    def cursor_rowcol(self) -> tuple[int, int]:
        """The cursor as a (row within the panel, 1-based column) pair.

        The input block is the last thing the panel draws, so its rows sit at
        the bottom whatever height the panel currently has.
        """
        rows = 1
        if not (self.confirm_prompt or self.steer_prompt or self.masked):
            rows = len(self._input_block()[0])
        rows = max(1, min(rows, self.height - 1))
        row = self.height - rows
        if not (self.confirm_prompt or self.steer_prompt or self.masked):
            row += min(self._input_block()[1], rows - 1)
        return row, self.cursor_col()

    def _input_block(self) -> tuple[list[str], int, int, list[int]]:
        """Wrap the input into rows, with the cursor's position and row offsets.

        The offsets are the buffer index each row starts at, which is what the
        selection is measured in. A long logical line wraps rather than scrolling
        sideways: with several lines in the buffer there is no single row to
        scroll, and seeing the whole of what you pasted is the point of accepting
        a paste at all.
        """
        prefix_w = len(self._prompt_prefix())
        indent = prefix_w + len(_PROMPT)
        avail = max(1, self.cols - indent)
        cur_line, cur_col = self.editor.cursor_line_col()
        rows: list[str] = []
        starts: list[int] = []
        cursor_row = 0
        cursor_x = 0
        off = 0
        for i, line in enumerate(self.editor.lines):
            segs = [line[j:j + avail] for j in range(0, len(line), avail)]
            if not segs:
                segs = [""]
            elif i == cur_line and cur_col == len(line) and cur_col % avail == 0:
                # The cursor sits just past a segment that filled the width, so
                # it belongs at the start of the next row rather than on top of
                # the last character of this one.
                segs.append("")
            if i == cur_line:
                seg = min(cur_col // avail, len(segs) - 1)
                cursor_row = len(rows) + seg
                cursor_x = cur_col - seg * avail
            starts += [off + j * avail for j in range(len(segs))]
            rows += segs
            off += len(line) + 1
        limit = max(1, min(INPUT_MAX_ROWS, self.height - 2))
        if len(rows) > limit:
            top = min(max(0, cursor_row - limit + 1), len(rows) - limit)
            rows = rows[top:top + limit]
            starts = starts[top:top + limit]
            cursor_row -= top
        return rows, cursor_row, min(self.cols, indent + cursor_x + 1), starts

    def _highlight(self, row: str, start: int) -> bytes:
        """Encode one input row, reverse-video over any selected part of it."""
        span = self.editor.selection()
        if span is None:
            return row.encode("utf-8", "replace")
        end = start + len(row)
        # A selected newline belongs to no row, so mark it with a trailing
        # block; without it a selection over blank lines would be invisible.
        tail = b""
        if span[0] <= end < span[1] and self.editor.text[end:end + 1] == "\n":
            tail = _REVERSE + b" " + _RESET
        a = max(span[0] - start, 0)
        b = min(span[1] - start, len(row))
        if b <= a:
            return row.encode("utf-8", "replace") + tail
        return (row[:a].encode("utf-8", "replace")
                + _REVERSE + row[a:b].encode("utf-8", "replace") + _RESET
                + row[b:].encode("utf-8", "replace") + tail)

    def _content_lines(self) -> list[bytes]:
        lines: list[bytes] = []
        for kind, text in self._messages:
            logical = text.split("\n")
            if kind == "you":
                segs: list[str] = []
                for para in logical:
                    segs += _wrap(para, max(1, self.cols - 2))
                for i, seg in enumerate(segs):
                    prefix = b"> " if i == 0 else b"  "
                    lines.append(
                        _CYAN + prefix + seg.encode("utf-8", "replace") + _RESET + _EOL
                    )
            elif kind == "info":
                for para in logical:
                    for seg in _wrap(para, self.cols):
                        lines.append(
                            _DIM + seg.encode("utf-8", "replace") + _RESET + _EOL
                        )
            elif kind == "system":
                for para in logical:
                    for seg in _wrap(para, self.cols):
                        lines.append(
                            _CYAN + _DIM + seg.encode("utf-8", "replace")
                            + _RESET + _EOL
                        )
            elif kind == "row":
                for para in logical:
                    lines.append(
                        _CYAN + _DIM
                        + _clip(para, self.cols).encode("utf-8", "replace")
                        + _RESET + _EOL
                    )
            elif kind == "summary":
                header = "\u2500\u2500 context compacted \u00b7 summary \u2500\u2500"
                lines.append(
                    _DIM + _CYAN + header.encode("utf-8", "replace") + _RESET + _EOL
                )
                for para in logical:
                    for seg in _wrap(para, self.cols):
                        lines.append(
                            _DIM + seg.encode("utf-8", "replace") + _RESET + _EOL
                        )
            else:
                for para in logical:
                    for seg in _wrap(para, self.cols):
                        lines.append(seg.encode("utf-8", "replace") + _RESET + _EOL)
        if self.interim:
            for para in self.interim.split("\n"):
                for seg in _wrap(para, self.cols):
                    lines.append(_DIM + seg.encode("utf-8", "replace") + _RESET + _EOL)
        if self.thinking:
            dots = _THINK_FRAMES[self.tick % len(_THINK_FRAMES)]
            base = self.activity
            if self.provider and base == "Thinking":
                base = f"Thinking ({self.provider})"
            if self.activity_elapsed is not None:
                base = f"{base} - {self.activity_elapsed:.0f}s"
            label = f"{base}{dots}"
            for seg in _wrap(label, self.cols):
                lines.append(_DIM + seg.encode("utf-8", "replace") + _RESET + _EOL)
        return lines

    def _header(self, more_above: int) -> bytes:
        label = f" ludvart · {self.provider} " if self.provider else " ludvart "
        if self.thinking:
            label += f"· {self.activity} "
        hints = "^O/Esc:close  M-Enter:newline  S-arrows/^Space:select  PgUp/Dn:scroll "
        if self.editor.mark:
            hints = "MARK - move to select, ^Space cancels  "
        if more_above > 0:
            hints = f"\u2191{more_above} more  " + hints
        text = (label + "· " + hints)[: self.cols].ljust(self.cols)
        return _REVERSE + text.encode("utf-8", "replace") + _RESET + _EOL

    def _input_line(self) -> list[bytes]:
        """The input block, one payload per row it occupies."""
        if self.confirm_prompt:
            text = self.confirm_prompt[: self.cols]
            return [_BOLD + _CYAN + text.encode("utf-8", "replace") + _RESET + _EOL]
        if self.steer_prompt:
            visible = self._input_view(prompt=self.steer_prompt, badge=False)[0]
            return [
                _BOLD + _CYAN + self.steer_prompt.encode("utf-8", "replace")
                + _RESET + visible.encode("utf-8", "replace") + _EOL
            ]
        prefix = self._prompt_prefix()
        badge = _DIM + prefix.encode("ascii") + _RESET if prefix else b""
        if self.masked:
            visible = self._input_view()[0]
            return [badge + _CYAN + _PROMPT.encode("ascii") + _RESET
                    + visible.encode("utf-8", "replace") + _EOL]
        rows, _, _, starts = self._input_block()
        out = []
        for i, row in enumerate(rows):
            if i == 0:
                head = badge + _CYAN + _PROMPT.encode("ascii") + _RESET
            else:
                # Continuations line up under the first row's text so a pasted
                # block reads as one thing rather than a stack of prompts.
                head = b" " * (len(prefix) + len(_PROMPT))
            out.append(head + self._highlight(row, starts[i]) + _EOL)
        return out

    def render(self, height: int, cols: int) -> list[bytes]:
        """Return exactly ``height`` drawable row payloads for the panel."""
        self.set_cols(cols)
        self.height = height
        input_rows = self._input_line()[: max(1, height - 2)]
        content_h = max(1, height - 1 - len(input_rows))

        lines = self._content_lines()
        max_start = max(0, len(lines) - content_h)
        self.scroll = max(0, min(self.scroll, max_start))
        start = max_start - self.scroll
        window = lines[start:start + content_h]
        while len(window) < content_h:
            window.append(_RESET + _EOL)

        rows = [self._header(start)]
        rows += window
        rows += input_rows
        if len(rows) > height:
            rows = rows[:height]
        while len(rows) < height:
            rows.append(_RESET + _EOL)
        return rows
