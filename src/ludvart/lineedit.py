"""A minimal, self-contained text editor buffer.

This deliberately avoids any terminal/UI concerns: it is just a string plus a
cursor position and the edit operations a line editor needs (insert, delete,
cursor movement, selection, word/line kills). The ludvart panel feeds it decoded
key events and renders it; nothing here reads or writes the terminal.

The text may contain newlines, in which case it is several logical lines and the
cursor moves between them; wrapping a long logical line across the width of the
panel is the renderer's business, not this module's.

The logic is intentionally simple and index-based (one index == one Unicode
code point, matching Python ``str`` indexing) so it maps directly onto a Rust
``Vec<char>`` if this is ported later. No regex, no external dependencies.
"""

from __future__ import annotations


class LineEditor:
    """Editable text with an insertion cursor and an optional selection.

    ``cursor`` is the code-point index in ``text`` where the next insertion
    happens; it ranges over ``0..=len(text)``. ``anchor`` is where a shifted
    movement started, so the selection is whatever lies between the two.
    """

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.cursor = len(text)
        self.anchor: int | None = None
        self.mark = False

    # -- selection -----------------------------------------------------------

    def toggle_mark(self) -> bool:
        """Turn a sticky selection anchor on or off; True if it is now on.

        Plenty of terminals never report a modifier, so a shifted arrow reaches
        us as a plain one and selecting would be impossible. With the mark set,
        ordinary movement extends the selection instead.
        """
        self.mark = not self.mark
        self.anchor = self.cursor if self.mark else None
        return self.mark

    def selection(self) -> tuple[int, int] | None:
        """The selected range as (start, end) offsets, or None if empty."""
        if self.anchor is None or self.anchor == self.cursor:
            return None
        return min(self.anchor, self.cursor), max(self.anchor, self.cursor)

    def selected_text(self) -> str:
        span = self.selection()
        return "" if span is None else self.text[span[0] : span[1]]

    def delete_selection(self) -> bool:
        """Remove the selected range; False if there was nothing selected."""
        span = self.selection()
        self.anchor = None
        self.mark = False
        if span is None:
            return False
        self.text = self.text[: span[0]] + self.text[span[1] :]
        self.cursor = span[0]
        return True

    def _mark(self, select: bool) -> None:
        """Start, keep or drop the anchor ahead of a cursor move."""
        if not (select or self.mark):
            self.anchor = None
        elif self.anchor is None:
            self.anchor = self.cursor

    # -- editing -------------------------------------------------------------

    def insert(self, s: str) -> None:
        """Insert ``s`` at the cursor, replacing the selection if there is one."""
        if not s:
            return
        self.delete_selection()
        self.text = self.text[: self.cursor] + s + self.text[self.cursor :]
        self.cursor += len(s)

    def backspace(self) -> None:
        """Delete the selection, or the character before the cursor."""
        if self.delete_selection():
            return
        if self.cursor > 0:
            self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
            self.cursor -= 1

    def delete(self) -> None:
        """Delete the selection, or the character under the cursor."""
        if self.delete_selection():
            return
        if self.cursor < len(self.text):
            self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]

    def delete_word_back(self) -> None:
        """Delete the whitespace-delimited word before the cursor (Ctrl-W)."""
        if self.delete_selection():
            return
        i = self.cursor
        while i > 0 and self.text[i - 1].isspace():
            i -= 1
        while i > 0 and not self.text[i - 1].isspace():
            i -= 1
        self.text = self.text[:i] + self.text[self.cursor :]
        self.cursor = i

    def kill_to_start(self) -> None:
        """Delete from the start of the current line up to the cursor (Ctrl-U)."""
        if self.delete_selection():
            return
        start = self.line_bounds()[0]
        self.text = self.text[:start] + self.text[self.cursor :]
        self.cursor = start

    def kill_to_end(self) -> None:
        """Delete from the cursor to the end of the current line (Ctrl-K)."""
        if self.delete_selection():
            return
        end = self.line_bounds()[1]
        self.text = self.text[: self.cursor] + self.text[end:]

    # -- lines ---------------------------------------------------------------

    def line_bounds(self, index: int | None = None) -> tuple[int, int]:
        """Offsets bounding the logical line that contains ``index``."""
        i = self.cursor if index is None else index
        start = self.text.rfind("\n", 0, i) + 1
        end = self.text.find("\n", i)
        return start, len(self.text) if end == -1 else end

    @property
    def lines(self) -> list[str]:
        return self.text.split("\n")

    def cursor_line_col(self) -> tuple[int, int]:
        """The cursor as a (logical line index, column) pair, both 0-based."""
        return self.text.count("\n", 0, self.cursor), self.cursor - self.line_bounds()[0]

    def up(self, select: bool = False) -> bool:
        """Move to the same column one line up; False if already on the first."""
        self._mark(select)
        start = self.line_bounds()[0]
        if start == 0:
            return False
        col = self.cursor - start
        prev_start = self.text.rfind("\n", 0, start - 1) + 1
        self.cursor = min(prev_start + col, start - 1)
        return True

    def down(self, select: bool = False) -> bool:
        """Move to the same column one line down; False if already on the last."""
        self._mark(select)
        start, end = self.line_bounds()
        if end >= len(self.text):
            return False
        col = self.cursor - start
        self.cursor = min(end + 1 + col, self.line_bounds(end + 1)[1])
        return True

    # -- cursor movement -----------------------------------------------------

    def left(self, select: bool = False) -> None:
        span = self.selection()
        self._mark(select)
        if span is not None and not (select or self.mark):
            self.cursor = span[0]  # an unshifted arrow collapses the selection
        elif self.cursor > 0:
            self.cursor -= 1

    def right(self, select: bool = False) -> None:
        span = self.selection()
        self._mark(select)
        if span is not None and not (select or self.mark):
            self.cursor = span[1]
        elif self.cursor < len(self.text):
            self.cursor += 1

    def home(self, select: bool = False) -> None:
        """Move to the start of the current line."""
        self._mark(select)
        self.cursor = self.line_bounds()[0]

    def end(self, select: bool = False) -> None:
        """Move to the end of the current line."""
        self._mark(select)
        self.cursor = self.line_bounds()[1]

    # -- whole-buffer --------------------------------------------------------

    def select_all(self) -> None:
        self.anchor = 0
        self.cursor = len(self.text)

    def set_text(self, text: str) -> None:
        """Replace the whole buffer with ``text`` and put the cursor at its end."""
        self.text = text
        self.cursor = len(text)
        self.anchor = None
        self.mark = False

    def clear(self) -> None:
        self.text = ""
        self.cursor = 0
        self.anchor = None
        self.mark = False

    def take(self) -> str:
        """Return the trimmed text and reset the buffer to empty."""
        value = self.text.strip()
        self.clear()
        return value
