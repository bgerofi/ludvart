"""A list the user arrows through, drawn in place of the panel transcript.

Choosing a profile, a session or a model all mean the same thing to the user --
"show me what there is and let me point at one" -- so one widget serves all
three. It holds only what a list needs (the items, where the cursor is, how far
the window has scrolled) and renders itself to panel rows; who fetched the items
and what happens to the chosen one are the panel's business, not its own.
"""

from __future__ import annotations

_RESET = b"\x1b[0m"
_EOL = b"\x1b[K"
_REVERSE = b"\x1b[7m"
_DIM = b"\x1b[2m"

#: What the chosen item turns into, per kind: the backend command that applies
#: it. The picker itself never sends these; the panel does.
APPLY_COMMAND = {
    "profile": "profile use",
    "session": "session load",
    "model": "model use",
}

TITLES = {
    "profile": "Profiles",
    "session": "Sessions",
    "model": "Models",
}


def _clip(text: str, width: int) -> str:
    width = max(1, width)
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "\u2026"


class Picker:
    """One list of choices, with a cursor and a scrolled window."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.title = TITLES.get(kind, kind)
        self.items: list[dict] = []
        self.index = 0
        self.top = 0
        # Items arrive from the backend a round trip after the key was pressed,
        # so the list opens empty and says so rather than flashing into view.
        self.loading = True
        self.error = ""

    # -- state ---------------------------------------------------------------

    def set_items(self, items: list[dict]) -> None:
        """Adopt the fetched list, starting on whatever is already in use."""
        self.items = list(items)
        self.loading = False
        self.index = 0
        for i, item in enumerate(self.items):
            if item.get("active"):
                self.index = i
                break
        self.top = 0

    def fail(self, message: str) -> None:
        self.loading = False
        self.error = message

    def move(self, delta: int) -> None:
        if not self.items:
            return
        self.index = max(0, min(len(self.items) - 1, self.index + delta))

    def to_end(self, last: bool) -> None:
        if not self.items:
            return
        self.index = len(self.items) - 1 if last else 0

    def selected(self) -> dict | None:
        if not self.items or not 0 <= self.index < len(self.items):
            return None
        return self.items[self.index]

    # -- rendering -----------------------------------------------------------

    def _window(self, rows: int) -> list[dict]:
        """Scroll the window the least amount that keeps the cursor inside."""
        rows = max(1, rows)
        self.top = max(0, min(self.top, max(0, len(self.items) - rows)))
        if self.index < self.top:
            self.top = self.index
        elif self.index >= self.top + rows:
            self.top = self.index - rows + 1
        return self.items[self.top:self.top + rows]

    def render(self, rows: int, cols: int) -> list[bytes]:
        """Return exactly ``rows`` payloads showing the list."""
        rows = max(1, rows)
        if self.loading:
            body = [_DIM + b"  Loading..." + _RESET + _EOL]
        elif self.error:
            body = [_clip("  " + self.error, cols).encode("utf-8", "replace") + _EOL]
        elif not self.items:
            body = [_DIM + b"  Nothing to choose from." + _RESET + _EOL]
        else:
            body = []
            for offset, item in enumerate(self._window(rows)):
                chosen = self.top + offset == self.index
                mark = "*" if item.get("active") else " "
                text = _clip(f"{mark} {item.get('label', '')}", cols).ljust(cols)
                painted = text.encode("utf-8", "replace")
                if chosen:
                    body.append(_REVERSE + painted + _RESET + _EOL)
                else:
                    body.append(_RESET + painted + _EOL)
        while len(body) < rows:
            body.append(_RESET + _EOL)
        return body[:rows]

    def header(self, cols: int) -> str:
        """The title bar text, including where the cursor is in a long list."""
        where = ""
        if self.items:
            where = f" {self.index + 1}/{len(self.items)}"
        return _clip(f" {self.title}{where} ", cols)

    @staticmethod
    def hint(cols: int) -> str:
        full = " Up/Down:move  PgUp/PgDn:page  Enter:choose  Esc:cancel "
        if len(full) <= cols:
            return full
        # Paging is discoverable from the arrows; choosing and escaping are not.
        return _clip(" Up/Dn:move  Enter:choose  Esc:cancel ", cols)
