"""Bracketed-paste routing test for the AI panel input.

Exercises Ludvart._panel_input directly (no PTY): a paste split across reads and
containing a newline must be inserted verbatim -- newlines and all -- into the
editor without submitting, and trailing bytes after the end marker are processed
as normal keys.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_ai_paste.py
"""

from ludvart.ludvart import Ludvart, _PASTE_START, _PASTE_END
from ludvart.panel import AiPanel


def make_ludvart():
    r = Ludvart(["true"])
    r._panel = AiPanel(cols=40, height=8, provider="test")
    r._panel_pasting = False
    r._panel_pastebuf = bytearray()
    return r


def test_paste_single_read():
    r = make_ludvart()
    r._panel_input(_PASTE_START + b"hello world" + _PASTE_END)
    assert r._panel.editor.text == "hello world", r._panel.editor.text
    assert not r._panel_pasting
    print("paste single read: OK")


def test_paste_with_newline_no_submit():
    """A pasted newline is text to keep, not an Enter to obey.

    Bracketed paste is what makes the distinction safe: the terminal has already
    told us these bytes are content, so the snippet can keep the shape it was
    written in and still be edited before it is sent.
    """
    r = make_ludvart()
    submitted = []
    r._panel_submit = lambda: submitted.append(True)
    r._panel_input(_PASTE_START + b"line1\r\nline2\rline3" + _PASTE_END)
    assert r._panel.editor.text == "line1\nline2\nline3", r._panel.editor.text
    assert submitted == [], "paste must not submit"
    # Control characters that are not newlines still cannot break the layout.
    r._panel.editor.clear()
    r._panel_input(_PASTE_START + b"a\tb\x00c" + _PASTE_END)
    assert r._panel.editor.text == "a b c", r._panel.editor.text
    print("paste newline no submit: OK")


def test_paste_split_across_reads():
    r = make_ludvart()
    # Start marker + first chunk in one read, remainder + end marker in another,
    # with the END marker itself split across the two reads.
    part1 = _PASTE_START + b"abc"
    mid = _PASTE_END[:2]  # split the end marker
    part2 = _PASTE_END[2:] + b"Z"  # rest of marker, then a normal keystroke
    r._panel_input(part1)
    assert r._panel_pasting
    r._panel_input(b"def" + mid)
    assert r._panel_pasting  # end marker not complete yet
    r._panel_input(part2)
    assert not r._panel_pasting
    assert r._panel.editor.text == "abcdefZ", r._panel.editor.text
    print("paste split across reads: OK")


def test_prefix_before_paste():
    r = make_ludvart()
    r._panel_input(b"hi " + _PASTE_START + b"there" + _PASTE_END)
    assert r._panel.editor.text == "hi there", r._panel.editor.text
    print("text before paste: OK")


def test_enter_submits_but_alt_enter_opens_a_line():
    """Enter keeps its meaning; the newline needs a key of its own.

    Enter is muscle memory and the only way to send a question, so the newline
    is bound to Alt-Enter instead. Ctrl-J was rejected as the binding because
    some terminals send \\n for Enter itself, which would make Enter ambiguous.
    """
    r = make_ludvart()
    submitted = []
    r._panel_submit = lambda: submitted.append(True)
    r._panel_input(b"one")
    r._panel_input(b"\x1b\r")
    r._panel_input(b"two")
    assert r._panel.editor.text == "one\ntwo", r._panel.editor.text
    assert submitted == [], "Alt-Enter must not submit"
    r._panel_input(b"\r")
    assert submitted == [True], "Enter must still submit"
    print("enter submits, alt-enter opens a line: OK")


def test_a_multiline_question_reaches_the_llm_intact():
    """Shape is meaning in a pasted snippet, so it must survive submission."""
    r = make_ludvart()
    asked = []
    r._start_ask = lambda q, user_echo=None: asked.append(q)
    r._panel.editor.set_text("why does this fail?\n\n    def f(a, b):\n        return a + b")
    r._panel_submit()
    assert asked == ["why does this fail?\n\n    def f(a, b):\n        return a + b"], asked
    print("multiline question reaches the llm intact: OK")


def test_every_shifted_arrow_extends_the_selection():
    """Check each binding on its own.

    A table of near-identical escape sequences is exactly where a typo survives
    a test that only exercises one of them, so every sequence gets its own case.
    Both the xterm and the older rxvt forms are bound, since which one arrives
    depends on the terminal rather than on ludvart.
    """
    # "ab\ncd": a=0 b=1 \n=2 c=3 d=4
    cases = [
        (b"\x1b[1;2D", 5, "d"),      # Shift-Left
        (b"\x1b[d",    5, "d"),
        (b"\x1b[1;2C", 0, "a"),      # Shift-Right
        (b"\x1b[c",    0, "a"),
        (b"\x1b[1;2A", 4, "b\nc"),   # Shift-Up
        (b"\x1b[a",    4, "b\nc"),
        (b"\x1b[1;2B", 1, "b\nc"),   # Shift-Down
        (b"\x1b[b",    1, "b\nc"),
        (b"\x1b[1;2H", 5, "cd"),     # Shift-Home
        (b"\x1b[1;2~", 5, "cd"),
        (b"\x1b[7$",   5, "cd"),
        (b"\x1b[1;2F", 3, "cd"),     # Shift-End
        (b"\x1b[4;2~", 3, "cd"),
        (b"\x1b[8$",   3, "cd"),
    ]
    for key, start, want in cases:
        r = make_ludvart()
        r._panel.editor.set_text("ab\ncd")
        r._panel.editor.cursor = start
        r._panel_input(key)
        got = r._panel.editor.selected_text()
        assert got == want, (key, got, want)
    print("every shifted arrow extends the selection: OK")


def test_shift_arrows_select_a_block_that_one_key_then_deletes():
    """Selecting a pasted block and dropping it should not be a Backspace vigil."""
    for up, home in ((b"\x1b[1;2A", b"\x1b[1;2H"), (b"\x1b[a", b"\x1b[7$")):
        r = make_ludvart()
        r._panel_input(_PASTE_START + b"keep me\nline one\nline two" + _PASTE_END)
        r._panel_input(up)    # from the end of the last line to the one above
        r._panel_input(home)  # ...and back to its start
        assert r._panel.editor.selected_text() == "line one\nline two", (
            r._panel.editor.selected_text())
        r._panel_input(b"\x7f")  # Backspace
        assert r._panel.editor.text == "keep me\n", r._panel.editor.text

    # A shifted arrow at the top of the input selects rather than scrolling:
    # the user is selecting, not navigating.
    r = make_ludvart()
    r._panel.scroll = 0
    r._panel_input(b"one")
    r._panel_input(b"\x1b[1;2A")
    assert r._panel.scroll == 0 and r._panel.editor.selection() is None
    print("shift arrows select a block: OK")


def test_the_mark_lets_plain_arrows_select():
    """Not every terminal reports a modifier, and a shifted arrow then arrives
    as a plain one; without a mark those users could not select at all."""
    r = make_ludvart()
    r._panel.scroll = 3
    r._panel_input(_PASTE_START + b"keep me\nline one\nline two" + _PASTE_END)
    r._panel_input(b"\x00")  # Ctrl-Space sets the mark
    assert r._panel.editor.mark
    assert b"MARK" in r._panel._header(0)

    r._panel_input(b"\x1b[A")  # a *plain* Up now extends the selection
    r._panel_input(b"\x1b[H")  # ...and a plain Home too
    assert r._panel.editor.selected_text() == "line one\nline two", (
        r._panel.editor.selected_text())
    r._panel_input(b"\x7f")
    assert r._panel.editor.text == "keep me\n", r._panel.editor.text
    # Deleting the block leaves the mark off, so movement is movement again.
    assert not r._panel.editor.mark
    assert b"MARK" not in r._panel._header(0)
    r._panel_input(b"\x1b[A")
    assert r._panel.editor.selection() is None

    # Repeated movement keeps extending: the mark holds the selection open, so
    # a plain arrow must not take the branch that collapses it to its edge.
    r._panel.editor.set_text("abcd")
    r._panel_input(b"\x00")
    for _ in range(3):
        r._panel_input(b"\x1b[D")
    assert r._panel.editor.selected_text() == "bcd", r._panel.editor.selected_text()
    r._panel_input(b"\x1b[C")
    assert r._panel.editor.selected_text() == "cd", r._panel.editor.selected_text()
    r._panel_input(b"\x00")  # cancel the mark again

    # Ctrl-Space toggles, and a marked Up at the top selects rather than
    # scrolling the transcript away under the user.
    r._panel_input(b"\x00")
    assert r._panel.editor.mark
    r._panel_input(b"\x00")
    assert not r._panel.editor.mark and r._panel.editor.selection() is None
    r._panel.scroll = 0
    r._panel.editor.set_text("solo")
    r._panel_input(b"\x00")
    r._panel_input(b"\x1b[A")
    assert r._panel.scroll == 0, r._panel.scroll
    print("the mark lets plain arrows select: OK")


if __name__ == "__main__":
    test_paste_single_read()
    test_paste_with_newline_no_submit()
    test_paste_split_across_reads()
    test_prefix_before_paste()
    test_enter_submits_but_alt_enter_opens_a_line()
    test_a_multiline_question_reaches_the_llm_intact()
    test_every_shifted_arrow_extends_the_selection()
    test_shift_arrows_select_a_block_that_one_key_then_deletes()
    test_the_mark_lets_plain_arrows_select()
    print("all paste tests passed")
