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


if __name__ == "__main__":
    test_paste_single_read()
    test_paste_with_newline_no_submit()
    test_paste_split_across_reads()
    test_prefix_before_paste()
    test_enter_submits_but_alt_enter_opens_a_line()
    test_a_multiline_question_reaches_the_llm_intact()
    print("all paste tests passed")
