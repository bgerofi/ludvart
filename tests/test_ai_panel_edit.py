"""Unit tests for the panel line editor and bracketed-paste input handling.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_ai_panel_edit.py
"""

from ludvart.lineedit import LineEditor
from ludvart.panel import _RESET, _REVERSE, INPUT_MAX_ROWS, AiPanel


def test_line_editor():
    ed = LineEditor()
    ed.insert("hello")
    assert ed.text == "hello" and ed.cursor == 5

    # move left twice and insert in the middle
    ed.left(); ed.left()
    assert ed.cursor == 3
    ed.insert("XY")
    assert ed.text == "helXYlo" and ed.cursor == 5

    # backspace removes before cursor
    ed.backspace()
    assert ed.text == "helXlo" and ed.cursor == 4

    # forward delete removes under cursor
    ed.delete()
    assert ed.text == "helXo" and ed.cursor == 4

    # home / end
    ed.home(); assert ed.cursor == 0
    ed.end(); assert ed.cursor == 5

    # right at end is a no-op; left at start is a no-op
    ed.right(); assert ed.cursor == 5
    ed.home(); ed.left(); assert ed.cursor == 0

    # kill to end / start
    ed2 = LineEditor("one two three")
    ed2.home()
    for _ in range(4):
        ed2.right()  # cursor after "one "
    ed2.kill_to_start()
    assert ed2.text == "two three" and ed2.cursor == 0
    ed2.end(); ed2.left(); ed2.left()  # inside "three"
    ed2.kill_to_end()
    assert ed2.text == "two thr"

    # delete word back
    ed3 = LineEditor("foo bar baz")
    ed3.delete_word_back()
    assert ed3.text == "foo bar " and ed3.cursor == len("foo bar ")
    ed3.delete_word_back()
    assert ed3.text == "foo " and ed3.cursor == 4

    # take strips and clears
    ed4 = LineEditor("  spaced  ")
    assert ed4.take() == "spaced"
    assert ed4.text == "" and ed4.cursor == 0
    print("line editor: OK")


def test_input_view_scroll():
    panel = AiPanel(cols=20, height=8, provider="test")
    # prompt "ludvart> " is 9 chars -> avail = 11
    panel.editor.insert("abcdefghij")  # 10 chars, fits
    visible, col = panel._input_view()
    assert visible == "abcdefghij"
    assert col == 9 + 10 + 1  # after last char (capped at cols=20)

    panel.editor.insert("klmnopqrst")  # now 20 chars, wider than avail=11
    visible, col = panel._input_view()
    assert len(visible) == 11
    # cursor is at end -> window shows the tail, cursor clamps to last column
    assert visible.endswith("t")
    assert col == 20

    # move to home: window should show the head, cursor at first input col
    panel.editor.home()
    visible, col = panel._input_view()
    assert visible.startswith("a")
    assert col == 10  # len(prompt)+1
    print("input view scroll: OK")


def test_editor_moves_between_lines():
    """The buffer is a document now, so home/end/kill act on one line of it."""
    ed = LineEditor("alpha\nbetas\ngamma")
    ed.cursor = 0
    assert ed.cursor_line_col() == (0, 0)
    assert ed.down() and ed.cursor_line_col() == (1, 0)
    ed.end(); assert ed.cursor_line_col() == (1, 5)
    ed.home(); assert ed.cursor_line_col() == (1, 0)

    # A column is kept across the move, clamped by a shorter line.
    ed.set_text("longer line\nab\ntail")
    ed.cursor = 9  # on the first line, past the width of the second
    assert ed.down() and ed.cursor_line_col() == (1, 2)

    # The ends of the document report no further line to move to.
    ed.cursor = 0
    assert ed.up() is False and ed.cursor == 0
    ed.cursor = len(ed.text)
    assert ed.down() is False and ed.cursor == len(ed.text)

    # Ctrl-U/Ctrl-K stay inside the current line.
    ed.set_text("one\ntwo three\nfour")
    ed.cursor = ed.text.index("three")
    ed.kill_to_end()
    assert ed.text == "one\ntwo \nfour", ed.text
    ed.kill_to_start()
    assert ed.text == "one\n\nfour", ed.text

    # Ctrl-W crosses a newline rather than stalling on it.
    ed.set_text("word\n")
    ed.cursor = len(ed.text)
    ed.delete_word_back()
    assert ed.text == "" and ed.cursor == 0
    print("editor moves between lines: OK")


def test_input_block_wraps_and_bounds_its_height():
    panel = AiPanel(cols=20, height=8, provider="test")
    # prompt is 9 wide -> 11 columns of text per row
    panel.editor.set_text("abc\ndefghijklmnopqrstuv")
    rows, cur_row, cur_col, starts = panel._input_block()
    assert rows == ["abc", "defghijklmn", "opqrstuv"], rows
    assert cur_row == 2 and cur_col == 9 + len("opqrstuv") + 1
    # Each row knows the buffer offset it starts at, which is what the selection
    # is measured in; "abc\n" is 4 characters, so the wrapped line starts at 4.
    assert starts == [0, 4, 15], starts
    for row, off in zip(rows, starts):
        assert panel.editor.text[off:off + len(row)] == row

    # The block never crowds the transcript out of the panel: in a short panel
    # the panel's own height is the binding limit...
    panel.editor.set_text("\n".join(str(i) for i in range(40)))
    rows, cur_row, _, starts = panel._input_block()
    assert len(rows) == panel.height - 2 == len(starts), len(rows)
    assert rows[cur_row] == "39", rows  # the cursor's row is the one kept
    assert panel.cursor_rowcol()[0] == panel.height - 1

    # ...and in a tall one INPUT_MAX_ROWS is, so a big paste still leaves most
    # of the panel showing the conversation it is about.
    tall = AiPanel(cols=20, height=20, provider="test")
    tall.editor.set_text("\n".join(str(i) for i in range(40)))
    rows, cur_row, _, _ = tall._input_block()
    assert len(rows) == INPUT_MAX_ROWS, len(rows)
    assert rows[cur_row] == "39", rows

    # Rendering still yields exactly `height` rows, transcript squeezed to fit.
    drawn = panel.render(panel.height, panel.cols)
    assert len(drawn) == panel.height
    assert b"39" in drawn[-1]
    print("input block wraps and bounds its height: OK")


def test_a_masked_key_never_wraps_onto_a_second_row():
    """An API key is masked precisely so it does not sit on screen; wrapping it
    across rows would put the whole of it there in asterisks and, worse, leave
    part behind when the block shrank back."""
    panel = AiPanel(cols=20, height=8, provider="test")
    panel.masked = True
    panel.editor.set_text("k" * 60)
    assert len(panel._input_line()) == 1
    assert panel.cursor_rowcol()[0] == panel.height - 1
    print("a masked key stays on one row: OK")


def test_shift_movement_selects_and_editing_replaces_the_selection():
    ed = LineEditor("alpha\nbeta\ngamma")
    assert ed.selection() is None

    # Shift-Home from the end of the last line selects that line's text.
    ed.home(select=True)
    assert ed.selected_text() == "gamma"
    # An unshifted arrow collapses the selection instead of moving off its edge.
    ed.right()
    assert ed.selection() is None and ed.cursor == len(ed.text)

    # Shift-Up spans the newline between two lines.
    ed.home()
    ed.up(select=True)
    ed.home(select=True)
    assert ed.selected_text() == "beta\n"
    ed.backspace()
    assert ed.text == "alpha\ngamma" and ed.cursor == 6, (ed.text, ed.cursor)
    assert ed.selection() is None

    # Typing over a selection replaces it, as it does in any GUI editor.
    ed.set_text("keep this\ndrop that")
    ed.home()
    ed.end(select=True)
    ed.insert("kept")
    assert ed.text == "keep this\nkept", ed.text

    # An anchor left where the cursor already is is not a selection.
    ed.set_text("x")
    ed.left(select=True)
    ed.right(select=True)
    assert ed.selection() is None
    print("shift movement selects and editing replaces it: OK")


def test_a_selection_is_shown_in_reverse_video():
    """The block is only deletable-with-confidence if you can see what it is."""
    panel = AiPanel(cols=20, height=8, provider="test")
    panel.editor.set_text("abc\n\ndef")
    panel.editor.home()
    for _ in range(3):
        panel.editor.up(select=True)
    panel.editor.home(select=True)
    assert panel.editor.selected_text() == "abc\n\n"

    rows = panel._input_line()
    assert _REVERSE + b"abc" + _RESET in rows[0], rows[0]
    # The two selected newlines are in no row at all, so each is drawn as a
    # trailing block; without that the blank line would look unselected.
    assert rows[0].count(_REVERSE) == 2, rows[0]
    assert _REVERSE + b" " + _RESET in rows[1], rows[1]
    assert _REVERSE not in rows[2], rows[2]

    panel.editor.anchor = None
    assert all(_REVERSE not in r for r in panel._input_line())
    print("a selection is shown in reverse video: OK")


if __name__ == "__main__":
    test_line_editor()
    test_input_view_scroll()
    test_editor_moves_between_lines()
    test_input_block_wraps_and_bounds_its_height()
    test_a_masked_key_never_wraps_onto_a_second_row()
    test_shift_movement_selects_and_editing_replaces_the_selection()
    test_a_selection_is_shown_in_reverse_video()
    print("all panel-edit tests passed")
