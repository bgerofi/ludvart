"""End-to-end: open the AI panel, verify bracketed paste and cursor editing land
in the prompt correctly through the real ludvart binary.

Drives a real PTY:
  1. Open the panel (Ctrl-O).
  2. Type "hello world".
  3. Bracketed-paste " PASTED\ntext" -> the newline is kept and nothing submits.
  4. Move the cursor left and insert a char mid-line.
  5. Alt-Enter opens a third line and typing continues on it.
  6. Up/Up + Home + Ctrl-K clears only the line the cursor is on.
Renders ludvart's output through pyte and checks the input block.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_ai_paste_e2e.py
"""

import errno, fcntl, os, pty, select, struct, termios, time
import pyte

from e2e_util import Checks, ludvart_argv, screen_text, wait_for, wait_until_started

ROWS, COLS = 24, 90
PASTE_START = b"\x1b[200~"
PASTE_END = b"\x1b[201~"


def pump(fd, stream, seconds):
    end = time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if fd in r:
            try:
                d = os.read(fd, 65536)
            except OSError as e:
                if e.errno == errno.EIO:
                    break
                raise
            if not d:
                break
            stream.feed(d)


def input_line(screen):
    """Return the panel input row (the one starting with 'ludvart> ')."""
    for row in screen.display:
        if "ludvart>" in row:
            return row.rstrip()
    return ""


def input_block(screen):
    """The input rows: the 'ludvart>' row and every row below it.

    Continuations carry no prompt of their own -- they are indented to line up
    under the first row's text -- so they are found by position, not by marker.
    """
    rows = [r.rstrip() for r in screen.display]
    for i, row in enumerate(rows):
        if "ludvart>" in row:
            return [r.strip() for r in rows[i:]]
    return []


def reversed_text(screen):
    """The characters drawn in reverse video within the input block.

    The panel header is reverse video too, so the scan starts at the input row.
    """
    rows = [r.rstrip() for r in screen.display]
    top = next((i for i, r in enumerate(rows) if "ludvart>" in r), len(rows))
    out = []
    for y in range(top, screen.lines):
        line = screen.buffer[y]
        out.append("".join(
            line[x].data for x in range(screen.columns) if line[x].reverse
        ).rstrip())
    return "".join(out)


def main():
    pid, m = pty.fork()
    if pid == 0:
        os.environ["PS1"] = "$ "
        os.environ["TERM"] = "xterm"
        argv = ludvart_argv()
        os.execvp(argv[0], argv)
    fcntl.ioctl(m, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)
    checks = Checks()

    checks.add("ludvart finished starting up", wait_until_started(m, stream.feed, screen))
    os.write(m, b"\x0f")  # Ctrl-O: open panel
    checks.add("panel opened", wait_for(m, stream.feed, lambda: "ludvart>" in screen_text(screen), 10, settle=0.3))

    def expect(name, want):
        """Wait for the input line to read ``want``, and record the check."""
        wait_for(m, stream.feed, lambda: input_line(screen).endswith(want), 5)
        line = input_line(screen)
        checks.add(name, line.endswith(want), f"line is {line!r}, wanted it to end with {want!r}")

    def expect_block(name, want):
        """Wait for the input block to read ``want``, and record the check."""
        wait_for(m, stream.feed, lambda: input_block(screen) == want, 5)
        got = input_block(screen)
        checks.add(name, got == want, f"block is {got!r}, wanted {want!r}")

    # 1. plain typing
    os.write(m, b"hello world")
    expect("typing appears in the input line", "hello world")

    # 2. bracketed paste with an embedded newline: kept as a newline, no submit
    os.write(m, PASTE_START + b" PASTED\ntext" + PASTE_END)
    expect_block(
        "a pasted newline stays a newline and does not submit",
        ["ludvart> hello world PASTED", "text"],
    )

    # 3. move left 4 (into "text") and insert 'Z'
    for _ in range(4):
        os.write(m, b"\x1b[D")  # Left
        pump(m, stream, 0.2)
    os.write(m, b"Z")
    expect_block(
        "editing mid-line inserts at the cursor",
        ["ludvart> hello world PASTED", "Ztext"],
    )

    # 4. Alt-Enter opens another line rather than submitting. The cursor is
    # still between the 'Z' and "text", and that is where the line breaks --
    # Alt-Enter is a newline typed at the cursor, not an append at the end.
    os.write(m, b"\x1b\r")
    pump(m, stream, 0.3)
    os.write(m, b"third")
    expect_block(
        "Alt-Enter opens a line at the cursor instead of submitting",
        ["ludvart> hello world PASTED", "Z", "thirdtext"],
    )

    # 5. Up/Up + Home + Ctrl-K kills one line, not the whole buffer
    for _ in range(2):
        os.write(m, b"\x1b[A")  # Up
        pump(m, stream, 0.2)
    os.write(m, b"\x1b[H")   # Home
    pump(m, stream, 0.3)
    os.write(m, b"\x0b")     # Ctrl-K kill-to-end
    expect_block(
        "Ctrl-K clears the line the cursor is on and leaves the rest",
        ["ludvart>", "Z", "thirdtext"],
    )

    # 6. Shift-Down/Shift-End select a block, reverse-video shows it, and one
    # Backspace removes it. pyte tracks the attribute per cell, so this is the
    # real screen state and not just the bytes we hoped we wrote.
    for _ in range(2):
        os.write(m, b"\x1b[1;2B")  # Shift-Down
        pump(m, stream, 0.2)
    os.write(m, b"\x1b[1;2F")  # Shift-End
    pump(m, stream, 0.4)
    checks.add(
        "the selected block is drawn in reverse video",
        reversed_text(screen) == "Zthirdtext",
        f"reversed cells read {reversed_text(screen)!r}",
    )
    os.write(m, b"\x7f")  # Backspace
    expect_block("Backspace deletes the whole selected block", ["ludvart>"])
    checks.add(
        "nothing is left highlighted once the block is gone",
        reversed_text(screen) == "",
        f"reversed cells read {reversed_text(screen)!r}",
    )

    os.write(m, b"\x0f")  # close panel
    time.sleep(0.2)
    os.write(m, b"\x03")
    time.sleep(0.3)
    checks.report()


if __name__ == "__main__":
    main()
