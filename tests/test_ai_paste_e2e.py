"""End-to-end: open the AI panel, verify bracketed paste and cursor editing land
in the prompt correctly through the real ludvart binary.

Drives a real PTY:
  1. Open the panel (Ctrl-O).
  2. Type "hello world".
  3. Bracketed-paste " PASTED\ntext" -> newline folds to a space, no submit.
  4. Move the cursor left and insert a char mid-line.
  5. Home + Ctrl-K erases everything.
Renders ludvart's output through pyte and checks the "ludvart> " input line.

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

    # 1. plain typing
    os.write(m, b"hello world")
    expect("typing appears in the input line", "hello world")

    # 2. bracketed paste with an embedded newline (must fold to space, no submit)
    os.write(m, PASTE_START + b" PASTED\ntext" + PASTE_END)
    expect("pasted newline folds to a space and does not submit", "hello world PASTED text")

    # 3. move left 4 (into "text") and insert 'Z'
    for _ in range(4):
        os.write(m, b"\x1b[D")  # Left
        pump(m, stream, 0.2)
    os.write(m, b"Z")
    expect("editing mid-line inserts at the cursor", "hello world PASTED Ztext")

    # 4. Home then Ctrl-K clears the whole line
    os.write(m, b"\x1b[H")   # Home
    pump(m, stream, 0.3)
    os.write(m, b"\x0b")     # Ctrl-K kill-to-end
    wait_for(m, stream.feed, lambda: input_line(screen).strip() == "ludvart>", 5)
    line = input_line(screen)
    checks.add("Home + Ctrl-K clears the line", line.strip() == "ludvart>", f"line is {line!r}")

    os.write(m, b"\x0f")  # close panel
    time.sleep(0.2)
    os.write(m, b"\x03")
    time.sleep(0.3)
    checks.report()


if __name__ == "__main__":
    main()
