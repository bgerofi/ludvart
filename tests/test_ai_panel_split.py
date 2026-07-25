"""Drive the bottom AI panel through a PTY, emulating a real terminal via pyte.

The compositor emits absolute-positioned row diffs, so the harness must render
them like a terminal. We feed all of ludvart's output into a pyte screen and print
snapshots at each step.
"""

import fcntl, os, pty, struct, termios, time
import pyte

from e2e_util import Checks, ludvart_argv, screen_text, wait_for, wait_until_started

ROWS, COLS = 24, 80

#: The panel header always ends with this hint strip, so the row it lands on
#: tells us where the panel starts and therefore how tall it is.
_HEADER = "^O/Esc:close"


def panel_height(screen):
    """Rows occupied by the panel, or 0 when the panel is closed."""
    for i, line in enumerate(screen.display):
        if _HEADER in line:
            return len(screen.display) - i
    return 0


def show(screen, label):
    print(f"\n===== {label} =====")
    for i, line in enumerate(screen.display):
        r = line.rstrip()
        if r:
            print(f"row{i:2d}|{r}")


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

    def key(seq, settle=0.5):
        os.write(m, seq)
        wait_for(m, stream.feed, lambda: False, settle)

    checks.add("ludvart finished starting up", wait_until_started(m, stream.feed, screen))
    os.write(m, b"echo HELLO_42\n")
    checks.add(
        "shell output visible before opening the panel",
        wait_for(m, stream.feed, lambda: "HELLO_42" in screen_text(screen), 10, settle=0.3),
    )
    show(screen, "before panel")

    key(b"\x07", 0.3)   # Ctrl-G
    os.write(m, b"a")   # open panel
    checks.add(
        "panel opened",
        wait_for(m, stream.feed, lambda: panel_height(screen) > 0, 10, settle=0.3),
    )
    show(screen, "panel open")

    os.write(m, b"say hi in three words")
    time.sleep(0.4)
    os.write(m, b"\r")   # submit question, await reply
    wait_for(m, stream.feed, lambda: "Thinking" in screen_text(screen), 20)
    checks.add(
        "the reply finished",
        wait_for(m, stream.feed, lambda: "Thinking" not in screen_text(screen), 90, settle=1.0),
    )
    show(screen, "after reply")
    opened = panel_height(screen)

    key(b"\x07", 0.2); key(b"\x1b[A", 0.3)  # Ctrl-G Up -> grow
    key(b"\x07", 0.2); key(b"\x1b[A", 0.4)  # Ctrl-G Up -> grow
    show(screen, "after growing panel x2")
    grown = panel_height(screen)
    checks.add("Ctrl-G Up grew the panel", grown > opened, f"{opened} -> {grown}")

    key(b"\x07", 0.2); key(b"\x1b[B", 0.4)  # Ctrl-G Down -> shrink
    show(screen, "after shrinking panel x1")
    shrunk = panel_height(screen)
    checks.add("Ctrl-G Down shrank the panel", shrunk < grown, f"{grown} -> {shrunk}")

    key(b"\x1b[5~", 0.4)  # PageUp scroll
    show(screen, "after PageUp scroll")
    checks.add("scrolling did not resize the panel", panel_height(screen) == shrunk)

    key(b"\x07", 0.2); os.write(m, b"a")  # close panel
    closed = wait_for(m, stream.feed, lambda: panel_height(screen) == 0, 10, settle=0.4)
    show(screen, "panel closed (app restored)")
    checks.add("panel closed and the app screen came back", closed)
    checks.add("the shell screen was restored", "HELLO_42" in screen_text(screen))

    os.write(m, b"echo BACK_TO_SHELL\n")
    checks.add(
        "the shell is usable after closing the panel",
        wait_for(m, stream.feed, lambda: "BACK_TO_SHELL" in screen_text(screen), 10, settle=0.3),
    )
    show(screen, "shell usable after close")

    key(b"\x07", 0.2); os.write(m, b"a")  # re-open panel
    wait_for(m, stream.feed, lambda: panel_height(screen) > 0, 10, settle=0.4)
    show(screen, "panel re-opened (transcript should persist)")
    checks.add(
        "the transcript survived closing and re-opening",
        "say hi in three words" in screen_text(screen),
        f"panel showed:\n{screen_text(screen)}",
    )

    os.write(m, b"\x07a\x03exit\n")
    time.sleep(0.3)
    checks.report()


if __name__ == "__main__":
    main()
