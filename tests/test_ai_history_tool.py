"""The model should use capture_screen_history to find output above the screen."""

import fcntl, os, pty, struct, termios, time
import pyte
from e2e_util import (
    Approver,
    Checks,
    ludvart_argv,
    screen_text,
    wait_for,
    wait_until_started,
)
ROWS, COLS = 24, 90


def show(screen, label):
    print(f"\n== {label} ==")
    for l in screen.display:
        r = l.rstrip()
        if r:
            print(r)


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
    approver = Approver(m)

    checks.add("ludvart finished starting up", wait_until_started(m, stream.feed, screen))
    # Emit a marker line, then push it far above the visible screen.
    os.write(m, b"echo MARKER_ALPHA_42; for i in $(seq 1 60); do echo filler_$i; done\r")
    scrolled = wait_for(
        m,
        stream.feed,
        lambda: "filler_60" in screen_text(screen) and "MARKER_ALPHA_42" not in screen_text(screen),
        15,
        settle=0.3,
    )
    checks.add("the marker scrolled off the visible screen", scrolled)

    os.write(m, b"\x07")
    time.sleep(0.3)
    os.write(m, b"a")
    checks.add(
        "panel opened",
        wait_for(m, stream.feed, lambda: "ludvart>" in screen_text(screen), 10, settle=0.3),
    )

    os.write(m, b"A while ago I printed a line containing MARKER_ALPHA followed by a "
                b"number. It has scrolled off screen. Look back through the terminal "
                b"history and tell me the exact number.")
    time.sleep(0.4)
    os.write(m, b"\r")
    wait_for(m, stream.feed, lambda: "Thinking" in screen_text(screen), 20, approver=approver)
    done = wait_for(
        m,
        stream.feed,
        lambda: "Thinking" not in screen_text(screen) and "Calling" not in screen_text(screen),
        120,
        approver=approver,
        settle=1.0,
    )
    show(screen, "reply (should recover 42 from history)")
    panel = screen_text(screen)
    checks.add("the turn finished", done)
    checks.add(
        "the model recovered the marker number from scrollback",
        "42" in panel,
        f"panel showed:\n{panel}",
    )

    os.write(m, b"\x07a\x03")
    time.sleep(0.3)
    checks.report()


if __name__ == "__main__":
    main()
