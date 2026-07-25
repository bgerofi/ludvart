"""End-to-end: verify the AI panel opens at half the screen height by default,
Ctrl-G PageUp resizes it to half the screen, and Ctrl-G PageDown restores the
previous height.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_ai_half_resize_e2e.py
"""

import errno, fcntl, os, pty, select, struct, termios, time
import pyte

from e2e_util import Checks, ludvart_argv, wait_for, wait_until_started

ROWS, COLS = 24, 90
PREFIX = b"\x07"      # Ctrl-G
PGUP = b"\x1b[5~"
PGDN = b"\x1b[6~"
DOWN = b"\x1b[B"      # Ctrl-G Down -> shrink panel by one row


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


def header_row(screen):
    """Index of the panel header row (the reverse-video 'ludvart ·' bar)."""
    for i, row in enumerate(screen.display):
        if "ludvart" in row and ("close" in row or "resize" in row):
            return i
    return -1


def panel_height(screen):
    """Panel height = rows from the header down to the bottom of the screen."""
    h = header_row(screen)
    return -1 if h < 0 else ROWS - h


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
    os.write(m, b"\x0f")  # open panel (defaults to half the screen height)
    checks.add("panel opened", wait_for(m, stream.feed, lambda: header_row(screen) >= 0, 10, settle=0.3))

    default = panel_height(screen)  # new default: half the screen

    # Shrink several rows so the half-resize and restore are observable (the
    # panel already opens at half, so PageUp from there would be a no-op).
    expect_half = ROWS // 2
    for _ in range(4):
        os.write(m, PREFIX + DOWN)  # Ctrl-G Down -> shrink by 1
        pump(m, stream, 0.4)
    wait_for(m, stream.feed, lambda: panel_height(screen) == expect_half - 4, 5)
    original = panel_height(screen)

    os.write(m, PREFIX + PGUP)  # -> half screen
    wait_for(m, stream.feed, lambda: panel_height(screen) == expect_half, 5, settle=0.3)
    half = panel_height(screen)

    os.write(m, PREFIX + PGDN)  # -> restore
    wait_for(m, stream.feed, lambda: panel_height(screen) == original, 5, settle=0.3)
    restored = panel_height(screen)

    print(
        f"default={default} original={original} half={half} "
        f"restored={restored}  (rows={ROWS})"
    )
    checks.add("panel opens at half the screen", default == expect_half, f"got {default}, want {expect_half}")
    checks.add("Ctrl-G Down shrinks by one row each", original == expect_half - 4, f"got {original}, want {expect_half - 4}")
    checks.add("Ctrl-G PageUp snaps to half", half == expect_half, f"got {half}, want {expect_half}")
    checks.add("Ctrl-G PageDown restores the prior height", restored == original, f"got {restored}, want {original}")

    os.write(m, b"\x0f")
    time.sleep(0.2)
    os.write(m, b"\x03")
    time.sleep(0.3)
    checks.report()


if __name__ == "__main__":
    main()
