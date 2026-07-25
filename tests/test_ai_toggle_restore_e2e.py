"""End-to-end: fill the screen, toggle the AI panel on and off, and confirm the
user screen returns to its exact prior state (including trailing blank lines)."""

import fcntl, os, pty, struct, termios, time
import pyte

from e2e_util import Checks, ludvart_argv, screen_text, wait_for, wait_until_started

ROWS, COLS = 24, 90


def disp(screen):
    return [row.rstrip() for row in screen.display]


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
    # Fill the screen with numbered lines.
    os.write(m, b"for i in $(seq 1 40); do echo filler_line_$i; done\r")
    filled = wait_for(m, stream.feed, lambda: "filler_line_40" in screen_text(screen), 15, settle=0.5)
    checks.add("screen filled with output", filled)
    before = disp(screen)
    print("== before toggle ==")
    for r in before:
        if r:
            print(r)

    os.write(m, b"\x0f")   # Ctrl-O open
    opened = wait_for(m, stream.feed, lambda: "^O/Esc:close" in screen_text(screen), 10, settle=0.3)
    checks.add("panel opened", opened)
    os.write(m, b"\x0f")   # Ctrl-O close
    closed = wait_for(m, stream.feed, lambda: "^O/Esc:close" not in screen_text(screen), 10, settle=0.3)
    checks.add("panel closed", closed)
    after = disp(screen)

    print("\n== after toggle ==")
    for r in after:
        if r:
            print(r)

    if before != after:
        for i, (b, a) in enumerate(zip(before, after)):
            if b != a:
                print(f"  row {i}: before={b!r} after={a!r}")
    checks.add("screen restored exactly after toggle", before == after)

    os.write(m, b"\x03")
    time.sleep(0.3)
    checks.report()


if __name__ == "__main__":
    main()
