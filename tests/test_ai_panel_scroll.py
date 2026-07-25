"""Verify panel scrollback with a reply longer than the panel."""

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

ROWS, COLS = 24, 80


def panel_rows(screen):
    return [l.rstrip() for l in screen.display if l.strip()]


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
    # Only a deadlock guard: the question below asks for a direct answer, but a
    # stray tool call would otherwise park on the approval prompt forever.
    approver = Approver(m)

    def content(label):
        print(f"\n== {label} ==")
        for l in screen.display:
            r = l.rstrip()
            if r:
                print(r)

    def key(seq):
        os.write(m, seq)
        wait_for(m, stream.feed, lambda: False, 0.6, approver=approver)
        return screen_text(screen)

    checks.add("ludvart finished starting up", wait_until_started(m, stream.feed, screen))
    os.write(m, b"\x07")
    time.sleep(0.3)
    os.write(m, b"a")  # open panel
    checks.add(
        "panel opened",
        wait_for(m, stream.feed, lambda: "ludvart>" in screen_text(screen), 10, settle=0.3),
    )

    # The reply has to land in the panel, not in the terminal, or there is
    # nothing to scroll through.
    os.write(m, b"Without running any command, answer directly: list the numbers "
                b"one through forty, each on its own line.")
    time.sleep(0.4)
    os.write(m, b"\r")
    wait_for(m, stream.feed, lambda: "Thinking" in screen_text(screen), 20, approver=approver)
    checks.add(
        "the long reply finished",
        wait_for(
            m,
            stream.feed,
            lambda: "Thinking" not in screen_text(screen)
            and "Calling" not in screen_text(screen),
            90,
            approver=approver,
            settle=1.0,
        ),
    )

    content("bottom of transcript (newest)")
    bottom = screen_text(screen)
    up1 = key(b"\x1b[5~")  # PageUp
    content("after PageUp")
    checks.add("PageUp scrolled the panel back", up1 != bottom)
    up2 = key(b"\x1b[5~")  # PageUp again
    content("after PageUp x2")
    checks.add("a second PageUp scrolled further", up2 != up1)
    key(b"\x1b[6~")
    back = key(b"\x1b[6~")  # PageDown back
    content("after PageDown x2 (back to bottom)")
    checks.add(
        "PageDown returned to the exact bottom view",
        back == bottom,
        "the panel did not land back on the newest lines",
    )

    os.write(m, b"\x07a")
    time.sleep(0.3)
    checks.report()


if __name__ == "__main__":
    main()
