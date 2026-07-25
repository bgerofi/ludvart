"""Capture the panel indicator; it should show 'Calling inject_input' mid-tool."""

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
    # ludvart asks before typing into the terminal; nothing would answer it here
    # and the whole turn would block until it timed out.
    approver = Approver(m)
    hits = []

    def sink(data):
        stream.feed(data)
        if "Calling" in screen_text(screen):
            hits.append(True)

    checks.add("ludvart finished starting up", wait_until_started(m, sink, screen))
    os.write(m, b"\x07")
    time.sleep(0.3)
    os.write(m, b"a")
    checks.add("panel opened", wait_for(m, sink, lambda: "ludvart>" in screen_text(screen), 10, settle=0.3))

    os.write(m, b"Run 'ls' in the terminal for me.")
    time.sleep(0.4)
    os.write(m, b"\r")
    # The turn is over once the spinner stops and the panel is idle again.
    wait_for(m, sink, lambda: bool(hits) and approver.approved, 60, approver=approver)

    print("== final screen ==")
    for l in screen.display:
        r = l.rstrip()
        if r:
            print(r)
    checks.add("panel showed a 'Calling <tool>' indicator", bool(hits))
    checks.add("inject_input asked for approval before typing", approver.approved)

    os.write(m, b"\x07a\x03")
    time.sleep(0.3)
    checks.report()


if __name__ == "__main__":
    main()
