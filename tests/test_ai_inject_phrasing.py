"""Repro of the user's phrasing: 'Use the inject tool to display files.'"""

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
    os.write(m, b"echo hello > file_a.txt; echo hi > file_b.txt\r")
    wait_for(m, stream.feed, lambda: "file_b.txt" in screen_text(screen), 8, settle=0.3)
    os.write(m, b"\x07")
    time.sleep(0.3)
    os.write(m, b"a")  # open panel
    checks.add(
        "panel opened",
        wait_for(m, stream.feed, lambda: "ludvart>" in screen_text(screen), 10, settle=0.3),
    )

    os.write(m, b"Use the inject tool to display files.")
    time.sleep(0.4)
    os.write(m, b"\r")
    # This phrasing used to make the model answer in prose instead of calling
    # the tool, so the approval prompt firing *is* the thing under test.
    used_tool = wait_for(m, stream.feed, lambda: approver.approved, 60, approver=approver, settle=1.0)
    show(screen, "after 'Use the inject tool to display files.'")
    checks.add("the model called inject_input for this phrasing", used_tool)
    checks.add(
        "the listing came back into the panel",
        "file_a.txt" in screen_text(screen),
        f"panel showed:\n{screen_text(screen)}",
    )

    os.write(m, b"\x07a")
    time.sleep(0.5)
    os.write(m, b"rm -f file_a.txt file_b.txt\r")
    time.sleep(0.3)
    os.write(m, b"\x03")
    time.sleep(0.3)
    checks.report()


if __name__ == "__main__":
    main()
