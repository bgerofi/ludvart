"""The model should cat README, get the settled screen back, and summarize it."""

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
ROWS, COLS = 30, 90


def show(screen, label):
    print(f"\n== {label} ==")
    for l in screen.display:
        r = l.rstrip()
        if r:
            print(r)


def main():
    # Our own README in the throwaway cwd, holding a number the answer has to be
    # derived from: the marker alone would also appear in the cat output, so
    # only arithmetic proves the settled screen actually came back.
    with open("README.md", "w") as fh:
        fh.write("Zorblax logs 7 stork sightings per hour.\n")

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
    os.write(m, b"\x07")
    time.sleep(0.3)
    os.write(m, b"a")
    checks.add(
        "panel opened",
        wait_for(m, stream.feed, lambda: "ludvart>" in screen_text(screen), 10, settle=0.3),
    )

    os.write(m, b"According to the README, how many stork sightings per hour, "
                b"times 6? Read the README in the terminal first.")
    time.sleep(0.4)
    os.write(m, b"\r")
    idle = wait_for(
        m,
        stream.feed,
        lambda: approver.approved and "Thinking" not in screen_text(screen)
        and "Calling" not in screen_text(screen),
        120,
        approver=approver,
        settle=1.0,
    )
    show(screen, "reply (should answer 42 from the README, not ask user)")
    panel = screen_text(screen)
    checks.add("the model read the README through the terminal", approver.approved)
    checks.add("the turn settled and finished", idle)
    # The point of the test: the settled screen came back, so the model answers
    # instead of asking the user to paste the file.
    checks.add(
        "the model answered from the file it read (7 * 6 = 42)",
        "42" in panel,
        f"panel showed:\n{panel}",
    )

    os.write(m, b"\x07a\x03")
    time.sleep(0.3)
    checks.report()


if __name__ == "__main__":
    main()
