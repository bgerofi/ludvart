"""Verify the LLM receives multi-turn history (second Q depends on first)."""

import fcntl, os, pty, struct, termios, time
import pyte

from e2e_util import Checks, ludvart_argv, screen_text, wait_for, wait_until_started

ROWS, COLS = 24, 80


def content(screen, label):
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

    def ask(question, timeout=90):
        os.write(m, question)
        time.sleep(0.4)
        os.write(m, b"\r")
        # Wait for the spinner to appear and then go away again.
        wait_for(m, stream.feed, lambda: "Thinking" in screen_text(screen), 20)
        return wait_for(
            m,
            stream.feed,
            lambda: "Thinking" not in screen_text(screen),
            timeout,
            settle=1.0,
        )

    checks.add("ludvart finished starting up", wait_until_started(m, stream.feed, screen))
    os.write(m, b"\x07")
    time.sleep(0.3)
    os.write(m, b"a")  # open panel
    checks.add(
        "panel opened",
        wait_for(m, stream.feed, lambda: "ludvart>" in screen_text(screen), 10, settle=0.3),
    )

    checks.add("first turn completed", ask(b"My favorite number is 7. Just acknowledge."))
    content(screen, "after Q1")
    checks.add("second turn completed", ask(b"What is my favorite number times 6?"))
    content(screen, "after Q2 (should know 7 -> 42)")
    panel = screen_text(screen)
    checks.add(
        "the second answer used the first turn (7 * 6 = 42)",
        "42" in panel,
        f"panel showed:\n{panel}",
    )

    os.write(m, b"\x07a\x03")
    time.sleep(0.3)
    checks.report()


if __name__ == "__main__":
    main()
