"""Drive ludvart through a PTY to test the Ctrl-G a AI panel end-to-end.

Uses select() with timeouts and an overall wall-clock cap so it can never hang.
Prints a transcript tail of what ludvart rendered, then a PASS/FAIL summary.
"""

import fcntl
import os
import pty
import struct
import sys
import termios
import time

import pyte

from e2e_util import (
    Checks,
    ludvart_argv,
    screen_text,
    tail,
    wait_for,
    wait_until_started,
)

OVERALL_TIMEOUT = 90.0  # hard cap for the whole test
IDLE_READ = 0.5
ROWS, COLS = 24, 80


def main():
    argv = ludvart_argv()
    pid, master = pty.fork()
    if pid == 0:
        os.environ["PS1"] = "$ "
        os.environ["TERM"] = "xterm"
        os.execvp(argv[0], argv)

    # Give the PTY a realistic window size (24x80).
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))

    transcript = bytearray()
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)
    checks = Checks()
    saw_thinking = []

    def sink(data):
        transcript.extend(data)
        stream.feed(data)
        if "Thinking" in screen_text(screen):
            saw_thinking.append(True)

    start = time.time()
    try:
        checks.add("ludvart finished starting up", wait_until_started(master, sink, screen))
        os.write(master, b"echo HELLO_FROM_SCREEN_42\n")   # screen content
        wait_for(master, sink, lambda: "HELLO_FROM_SCREEN_42" in screen_text(screen), 10)
        os.write(master, b"\x07")                          # Ctrl-G
        time.sleep(0.3)
        os.write(master, b"a")                             # AI panel
        checks.add(
            "panel prompt shown",
            wait_for(master, sink, lambda: "ludvart>" in screen_text(screen), 10, settle=0.3),
        )
        os.write(master, b"What number appears on the screen?")
        time.sleep(0.5)
        os.write(master, b"\r")                            # submit
        # The reply is done when the spinner has stopped and text has landed.
        answered = wait_for(
            master,
            sink,
            lambda: saw_thinking and "Thinking" not in screen_text(screen),
            60,
            settle=1.0,
        )
        checks.add("thinking indicator shown while working", bool(saw_thinking))
        checks.add("turn completed within the timeout", answered)
        panel = screen_text(screen)
        checks.add("question echoed into the transcript", "What number appears" in panel)
        checks.add("answer mentions 42", "42" in panel, f"panel was:\n{panel}")
        os.write(master, b"q")                             # close reply
        time.sleep(0.5)
        os.write(master, b"exit\n")                        # exit bash
        time.sleep(1.0)
    finally:
        try:
            os.write(master, b"\x03exit\n")
        except OSError:
            pass

    text = transcript.decode("utf-8", "replace")
    sys.stdout.write("===== RAW TRANSCRIPT (tail) =====\n")
    sys.stdout.write(tail(text))
    sys.stdout.write(f"\n  elapsed: {time.time() - start:.1f}s\n")
    checks.report()


if __name__ == "__main__":
    main()
