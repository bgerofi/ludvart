"""Drive ludvart through a PTY to test the inline Ctrl-G a AI chat end-to-end.

Uses select() with timeouts and an overall wall-clock cap so it can never hang.
Verifies the inline exchange appears in the scroll flow and that ludvart did NOT
switch to the alternate screen for a line-oriented (shell) session.
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

OVERALL_TIMEOUT = 90.0
IDLE_READ = 0.5
ROWS, COLS = 24, 80


def main():
    argv = ludvart_argv()
    pid, master = pty.fork()
    if pid == 0:
        os.environ["PS1"] = "$ "
        os.environ["TERM"] = "xterm"
        os.execvp(argv[0], argv)

    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))

    transcript = bytearray()
    # The raw byte stream carries the assertions; the pyte screen is only used
    # to tell when ludvart is ready and when a turn has finished.
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)
    checks = Checks()
    start = time.time()

    def sink(data):
        transcript.extend(data)
        stream.feed(data)

    try:
        checks.add("ludvart finished starting up", wait_until_started(master, sink, screen))
        os.write(master, b"echo HELLO_FROM_SCREEN_42\n")   # screen content
        wait_for(master, sink, lambda: "HELLO_FROM_SCREEN_42" in screen_text(screen), 10)
        os.write(master, b"\x07")                          # Ctrl-G
        time.sleep(0.3)
        os.write(master, b"a")                             # inline AI
        wait_for(master, sink, lambda: "ludvart>" in screen_text(screen), 10, settle=0.3)
        os.write(master, b"What number appears on the screen?")
        time.sleep(0.5)
        os.write(master, b"\r")                            # submit
        wait_for(master, sink, lambda: "Thinking" in screen_text(screen), 20)
        checks.add(
            "the turn finished",
            wait_for(
                master, sink, lambda: "Thinking" not in screen_text(screen), 90, settle=1.0
            ),
        )
        os.write(master, b"\x07a")                         # back to the shell
        time.sleep(0.5)
        os.write(master, b"echo AFTER_AI_OK\n")            # shell still usable
        wait_for(master, sink, lambda: "AFTER_AI_OK" in screen_text(screen), 10, settle=0.3)
        os.write(master, b"exit\n")
        time.sleep(0.5)
    finally:
        try:
            os.write(master, b"\x03exit\n")
        except OSError:
            pass

    text = transcript.decode("utf-8", "replace")
    lower = text.lower()
    sys.stdout.write("===== RAW TRANSCRIPT (tail) =====\n")
    sys.stdout.write(repr(tail(text, 2500)))
    sys.stdout.write(f"\n  elapsed: {time.time() - start:.1f}s\n")
    checks.add("inline prompt shown", "ludvart> " in text)
    checks.add("question echoed", "What number appears" in text)
    checks.add("thinking indicator", "thinking" in lower)
    checks.add("answer mentions 42", "42" in lower.split("thinking")[-1])
    checks.add("no alt-screen switch", "\x1b[?1049h" not in text)
    checks.add("shell usable after", "AFTER_AI_OK" in text)
    checks.report()


if __name__ == "__main__":
    main()
