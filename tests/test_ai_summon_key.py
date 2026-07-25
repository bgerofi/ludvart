"""A single Ctrl-O opens the AI panel; a second Ctrl-O closes it."""

import fcntl, os, pty, struct, termios, time
import pyte

from e2e_util import Checks, ludvart_argv, screen_text, wait_for, wait_until_started

ROWS, COLS = 24, 90


def panel_open(screen):
    text = screen_text(screen)
    return "ludvart>" in text or "^O/Esc:close" in text


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

    # Keys sent before ludvart is up are echoed by the tty instead of reaching
    # it, which used to look like "the panel refused to open".
    checks.add("ludvart finished starting up", wait_until_started(m, stream.feed, screen))
    checks.add("panel closed before Ctrl-O", not panel_open(screen))

    os.write(m, b"\x0f")  # Ctrl-O -> summon
    checks.add("panel opens on 1st Ctrl-O", wait_for(m, stream.feed, lambda: panel_open(screen), 10))

    os.write(m, b"\x0f")  # Ctrl-O -> close
    checks.add("panel closes on 2nd Ctrl-O", wait_for(m, stream.feed, lambda: not panel_open(screen), 10))

    os.write(m, b"echo READY\r")
    checks.add("shell usable after close", wait_for(m, stream.feed, lambda: "READY" in screen_text(screen), 10))

    os.write(m, b"\x03")
    time.sleep(0.3)
    print("== final screen ==")
    print(screen_text(screen))
    checks.report()


if __name__ == "__main__":
    main()
