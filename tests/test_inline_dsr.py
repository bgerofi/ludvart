"""Inline AI test harness that emulates a real terminal (answers DSR ESC[6n).

ludvart now asks the terminal for the true cursor position; a dumb pipe never
answers, so this harness feeds all child output through pyte and replies to
ESC[6n with the pyte cursor position, exactly like a real terminal.
"""

import fcntl, os, pty, re, struct, termios, time
import pyte

from e2e_util import (
    Approver,
    Checks,
    ludvart_argv,
    screen_text,
    wait_for,
    wait_until_started,
)

_DSR = re.compile(rb"\x1b\[6n")


def scenario(checks, ps1, label, partial=b""):
    pid, m = pty.fork()
    if pid == 0:
        os.environ["PS1"] = ps1
        os.environ["TERM"] = "xterm"
        argv = ludvart_argv()
        os.execvp(argv[0], argv)
    fcntl.ioctl(m, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
    screen = pyte.Screen(80, 24)
    stream = pyte.ByteStream(screen)
    approver = Approver(m)

    def sink(data):
        """Feed pyte and answer cursor-position queries like a real terminal."""
        stream.feed(data)
        if _DSR.search(data):
            row, col = screen.cursor.y + 1, screen.cursor.x + 1
            os.write(m, f"\x1b[{row};{col}R".encode("ascii"))

    def add(name, ok, detail=""):
        checks.add(f"{label}: {name}", ok, detail)

    add("ludvart finished starting up", wait_until_started(m, sink, screen))
    os.write(m, b"echo HELLO_42\n")
    wait_for(m, sink, lambda: "HELLO_42" in screen_text(screen), 10, settle=0.3)
    if partial:
        os.write(m, partial)
        wait_for(m, sink, lambda: False, 0.4)
    os.write(m, b"\x07")
    time.sleep(0.3)
    os.write(m, b"a")
    add("panel opened", wait_for(m, sink, lambda: "ludvart>" in screen_text(screen), 10, settle=0.3))
    os.write(m, b"What number is shown?")
    time.sleep(0.4)
    os.write(m, b"\r")
    wait_for(m, sink, lambda: "Thinking" in screen_text(screen), 20, approver=approver)
    add(
        "the turn finished",
        wait_for(
            m,
            sink,
            lambda: "Thinking" not in screen_text(screen)
            and "Calling" not in screen_text(screen),
            90,
            approver=approver,
            settle=1.0,
        ),
    )
    add("the model read the screen through the emulated terminal", "42" in screen_text(screen))

    os.write(m, b"Z")  # prove line editor intact
    wait_for(m, sink, lambda: False, 0.6)

    print(f"\n===== {label} =====")
    for i, line in enumerate(screen.display):
        r = line.rstrip()
        if r:
            print(f"row{i:2d}|{r}")
    add(
        "the line editor still accepts input after the reply",
        any(l.rstrip().endswith("Z") for l in screen.display),
        "no row ends with the typed 'Z'",
    )
    os.write(m, b"\x03exit\n")
    time.sleep(0.3)


def main():
    checks = Checks()
    scenario(checks, "$ ", "SINGLE-LINE PROMPT")
    scenario(checks, "[demo]\\n$ ", "TWO-LINE PROMPT")
    scenario(checks, "$ ", "SINGLE-LINE + PARTIAL BUFFER 'ls -la'", partial=b"ls -la")
    checks.report()


if __name__ == "__main__":
    main()
