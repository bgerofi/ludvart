"""Verify the LLM can call the inject_input tool to run a command in the shell."""

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
    # inject_input needs a human "yes"; without one the turn blocks until it
    # times out and the test would fail for the wrong reason.
    approver = Approver(m)

    checks.add("ludvart finished starting up", wait_until_started(m, stream.feed, screen))
    os.write(m, b"\x07")
    time.sleep(0.3)
    os.write(m, b"a")  # open panel
    checks.add(
        "panel opened",
        wait_for(m, stream.feed, lambda: "ludvart>" in screen_text(screen), 10, settle=0.3),
    )

    os.write(m, b"Please create a file called ludvart_tool_ok.txt in the current "
                b"directory using the touch command. Run it for me.")
    time.sleep(0.4)
    os.write(m, b"\r")
    # The turn is done once the spinner has stopped again.
    done = wait_for(
        m,
        stream.feed,
        lambda: approver.approved and "Thinking" not in screen_text(screen)
        and "Calling" not in screen_text(screen),
        90,
        approver=approver,
        settle=1.0,
    )
    show(screen, "after tool-driven request")
    checks.add("inject_input asked for approval", approver.approved)
    checks.add("the turn finished", done)

    # Close the panel and verify the file was actually created in the shell.
    os.write(m, b"\x07a")
    time.sleep(0.5)
    os.write(m, b"ls ludvart_tool_ok.txt\r")
    wait_for(m, stream.feed, lambda: "ludvart_tool_ok.txt" in screen_text(screen), 8, settle=0.5)
    show(screen, "verify file exists in shell")
    shell = screen_text(screen)
    checks.add(
        "touch actually ran in the shell",
        "ludvart_tool_ok.txt" in shell and "No such file" not in shell,
        f"shell showed:\n{shell}",
    )

    os.write(m, b"rm -f ludvart_tool_ok.txt\r")
    time.sleep(0.3)
    os.write(m, b"\x03")
    time.sleep(0.3)
    checks.report()


if __name__ == "__main__":
    main()
