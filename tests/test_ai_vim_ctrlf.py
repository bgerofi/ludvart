"""In vim, the model should page down (Ctrl-F) without modifying the buffer."""

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


def show(screen, label):
    print(f"\n== {label} ==")
    for l in screen.display:
        r = l.rstrip()
        if r:
            print(r)


def main():
    # Build a long numbered file to make paging visible.
    with open("/tmp/ludvart_vim_test.txt", "w") as f:
        for i in range(1, 201):
            f.write(f"line_{i:03d}\n")

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
    os.write(m, b"vim -u NONE /tmp/ludvart_vim_test.txt\r")
    checks.add(
        "vim opened at the top of the file",
        wait_for(m, stream.feed, lambda: "line_001" in screen_text(screen), 15, settle=0.5),
    )
    show(screen, "vim opened (top of file)")

    os.write(m, b"\x07")
    time.sleep(0.3)
    os.write(m, b"a")
    checks.add(
        "panel opened",
        wait_for(m, stream.feed, lambda: "ludvart>" in screen_text(screen), 10, settle=0.3),
    )
    os.write(m, b"Page down one screen in vim (this is a read-only view, do NOT "
                b"modify the buffer).")
    time.sleep(0.4)
    os.write(m, b"\r")
    wait_for(
        m,
        stream.feed,
        lambda: approver.approved and "Thinking" not in screen_text(screen)
        and "Calling" not in screen_text(screen),
        120,
        approver=approver,
        settle=1.0,
    )
    checks.add("the model drove vim through inject_input", approver.approved)

    # Back to vim to inspect the result.
    os.write(m, b"\x07a")
    wait_for(m, stream.feed, lambda: "ludvart>" not in screen_text(screen), 8, settle=0.5)
    show(screen, "after model pages down (expect later lines, no [+] modified)")
    checks.add(
        "the view paged past the first screenful",
        "line_001" not in screen_text(screen),
        f"vim showed:\n{screen_text(screen)}",
    )

    # A plain :q is the buffer-modified oracle: vim refuses with E37 if the
    # model typed anything into the file.
    os.write(m, b"\x1b:q\r")
    wait_for(m, stream.feed, lambda: "E37" in screen_text(screen), 4, settle=0.5)
    modified = "E37" in screen_text(screen)
    checks.add("the buffer was left unmodified", not modified, "vim refused :q with E37")

    os.write(m, b"\x1b:q!\r")
    time.sleep(0.5)
    os.write(m, b"\x03")
    time.sleep(0.3)
    checks.report()


if __name__ == "__main__":
    main()
