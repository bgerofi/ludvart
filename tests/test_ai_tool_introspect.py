"""Ask the model what tools it can invoke; it should name inject_input."""

import fcntl, os, pty, struct, termios, time
import pyte

from e2e_util import Checks, ludvart_argv, screen_text, wait_for, wait_until_started

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

    checks.add("ludvart finished starting up", wait_until_started(m, stream.feed, screen))
    os.write(m, b"\x07")
    time.sleep(0.3)
    os.write(m, b"a")
    checks.add(
        "panel opened",
        wait_for(m, stream.feed, lambda: "ludvart>" in screen_text(screen), 10, settle=0.3),
    )

    os.write(m, b"What tools can you invoke? List their names.")
    time.sleep(0.4)
    os.write(m, b"\r")
    wait_for(m, stream.feed, lambda: "Thinking" in screen_text(screen), 20)
    done = wait_for(
        m, stream.feed, lambda: "Thinking" not in screen_text(screen), 90, settle=1.0
    )
    show(screen, "reply to 'What tools can you invoke?'")
    checks.add("the turn finished", done)
    # A tool listing is easily longer than the panel, so page back through the
    # whole reply instead of only looking at what happens to be on screen.
    panel = screen_text(screen)
    seen = [panel]
    for _ in range(12):
        os.write(m, b"\x1b[5~")  # PageUp
        wait_for(m, stream.feed, lambda: False, 0.5, settle=0.2)
        page = screen_text(screen)
        if page == seen[-1]:
            break
        seen.append(page)
    panel = "\n".join(seen)
    # If the tool specs never reached the model it cannot name them.
    checks.add(
        "the model names inject_input among its tools",
        "inject_input" in panel,
        f"panel showed:\n{panel}",
    )

    os.write(m, b"\x07a\x03")
    time.sleep(0.3)
    checks.report()


if __name__ == "__main__":
    main()
