"""Reproduce: last number disappears when toggling the panel during output."""

import fcntl, os, pty, struct, termios, time
import pyte

from e2e_util import Checks, ludvart_argv, screen_text, wait_for

ROWS, COLS = 24, 80


def dump(screen, label):
    print(f"\n== {label} ==")
    for i, line in enumerate(screen.display):
        r = line.rstrip()
        if r:
            print(f"{i:2d}|{r}")


def numbers(screen):
    """The loop counter values currently visible, in screen order."""
    out = []
    for line in screen.display:
        text = line.strip()
        if text.isdigit():
            out.append(int(text))
    return out


def contiguous(seen):
    """True when no printed number went missing from the visible run."""
    return bool(seen) and all(b == a + 1 for a, b in zip(seen, seen[1:]))


def main():
    pid, m = pty.fork()
    if pid == 0:
        os.environ["PS1"] = "$ "
        os.environ["TERM"] = "xterm"
        argv = ludvart_argv("--no-llm")
        os.execvp(argv[0], argv)
    fcntl.ioctl(m, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)
    checks = Checks()

    # --no-llm prints no model banner, so round-trip a marker instead to prove
    # ludvart is between us and the shell before sending any hotkeys.
    os.write(m, b"echo TOGGLE_READY\n")
    checks.add(
        "ludvart is passing input through to the shell",
        wait_for(m, stream.feed, lambda: "TOGGLE_READY" in screen_text(screen), 20, settle=0.3),
    )

    os.write(m, b'for i in $(seq 1 100); do echo "$i"; sleep 0.4; done\n')
    wait_for(m, stream.feed, lambda: len(numbers(screen)) >= 8, 15)
    dump(screen, "running loop (before toggle)")
    before = numbers(screen)
    checks.add("the loop is printing", bool(before), f"saw {before}")

    os.write(m, b"\x07")
    time.sleep(0.2)
    os.write(m, b"a")  # open panel mid-output
    wait_for(m, stream.feed, lambda: False, 0.8)
    dump(screen, "just after opening panel")
    just_after = numbers(screen)
    checks.add(
        "no number was lost when the panel opened",
        contiguous(just_after),
        f"visible numbers were {just_after}",
    )

    wait_for(m, stream.feed, lambda: False, 2.0)
    dump(screen, "a couple numbers later (panel open)")
    later = numbers(screen)
    checks.add(
        "output keeps flowing into the shrunk app area",
        bool(later) and max(later) > max(just_after or [0]),
        f"{just_after} -> {later}",
    )
    checks.add(
        "the numbers stay contiguous while the panel is open",
        contiguous(later),
        f"visible numbers were {later}",
    )

    os.write(m, b"\x07a\x03")
    time.sleep(0.3)
    checks.report()


if __name__ == "__main__":
    main()
