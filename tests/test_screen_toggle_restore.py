"""LudvartScreen shrink->grow restores the exact prior viewport (incl. trailing
blank lines) and stays correct when content changes while shrunk."""

from ludvart.screen import LudvartScreen

from e2e_util import Checks

COLS, ROWS, PANEL = 80, 24, 10
APP = ROWS - PANEL


def feed_lines(screen, texts):
    import pyte
    stream = pyte.ByteStream(screen)
    for t in texts:
        stream.feed((t + "\r\n").encode())


def disp(screen):
    return [row.rstrip() for row in screen.display]


def scenario_full_screen(checks):
    s = LudvartScreen(COLS, ROWS)
    feed_lines(s, [f"line_{i:02d}" for i in range(30)])  # scrolls
    before, cur_before = disp(s), (s.cursor.y, s.cursor.x)
    s.resize(APP, COLS)      # open panel
    s.resize(ROWS, COLS)     # close panel
    after, cur_after = disp(s), (s.cursor.y, s.cursor.x)
    checks.add("full screen: viewport identical after toggle", after == before)
    checks.add("full screen: cursor identical", cur_before == cur_after,
               f"{cur_before} -> {cur_after}")


def scenario_partial_with_blanks(checks):
    s = LudvartScreen(COLS, ROWS)
    import pyte
    stream = pyte.ByteStream(s)
    stream.feed(b"A0\r\nA1\r\nA2\r\nA3\r\nA4")  # 5 rows, cursor row 4, blanks below
    before, cur_before = disp(s), (s.cursor.y, s.cursor.x)
    s.resize(APP, COLS)
    s.resize(ROWS, COLS)
    after, cur_after = disp(s), (s.cursor.y, s.cursor.x)
    checks.add("partial: trailing blanks preserved", after == before)
    checks.add("partial: cursor identical", cur_before == cur_after,
               f"{cur_before} -> {cur_after}")


def scenario_changes_while_open(checks):
    # ludvart path: full screen -> shrink -> new output arrives -> grow.
    s = LudvartScreen(COLS, ROWS)
    feed_lines(s, [f"orig_{i:02d}" for i in range(24)])
    s.resize(APP, COLS)                        # open panel
    feed_lines(s, [f"new_{i:02d}" for i in range(5)])  # output while shrunk
    s.resize(ROWS, COLS)                       # close panel
    # reference: same total output on an always-full screen.
    ref = LudvartScreen(COLS, ROWS)
    feed_lines(ref, [f"orig_{i:02d}" for i in range(24)] +
               [f"new_{i:02d}" for i in range(5)])
    checks.add("changed-while-open: matches always-full-size screen",
               disp(s) == disp(ref))
    checks.add("changed-while-open: cursor matches",
               (s.cursor.y, s.cursor.x) == (ref.cursor.y, ref.cursor.x),
               f"{(s.cursor.y, s.cursor.x)} != {(ref.cursor.y, ref.cursor.x)}")


def main():
    checks = Checks()
    scenario_full_screen(checks)
    scenario_partial_with_blanks(checks)
    scenario_changes_while_open(checks)
    checks.report()


if __name__ == "__main__":
    main()
