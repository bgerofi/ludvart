"""Unit test: the client notices a terminal that is gone and stops holding on.

Two ways a terminal can leave. It can hang up, which arrives as EOF on stdin --
the relay used to ignore that and spin on it forever. Or it can leave the pty
open behind it (sshd does this when the network under an ssh session drops), in
which case nothing arrives and nothing closes; the only way to tell is to ask
the terminal a question and notice that no answer comes back.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_client_liveness.py
"""

import os
import threading

from ludvart.ludvart import Ludvart, TerminalLiveness


# -- the timing rules, on a clock we control --------------------------------


def test_the_wait_tracks_whichever_deadline_is_next():
    now = [0.0]
    live = TerminalLiveness(60.0, 5.0, 3, clock=lambda: now[0])
    assert live.timeout() == 60.0
    now[0] += 59.0
    assert live.timeout() == 1.0
    now[0] += 1.0
    assert live.check() == "probe"
    # An outstanding question is the nearer deadline of the two.
    assert live.timeout() == 5.0


def test_only_repeated_silence_counts_as_death():
    now = [1000.0]
    live = TerminalLiveness(60.0, 5.0, 3, clock=lambda: now[0])
    assert live.check() == ""
    now[0] += 60.0
    assert live.check() == "probe"
    # An answer clears the score, however late in the window it arrives.
    now[0] += 4.0
    assert live.answered(b"\x1b[3;7R") == b""
    assert live.check() == ""
    # Silence: each expired question costs one miss, and only the last gives up.
    now[0] += 60.0
    assert live.check() == "probe"
    now[0] += 5.0
    assert live.check() == "probe"  # miss 1
    now[0] += 5.0
    assert live.check() == "probe"  # miss 2
    now[0] += 5.0
    assert live.check() == "dead"  # miss 3


def test_a_slow_answer_still_saves_the_session():
    now = [0.0]
    live = TerminalLiveness(60.0, 5.0, 3, clock=lambda: now[0])
    now[0] += 60.0
    assert live.check() == "probe"
    now[0] += 5.0
    assert live.check() == "probe"  # miss 1
    live.answered(b"\x1b[1;1R")
    now[0] += 60.0
    assert live.check() == "probe"
    now[0] += 5.0
    assert live.check() == "probe"  # miss 1 again, not miss 2
    now[0] += 5.0
    assert live.check() == "probe"  # miss 2


# -- who the answer belongs to ----------------------------------------------


def test_the_answer_to_our_question_is_kept_from_the_child():
    now = [0.0]
    live = TerminalLiveness(0.0, 5.0, 3, clock=lambda: now[0])
    assert live.check() == "probe"
    # Whatever was typed alongside it is still the child's.
    assert live.answered(b"ls\x1b[3;7R\r") == b"ls\r"


def test_a_cursor_report_we_never_asked_for_reaches_the_child():
    # vim asks the terminal the same question; that reply is not ours to eat.
    live = TerminalLiveness(60.0, 5.0, 3, clock=lambda: 0.0)
    assert live.answered(b"\x1b[3;7R") == b"\x1b[3;7R"


# -- the relay itself --------------------------------------------------------


def make_relay(idle=300.0, reply_wait=10.0, misses=3):
    """A Ludvart wired to plain pipes instead of a terminal and a child."""
    r = Ludvart(["true"])
    r.LIVENESS_IDLE = idle
    r.LIVENESS_REPLY_WAIT = reply_wait
    r.LIVENESS_MAX_MISSES = misses
    master_r, master_w = os.pipe()
    stdin_r, stdin_w = os.pipe()
    r._master_fd, r._stdin_fd = master_r, stdin_r
    r._stdout_fd = master_w  # unused unless a test stubs _write_all
    return r, stdin_w


def run_loop(r):
    """Start ``_loop`` on a thread so a relay that never stops is detectable."""
    out = []
    t = threading.Thread(target=lambda: out.append(r._loop()), daemon=True)
    t.start()
    return t, out


def test_a_hangup_on_stdin_ends_the_session():
    r, stdin_w = make_relay()
    hung = []
    r._hang_up_child = lambda: hung.append(True)
    r._reap_child = lambda: 7
    os.close(stdin_w)  # the terminal hangs up
    t, out = run_loop(r)
    t.join(5)
    assert not t.is_alive(), "the relay spun on stdin EOF instead of shutting down"
    assert out == [7]
    assert hung == [True], "the child was left waiting for a terminal that is gone"


def test_an_unanswered_terminal_ends_the_session():
    r, stdin_w = make_relay(idle=0.02, reply_wait=0.02, misses=2)
    probes = []
    r._write_all = lambda fd, data: probes.append(data)
    r._hang_up_child = lambda: None
    r._reap_child = lambda: 0
    t, out = run_loop(r)
    t.join(5)
    assert not t.is_alive(), "a terminal that never answers kept the session open"
    assert out == [0]
    assert probes == [TerminalLiveness.QUERY] * 2
    os.close(stdin_w)


def test_an_answering_terminal_keeps_the_session():
    r, stdin_w = make_relay(idle=0.02, reply_wait=0.05, misses=2)
    probes = []
    typed = []

    def write_all(fd, data):
        if data == TerminalLiveness.QUERY:
            probes.append(data)
            os.write(stdin_w, b"\x1b[12;34R")

    r._write_all = write_all
    r._handle_input = lambda data: typed.append(data)
    r._hang_up_child = lambda: None
    r._reap_child = lambda: 0
    t, out = run_loop(r)
    t.join(0.6)
    assert t.is_alive(), "an answering terminal was given up on"
    assert len(probes) >= 3, "the relay stopped checking"
    assert typed == [], "the terminal's answer was forwarded to the child"
    os.close(stdin_w)
    t.join(5)
    assert not t.is_alive()
    assert out == [0]


if __name__ == "__main__":
    test_the_wait_tracks_whichever_deadline_is_next()
    test_only_repeated_silence_counts_as_death()
    test_a_slow_answer_still_saves_the_session()
    test_the_answer_to_our_question_is_kept_from_the_child()
    test_a_cursor_report_we_never_asked_for_reaches_the_child()
    test_a_hangup_on_stdin_ends_the_session()
    test_an_unanswered_terminal_ends_the_session()
    test_an_answering_terminal_keeps_the_session()
    print("all client liveness checks passed")
