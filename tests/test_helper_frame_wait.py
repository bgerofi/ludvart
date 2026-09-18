"""run_shell_command waits for the helper's own END frame, not for a guess.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_helper_frame_wait.py
"""

import threading
import time

from ludvart.agent_core import AgentCore
from ludvart.ludvart import Ludvart as RelayPTY


class Snapshots:
    """Hands out a scripted sequence of screens, repeating the last one."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.taken = 0

    def __call__(self):
        i = min(self.taken, len(self.texts) - 1)
        self.taken += 1
        return self.texts[i]


def make_relay(texts=("screen",), idle=0.3):
    relay = RelayPTY.__new__(RelayPTY)
    relay._helper_frames = 0
    relay._frame_carry = b""
    relay._ask_cancel = threading.Event()
    relay._safe_snapshot = Snapshots(texts)
    relay.HELPER_IDLE_TIMEOUT = idle
    relay.SETTLE_POLL = 0.01
    return relay


def test_frames_are_counted_as_they_stream_past():
    relay = make_relay()
    relay._count_helper_frames(b"hi\n<<<LUDVART:END op=run exit=0>>>\n")
    assert relay._helper_frames == 1, relay._helper_frames
    relay._count_helper_frames(b"<<<LUDVART:END op=run exit=1>>>\n")
    assert relay._helper_frames == 2, relay._helper_frames
    print("frames are counted as they stream past: OK")


def test_a_frame_split_across_two_reads_is_counted_once():
    # A read boundary can fall anywhere, and a frame counted twice (or not at
    # all) would make a call wait for someone else's completion.
    for cut in range(1, len("<<<LUDVART:END")):
        relay = make_relay()
        blob = b"out\n<<<LUDVART:END op=run exit=0>>>\n"
        head = b"out\n" + b"<<<LUDVART:END"[:cut]
        tail = blob[len(head):]
        relay._count_helper_frames(head)
        relay._count_helper_frames(tail)
        assert relay._helper_frames == 1, (cut, relay._helper_frames)
    print("a frame split across two reads is counted once: OK")


def test_the_wait_ends_the_moment_the_frame_arrives():
    relay = make_relay(texts=["before", "after"])

    def finish():
        time.sleep(0.05)
        relay._count_helper_frames(b"<<<LUDVART:END op=run exit=0>>>\n")

    threading.Thread(target=finish, daemon=True).start()
    started = time.time()
    snapshot, done = relay._wait_for_helper_frame(since=0)
    elapsed = time.time() - started
    assert done is True, snapshot
    assert elapsed < relay.HELPER_IDLE_TIMEOUT, elapsed
    print("the wait ends the moment the frame arrives: OK")


def test_an_earlier_calls_frame_does_not_end_this_wait():
    """The previous command's frame is still on screen; it must not count."""
    relay = make_relay(idle=0.2)
    relay._count_helper_frames(b"<<<LUDVART:END op=run exit=0>>>\n")
    since = relay._helper_frames
    _, done = relay._wait_for_helper_frame(since=since)
    assert done is False, "a stale frame was mistaken for this call's"
    print("an earlier call's frame does not end this wait: OK")


def test_a_silent_command_is_reported_as_still_running():
    relay = make_relay(idle=0.2)
    started = time.time()
    _, done = relay._wait_for_helper_frame(since=0)
    assert done is False
    assert time.time() - started >= 0.2
    print("a silent command is reported as still running: OK")


def test_output_keeps_a_slow_command_alive():
    """Anything printed restarts the clock: only silence is a hang."""
    relay = make_relay(texts=[f"line {i}" for i in range(40)], idle=0.2)
    started = time.time()
    _, done = relay._wait_for_helper_frame(since=0)
    elapsed = time.time() - started
    assert done is False
    # 40 changing screens at a 0.01s poll outlast a 0.2s idle window only if
    # each change pushed the deadline back.
    assert elapsed > 0.35, elapsed
    print("output keeps a slow command alive: OK")


def test_a_cancelled_turn_stops_waiting():
    relay = make_relay(idle=30.0)
    relay._ask_cancel.set()
    started = time.time()
    _, done = relay._wait_for_helper_frame(since=0)
    assert done is False
    assert time.time() - started < 1.0, "cancel did not break the wait"
    print("a cancelled turn stops waiting: OK")


def make_injector(screen, idle=0.2, emits=None):
    """A relay where typing the helper line makes the frame appear, as it does."""
    relay = make_relay(texts=[screen], idle=idle)
    relay._panel = object()  # open, so the injection is not parked
    relay._inject_approval_all = True
    relay._master_fd = -1

    def write(fd, data):
        if emits is not None:
            relay._count_helper_frames(emits)

    relay._write_all = write
    relay._injection_prompt_prefix = lambda submitted: ""
    relay._wait_for_injection_to_settle = lambda injected, prompt_prefix="": screen
    return relay


def test_a_finished_command_reports_its_exit_status():
    relay = make_injector(
        "$ make\nok\n<<<LUDVART:END op=run exit=0>>>",
        emits=b"<<<LUDVART:END op=run exit=0>>>",
    )
    out = relay._tool_inject_input(
        {"text": "helper run", "submit": True, "await_frame": True}
    )
    assert "ran to completion" in out, out
    assert "exit=0" in out, out
    print("a finished command reports its exit status: OK")


def test_a_failed_command_is_not_described_as_success():
    relay = make_injector(
        "$ make\nboom\n<<<LUDVART:END op=run exit=2>>>",
        emits=b"<<<LUDVART:END op=run exit=2>>>",
    )
    out = relay._tool_inject_input(
        {"text": "helper run", "submit": True, "await_frame": True}
    )
    assert "FAILED (exit=2)" in out, out
    print("a failed command is not described as success: OK")


def test_an_interrupted_command_says_so_in_words():
    """130 has to be spelled out; a bare number invites 'it failed, retry'."""
    relay = make_injector(
        "$ make\n^C\n<<<LUDVART:END op=run exit=130>>>",
        emits=b"<<<LUDVART:END op=run exit=130>>>",
    )
    out = relay._tool_inject_input(
        {"text": "helper run", "submit": True, "await_frame": True}
    )
    assert "INTERRUPTED" in out, out
    assert "Ctrl-C" in out, out
    assert "did not do its work" in out, out
    print("an interrupted command says so in words: OK")


def test_a_hung_command_is_never_called_settled():
    relay = make_injector("$ make\nbuilding...", idle=0.2)
    out = relay._tool_inject_input(
        {"text": "helper run", "submit": True, "await_frame": True}
    )
    assert "STILL RUNNING" in out, out
    assert "settled" not in out, out
    assert "ran to completion" not in out, out
    assert "\\x03" in out, out  # the model is told how to interrupt it
    print("a hung command is never called settled: OK")


def test_background_returns_without_waiting():
    relay = make_injector("$ make\nbuilding...", idle=30.0)
    started = time.time()
    out = relay._tool_inject_input(
        {"text": "helper run", "submit": True, "no_wait": True}
    )
    assert time.time() - started < 1.0, "background call waited anyway"
    assert "NOT waited for" in out, out
    assert "no exit status yet" in out, out
    print("background returns without waiting: OK")


def test_a_stale_frame_on_screen_does_not_end_the_next_call():
    """Back-to-back commands: the previous frame is still there to be seen."""
    relay = make_injector(
        "$ make\n<<<LUDVART:END op=run exit=0>>>\n$ make again\nbuilding...",
        idle=0.2,
    )
    relay._count_helper_frames(b"<<<LUDVART:END op=run exit=0>>>")
    out = relay._tool_inject_input(
        {"text": "helper run", "submit": True, "await_frame": True}
    )
    assert "STILL RUNNING" in out, out
    print("a stale frame on screen does not end the next call: OK")


def run_command_args(**args):
    """The inject_input arguments one run_shell_command call produces."""
    seen = {}

    class Host:
        def run_terminal_tool(self, name, payload):
            seen.update(payload)
            seen["tool"] = name
            return "ok"

    core = AgentCore.__new__(AgentCore)
    core.host = Host()
    core._tool_run_command({"command": "make", **args})
    return seen


def test_a_plain_command_asks_the_terminal_to_wait():
    seen = run_command_args()
    assert seen["tool"] == "inject_input"
    assert seen["await_frame"] is True, seen
    assert seen["no_wait"] is False, seen
    print("a plain command asks the terminal to wait: OK")


def test_background_turns_the_wait_off():
    seen = run_command_args(background=True)
    assert seen["await_frame"] is False, seen
    assert seen["no_wait"] is True, seen
    # Models sometimes send the JSON boolean as a string.
    assert run_command_args(background="true")["no_wait"] is True
    assert run_command_args(background="false")["no_wait"] is False
    print("background turns the wait off: OK")


def main():
    test_frames_are_counted_as_they_stream_past()
    test_a_frame_split_across_two_reads_is_counted_once()
    test_the_wait_ends_the_moment_the_frame_arrives()
    test_an_earlier_calls_frame_does_not_end_this_wait()
    test_a_silent_command_is_reported_as_still_running()
    test_output_keeps_a_slow_command_alive()
    test_a_cancelled_turn_stops_waiting()
    test_a_finished_command_reports_its_exit_status()
    test_a_failed_command_is_not_described_as_success()
    test_an_interrupted_command_says_so_in_words()
    test_a_hung_command_is_never_called_settled()
    test_a_stale_frame_on_screen_does_not_end_the_next_call()
    test_background_returns_without_waiting()
    test_a_plain_command_asks_the_terminal_to_wait()
    test_background_turns_the_wait_off()
    print("\nALL helper frame wait tests passed.")


if __name__ == "__main__":
    main()
