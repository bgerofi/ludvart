"""Unit test: injection settle uses the fast TUI path in the alternate screen.

Exercises RelayPTY._wait_for_injection_to_settle without spawning a child or an
LLM. It fakes a screen model whose ``in_alt_screen`` flag and text we control,
and asserts:
  1) In a full-screen (alternate-buffer) app the method returns after a short
     unchanged window (<= SETTLE_TUI_MAX_WAIT), never invoking the LLM check --
     even though no shell prompt is ever learned/returned. This is the screen /
     tmux / vim case that used to hang up to SETTLE_MAX_WAIT (120s).
  2) The TUI cap is far below the normal cap.

Run: python3 tools/test_tui_settle.py   (exit 0 = pass)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ludvart.ludvart import Ludvart as RelayPTY


class FakeScreen:
    def __init__(self, in_alt_screen):
        self.in_alt_screen = in_alt_screen


def make_relay(in_alt_screen, texts):
    """Build a RelayPTY without running __init__ (no PTY, no threads)."""
    relay = RelayPTY.__new__(RelayPTY)
    relay._backend_client = None  # if the model path were taken it short-circuits
    relay.screen = FakeScreen(in_alt_screen)
    # Feed a deterministic sequence of snapshots: it changes once, then stays
    # constant so the quiescence window elapses.
    seq = list(texts)

    def _safe_snapshot():
        return seq.pop(0) if len(seq) > 1 else seq[0]

    relay._safe_snapshot = _safe_snapshot
    # Guard: the TUI path must NOT consult the learned prompt or the LLM.
    def _prompt_returned(_prefix):
        raise AssertionError("_prompt_returned must not be called in TUI mode")

    def _injection_finished(_inj, _txt):
        raise AssertionError("_injection_finished (LLM) must not run in TUI mode")

    relay._prompt_returned = _prompt_returned
    relay._injection_finished = _injection_finished
    return relay


def test_tui_returns_fast_without_prompt_or_llm():
    relay = make_relay(True, ["initial", "changed", "changed"])
    start = time.time()
    out = relay._wait_for_injection_to_settle("\x01n", prompt_prefix="")
    elapsed = time.time() - start
    assert out == "changed", out
    assert elapsed <= RelayPTY.SETTLE_TUI_MAX_WAIT + 1.0, elapsed
    # Should settle around the short quiet window, well under the normal cap.
    assert elapsed < RelayPTY.SETTLE_MAX_WAIT, elapsed
    print(f"ok: TUI settle returned in {elapsed:.2f}s (cap {RelayPTY.SETTLE_TUI_MAX_WAIT}s)")


def test_caps_are_sane():
    assert RelayPTY.SETTLE_TUI_MAX_WAIT < RelayPTY.SETTLE_MAX_WAIT
    assert RelayPTY.SETTLE_TUI_QUIET_WINDOW < RelayPTY.SETTLE_QUIET_WINDOW
    assert RelayPTY.SETTLE_MAX_WAIT <= 30.0  # no more 120s hang
    print("ok: settle caps are sane "
          f"(tui max {RelayPTY.SETTLE_TUI_MAX_WAIT}s, normal max {RelayPTY.SETTLE_MAX_WAIT}s)")


# (test runner is defined at the end of this file, after all test functions)
def test_injection_finished_receives_before_and_after():
    """The status check must be shown BEFORE, injected input, and AFTER."""
    captured = {}

    class FakeBackend:
        def backend_request(self, method, params):
            captured["method"] = method
            messages = params["messages"]
            captured["system"] = messages[0]["content"]
            captured["user"] = messages[1]["content"]
            return "DONE"

    relay = RelayPTY.__new__(RelayPTY)
    relay._backend_client = FakeBackend()
    out = relay._injection_finished(
        "\x01n", screen_text="AFTER-CONTENT", before_text="BEFORE-CONTENT"
    )
    assert out is True, out
    assert captured["method"] == "complete", captured["method"]
    u = captured["user"]
    assert "BEFORE-CONTENT" in u, u
    assert "AFTER-CONTENT" in u, u
    assert "BEFORE" in u and "AFTER" in u
    # The injected input repr must be present so the model knows what was sent.
    assert "\\x01n" in u or "x01n" in u, u
    print("ok: _injection_finished sends before + injected + after to the model")


def test_injection_finished_running_keeps_waiting():
    class FakeBackend:
        def backend_request(self, method, params):
            return "RUNNING"

    relay = RelayPTY.__new__(RelayPTY)
    relay._backend_client = FakeBackend()
    out = relay._injection_finished("x", "after", "before")
    assert out is False, out
    print("ok: RUNNING verdict keeps waiting")


def test_injection_finished_without_a_backend_does_not_hang():
    relay = RelayPTY.__new__(RelayPTY)
    relay._backend_client = None
    assert relay._injection_finished("x", "after", "before") is True
    print("ok: no backend -> reports finished")


def test_the_status_check_does_not_retry():
    """It only asks whether to keep waiting, and a failure already means DONE.

    Inheriting a conversational turn's retry budget would make one stalled
    request cost several request timeouts, all to re-ask a question whose
    fallback answer we already have.
    """
    seen = {}

    class FakeBackend:
        def backend_request(self, method, params):
            seen.update(params)
            return "DONE"

    relay = RelayPTY.__new__(RelayPTY)
    relay._backend_client = FakeBackend()
    relay._injection_finished("x", "after", "before")
    assert seen.get("max_retries") == 0, seen.get("max_retries")
    print("ok: the status check asks for no retries")


class PromptScreen:
    """A screen whose cursor line and column we set directly."""

    in_alt_screen = False

    def __init__(self):
        self.display = [""]
        self.cursor = type("Cursor", (), {"x": 0, "y": 0})()

    def at(self, line):
        self.display = [line]
        self.cursor.x = len(line)


def make_injector():
    """A relay wired so _tool_inject_input reports the prompt it will watch."""
    relay = RelayPTY.__new__(RelayPTY)
    relay.screen = PromptScreen()
    relay._partial_line_prompt = None
    relay._inject_approval_all = True
    relay._master_fd = -1
    relay._write_all = lambda fd, data: None
    relay._wait_for_injection_to_settle = lambda injected, prompt_prefix="": (
        "WATCHING:" + prompt_prefix
    )
    return relay


def watched_prompt(relay, **args):
    out = relay._tool_inject_input(args)
    return out.split("WATCHING:", 1)[1].split("\n", 1)[0]


def test_the_prompt_survives_a_command_line_typed_in_chunks():
    """A long command line arrives as several injections; only the last submits.

    By then the cursor line is the prompt plus a half-typed command -- and once
    it has wrapped, the cursor row does not even show the prompt any more. A
    prefix learned at that moment can never match the prompt that returns when
    the command finishes, so the LLM-free fast path would be dead for exactly
    the longest commands.
    """
    relay = make_injector()
    prompt = "user@host:~ $ "

    relay.screen.at(prompt)
    watched_prompt(relay, text="ludvart_helper structured-patch /tmp/f.py ")
    # The typed line has grown past the screen width and wrapped.
    relay.screen.at("QUFBQkJC" * 9)
    watched_prompt(relay, text="--new-b64 QUFB")
    relay.screen.at("Q0NDRERE" * 9)
    final = watched_prompt(relay, text="QkJC", submit=True)
    assert final == prompt, final

    # Submitting ended the sequence, so the next command learns afresh.
    relay.screen.at("other@box:/tmp $ ")
    assert watched_prompt(relay, text="ls", submit=True) == "other@box:/tmp $ "
    print("ok: a chunked command line still watches for the real prompt")


def test_a_trailing_newline_ends_a_chunked_line_too():
    """submit=true is not the only way to execute; a trailing newline is one."""
    relay = make_injector()
    relay.screen.at("$ ")
    watched_prompt(relay, text="echo ")
    relay.screen.at("$ echo ")
    watched_prompt(relay, text="hi\\n")
    relay.screen.at("# ")
    assert watched_prompt(relay, text="ls", submit=True) == "# "
    print("ok: a trailing newline closes the chunk sequence")


def test_the_settle_cap_is_not_blown_by_a_status_check():
    """The cap only holds if we refuse to start a check we have no room for.

    An LLM round-trip cannot be called back once sent, so a check begun near the
    deadline runs past it however long it takes.
    """
    relay = RelayPTY.__new__(RelayPTY)
    relay.screen = PromptScreen()
    relay._backend_client = object()
    seq = ["before", "after", "after"]
    relay._safe_snapshot = lambda: seq.pop(0) if len(seq) > 1 else seq[0]
    relay._prompt_returned = lambda prefix: False
    checks = []

    def slow_check(injected, text, before=""):
        checks.append(time.time())
        time.sleep(1.0)
        return False  # never finished, so it keeps asking

    relay._injection_finished = slow_check

    start = time.time()
    relay._wait_for_injection_to_settle("x", prompt_prefix="p")
    elapsed = time.time() - start
    assert checks, "the status check never ran"
    assert elapsed <= RelayPTY.SETTLE_MAX_WAIT, (
        f"waited {elapsed:.1f}s against a {RelayPTY.SETTLE_MAX_WAIT}s cap"
    )
    # The last check must have started with room to finish inside the cap.
    room = RelayPTY.SETTLE_MAX_WAIT - (checks[-1] - start)
    assert room >= RelayPTY.SETTLE_CHECK_RESERVE - 0.5, room
    print(f"ok: {len(checks)} status checks fitted inside {elapsed:.1f}s")


if __name__ == "__main__":
    test_caps_are_sane()
    test_tui_returns_fast_without_prompt_or_llm()
    test_injection_finished_receives_before_and_after()
    test_injection_finished_running_keeps_waiting()
    test_injection_finished_without_a_backend_does_not_hang()
    test_the_status_check_does_not_retry()
    test_the_prompt_survives_a_command_line_typed_in_chunks()
    test_a_trailing_newline_ends_a_chunked_line_too()
    test_the_settle_cap_is_not_blown_by_a_status_check()
    print("ALL PASS")
