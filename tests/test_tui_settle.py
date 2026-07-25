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


if __name__ == "__main__":
    test_caps_are_sane()
    test_tui_returns_fast_without_prompt_or_llm()
    test_injection_finished_receives_before_and_after()
    test_injection_finished_running_keeps_waiting()
    test_injection_finished_without_a_backend_does_not_hang()
    print("ALL PASS")
