"""Live activity progress: elapsed-time hint on the spinner during waits.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_ai_activity_progress.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart.panel import AiPanel  # noqa: E402
from ludvart.ludvart import Ludvart  # noqa: E402


def _make_ludvart():
    runner = Ludvart(["true"])
    runner._panel = AiPanel(cols=80, height=10, provider="openai")
    runner._render_split = lambda: None
    return runner


# -- panel rendering --------------------------------------------------------


def test_spinner_shows_elapsed_when_set():
    panel = AiPanel(cols=60, height=8, provider="openai")
    panel.thinking = True
    panel.activity = "Thinking"
    panel.activity_elapsed = 8.0
    blob = b"".join(panel.render(8, 60))
    assert b"Thinking (openai)" in blob, blob
    assert b"Thinking (openai) - 8s" in blob, blob
    print("spinner shows provider and elapsed seconds: OK")


def test_spinner_hides_elapsed_when_none():
    panel = AiPanel(cols=60, height=8, provider="openai")
    panel.thinking = True
    panel.activity = "Thinking"
    panel.activity_elapsed = None
    blob = b"".join(panel.render(8, 60))
    assert b"Thinking (openai)" in blob, blob
    assert b"Thinking (openai) - " not in blob, blob
    print("spinner hides elapsed when unset: OK")


def test_spinner_elapsed_on_tool_label():
    panel = AiPanel(cols=60, height=8, provider="openai")
    panel.thinking = True
    panel.activity = "Calling inject_input"
    panel.activity_elapsed = 12.0
    blob = b"".join(panel.render(8, 60))
    assert b"Calling inject_input" in blob, blob
    assert b"Calling inject_input - 12s" in blob, blob
    print("spinner shows elapsed for a running tool: OK")


# -- controller wait lifecycle ---------------------------------------------


def test_begin_wait_sets_clock_and_label():
    runner = _make_ludvart()
    runner._begin_wait("Thinking")
    assert runner._panel.activity == "Thinking"
    assert runner._panel.activity_elapsed is None
    assert runner._wait_since is not None
    assert runner._wait_streaming is False
    print("begin_wait sets the label and starts the clock: OK")


def test_refresh_wait_below_threshold_hides_elapsed():
    runner = _make_ludvart()
    runner._panel.thinking = True
    runner._begin_wait("Thinking")
    runner._refresh_wait()
    assert runner._panel.activity_elapsed is None
    print("refresh keeps elapsed hidden below the threshold: OK")


def test_refresh_wait_past_threshold_shows_elapsed():
    runner = _make_ludvart()
    runner._panel.thinking = True
    runner._begin_wait("Thinking")
    runner._wait_since = time.monotonic() - 5.0
    runner._refresh_wait()
    assert runner._panel.activity_elapsed is not None
    assert runner._panel.activity_elapsed >= runner.ACTIVITY_ELAPSED_HINT
    print("refresh reveals elapsed once the threshold passes: OK")


def test_streaming_suppresses_elapsed():
    runner = _make_ludvart()
    runner._panel.thinking = True
    runner._begin_wait("Thinking")
    runner._wait_since = time.monotonic() - 5.0
    runner._mark_wait_streaming()
    runner._refresh_wait()
    assert runner._wait_streaming is True
    assert runner._panel.activity_elapsed is None
    print("streaming output suppresses the elapsed counter: OK")


def test_end_wait_clears_clock_and_elapsed():
    runner = _make_ludvart()
    runner._panel.thinking = True
    runner._begin_wait("Thinking")
    runner._wait_since = time.monotonic() - 5.0
    runner._refresh_wait()
    assert runner._panel.activity_elapsed is not None
    runner._end_wait()
    assert runner._wait_since is None
    assert runner._panel.activity_elapsed is None
    print("end_wait stops and hides the elapsed counter: OK")


def test_cancel_ask_clears_wait():
    runner = _make_ludvart()
    runner._panel.thinking = True
    runner._llm_request_in_flight = True
    runner._begin_wait("Calling inject_input")
    runner._wait_since = time.monotonic() - 5.0
    runner._refresh_wait()
    assert runner._panel.activity_elapsed is not None
    runner._cancel_ask()
    assert runner._wait_since is None
    assert runner._panel.activity_elapsed is None
    assert runner._panel.thinking is False
    print("cancelling a request clears the wait indicator: OK")


# -- backend (split) mode: the clock still runs on the client ---------------


def test_backend_activity_starts_the_client_clock():
    """A backend "activity" update begins a locally timed waiting phase."""
    from ludvart.ludvart import _ClientTerminalHost

    runner = _make_ludvart()
    runner._panel.thinking = True
    host = _ClientTerminalHost(runner)

    host.set_activity("Thinking")
    assert runner._panel.activity == "Thinking"
    assert runner._wait_since is not None
    # The elapsed seconds are measured here, not sent by the backend.
    runner._wait_since = time.monotonic() - 5.0
    runner._refresh_wait()
    assert runner._panel.activity_elapsed is not None
    print("backend activity starts the client-side elapsed clock: OK")


def test_backend_narration_suppresses_elapsed():
    from ludvart.ludvart import _ClientTerminalHost

    runner = _make_ludvart()
    runner._panel.thinking = True
    host = _ClientTerminalHost(runner)
    host.set_activity("Thinking")
    runner._wait_since = time.monotonic() - 5.0

    host.narrate("streaming...")
    runner._refresh_wait()
    assert runner._wait_streaming is True
    assert runner._panel.activity_elapsed is None
    assert runner._panel.interim == "streaming..."
    print("backend narration suppresses the elapsed counter: OK")


def test_backend_tool_activity_restarts_the_clock():
    from ludvart.ludvart import _ClientTerminalHost

    runner = _make_ludvart()
    runner._panel.thinking = True
    host = _ClientTerminalHost(runner)
    host.set_activity("Thinking")
    host.narrate("streaming...")
    runner._wait_since = time.monotonic() - 5.0

    host.set_activity("Calling inject_input")
    assert runner._panel.activity == "Calling inject_input"
    assert runner._wait_streaming is False  # a fresh wait, elapsed shows again
    assert runner._panel.activity_elapsed is None
    runner._wait_since = time.monotonic() - 5.0
    runner._refresh_wait()
    assert runner._panel.activity_elapsed is not None
    print("a backend tool step restarts the elapsed clock: OK")


def test_backend_context_pct_updates_the_badge():
    from ludvart.ludvart import _ClientTerminalHost

    runner = _make_ludvart()
    host = _ClientTerminalHost(runner)

    host.set_context_usage(3200, 42.0)
    assert runner._panel.context_pct == 42.0
    # Also cached so the badge survives a panel toggle.
    assert runner._panel_context_pct == 42.0
    blob = b"".join(runner._panel.render(10, 80))
    assert b"[c:3.2k(42%)]" in blob, blob
    print("backend context usage updates the client badge: OK")


def main():
    test_spinner_shows_elapsed_when_set()
    test_spinner_hides_elapsed_when_none()
    test_spinner_elapsed_on_tool_label()
    test_begin_wait_sets_clock_and_label()
    test_refresh_wait_below_threshold_hides_elapsed()
    test_refresh_wait_past_threshold_shows_elapsed()
    test_streaming_suppresses_elapsed()
    test_end_wait_clears_clock_and_elapsed()
    test_cancel_ask_clears_wait()
    test_backend_activity_starts_the_client_clock()
    test_backend_narration_suppresses_elapsed()
    test_backend_tool_activity_restarts_the_clock()
    test_backend_context_pct_updates_the_badge()
    print("\nALL activity progress tests passed.")


if __name__ == "__main__":
    main()
