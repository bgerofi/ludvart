"""Hiding the AI panel parks the turn instead of killing it.

The user hides the panel to get at a terminal the agent is using -- usually to
Ctrl-C something wedged. The turn has to survive that, and it must not keep
typing while they are in there.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_panel_hide_park.py
"""

import threading
import time

from ludvart.ludvart import Ludvart as RelayPTY
from ludvart.ludvart import _duration
from ludvart.panel import AiPanel


def make_relay(thinking=True):
    relay = RelayPTY.__new__(RelayPTY)
    relay._panel = AiPanel(80, 8, "test")
    relay._panel.thinking = thinking
    relay._llm_request_in_flight = thinking
    relay._panel_closing = False
    relay._confirm_close = False
    relay._panel_hidden_at = None
    relay._park_expired = None
    relay._ask_cancel = threading.Event()
    relay._ask_done = threading.Event()
    relay._inject_approval_all = False
    relay._inject_approval_pending = False
    relay._panel_messages = []
    relay.SETTLE_POLL = 0.01
    relay.PARK_MAX_WAIT = 0.3
    relay._cancelled = []
    relay._cancel_ask = lambda: (
        relay._ask_cancel.set(),
        relay._cancelled.append(True),
    )
    return relay


def test_the_close_prompt_offers_a_way_out_that_keeps_the_turn():
    relay = make_relay()
    relay._request_toggle_close()
    prompt = relay._panel.confirm_prompt
    assert "(h)ide" in prompt, prompt
    assert "(a)bort" in prompt, prompt
    print("the close prompt offers a way out that keeps the turn: OK")


def test_hiding_closes_the_panel_without_cancelling():
    relay = make_relay()
    relay._request_toggle_close()
    relay._handle_confirm_close(b"h")
    assert relay._panel_closing is True
    assert relay._cancelled == [], "hiding must not cancel the turn"
    assert relay._ask_cancel.is_set() is False
    assert relay._panel_hidden_at is not None
    print("hiding closes the panel without cancelling: OK")


def test_aborting_still_cancels():
    """The old answer must keep its old meaning."""
    relay = make_relay()
    relay._request_toggle_close()
    relay._handle_confirm_close(b"a")
    assert relay._panel_closing is True
    assert relay._cancelled == [True]
    assert relay._panel_hidden_at is None
    print("aborting still cancels: OK")


def test_nothing_is_typed_while_the_panel_is_hidden():
    """The whole point: one writer on the terminal at a time."""
    relay = make_relay()
    relay._handle_confirm_close(b"h")
    relay._panel = None  # _leave_split has run
    approved = []

    def worker():
        approved.append(relay._await_inject_approval("make clean"))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.1)
    assert approved == [], "the tool call was allowed through a hidden panel"
    t.join(timeout=2)
    print("nothing is typed while the panel is hidden: OK")


def test_approve_everything_does_not_open_the_gate():
    """'always approve' answers a different question than 'is anyone there'."""
    relay = make_relay()
    relay._inject_approval_all = True
    relay._panel = None
    approved = []
    t = threading.Thread(
        target=lambda: approved.append(relay._await_inject_approval("rm -rf x")),
        daemon=True,
    )
    t.start()
    time.sleep(0.1)
    assert approved == [], "blanket approval bypassed the park"
    t.join(timeout=2)
    print("approve everything does not open the gate: OK")


def test_the_turn_carries_on_when_the_panel_comes_back():
    relay = make_relay()
    relay._inject_approval_all = True
    relay._panel = None
    approved = []
    t = threading.Thread(
        target=lambda: approved.append(relay._await_inject_approval("make")),
        daemon=True,
    )
    t.start()
    time.sleep(0.05)
    relay._panel = AiPanel(80, 8, "test")  # the user opened it again
    t.join(timeout=2)
    assert approved == [True], approved
    print("the turn carries on when the panel comes back: OK")


def test_a_turn_nobody_comes_back_to_is_dropped():
    relay = make_relay()
    relay._inject_approval_all = True
    relay._panel = None
    approved = []
    started = time.time()
    t = threading.Thread(
        target=lambda: approved.append(relay._await_inject_approval("make")),
        daemon=True,
    )
    t.start()
    t.join(timeout=5)
    assert approved == [False], approved
    assert time.time() - started >= relay.PARK_MAX_WAIT
    assert relay._cancelled == [True], "the parked turn was left standing"
    assert relay._park_expired is not None
    print("a turn nobody comes back to is dropped: OK")


def test_a_cancelled_turn_does_not_sit_in_the_park():
    relay = make_relay()
    relay._inject_approval_all = True
    relay._panel = None
    relay._ask_cancel.set()
    started = time.time()
    assert relay._await_inject_approval("make") is False
    assert time.time() - started < relay.PARK_MAX_WAIT
    print("a cancelled turn does not sit in the park: OK")


def open_again(relay):
    relay._panel = AiPanel(80, 8, "test")
    relay._resume_hidden_turn()
    return [text for who, text in relay._panel.messages]


def test_the_user_is_told_the_turn_was_picked_back_up():
    relay = make_relay()
    relay._handle_confirm_close(b"h")
    relay._panel_hidden_at = time.monotonic() - 75
    relay._panel = None
    said = open_again(relay)
    assert any("1m15s" in t for t in said), said
    assert any("Nothing was typed" in t for t in said), said
    assert relay._panel.thinking is True, "the spinner never restarted"
    print("the user is told the turn was picked back up: OK")


def test_the_user_is_told_when_the_parked_turn_expired():
    relay = make_relay()
    relay._handle_confirm_close(b"h")
    relay._panel = None
    relay._park_expired = "The turn was dropped: the panel stayed hidden."
    said = open_again(relay)
    assert any("dropped" in t for t in said), said
    assert relay._park_expired is None, "the notice would repeat on every open"
    print("the user is told when the parked turn expired: OK")


def test_an_ordinary_toggle_says_nothing():
    """Closing with no turn running must not produce a resume notice."""
    relay = make_relay(thinking=False)
    relay._request_toggle_close()
    assert relay._panel_closing is True
    assert relay._panel_hidden_at is None
    relay._panel = None
    assert open_again(relay) == []
    print("an ordinary toggle says nothing: OK")


def test_durations_read_like_durations():
    assert _duration(0) == "0s"
    assert _duration(45.7) == "45s"
    assert _duration(60) == "1m00s"
    assert _duration(605) == "10m05s"
    print("durations read like durations: OK")


def main():
    test_the_close_prompt_offers_a_way_out_that_keeps_the_turn()
    test_hiding_closes_the_panel_without_cancelling()
    test_aborting_still_cancels()
    test_nothing_is_typed_while_the_panel_is_hidden()
    test_approve_everything_does_not_open_the_gate()
    test_the_turn_carries_on_when_the_panel_comes_back()
    test_a_turn_nobody_comes_back_to_is_dropped()
    test_a_cancelled_turn_does_not_sit_in_the_park()
    test_the_user_is_told_the_turn_was_picked_back_up()
    test_the_user_is_told_when_the_parked_turn_expired()
    test_an_ordinary_toggle_says_nothing()
    test_durations_read_like_durations()
    print("\nALL panel hide/park tests passed.")


if __name__ == "__main__":
    main()
