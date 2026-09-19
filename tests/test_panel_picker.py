"""Ctrl-P / Ctrl-S / Ctrl-L open a list to pick a profile, session or model.

Typing "/session list", reading the numbers, then typing "/session load 7" is
three steps to do one thing. The picker is the same three registries with the
reading and the retyping taken out.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_panel_picker.py
"""

import threading
import time

from ludvart.ludvart import PICKER_KEYS
from ludvart.ludvart import Ludvart as RelayPTY
from ludvart.panel import AiPanel
from ludvart.picker import APPLY_COMMAND, Picker


def items(*names, active=None):
    return [
        {"label": n, "ref": str(i), "active": n == active}
        for i, n in enumerate(names, 1)
    ]


def make_relay(thinking=False, reply=None, fail=None):
    relay = RelayPTY.__new__(RelayPTY)
    relay._panel = AiPanel(80, 8, "test")
    relay._panel.thinking = thinking
    relay._model_add = None
    relay._profile_add = None
    relay._rendered = []
    relay._render_split = lambda: relay._rendered.append(True)
    relay._forwarded = []
    relay._forward_command_to_backend = lambda line: relay._forwarded.append(line)

    class _Backend:
        def request(self, line, host):
            if fail is not None:
                raise ConnectionError(fail)
            return {"items": reply or []}

    relay._backend_client = None if reply is None and fail is None else _Backend()
    return relay


def open_and_settle(relay, key):
    relay._panel_key(key)
    for _ in range(200):
        picker = relay._panel.picker
        if picker is None or not picker.loading:
            break
        time.sleep(0.01)
    return relay._panel.picker


# -- the widget ----------------------------------------------------------


def test_the_cursor_starts_on_what_is_already_in_use():
    p = Picker("profile")
    p.set_items(items("alpha", "beta", "gamma", active="beta"))
    assert p.selected()["label"] == "beta"
    print("the cursor starts on what is already in use: OK")


def test_the_cursor_cannot_leave_the_list():
    p = Picker("model")
    p.set_items(items("one", "two"))
    p.move(-5)
    assert p.index == 0
    p.move(50)
    assert p.index == 1
    print("the cursor cannot leave the list: OK")


def test_a_long_list_scrolls_to_follow_the_cursor():
    p = Picker("session")
    p.set_items(items(*[f"s{i}" for i in range(30)]))
    rows = 5
    p.render(rows, 40)
    assert p.top == 0
    p.move(29)
    p.render(rows, 40)
    assert p.top == 25, p.top
    shown = [r for r in p.render(rows, 40)]
    assert b"s29" in shown[-1], shown[-1]
    p.to_end(False)
    p.render(rows, 40)
    assert p.top == 0
    print("a long list scrolls to follow the cursor: OK")


def test_the_highlighted_row_is_the_only_one_inverted():
    p = Picker("model")
    p.set_items(items("one", "two", "three"))
    p.move(1)
    rows = p.render(3, 40)
    inverted = [i for i, r in enumerate(rows) if b"\x1b[7m" in r]
    assert inverted == [1], inverted
    print("the highlighted row is the only one inverted: OK")


def test_an_empty_list_says_so_instead_of_looking_broken():
    p = Picker("profile")
    p.set_items([])
    rows = p.render(3, 40)
    assert b"Nothing to choose" in rows[0], rows[0]
    assert p.selected() is None
    print("an empty list says so: OK")


def test_a_list_that_never_arrives_says_why():
    p = Picker("session")
    p.fail("Could not reach the backend: gone")
    rows = p.render(3, 40)
    assert b"gone" in rows[0], rows[0]
    assert p.loading is False
    print("a list that never arrives says why: OK")


def test_the_picker_always_fills_its_rows():
    p = Picker("model")
    p.set_items(items("only"))
    assert len(p.render(6, 40)) == 6
    p.set_items(items(*[f"m{i}" for i in range(20)]))
    assert len(p.render(6, 40)) == 6
    print("the picker always fills its rows: OK")


# -- the panel -----------------------------------------------------------


def test_the_picker_takes_over_the_panel():
    panel = AiPanel(40, 6, "test")
    panel.add_user("a question nobody should see now")
    panel.picker = Picker("session")
    panel.picker.set_items(items("one", "two"))
    rows = panel.render(height=6, cols=40)
    assert len(rows) == 6
    body = b"".join(rows)
    assert b"a question nobody" not in body
    assert b"Sessions" in body
    assert b"Enter:choose" in body
    print("the picker takes over the panel: OK")


def test_the_cursor_sits_on_the_highlighted_row():
    panel = AiPanel(40, 6, "test")
    panel.picker = Picker("model")
    panel.picker.set_items(items("one", "two", "three"))
    panel.picker.move(2)
    panel.render(height=6, cols=40)
    row, col = panel.cursor_rowcol()
    assert (row, col) == (3, 1), (row, col)
    print("the cursor sits on the highlighted row: OK")


def test_the_keys_are_advertised():
    panel = AiPanel(120, 6, "test")
    header = panel.render(height=6, cols=120)[0]
    assert b"^P/^S/^L" in header, header
    print("the keys are advertised: OK")


# -- the wiring ----------------------------------------------------------


def test_each_key_opens_its_own_list():
    for key, kind in PICKER_KEYS.items():
        relay = make_relay(reply=items("a"))
        picker = open_and_settle(relay, key)
        assert picker is not None and picker.kind == kind, key
    assert b"\x0d" not in PICKER_KEYS, "Ctrl-M is Enter and cannot be bound"
    print("each key opens its own list: OK")


def test_the_keys_do_nothing_mid_turn():
    """A choice replaces what the turn is standing on, so not while it runs."""
    for key in PICKER_KEYS:
        relay = make_relay(thinking=True, reply=items("a"))
        relay._panel_key(key)
        assert relay._panel.picker is None, key
    print("the keys do nothing mid-turn: OK")


def test_the_list_is_fetched_without_freezing_the_panel():
    relay = make_relay(reply=items("alpha", "beta", active="beta"))
    started = threading.Event()

    class _Slow:
        def request(self, line, host):
            started.set()
            time.sleep(0.05)
            return {"items": items("alpha", "beta", active="beta")}

    relay._backend_client = _Slow()
    relay._panel_key(b"\x10")
    # The picker is on screen before the answer is, saying what it is doing.
    assert relay._panel.picker.loading is True
    assert b"Loading" in relay._panel.picker.render(2, 40)[0]
    assert started.wait(2)
    for _ in range(200):
        if not relay._panel.picker.loading:
            break
        time.sleep(0.01)
    assert relay._panel.picker.selected()["label"] == "beta"
    print("the list is fetched without freezing the panel: OK")


def test_choosing_runs_the_command_you_would_have_typed():
    for kind, key in ((k, b) for b, k in PICKER_KEYS.items()):
        relay = make_relay(reply=items("one", "two", "three"))
        open_and_settle(relay, key)
        relay._handle_picker_input(b"\x1b[B")  # down to the second entry
        relay._handle_picker_input(b"\r")
        assert relay._forwarded == [f"{APPLY_COMMAND[kind]} 2"], (kind, relay._forwarded)
        assert relay._panel.picker is None, "the list stayed up after choosing"
    print("choosing runs the command you would have typed: OK")


def test_escape_leaves_everything_alone():
    relay = make_relay(reply=items("one", "two"))
    open_and_settle(relay, b"\x13")
    relay._handle_picker_input(b"\x1b")
    assert relay._panel.picker is None
    assert relay._forwarded == []
    print("escape leaves everything alone: OK")


def test_keys_go_to_the_list_and_not_the_input_line():
    relay = make_relay(reply=items("one", "two"))
    relay._inject_approval_pending = False
    open_and_settle(relay, b"\x10")
    relay._panel_input(b"x")
    assert relay._panel.editor.text == "", "typing leaked into the draft"
    relay._panel_input(b"\x1b[B")
    assert relay._panel.picker.index == 1
    print("keys go to the list and not the input line: OK")


def test_a_list_that_arrives_after_escape_is_dropped():
    """The fetch outlives the picker it was for; it must not raise the dead."""
    relay = make_relay()
    picker = Picker("profile")
    relay._panel.picker = picker
    relay._panel.picker = None

    class _Late:
        def request(self, line, host):
            return {"items": items("late")}

    relay._backend_client = _Late()
    relay._fill_picker(picker)
    assert relay._panel.picker is None
    print("a list that arrives after escape is dropped: OK")


def test_a_backend_that_is_not_there_is_reported():
    relay = make_relay(fail="link down")
    picker = open_and_settle(relay, b"\x0c")
    assert "link down" in picker.error, picker.error
    print("a backend that is not there is reported: OK")


def test_the_loop_waits_for_a_list_that_is_still_coming():
    """Idle, the loop sleeps in select; nothing would draw the list that lands."""
    relay = make_relay()
    relay._panel.thinking = False
    assert relay._picker_is_loading() is False
    relay._panel.picker = Picker("profile")
    assert relay._picker_is_loading() is True
    relay._panel.picker.set_items(items("a"))
    assert relay._picker_is_loading() is False
    print("the loop waits for a list that is still coming: OK")


# -- the backend side ----------------------------------------------------


def test_the_backend_numbers_models_the_way_model_use_expects():
    from types import SimpleNamespace

    from ludvart.server import _handle_pick

    manager = SimpleNamespace(
        models=[
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "anthropic", "model": "claude"},
        ],
        available=[True, False],
        active_index=lambda: 1,
    )
    picked = _handle_pick(["model"], manager, None)["items"]
    assert [i["ref"] for i in picked] == ["1", "2"], picked
    assert [i["active"] for i in picked] == [False, True]
    assert "unavailable" in picked[1]["label"]
    assert "unavailable" not in picked[0]["label"]
    print("the backend numbers models the way model use expects: OK")


def test_the_backend_numbers_sessions_from_one_and_remembers_the_list():
    from types import SimpleNamespace

    import ludvart.session as session_mod
    from ludvart.server import _handle_pick

    listed = [
        {"id": "aaa", "count": 3, "title": "first  thing"},
        {"id": "bbb", "count": 9, "title": "second"},
    ]
    core = SimpleNamespace(session_list=[], session=SimpleNamespace(session_id="bbb"))
    saved = session_mod.list_sessions
    session_mod.list_sessions = lambda: listed
    try:
        picked = _handle_pick(["session"], None, core)["items"]
    finally:
        session_mod.list_sessions = saved
    assert [i["ref"] for i in picked] == ["1", "2"], picked
    assert picked[1]["active"] is True
    assert "first thing" in picked[0]["label"], picked[0]["label"]
    # The refs are positions in this list, so the backend has to be holding it.
    assert core.session_list == listed
    print("the backend numbers sessions from one and remembers the list: OK")


def test_the_profile_list_offers_running_without_one():
    import ludvart.profiles as profiles_mod
    from ludvart.server import _handle_pick

    saved = profiles_mod.load_profiles
    profiles_mod.load_profiles = lambda: [
        {"name": "work", "dir": "work", "active": False},
        {"name": "play", "dir": "play", "active": True},
    ]
    try:
        picked = _handle_pick(["profile"], None, None)["items"]
    finally:
        profiles_mod.load_profiles = saved
    assert picked[0] == {"label": "(Clear profile)", "ref": "none", "active": False}
    # The real profiles keep the numbers /profile use already gave them.
    assert [i["ref"] for i in picked[1:]] == ["1", "2"]
    assert picked[2]["active"] is True
    print("the profile list offers running without one: OK")


def test_clearing_is_marked_current_when_no_profile_is_loaded():
    import ludvart.profiles as profiles_mod
    from ludvart.server import _handle_pick

    saved = profiles_mod.load_profiles
    profiles_mod.load_profiles = lambda: [{"name": "work", "dir": "work"}]
    try:
        picked = _handle_pick(["profile"], None, None)["items"]
    finally:
        profiles_mod.load_profiles = saved
    assert picked[0]["active"] is True
    print("clearing is marked current when no profile is loaded: OK")


def test_an_unknown_kind_is_an_empty_list_not_a_crash():
    from ludvart.server import _handle_pick

    assert _handle_pick(["nonsense"], None, None) == {"items": []}
    assert _handle_pick([], None, None) == {"items": []}
    print("an unknown kind is an empty list not a crash: OK")


def main():
    test_the_cursor_starts_on_what_is_already_in_use()
    test_the_cursor_cannot_leave_the_list()
    test_a_long_list_scrolls_to_follow_the_cursor()
    test_the_highlighted_row_is_the_only_one_inverted()
    test_an_empty_list_says_so_instead_of_looking_broken()
    test_a_list_that_never_arrives_says_why()
    test_the_picker_always_fills_its_rows()
    test_the_picker_takes_over_the_panel()
    test_the_cursor_sits_on_the_highlighted_row()
    test_the_keys_are_advertised()
    test_each_key_opens_its_own_list()
    test_the_keys_do_nothing_mid_turn()
    test_the_list_is_fetched_without_freezing_the_panel()
    test_choosing_runs_the_command_you_would_have_typed()
    test_escape_leaves_everything_alone()
    test_keys_go_to_the_list_and_not_the_input_line()
    test_a_list_that_arrives_after_escape_is_dropped()
    test_a_backend_that_is_not_there_is_reported()
    test_the_loop_waits_for_a_list_that_is_still_coming()
    test_the_backend_numbers_models_the_way_model_use_expects()
    test_the_backend_numbers_sessions_from_one_and_remembers_the_list()
    test_the_profile_list_offers_running_without_one()
    test_clearing_is_marked_current_when_no_profile_is_loaded()
    test_an_unknown_kind_is_an_empty_list_not_a_crash()
    print("\nALL panel picker tests passed.")


if __name__ == "__main__":
    main()
