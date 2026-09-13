"""The one client-side part of ``/profile``: asking what to call a new profile.

Everything else about profiles lives on the backend (see
``test_backend_profiles.py``), because the files are on the backend host. Only
``add`` needs a prompt, so these tests cover that prompt and the payload it
forwards.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_ai_profile_command.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart.ludvart import _BACKEND_COMMANDS, Ludvart  # noqa: E402
from ludvart.panel import AiPanel  # noqa: E402


class _Backend:
    def request(self, line, host, payload=None):
        return {}


def _make_ludvart():
    r = Ludvart(["true"])
    r._panel = AiPanel(cols=80, height=10, provider="openai")
    r._render_split = lambda: None
    r._backend_client = _Backend()
    r.forwarded = {}
    r._forward_command_to_backend = lambda line, payload=None: r.forwarded.update(
        {"line": line, "payload": payload}
    )
    return r


def _systems(r):
    return [t for kind, t in r._panel.messages if kind == "system"]


def test_profile_is_a_backend_command():
    """Otherwise the client would try to answer /profile itself."""
    assert "profile" in _BACKEND_COMMANDS
    print("/profile is forwarded to the backend: OK")


def test_add_asks_for_a_name_then_forwards_it():
    r = _make_ludvart()
    r._profile_add_start(["investigator"])
    assert r._profile_add == {"dir": "investigator"}
    assert "investigator" in _systems(r)[-1]

    r._feed_profile_add("StabilityDB investigator")
    assert r._profile_add is None
    assert r.forwarded == {
        "line": "profile add",
        "payload": {
            "name": "StabilityDB investigator",
            "dir": "investigator",
        },
    }, r.forwarded
    print("/profile add asks for a name and forwards it: OK")


def test_a_trailing_slash_is_accepted():
    """Tab-completing a folder in the shell leaves one behind."""
    r = _make_ludvart()
    r._profile_add_start(["investigator/"])
    assert r._profile_add == {"dir": "investigator"}
    print("/profile add accepts a trailing slash on the folder: OK")


def test_add_without_a_folder_shows_the_usage():
    r = _make_ludvart()
    r._profile_add_start([])
    assert r._profile_add is None
    assert "Usage: /profile add" in _systems(r)[-1]
    assert not r.forwarded, r.forwarded
    print("/profile add without a folder shows the usage: OK")


def test_add_can_be_cancelled():
    r = _make_ludvart()
    r._profile_add_start(["x"])
    r._feed_profile_add("cancel")
    assert r._profile_add is None
    assert not r.forwarded, r.forwarded
    assert "cancelled" in _systems(r)[-1].lower()
    print("/profile add can be cancelled: OK")


def test_an_empty_name_cancels_rather_than_registering_a_nameless_profile():
    r = _make_ludvart()
    r._profile_add_start(["x"])
    r._feed_profile_add("   ")
    assert r._profile_add is None
    assert not r.forwarded, r.forwarded
    print("/profile add with an empty name cancels: OK")


def test_add_needs_a_backend():
    """The profile folders live on the backend host, so there is nothing to do."""
    r = _make_ludvart()
    r._backend_client = None
    r._profile_add_start(["x"])
    assert r._profile_add is None
    print("/profile add needs a backend: OK")


def test_the_slash_command_routes_add_to_the_prompt():
    r = _make_ludvart()
    r._handle_slash_command("/profile add notes")
    # The prompt is waiting; nothing has reached the backend yet.
    assert r._profile_add == {"dir": "notes"}
    assert not r.forwarded, r.forwarded
    print("/profile add is routed to the client prompt: OK")


def test_the_other_subcommands_go_straight_to_the_backend():
    r = _make_ludvart()
    r._handle_slash_command("/profile list")
    assert r._profile_add is None
    assert r.forwarded["line"] == "profile list", r.forwarded
    print("the other /profile subcommands are forwarded as-is: OK")


def main():
    test_profile_is_a_backend_command()
    test_add_asks_for_a_name_then_forwards_it()
    test_a_trailing_slash_is_accepted()
    test_add_without_a_folder_shows_the_usage()
    test_add_can_be_cancelled()
    test_an_empty_name_cancels_rather_than_registering_a_nameless_profile()
    test_add_needs_a_backend()
    test_the_slash_command_routes_add_to_the_prompt()
    test_the_other_subcommands_go_straight_to_the_backend()
    print("\nALL /profile client tests passed.")


if __name__ == "__main__":
    main()
