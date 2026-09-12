"""Backend-side /profile command round-trips and session profile restore.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_backend_profiles.py
"""

import contextlib
import os
import shutil
import tempfile
import threading
from pathlib import Path

from ludvart.backend_client import BackendClient
from ludvart.profiles import (
    LARGE_PROFILE_TOKENS,
    active_profile,
    add_profile,
    load_profiles,
    save_profiles,
    set_active,
)
from ludvart.protocol import FrameChannel
from ludvart.server import _FakeBackendLLM, serve
from ludvart.session import SessionStore

from test_backend_sessions import RecordingHost, _pipe_pair, _tmp_sessions


@contextlib.contextmanager
def _tmp_profiles():
    """Point the profile store at a throwaway directory for the duration."""
    old = os.environ.get("LUDVART_PROFILES_DIR")
    root = tempfile.mkdtemp(prefix="ludvart_prof_")
    os.environ["LUDVART_PROFILES_DIR"] = root
    try:
        yield Path(root)
    finally:
        if old is None:
            os.environ.pop("LUDVART_PROFILES_DIR", None)
        else:
            os.environ["LUDVART_PROFILES_DIR"] = old
        shutil.rmtree(root, ignore_errors=True)


def _run_commands(commands, session=None):
    """Run ``(line, payload)`` pairs against a live backend, return the host."""
    session = session or SessionStore.create_new()
    client_ch, backend_ch = _pipe_pair()
    t = threading.Thread(
        target=lambda: serve(backend_ch, llm=_FakeBackendLLM(), session=session),
        daemon=True,
    )
    t.start()
    client = BackendClient(client_ch)
    host = RecordingHost()
    assert client_ch.recv()["type"] == "hello"
    for entry in commands:
        line, payload = entry if isinstance(entry, tuple) else (entry, None)
        client.command(line, host, payload=payload)
    client_ch.close()
    t.join(timeout=2)
    backend_ch.close()
    return host


def test_profile_list_is_helpful_when_empty():
    with _tmp_sessions(), _tmp_profiles():
        host = _run_commands(["profile list"])
        assert any("No profiles registered" in s for s in host.systems), host.systems
    print("/profile list explains how to add one when empty: OK")


def test_profile_add_registers_the_file():
    with _tmp_sessions(), _tmp_profiles() as root:
        (root / "inv.md").write_text("how to investigate", encoding="utf-8")
        host = _run_commands(
            [("profile add", {"name": "investigator", "file": "inv.md"})]
        )

        assert any("Registered profile" in s for s in host.systems), host.systems
        profiles = load_profiles()
        assert [p["name"] for p in profiles] == ["investigator"]
        # Adding does not activate; /profile use is the explicit second step.
        assert active_profile(profiles) is None
    print("/profile add registers the file without activating it: OK")


def test_profile_add_refuses_a_file_that_is_not_there():
    with _tmp_sessions(), _tmp_profiles():
        host = _run_commands([("profile add", {"name": "x", "file": "gone.md"})])
        assert any("gone.md" in s for s in host.systems), host.systems
        assert load_profiles() == []
    print("/profile add refuses a file that is not in the profiles dir: OK")


def test_profile_add_refuses_a_path_outside_the_profiles_dir():
    with _tmp_sessions(), _tmp_profiles():
        host = _run_commands(
            [("profile add", {"name": "x", "file": "../../etc/passwd"})]
        )
        assert any("Not a profile filename" in s for s in host.systems), host.systems
        assert load_profiles() == []
    print("/profile add refuses a path outside the profiles dir: OK")


def test_profile_list_shows_index_name_file_and_cost():
    with _tmp_sessions(), _tmp_profiles() as root:
        (root / "a.md").write_text("x" * 4000, encoding="utf-8")
        (root / "b.md").write_text("y" * 400, encoding="utf-8")
        save_profiles(
            set_active(add_profile(add_profile([], "alpha", "a.md"), "beta", "b.md"), 1)
        )

        host = _run_commands(["profile list"])
        listing = "\n".join(host.systems)
        assert "1. alpha" in listing, listing
        assert "(a.md)" in listing and "(b.md)" in listing, listing
        assert "1.0k tokens" in listing, listing
        assert "100 tokens" in listing, listing
        # The active one is marked.
        assert any(s.startswith("*2. beta") for s in host.systems), host.systems
    print("/profile list shows the index, name, file and token cost: OK")


def test_profile_list_warns_about_a_large_profile():
    with _tmp_sessions(), _tmp_profiles() as root:
        (root / "big.md").write_text("x" * (LARGE_PROFILE_TOKENS * 4 + 8), "utf-8")
        (root / "small.md").write_text("small", encoding="utf-8")
        save_profiles(add_profile(add_profile([], "big", "big.md"), "small", "small.md"))

        host = _run_commands(["profile list"])
        big = next(s for s in host.systems if "big.md" in s)
        small = next(s for s in host.systems if "small.md" in s)
        assert "large" in big and "compacting" in big, big
        assert "large" not in small, small
    print("/profile list warns about a profile that is too large: OK")


def test_profile_list_flags_a_missing_file():
    with _tmp_sessions(), _tmp_profiles():
        save_profiles(add_profile([], "ghost", "ghost.md"))
        host = _run_commands(["profile list"])
        assert any("missing" in s for s in host.systems), host.systems
    print("/profile list flags a profile whose file has gone: OK")


def test_profile_use_switches_and_none_clears():
    with _tmp_sessions(), _tmp_profiles() as root:
        (root / "a.md").write_text("A", encoding="utf-8")
        (root / "b.md").write_text("B", encoding="utf-8")
        save_profiles(add_profile(add_profile([], "alpha", "a.md"), "beta", "b.md"))

        _run_commands(["profile use 2"])
        assert active_profile(load_profiles())["name"] == "beta"
        # By name too.
        _run_commands(["profile use alpha"])
        assert active_profile(load_profiles())["name"] == "alpha"
        # And switched off again.
        host = _run_commands(["profile use none"])
        assert active_profile(load_profiles()) is None
        assert any("No profile in use" in s for s in host.systems), host.systems
    print("/profile use switches by index or name, and none clears: OK")


def test_profile_use_reports_an_unknown_profile():
    with _tmp_sessions(), _tmp_profiles():
        host = _run_commands(["profile use 7"])
        assert any("No profile matches" in s for s in host.systems), host.systems
    print("/profile use reports an unknown profile: OK")


def test_profile_delete_unregisters_but_keeps_the_file():
    with _tmp_sessions(), _tmp_profiles() as root:
        (root / "a.md").write_text("A", encoding="utf-8")
        save_profiles(set_active(add_profile([], "alpha", "a.md"), 0))

        host = _run_commands(["profile delete 1"])
        assert load_profiles() == []
        assert (root / "a.md").is_file()
        assert any("a.md" in s for s in host.systems), host.systems
    print("/profile delete unregisters but leaves the markdown: OK")


def test_unknown_profile_subcommand_is_reported():
    with _tmp_sessions(), _tmp_profiles():
        host = _run_commands(["profile frobnicate"])
        assert any("Unknown subcommand" in s for s in host.systems), host.systems
    print("/profile rejects an unknown subcommand: OK")


def test_session_load_restores_the_session_profile():
    with _tmp_sessions(), _tmp_profiles() as root:
        (root / "a.md").write_text("A", encoding="utf-8")
        (root / "b.md").write_text("B", encoding="utf-8")
        save_profiles(add_profile(add_profile([], "alpha", "a.md"), "beta", "b.md"))

        saved = SessionStore.create_new()
        saved.save(
            [("you", "q"), ("ludvart", "a")],
            [{"role": "user", "content": "q"}],
            provider="custom",
            profile="beta",
        )
        _run_commands(["profile use alpha", f"session load {saved.session_id}"])

        assert active_profile(load_profiles())["name"] == "beta"
    print("/session load switches to the session's profile: OK")


def test_session_load_keeps_the_current_profile_when_unavailable():
    with _tmp_sessions(), _tmp_profiles() as root:
        (root / "a.md").write_text("A", encoding="utf-8")
        save_profiles(set_active(add_profile([], "alpha", "a.md"), 0))

        saved = SessionStore.create_new()
        saved.save([("you", "q")], [], provider="custom", profile="long-gone")
        host = _run_commands([f"session load {saved.session_id}"])

        assert active_profile(load_profiles())["name"] == "alpha"
        assert any("long-gone" in s for s in host.systems), host.systems
    print("/session load keeps the current profile when the saved one is gone: OK")


def test_session_without_a_profile_field_loads_unchanged():
    """Sessions saved before profiles existed must still load."""
    with _tmp_sessions(), _tmp_profiles() as root:
        (root / "a.md").write_text("A", encoding="utf-8")
        save_profiles(set_active(add_profile([], "alpha", "a.md"), 0))

        saved = SessionStore.create_new()
        saved.save([("you", "q")], [], provider="custom")
        host = _run_commands([f"session load {saved.session_id}"])

        assert active_profile(load_profiles())["name"] == "alpha"
        assert any("Loaded session" in s for s in host.systems), host.systems
    print("a session with no profile field loads and changes nothing: OK")


def main():
    test_profile_list_is_helpful_when_empty()
    test_profile_add_registers_the_file()
    test_profile_add_refuses_a_file_that_is_not_there()
    test_profile_add_refuses_a_path_outside_the_profiles_dir()
    test_profile_list_shows_index_name_file_and_cost()
    test_profile_list_warns_about_a_large_profile()
    test_profile_list_flags_a_missing_file()
    test_profile_use_switches_and_none_clears()
    test_profile_use_reports_an_unknown_profile()
    test_profile_delete_unregisters_but_keeps_the_file()
    test_unknown_profile_subcommand_is_reported()
    test_session_load_restores_the_session_profile()
    test_session_load_keeps_the_current_profile_when_unavailable()
    test_session_without_a_profile_field_loads_unchanged()
    print("\nALL backend profile tests passed.")


if __name__ == "__main__":
    main()
