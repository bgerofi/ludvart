"""Unit tests for the agent-profile registry (profiles.py).

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_profiles.py
"""

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from ludvart.profiles import (
    ActiveProfile,
    LARGE_PROFILE_TOKENS,
    active_profile,
    add_profile,
    config_path,
    estimate_tokens,
    find_profile,
    load_profiles,
    profile_path,
    profile_section,
    profiles_dir,
    read_profile_text,
    remove_profile,
    save_profiles,
    set_active,
    valid_filename,
)


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


def _write(root: Path, name: str, text: str) -> None:
    (root / name).write_text(text, encoding="utf-8")


def test_valid_filename_only_accepts_a_bare_md_name():
    assert valid_filename("investigator.md")
    assert valid_filename("stabilitydb-investigator.md")
    # A profile is a file *in* the profiles directory, never a path to one.
    assert not valid_filename("sub/dir.md")
    assert not valid_filename("../../etc/passwd.md")
    assert not valid_filename("..\\windows.md")
    assert not valid_filename("notes.txt")
    assert not valid_filename("")
    print("valid_filename accepts only a bare *.md name: OK")


def test_profile_path_refuses_to_leave_the_profiles_dir():
    with _tmp_profiles() as root:
        assert profile_path("ok.md") == root / "ok.md"
        assert profile_path("../ok.md") is None
        assert profile_path("/etc/passwd") is None
    print("profile_path refuses anything outside the profiles dir: OK")


def test_add_list_and_activate_round_trip():
    with _tmp_profiles() as root:
        _write(root, "a.md", "how to investigate A")
        _write(root, "b.md", "how to investigate B")

        profiles = add_profile([], "alpha", "a.md")
        profiles = add_profile(profiles, "beta", "b.md")
        # Adding never activates: choosing is a separate, explicit step.
        assert active_profile(profiles) is None, profiles
        save_profiles(profiles)

        loaded = load_profiles()
        assert [p["name"] for p in loaded] == ["alpha", "beta"]
        assert [p["file"] for p in loaded] == ["a.md", "b.md"]

        save_profiles(set_active(loaded, 1))
        assert active_profile(load_profiles())["name"] == "beta"
        # "No profile" is a legal state, unlike the model registry.
        save_profiles(set_active(load_profiles(), None))
        assert active_profile(load_profiles()) is None
    print("profiles round-trip through config.json: OK")


def test_add_rejects_a_duplicate_name_and_a_bad_file():
    with _tmp_profiles():
        profiles = add_profile([], "alpha", "a.md")
        for bad in (("alpha", "other.md"), ("", "a.md"), ("gamma", "a.txt"),
                    ("gamma", "../a.md")):
            try:
                add_profile(profiles, *bad)
            except ValueError:
                continue
            raise AssertionError(f"accepted {bad!r}")
    print("add_profile rejects duplicates and non-profile files: OK")


def test_remove_drops_the_entry_and_leaves_the_file():
    with _tmp_profiles() as root:
        _write(root, "a.md", "keep this file")
        save_profiles(add_profile([], "alpha", "a.md"))

        save_profiles(remove_profile(load_profiles(), 0))

        assert load_profiles() == []
        # Unregistering is not deleting the user's markdown.
        assert (root / "a.md").is_file()
    print("remove_profile unregisters but keeps the file: OK")


def test_find_profile_takes_an_index_or_a_name():
    profiles = add_profile(add_profile([], "alpha", "a.md"), "beta", "b.md")
    assert find_profile(profiles, "1") == 0
    assert find_profile(profiles, "2") == 1
    assert find_profile(profiles, "beta") == 1
    assert find_profile(profiles, "3") is None
    assert find_profile(profiles, "0") is None
    assert find_profile(profiles, "nope") is None
    assert find_profile(profiles, "") is None
    print("find_profile resolves an index or a name: OK")


def test_load_tolerates_a_missing_or_broken_config():
    with _tmp_profiles() as root:
        # Missing file.
        assert load_profiles() == []
        # Not even JSON.
        config_path().write_text("{not json", encoding="utf-8")
        assert load_profiles() == []
        # JSON of the wrong shape.
        config_path().write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        assert load_profiles() == []
        # Right shape, but entries that cannot be used are dropped rather than
        # taking the whole registry down with them.
        config_path().write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": [
                        {"name": "good", "file": "a.md", "active": True},
                        {"name": "no file"},
                        {"file": "b.md"},
                        {"name": "escapes", "file": "../x.md"},
                        "nonsense",
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert [p["name"] for p in load_profiles()] == ["good"]
        assert root.is_dir()
    print("load_profiles survives a missing or broken config: OK")


def test_only_one_profile_can_be_active():
    with _tmp_profiles():
        save_profiles(
            [
                {"name": "a", "file": "a.md", "active": True},
                {"name": "b", "file": "b.md", "active": True},
            ]
        )
        loaded = load_profiles()
        assert [p["active"] for p in loaded] == [True, False], loaded
    print("at most one profile is ever active: OK")


def test_read_profile_text_is_uncapped_and_safe():
    with _tmp_profiles() as root:
        big = "x" * (LARGE_PROFILE_TOKENS * 4 + 1000)
        _write(root, "big.md", big)
        # No truncation: a profile is the user's own context budget to spend.
        assert read_profile_text("big.md") == big
        assert estimate_tokens(read_profile_text("big.md")) > LARGE_PROFILE_TOKENS
        # A missing or unnameable file reads as empty instead of raising.
        assert read_profile_text("gone.md") == ""
        assert read_profile_text("../escape.md") == ""
    print("read_profile_text is uncapped and never raises: OK")


def test_profile_section_names_the_file_and_carries_the_text():
    with _tmp_profiles() as root:
        _write(root, "a.md", "Investigate by reading the logs first.")
        entry = {"name": "alpha", "file": "a.md", "active": True}

        section = profile_section(entry)
        assert "## Agent profile: alpha" in section
        assert "~/.ludvart/profiles/a.md" in section
        assert "Investigate by reading the logs first." in section

        # Nothing to say for no profile, a missing file, or an empty one.
        assert profile_section(None) == ""
        assert profile_section({"name": "x", "file": "gone.md"}) == ""
        _write(root, "blank.md", "   \n\n")
        assert profile_section({"name": "x", "file": "blank.md"}) == ""
    print("profile_section names the file and carries its text: OK")


def test_active_profile_is_re_read_every_time():
    """The agent loop asks per request, so edits must be picked up live."""
    with _tmp_profiles() as root:
        _write(root, "a.md", "first version")
        save_profiles(set_active(add_profile([], "alpha", "a.md"), 0))
        source = ActiveProfile()

        assert "first version" in source.section()
        assert source.name() == "alpha"

        _write(root, "a.md", "second version")
        assert "second version" in source.section()
        assert "first version" not in source.section()

        # Switching profiles is felt without rebuilding anything.
        save_profiles(set_active(load_profiles(), None))
        assert source.section() == ""
        assert source.name() == ""
    print("ActiveProfile re-reads the registry and the file: OK")


def test_profiles_dir_honours_the_env_override():
    with _tmp_profiles() as root:
        assert profiles_dir() == root
        assert config_path() == root / "config.json"
    print("profiles_dir honours LUDVART_PROFILES_DIR: OK")


def main():
    test_valid_filename_only_accepts_a_bare_md_name()
    test_profile_path_refuses_to_leave_the_profiles_dir()
    test_add_list_and_activate_round_trip()
    test_add_rejects_a_duplicate_name_and_a_bad_file()
    test_remove_drops_the_entry_and_leaves_the_file()
    test_find_profile_takes_an_index_or_a_name()
    test_load_tolerates_a_missing_or_broken_config()
    test_only_one_profile_can_be_active()
    test_read_profile_text_is_uncapped_and_safe()
    test_profile_section_names_the_file_and_carries_the_text()
    test_active_profile_is_re_read_every_time()
    test_profiles_dir_honours_the_env_override()
    print("\nALL profile registry tests passed.")


if __name__ == "__main__":
    main()
