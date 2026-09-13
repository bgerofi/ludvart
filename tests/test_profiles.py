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
    MEMORY_NAME,
    SELF_NAME,
    active_profile,
    add_profile,
    config_path,
    estimate_tokens,
    find_profile,
    load_profiles,
    memory_block,
    memory_path,
    profile_dir,
    profile_section,
    profile_tokens,
    profiles_dir,
    read_memory_text,
    read_profile_text,
    remove_profile,
    save_profiles,
    self_path,
    set_active,
    valid_dirname,
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


def _make(root: Path, dirname: str, briefing: str = "", memory: str | None = None):
    """Create a profile folder with a self.md and optionally a memory.md."""
    (root / dirname).mkdir(parents=True, exist_ok=True)
    (root / dirname / SELF_NAME).write_text(briefing, encoding="utf-8")
    if memory is not None:
        (root / dirname / MEMORY_NAME).write_text(memory, encoding="utf-8")


def test_valid_dirname_only_accepts_a_bare_folder_name():
    assert valid_dirname("investigator")
    assert valid_dirname("stabilitydb-investigator")
    # A profile is a folder *in* the profiles directory, never a path to one.
    assert not valid_dirname("sub/dir")
    assert not valid_dirname("../../etc")
    assert not valid_dirname("..")
    assert not valid_dirname(".")
    assert not valid_dirname("..\\windows")
    assert not valid_dirname("")
    print("valid_dirname accepts only a bare folder name: OK")


def test_paths_refuse_to_leave_the_profiles_dir():
    with _tmp_profiles() as root:
        assert profile_dir("ok") == root / "ok"
        assert self_path("ok") == root / "ok" / SELF_NAME
        assert memory_path("ok") == root / "ok" / MEMORY_NAME
        for bad in ("../ok", "/etc", ".."):
            assert profile_dir(bad) is None, bad
            assert self_path(bad) is None, bad
            assert memory_path(bad) is None, bad
    print("the profile paths refuse anything outside the profiles dir: OK")


def test_add_list_and_activate_round_trip():
    with _tmp_profiles() as root:
        _make(root, "a", "how to investigate A")
        _make(root, "b", "how to investigate B")

        profiles = add_profile([], "alpha", "a")
        profiles = add_profile(profiles, "beta", "b")
        # Adding never activates: choosing is a separate, explicit step.
        assert active_profile(profiles) is None, profiles
        save_profiles(profiles)

        loaded = load_profiles()
        assert [p["name"] for p in loaded] == ["alpha", "beta"]
        assert [p["dir"] for p in loaded] == ["a", "b"]

        save_profiles(set_active(loaded, 1))
        assert active_profile(load_profiles())["name"] == "beta"
        # "No profile" is a legal state, unlike the model registry.
        save_profiles(set_active(load_profiles(), None))
        assert active_profile(load_profiles()) is None
    print("profiles round-trip through config.json: OK")


def test_add_rejects_a_duplicate_name_and_a_bad_folder():
    with _tmp_profiles():
        profiles = add_profile([], "alpha", "a")
        for bad in (("alpha", "other"), ("", "a"), ("gamma", "../a"),
                    ("gamma", "")):
            try:
                add_profile(profiles, *bad)
            except ValueError:
                continue
            raise AssertionError(f"accepted {bad!r}")
    print("add_profile rejects duplicates and folders outside the dir: OK")


def test_remove_drops_the_entry_and_leaves_the_folder():
    with _tmp_profiles() as root:
        _make(root, "a", "keep this", memory="and this")
        save_profiles(add_profile([], "alpha", "a"))

        save_profiles(remove_profile(load_profiles(), 0))

        assert load_profiles() == []
        # Unregistering is not deleting the user's briefing or its memory.
        assert (root / "a" / SELF_NAME).is_file()
        assert (root / "a" / MEMORY_NAME).is_file()
    print("remove_profile unregisters but keeps the folder: OK")


def test_find_profile_takes_an_index_or_a_name():
    profiles = add_profile(add_profile([], "alpha", "a"), "beta", "b")
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
                        {"name": "good", "dir": "a", "active": True},
                        {"name": "no folder"},
                        {"dir": "b"},
                        {"name": "escapes", "dir": "../x"},
                        # The pre-folder shape, which named a file.
                        {"name": "old", "file": "a.md"},
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
                {"name": "a", "dir": "a", "active": True},
                {"name": "b", "dir": "b", "active": True},
            ]
        )
        loaded = load_profiles()
        assert [p["active"] for p in loaded] == [True, False], loaded
    print("at most one profile is ever active: OK")


def test_reads_are_uncapped_and_safe():
    with _tmp_profiles() as root:
        big = "x" * (LARGE_PROFILE_TOKENS * 4 + 1000)
        _make(root, "big", big)
        # No truncation: a profile is the user's own context budget to spend.
        assert read_profile_text("big") == big
        assert estimate_tokens(read_profile_text("big")) > LARGE_PROFILE_TOKENS
        # A missing folder or file reads as empty instead of raising.
        assert read_profile_text("gone") == ""
        assert read_profile_text("../escape") == ""
        assert read_memory_text("big") == ""
        assert read_memory_text("../escape") == ""
    print("the profile reads are uncapped and never raise: OK")


def test_profile_tokens_counts_the_briefing_and_the_memory():
    with _tmp_profiles() as root:
        _make(root, "a", "x" * 400)
        assert profile_tokens("a") == 100
        _make(root, "a", "x" * 400, memory="y" * 800)
        assert profile_tokens("a") == 300
    print("profile_tokens counts self.md plus memory.md: OK")


def test_profile_section_names_the_file_and_carries_the_text():
    with _tmp_profiles() as root:
        _make(root, "a", "Investigate by reading the logs first.")
        entry = {"name": "alpha", "dir": "a", "active": True}

        section = profile_section(entry)
        assert "## Agent profile: alpha" in section
        assert "~/.ludvart/profiles/a/self.md" in section
        assert "Investigate by reading the logs first." in section

        # Nothing to say for no profile, a missing folder, or an empty briefing.
        assert profile_section(None) == ""
        assert profile_section({"name": "x", "dir": "gone"}) == ""
        _make(root, "blank", "   \n\n")
        assert profile_section({"name": "x", "dir": "blank"}) == ""
    print("profile_section names the file and carries its text: OK")


def test_memory_block_is_separate_from_the_briefing():
    with _tmp_profiles() as root:
        _make(root, "a", "BRIEFING", memory="Learned that X is flaky.")
        entry = {"name": "alpha", "dir": "a", "active": True}

        block = memory_block(entry)
        assert block.startswith('<memory profile="alpha"'), block
        assert "~/.ludvart/profiles/a/memory.md" in block
        assert "Learned that X is flaky." in block
        assert block.endswith("</memory>"), block
        # The memory must not leak into the system-prompt section, which is the
        # cached prefix: it is written mid-conversation.
        assert "Learned that X is flaky." not in profile_section(entry)

        # memory.md is optional.
        _make(root, "b", "BRIEFING")
        assert memory_block({"name": "x", "dir": "b"}) == ""
        assert memory_block(None) == ""
        _make(root, "c", "BRIEFING", memory="  \n")
        assert memory_block({"name": "x", "dir": "c"}) == ""
    print("memory_block stays out of the briefing section: OK")


def test_memory_survives_a_missing_briefing():
    """A half-set-up folder should still hand over whatever it does have."""
    with _tmp_profiles() as root:
        (root / "a").mkdir()
        (root / "a" / MEMORY_NAME).write_text("kept", encoding="utf-8")
        entry = {"name": "alpha", "dir": "a"}
        assert profile_section(entry) == ""
        assert "kept" in memory_block(entry)
    print("a memory with no briefing still reaches the model: OK")


def test_active_profile_is_re_read_every_time():
    """The agent loop asks per request, so edits must be picked up live."""
    with _tmp_profiles() as root:
        _make(root, "a", "first version", memory="first memory")
        save_profiles(set_active(add_profile([], "alpha", "a"), 0))
        source = ActiveProfile()

        assert "first version" in source.section()
        assert "first memory" in source.memory()
        assert source.name() == "alpha"

        _make(root, "a", "second version", memory="second memory")
        assert "second version" in source.section()
        assert "first version" not in source.section()
        assert "second memory" in source.memory()

        # Switching profiles is felt without rebuilding anything.
        save_profiles(set_active(load_profiles(), None))
        assert source.section() == ""
        assert source.memory() == ""
        assert source.name() == ""
    print("ActiveProfile re-reads the registry and both files: OK")


def test_profiles_dir_honours_the_env_override():
    with _tmp_profiles() as root:
        assert profiles_dir() == root
        assert config_path() == root / "config.json"
    print("profiles_dir honours LUDVART_PROFILES_DIR: OK")


def main():
    test_valid_dirname_only_accepts_a_bare_folder_name()
    test_paths_refuse_to_leave_the_profiles_dir()
    test_add_list_and_activate_round_trip()
    test_add_rejects_a_duplicate_name_and_a_bad_folder()
    test_remove_drops_the_entry_and_leaves_the_folder()
    test_find_profile_takes_an_index_or_a_name()
    test_load_tolerates_a_missing_or_broken_config()
    test_only_one_profile_can_be_active()
    test_reads_are_uncapped_and_safe()
    test_profile_tokens_counts_the_briefing_and_the_memory()
    test_profile_section_names_the_file_and_carries_the_text()
    test_memory_block_is_separate_from_the_briefing()
    test_memory_survives_a_missing_briefing()
    test_active_profile_is_re_read_every_time()
    test_profiles_dir_honours_the_env_override()
    print("\nALL profile registry tests passed.")


if __name__ == "__main__":
    main()
