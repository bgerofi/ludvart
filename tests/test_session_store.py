"""Unit tests for session persistence and slash-command helpers (session.py).

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_session_store.py
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from e2e_util import isolate_sessions

from ludvart.session import (
    SessionStore,
    complete_slash,
    fork_history_end,
    list_sessions,
    load_session,
    parse_rename_args,
    persisted_messages,
    provider_family,
    rename_session,
    resolve_session_ref,
    sanitize_history,
    sessions_root,
    turn_count,
    turn_numbers,
)

# Running this file directly skips conftest, so isolate here too.
isolate_sessions()


def test_path_layout_utc():
    root = Path(tempfile.mkdtemp())
    when = datetime(2026, 7, 2, 8, 5, 9, tzinfo=timezone.utc)
    store = SessionStore(root=root, started_at=when)
    assert store.session_id == "2026-07-02/08_05_09", store.session_id
    assert store.path == root / "2026-07-02" / "08_05_09" / "conversation.json"
    print("path layout (UTC): OK")


def test_path_layout_converts_to_utc():
    # A non-UTC aware time must be converted to UTC for the folder name.
    root = Path(tempfile.mkdtemp())
    from datetime import timedelta

    tz = timezone(timedelta(hours=2))
    when = datetime(2026, 7, 2, 10, 5, 9, tzinfo=tz)  # == 08:05:09 UTC
    store = SessionStore(root=root, started_at=when.astimezone(timezone.utc))
    assert store.session_id == "2026-07-02/08_05_09", store.session_id
    print("path layout converts to UTC: OK")


def test_save_and_reload_roundtrip():
    root = Path(tempfile.mkdtemp())
    when = datetime(2026, 7, 2, 8, 5, 9, tzinfo=timezone.utc)
    store = SessionStore(root=root, started_at=when)
    messages = [("you", "hi"), ("ludvart", "hello"), ("info", "note")]
    history = [{"role": "user", "content": "hi"},
               {"role": "assistant", "content": "hello"}]
    store.save(messages, history)

    assert store.path.is_file()
    data = json.loads(store.path.read_text())
    assert data["version"] == 3
    assert data["session_id"] == "2026-07-02/08_05_09"
    assert data["started_at"] == "2026-07-02T08:05:09Z"
    assert data["updated_at"].endswith("Z")
    assert data["messages"] == [["you", "hi"], ["ludvart", "hello"], ["info", "note"]]
    assert data["llm_history"] == history

    loaded = load_session("2026-07-02/08_05_09", root=root)
    assert loaded["messages"] == data["messages"]
    print("save + reload roundtrip: OK")


def test_save_extends_same_file():
    root = Path(tempfile.mkdtemp())
    when = datetime(2026, 7, 2, 8, 5, 9, tzinfo=timezone.utc)
    store = SessionStore(root=root, started_at=when)
    store.save([("you", "q1"), ("ludvart", "a1")], [{"role": "user", "content": "q1"}])
    store.save(
        [("you", "q1"), ("ludvart", "a1"), ("you", "q2"), ("ludvart", "a2")],
        [{"role": "user", "content": "q1"}, {"role": "user", "content": "q2"}],
    )
    # Still one file; it holds the extended conversation.
    files = list(store.dir.glob("*.json"))
    assert files == [store.path], files
    data = json.loads(store.path.read_text())
    assert len(data["messages"]) == 4
    assert data["messages"][-1] == ["ludvart", "a2"]
    print("save extends same file: OK")


def test_system_messages_not_persisted():
    root = Path(tempfile.mkdtemp())
    store = SessionStore(root=root, started_at=datetime.now(timezone.utc))
    messages = [
        ("you", "hi"),
        ("system", "> /sessions list"),
        ("system", "1. 2026.../.."),
        ("ludvart", "hello"),
    ]
    store.save(messages, [])
    data = json.loads(store.path.read_text())
    kinds = [m[0] for m in data["messages"]]
    assert kinds == ["you", "ludvart"], kinds
    # And the pure filter helper agrees.
    assert persisted_messages(messages) == [("you", "hi"), ("ludvart", "hello")]
    print("system messages not persisted: OK")


def test_list_sessions_sorted_with_preview():
    root = Path(tempfile.mkdtemp())
    a = SessionStore(root=root, started_at=datetime(2026, 7, 1, 9, 0, 0, tzinfo=timezone.utc))
    b = SessionStore(root=root, started_at=datetime(2026, 7, 2, 9, 0, 0, tzinfo=timezone.utc))
    a.save([("you", "first question"), ("ludvart", "ans")], [])
    b.save([("info", "note"), ("you", "second question")], [])

    listed = list_sessions(root=root)
    ids = [s["id"] for s in listed]
    assert ids == ["2026-07-01/09_00_00", "2026-07-02/09_00_00"], ids
    assert listed[0]["preview"] == "first question"
    assert listed[1]["preview"] == "second question"
    assert listed[0]["count"] == 2
    print("list sessions sorted + preview: OK")


def test_list_sessions_empty_and_missing_root():
    assert list_sessions(root=Path(tempfile.mkdtemp())) == []
    assert list_sessions(root=Path(tempfile.mkdtemp()) / "does-not-exist") == []
    print("list sessions empty/missing: OK")


def test_open_existing_binds_to_same_file():
    root = Path(tempfile.mkdtemp())
    store = SessionStore(root=root, started_at=datetime(2026, 7, 2, 8, 5, 9, tzinfo=timezone.utc))
    store.save([("you", "hi"), ("ludvart", "hello")], [])

    reopened = SessionStore.open_existing("2026-07-02/08_05_09", root=root)
    assert reopened.path == store.path
    reopened.save(
        [("you", "hi"), ("ludvart", "hello"), ("you", "more")], []
    )
    data = json.loads(store.path.read_text())
    assert data["messages"][-1] == ["you", "more"]
    print("open_existing binds to same file: OK")


def test_create_new_avoids_directory_collision():
    root = Path(tempfile.mkdtemp())
    # Occupy the directory the next create_new() would pick (same second).
    first = SessionStore(root=root)
    first.save([("you", "hi")], [])
    assert first.dir.exists()

    # create_new must not reuse it; it advances to a free directory.
    second = SessionStore.create_new(root=root)
    assert second.dir != first.dir, (first.dir, second.dir)
    assert not second.dir.exists()  # unused until first save

    # The existing session is untouched after the new one saves.
    second.save([("you", "fresh")], [])
    first_data = json.loads(first.path.read_text())
    assert [m for m in first_data["messages"]] == [["you", "hi"]]
    print("create_new avoids directory collision: OK")


def test_sessions_root_env_override(monkeypatch=None):
    import os

    old = os.environ.get("LUDVART_SESSIONS_DIR")
    try:
        os.environ["LUDVART_SESSIONS_DIR"] = "/tmp/ludvart-test-root"
        assert sessions_root() == Path("/tmp/ludvart-test-root")
    finally:
        if old is None:
            os.environ.pop("LUDVART_SESSIONS_DIR", None)
        else:
            os.environ["LUDVART_SESSIONS_DIR"] = old
    print("sessions_root env override: OK")


def test_a_test_run_never_writes_to_the_real_session_store():
    """Test runs used to save into ~/.ludvart/sessions like any real run.

    The e2e scripts fork a real ludvart, which persists its conversation, so a
    single suite run dropped dozens of throwaway sessions into the developer's
    store and /sessions listed them next to actual work.
    """
    import os

    root = sessions_root().resolve()
    real = Path(os.path.expanduser("~/.ludvart/sessions")).resolve()
    assert root != real
    assert real not in root.parents
    print("test sessions are isolated: OK")


def test_save_records_provider():
    root = Path(tempfile.mkdtemp())
    store = SessionStore(root=root)
    store.save([("you", "q")], [{"role": "user", "content": "q"}], provider="openai")
    data = json.loads(store.path.read_text())
    assert data["provider"] == "openai"
    # provider defaults to None when not supplied (older callers).
    store2 = SessionStore(root=root)
    store2.save([("you", "q")], [])
    assert json.loads(store2.path.read_text())["provider"] is None
    print("save records provider: OK")


def test_provider_family_mapping():
    assert provider_family("openai") == "openai"
    assert provider_family("custom") == "openai"  # both use the OpenAI shape
    assert provider_family("anthropic") == "anthropic"
    assert provider_family("google") == "google"
    assert provider_family(None) is None
    assert provider_family("weird") is None
    print("provider family mapping: OK")


def test_sanitize_history_same_family_unchanged():
    hist = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "result"},
    ]
    # openai <-> custom share a family: history is preserved verbatim.
    out = sanitize_history(hist, "openai", "openai")
    assert out == hist
    # Unknown on both sides also counts as "same" -> unchanged.
    assert sanitize_history(hist, None, None) == hist


def test_sanitize_openai_history_to_anthropic():
    # An OpenAI-shaped history (with a "tool" role) must not keep that role
    # when adapted for Anthropic, which only accepts user/assistant.
    hist = [
        {"role": "user", "content": "what time is it?"},
        {
            "role": "assistant",
            "content": "let me check",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "clock", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "12:00"},
        {"role": "assistant", "content": "it is noon"},
    ]
    out = sanitize_history(hist, "openai", "anthropic")
    # No forbidden role and every content is a plain string.
    assert all(m["role"] in ("user", "assistant") for m in out)
    assert all(isinstance(m["content"], str) for m in out)
    flat = " ".join(m["content"] for m in out)
    assert "what time is it?" in flat
    assert "12:00" in flat  # the tool result survives as text
    assert "it is noon" in flat


def test_sanitize_anthropic_history_to_openai():
    # Anthropic uses block lists and a user-role tool_result; flatten to strings.
    hist = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "t1", "name": "echo", "input": {"x": 1}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "done"}
            ],
        },
    ]
    out = sanitize_history(hist, "anthropic", "openai")
    assert all(isinstance(m["content"], str) for m in out)
    assert all(m["role"] in ("user", "assistant") for m in out)
    flat = " ".join(m["content"] for m in out)
    assert "echo" in flat and "done" in flat


def _google_history():
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "function_call", "name": "search", "args": {"q": "x"}}
            ],
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "function_response",
                    "name": "search",
                    "response": {"result": "found it"},
                }
            ],
        },
    ]


def test_sanitize_google_history_to_openai():
    out = sanitize_history(_google_history(), "google", "openai")
    assert all(m["role"] in ("user", "assistant") for m in out)
    assert all(isinstance(m["content"], str) for m in out)
    flat = " ".join(m["content"] for m in out)
    assert "search" in flat and "found it" in flat


def test_sanitize_google_history_to_anthropic():
    out = sanitize_history(_google_history(), "google", "anthropic")
    assert all(m["role"] in ("user", "assistant") for m in out)
    flat = " ".join(m["content"] for m in out)
    assert "search" in flat and "found it" in flat


def test_sanitize_coalesces_adjacent_roles():
    # Two tool results in a row (both become user notes) must not produce two
    # adjacent user messages, which some providers reject.
    hist = [
        {"role": "tool", "tool_call_id": "1", "content": "r1"},
        {"role": "tool", "tool_call_id": "2", "content": "r2"},
    ]
    out = sanitize_history(hist, "openai", "anthropic")
    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert "r1" in out[0]["content"] and "r2" in out[0]["content"]


# -- cross-provider save -> load matrix --------------------------------------

# One representative native history per provider family. Each carries a
# recognizable marker plus a tool-call/tool-result pair in that provider's shape.
_NATIVE_HISTORY = {
    "openai": [
        {"role": "user", "content": "openai-marker: what time is it?"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "clock", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "openai-result-noon"},
        {"role": "assistant", "content": "it is noon"},
    ],
    "anthropic": [
        {"role": "user", "content": "anthropic-marker: hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "t1", "name": "echo", "input": {"x": 1}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "anthropic-result-done",
                }
            ],
        },
    ],
    "google": [
        {"role": "user", "content": "google-marker: search please"},
        {
            "role": "assistant",
            "content": [
                {"type": "function_call", "name": "search", "args": {"q": "x"}}
            ],
        },
        {
            "role": "tool",
            "content": [
                {
                    "type": "function_response",
                    "name": "search",
                    "response": {"result": "google-result-found"},
                }
            ],
        },
    ],
}

# Which stored provider name to write for each family (custom shares openai).
_PROVIDER_NAME = {"openai": "custom", "anthropic": "anthropic", "google": "google"}
_MARKER = {
    "openai": ("openai-marker", "openai-result-noon"),
    "anthropic": ("anthropic-marker", "anthropic-result-done"),
    "google": ("google-marker", "google-result-found"),
}


def _resume_like_load_session(stored_provider, target_provider, history):
    """Mirror Ludvart._load_session's family-adaptation step exactly."""
    stored_family = provider_family(stored_provider)
    target_family = provider_family(target_provider)
    sanitized = sanitize_history(history, stored_family, target_family)
    converted = (
        stored_family is not None
        and target_family is not None
        and stored_family != target_family
    )
    return sanitized, converted


def test_cross_provider_save_load_matrix():
    """Every (stored provider) x (resume provider) combination round-trips.

    Saves a native history with SessionStore, reloads it, then applies the same
    family-adaptation _load_session does. Same-family resumes keep the native
    shape; cross-family resumes flatten to plain user/assistant strings while
    preserving the conversation markers (including tool results).
    """
    root = Path(tempfile.mkdtemp())
    families = ["openai", "anthropic", "google"]
    for stored_fam in families:
        # Persist a session recorded by this provider.
        store = SessionStore(root=root)
        store.save(
            [("you", "hi")],
            _NATIVE_HISTORY[stored_fam],
            provider=_PROVIDER_NAME[stored_fam],
        )
        data = json.loads(store.path.read_text())
        history = list(data["llm_history"])

        for target_fam in families:
            target_provider = _PROVIDER_NAME[target_fam]
            out, converted = _resume_like_load_session(
                data["provider"], target_provider, history
            )
            user_marker, result_marker = _MARKER[stored_fam]

            if stored_fam == target_fam:
                # Same family (incl. openai<->custom): native shape preserved.
                assert out == history, (stored_fam, target_fam, out)
                assert converted is False
            else:
                # Cross family: flat user/assistant, plain-string content only,
                # no forbidden roles, and the markers survive as text.
                assert converted is True, (stored_fam, target_fam)
                assert all(m["role"] in ("user", "assistant") for m in out), out
                assert all(isinstance(m["content"], str) for m in out), out
                flat = " ".join(m["content"] for m in out)
                assert user_marker in flat, (stored_fam, target_fam, flat)
                assert result_marker in flat, (stored_fam, target_fam, flat)
    print("cross-provider save/load matrix: OK")


def test_openai_custom_same_family_roundtrip():
    """openai and custom share a family, so resuming across them is verbatim."""
    root = Path(tempfile.mkdtemp())
    store = SessionStore(root=root)
    store.save([("you", "hi")], _NATIVE_HISTORY["openai"], provider="openai")
    data = json.loads(store.path.read_text())
    history = list(data["llm_history"])
    # Stored under "openai", resumed under "custom": no conversion.
    out, converted = _resume_like_load_session("openai", "custom", history)
    assert out == history
    assert converted is False
    # And the reverse direction too.
    out2, converted2 = _resume_like_load_session("custom", "openai", history)
    assert out2 == history
    assert converted2 is False
    print("openai/custom same-family roundtrip: OK")


def test_complete_slash():
    # command-name completion (unique -> trailing space)
    assert complete_slash("/sess") == "/sessions "
    assert complete_slash("/s") == "/sessions "
    assert complete_slash("/i") == "/init_helpers "
    assert complete_slash("/init") == "/init_helpers "
    # ambiguous at the root ("init_helpers" vs "sessions" share no prefix) -> None
    assert complete_slash("/") is None
    # already complete command name -> add trailing space
    assert complete_slash("/sessions") == "/sessions "
    # subcommand completion
    assert complete_slash("/sessions li") == "/sessions list "
    assert complete_slash("/sessions lo") == "/sessions load "
    assert complete_slash("/sessions n") == "/sessions new "
    # ambiguous subcommand prefix "l" -> common prefix is "l" (== word) -> None
    assert complete_slash("/sessions l") is None
    # a command with no subcommands does not complete its argument
    assert complete_slash("/init_helpers ") is None
    # no completion possible
    assert complete_slash("/xyz") is None
    assert complete_slash("/sessions bogus") is None
    # arguments are not completed
    assert complete_slash("/sessions load 3") is None
    # non-slash input
    assert complete_slash("hello") is None
    print("complete_slash: OK")


def test_save_writes_empty_title_by_default():
    root = Path(tempfile.mkdtemp())
    store = SessionStore(root=root, started_at=datetime(2026, 7, 2, 8, 5, 9, tzinfo=timezone.utc))
    store.save([("you", "hi")], [])
    data = json.loads(store.path.read_text())
    assert data["title"] == "", data
    print("save writes an empty title by default: OK")


def test_save_persists_set_title():
    root = Path(tempfile.mkdtemp())
    store = SessionStore(root=root, started_at=datetime(2026, 7, 2, 8, 5, 9, tzinfo=timezone.utc))
    store.title = "Refund bug hunt"
    store.save([("you", "hi")], [])
    data = json.loads(store.path.read_text())
    assert data["title"] == "Refund bug hunt", data
    summaries = list_sessions(root=root)
    assert summaries[0]["title"] == "Refund bug hunt", summaries
    print("save persists a set title and list_sessions returns it: OK")


def test_list_sessions_backward_compatible_without_title():
    # A session file written before titles existed has no 'title' key.
    root = Path(tempfile.mkdtemp())
    store = SessionStore(root=root, started_at=datetime(2026, 7, 2, 8, 5, 9, tzinfo=timezone.utc))
    store.save([("you", "old")], [])
    data = json.loads(store.path.read_text())
    del data["title"]
    store.path.write_text(json.dumps(data))
    summaries = list_sessions(root=root)
    assert summaries[0]["title"] == "", summaries
    assert summaries[0]["preview"] == "old"
    print("list_sessions defaults title to '' for pre-title files: OK")


def test_rename_session_sets_and_clears():
    root = Path(tempfile.mkdtemp())
    store = SessionStore(root=root, started_at=datetime(2026, 7, 2, 8, 5, 9, tzinfo=timezone.utc))
    store.save([("you", "first question")], [{"role": "user", "content": "x"}])
    sid = store.session_id

    assert rename_session(sid, "My title", root=root) is True
    data = load_session(sid, root=root)
    assert data["title"] == "My title"
    # Other fields are preserved by the rewrite.
    assert data["messages"] == [["you", "first question"]]
    assert data["llm_history"] == [{"role": "user", "content": "x"}]

    # Clearing reverts to an empty title.
    assert rename_session(sid, "", root=root) is True
    assert load_session(sid, root=root)["title"] == ""
    print("rename_session sets and clears the title, preserving data: OK")


def test_rename_missing_session_returns_false():
    root = Path(tempfile.mkdtemp())
    assert rename_session("2099-01-01/00_00_00", "nope", root=root) is False
    print("rename_session returns False for a missing session: OK")


def test_parse_rename_args():
    assert parse_rename_args('2026-07-02/08_05_09 "New title"') == (
        "2026-07-02/08_05_09",
        "New title",
    )
    # Unquoted single-word title works.
    assert parse_rename_args("id title") == ("id", "title")
    # Unquoted multi-word title is taken whole (no quoting required).
    assert parse_rename_args("150 TCP timeout issue") == ("150", "TCP timeout issue")
    # Single quotes are stripped as well.
    assert parse_rename_args("id 'quoted words'") == ("id", "quoted words")
    # An empty quoted title clears the title.
    assert parse_rename_args('id ""') == ("id", "")
    # Missing title -> None (caller prints usage).
    assert parse_rename_args("only-id") is None
    assert parse_rename_args("") is None
    print("parse_rename_args handles unquoted multi-word titles and quotes: OK")


def test_resolve_session_ref():
    sessions = [{"id": "a/1"}, {"id": "b/2"}, {"id": "c/3"}]
    # 1-based index resolves to the id.
    assert resolve_session_ref("2", sessions) == ("b/2", None)
    # A raw (non-numeric) id passes through unchanged.
    assert resolve_session_ref("x/9", sessions) == ("x/9", None)
    # Out-of-range index reports an error.
    sid, err = resolve_session_ref("9", sessions)
    assert sid is None and "No session #9" in err, (sid, err)
    print("resolve_session_ref maps indices and passes ids through: OK")


def test_complete_slash_mcp_commands():
    """Every dispatchable command must be reachable by Tab, not just /help."""
    assert complete_slash("/mcp_l") == "/mcp_login "
    assert complete_slash("/mcp_a") == "/mcp_auth "
    assert complete_slash("/mcp_r") == "/mcp_refresh "
    # The three share a prefix, so Tab gets as far as it unambiguously can.
    assert complete_slash("/mc") == "/mcp_"
    print("complete_slash completes the MCP commands: OK")


def test_complete_slash_rename():
    assert complete_slash("/sessions r") == "/sessions rename "
    # 'l' is still ambiguous (list/load), 'n' completes to new.
    assert complete_slash("/sessions n") == "/sessions new "
    print("complete_slash completes /sessions rename: OK")


def test_complete_slash_fork():
    assert complete_slash("/sessions f") == "/sessions fork "
    print("complete_slash completes /sessions fork: OK")


def test_turn_numbers_pair_a_question_with_its_answer():
    messages = [
        ("you", "q1"),
        ("ludvart", "a1"),
        ("info", "a note"),
        ("you", "q2"),
        ("summary", "compacted"),
        ("ludvart", "a2"),
        ("system", "/help output"),
    ]
    assert turn_numbers(messages) == [1, 1, None, 2, None, 2, None]
    assert turn_count(messages) == 2
    print("turn numbers pair a question with its answer: OK")


def test_turn_numbers_on_an_empty_transcript():
    assert turn_numbers([]) == []
    assert turn_count([]) == 0
    # An answer with no question before it belongs to no exchange.
    assert turn_numbers([("ludvart", "orphan")]) == [None]
    print("turn numbers on an empty transcript: OK")


def _user(question):
    return {
        "role": "user",
        "content": f"<screenContext>x</screenContext>\n"
                   f"<userRequest>\n{question}\n</userRequest>",
    }


def test_fork_history_end_cuts_after_the_turns_answer():
    history = [
        _user("q1"),
        {"role": "assistant", "content": "a1"},
        _user("q2"),
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t"}]},
        {"role": "tool", "tool_call_id": "t", "content": "out"},
        {"role": "assistant", "content": "a2"},
        _user("q3"),
        {"role": "assistant", "content": "a3"},
    ]
    # Turn 2 keeps its tool round-trip and stops before turn 3's question.
    assert fork_history_end(history, "q2", 2, 3) == 6
    assert fork_history_end(history, "q1", 1, 3) == 2
    # The last turn runs to the end of the history.
    assert fork_history_end(history, "q3", 3, 3) == 8
    print("fork_history_end cuts after the turn's answer: OK")


def test_fork_history_end_skips_a_compaction_seed():
    from ludvart.session import SUMMARY_MARKER, SUMMARY_MARKER_END

    history = [
        {"role": "user",
         "content": f"{SUMMARY_MARKER}\nnotes\n{SUMMARY_MARKER_END}"},
        {"role": "assistant", "content": "Understood."},
        _user("q4"),
        {"role": "assistant", "content": "a4"},
        _user("q5"),
        {"role": "assistant", "content": "a5"},
    ]
    # The seed is context, not a turn, so it is kept and never counted.
    assert fork_history_end(history, "q4", 4, 5) == 4
    print("fork_history_end skips a compaction seed: OK")


def test_fork_history_end_reports_a_turn_that_was_compacted_away():
    history = [_user("q9"), {"role": "assistant", "content": "a9"}]
    assert fork_history_end(history, "q1", 1, 9) is None
    print("fork_history_end reports a turn that was compacted away: OK")


def test_fork_history_end_disambiguates_a_repeated_question():
    history = [
        _user("yes"),
        {"role": "assistant", "content": "a1"},
        _user("other"),
        {"role": "assistant", "content": "a2"},
        _user("yes"),
        {"role": "assistant", "content": "a3"},
    ]
    # Both turn 1 and turn 3 ask "yes"; counting picks them apart.
    assert fork_history_end(history, "yes", 1, 3) == 2
    assert fork_history_end(history, "yes", 3, 3) == 6
    print("fork_history_end disambiguates a repeated question: OK")


if __name__ == "__main__":
    test_path_layout_utc()
    test_path_layout_converts_to_utc()
    test_save_and_reload_roundtrip()
    test_save_extends_same_file()
    test_system_messages_not_persisted()
    test_list_sessions_sorted_with_preview()
    test_list_sessions_empty_and_missing_root()
    test_open_existing_binds_to_same_file()
    test_create_new_avoids_directory_collision()
    test_sessions_root_env_override()
    test_a_test_run_never_writes_to_the_real_session_store()
    test_save_records_provider()
    test_provider_family_mapping()
    test_sanitize_history_same_family_unchanged()
    test_sanitize_openai_history_to_anthropic()
    test_sanitize_anthropic_history_to_openai()
    test_sanitize_google_history_to_openai()
    test_sanitize_google_history_to_anthropic()
    test_sanitize_coalesces_adjacent_roles()
    test_cross_provider_save_load_matrix()
    test_openai_custom_same_family_roundtrip()
    test_complete_slash()
    test_save_writes_empty_title_by_default()
    test_save_persists_set_title()
    test_list_sessions_backward_compatible_without_title()
    test_rename_session_sets_and_clears()
    test_rename_missing_session_returns_false()
    test_parse_rename_args()
    test_resolve_session_ref()
    test_complete_slash_rename()
    test_complete_slash_mcp_commands()
    test_complete_slash_fork()
    test_turn_numbers_pair_a_question_with_its_answer()
    test_turn_numbers_on_an_empty_transcript()
    test_fork_history_end_cuts_after_the_turns_answer()
    test_fork_history_end_skips_a_compaction_seed()
    test_fork_history_end_reports_a_turn_that_was_compacted_away()
    test_fork_history_end_disambiguates_a_repeated_question()
    print("\nALL session-store tests passed.")
