"""get_past_snapshot: retrieve a past terminal screenshot by its timestamp.

Every user turn embeds a ``<screenContext ts="...">`` snapshot stamped with a
UTC nanosecond timestamp (see ``AgentCore.run_turn`` / ``_utc_ns_timestamp``).
Older snapshots are stripped from the model-facing context and collapsed to a
breadcrumb that keeps the timestamp, and ``get_past_snapshot`` fetches the full
snapshot back from the unstripped log.

The agent loop lives in the backend process, so these exercise
:class:`ludvart.agent_core.AgentCore` directly.

Run:
    cd ~/src/ludvart && source .venv/bin/activate \
        && python tests/test_get_past_snapshot.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart.agent_core import AgentCore  # noqa: E402


class _NullLLM:
    """Never called: these tests only exercise the snapshot bookkeeping."""

    name = "fake"
    model = "fake"


class _NullHost:
    def snapshot(self) -> str:
        return "SCREEN"


def _core() -> AgentCore:
    return AgentCore(_NullLLM(), _NullHost(), system_prompt="SYS")


def _user_turn(ts: str, screen: str, question: str) -> dict:
    return {
        "role": "user",
        "content": (
            f'<screenContext ts="{ts}">\n'
            f"{screen}\n"
            "</screenContext>\n"
            f"<userRequest>\n{question}\n</userRequest>"
        ),
    }


def test_utc_ns_timestamp_format():
    ts = AgentCore._utc_ns_timestamp()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}", ts), ts
    ts2 = AgentCore._utc_ns_timestamp()
    assert ts != ts2 or AgentCore._utc_ns_timestamp() != ts, "timestamps not unique"
    print("utc ns timestamp format: OK")


def test_get_past_snapshot_returns_stored_body():
    r = _core()
    ts_a = "2026-07-06T10:00:00.000000001"
    ts_b = "2026-07-06T10:05:00.000000002"
    r.history = [
        _user_turn(ts_a, "SCREEN ALPHA line1\nSCREEN ALPHA line2", "first"),
        {"role": "assistant", "content": "ok first"},
        _user_turn(ts_b, "SCREEN BETA line1\nSCREEN BETA line2", "second"),
        {"role": "assistant", "content": "ok second"},
    ]

    out_a = r._tool_get_past_snapshot({"timestamp": ts_a})
    assert "SCREEN ALPHA line1" in out_a and "SCREEN ALPHA line2" in out_a, out_a
    assert "SCREEN BETA" not in out_a, out_a
    assert ts_a in out_a, out_a
    assert "first" not in out_a.replace(ts_a, ""), out_a

    out_b = r._tool_get_past_snapshot({"timestamp": ts_b})
    assert "SCREEN BETA line1" in out_b and "SCREEN BETA line2" in out_b, out_b
    assert "SCREEN ALPHA" not in out_b, out_b
    print("get_past_snapshot returns stored body: OK")


def test_get_past_snapshot_tolerates_whitespace():
    r = _core()
    ts = "2026-07-06T11:00:00.123456789"
    r.history = [_user_turn(ts, "HELLO WORLD", "q")]
    out = r._tool_get_past_snapshot({"timestamp": f"  {ts}  "})
    assert "HELLO WORLD" in out, out
    print("get_past_snapshot tolerates surrounding whitespace: OK")


def test_get_past_snapshot_unknown_timestamp_errors():
    r = _core()
    ts = "2026-07-06T12:00:00.000000000"
    r.history = [_user_turn(ts, "ONLY SCREEN", "q")]
    out = r._tool_get_past_snapshot({"timestamp": "2026-01-01T00:00:00.000000000"})
    low = out.lower()
    assert "no snapshot found" in low, out
    assert "valid" in low, out
    assert "ONLY SCREEN" not in out, out
    print("get_past_snapshot unknown timestamp errors: OK")


def test_get_past_snapshot_missing_timestamp_errors():
    r = _core()
    r.history = [_user_turn("2026-07-06T12:00:00.0", "S", "q")]
    for bad in ({}, {"timestamp": ""}, {"timestamp": "   "}, {"timestamp": 5}):
        out = r._tool_get_past_snapshot(bad)
        assert "get_past_snapshot" in out and "valid" in out.lower(), (bad, out)
    print("get_past_snapshot missing timestamp errors: OK")


def test_stripping_keeps_timestamp_breadcrumb_and_snapshot_retrievable():
    r = _core()
    ts_old = "2026-07-06T09:00:00.000000001"
    ts_new = "2026-07-06T09:30:00.000000002"
    r.history = [
        _user_turn(ts_old, "OLD SCREEN BODY", "old q"),
        {"role": "assistant", "content": "ok"},
        _user_turn(ts_new, "NEW SCREEN BODY", "new q"),
    ]

    stripped = AgentCore._strip_old_screenshots(r.history)

    old_msg = stripped[0]["content"]
    new_msg = stripped[2]["content"]

    assert "OLD SCREEN BODY" not in old_msg, old_msg
    assert ts_old in old_msg, old_msg
    assert f"get_past_snapshot({ts_old})" in old_msg, old_msg
    assert "old q" in old_msg, old_msg
    assert "NEW SCREEN BODY" in new_msg, new_msg

    assert "OLD SCREEN BODY" in r.history[0]["content"]

    out = r._tool_get_past_snapshot({"timestamp": ts_old})
    assert "OLD SCREEN BODY" in out, out
    print("stripping keeps ts breadcrumb + snapshot retrievable: OK")


def main():
    test_utc_ns_timestamp_format()
    test_get_past_snapshot_returns_stored_body()
    test_get_past_snapshot_tolerates_whitespace()
    test_get_past_snapshot_unknown_timestamp_errors()
    test_get_past_snapshot_missing_timestamp_errors()
    test_stripping_keeps_timestamp_breadcrumb_and_snapshot_retrievable()
    print("\nALL get_past_snapshot tests passed.")


if __name__ == "__main__":
    main()
