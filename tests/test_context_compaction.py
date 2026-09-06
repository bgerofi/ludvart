"""Tests for automatic context compaction (summary-based history compression).

This exercises the *infrastructure* only -- it never calls a real LLM. A
``FakeLLM`` stands in for the model: the conversation turns and the compacted
summary are hand-written to look like something an LLM would produce, so the
tests can verify how ludvart stores, purges, marks, persists and reloads them.

The agent loop lives in the backend process, so the compaction tests drive
:class:`ludvart.agent_core.AgentCore` directly. The ``/compact`` command itself
is covered over the wire in ``test_backend_rpc.py``.

Covers:
- ``working_history`` slicing to the latest summary marker.
- ``maybe_compact`` firing only past the threshold and reseeding the context.
- End-to-end: the on-disk log has the right entries and resuming a session
  picks up from the latest summary while keeping the full visible transcript.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_context_compaction.py
"""

import json
import os
import tempfile
from pathlib import Path

from ludvart.agent_core import AgentCore
from ludvart.session import (
    SUMMARY_MARKER,
    SessionStore,
    load_session,
    neutralize_history,
    provider_family,
    working_history,
)
from ludvart.llm import Turn
from ludvart.terminal_host import TerminalHost


# A made-up multi-turn conversation (user request -> assistant answer), written
# to resemble a real ludvart debugging session. Used to populate the history the
# way a live conversation would before a compaction.
SAMPLE_CONVERSATION = [
    (
        "What is failing in the test suite?",
        "The suite has one failing test, `test_refund_idempotency` in "
        "tests/test_payments.py. It fails intermittently with a "
        "`DuplicateRefundError`, which points to a race condition rather than a "
        "logic bug.",
    ),
    (
        "Can you reproduce it reliably?",
        "Running `pytest -k refund_idempotency -p no:randomly` reproduces it "
        "about 1 in 3 runs. The race is on the shared `RefundLedger` singleton "
        "in src/payments/ledger.py, whose `record()` reads-then-writes with no "
        "lock.",
    ),
    (
        "Fix it with a lock, keep the change minimal.",
        "I've added `self._lock = threading.Lock()` to `RefundLedger.__init__`. "
        "Next I'll wrap the body of `record()` in `with self._lock:` and re-run "
        "the test 20 times to confirm stability.",
    ),
]

# A made-up compaction summary, as if the model wrote it when asked to condense
# the conversation above into a resumable brief.
SAMPLE_SUMMARY = (
    "CONVERSATION SUMMARY (for continuation)\n\n"
    "Goal:\n"
    "- Fix the intermittently failing `test_refund_idempotency` test in the "
    "payments service with a minimal change (user prefers a lock over a DB "
    "transaction).\n\n"
    "Findings:\n"
    "- Failure is a race on the shared `RefundLedger` singleton in "
    "src/payments/ledger.py; `record()` does a read-then-write with no lock.\n"
    "- Reproduced with: pytest -k refund_idempotency -p no:randomly (~1/3 runs)."
    "\n\n"
    "State:\n"
    "- Added `self._lock = threading.Lock()` to RefundLedger.__init__.\n"
    "- `record()` is NOT yet wrapped in the lock.\n\n"
    "Next steps:\n"
    "1. Wrap the body of RefundLedger.record() in `with self._lock:`.\n"
    "2. Re-run `pytest -k refund_idempotency` ~20x to confirm stability.\n"
    "3. Update CHANGELOG.md with the fix."
)


class FakeLLM:
    """Stands in for the model: records calls, returns canned realistic text.

    A summarization request (identified by the compaction instruction) returns
    :data:`SAMPLE_SUMMARY`; any other request returns a short canned answer.
    """

    name = "fake"
    model = "opus-fake"
    context_window = 1000

    def __init__(self):
        self.on_retry = None
        self.calls = []

    def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
        self.calls.append(list(messages))
        last = messages[-1]["content"] if messages else ""
        if isinstance(last, str) and "Summarize the ENTIRE" in last:
            return Turn(
                text=SAMPLE_SUMMARY,
                assistant_message={"role": "assistant", "content": SAMPLE_SUMMARY},
                usage=None,
            )
        if on_text is not None:
            on_text("(canned answer)")
        return Turn(
            text="(canned answer)",
            assistant_message={"role": "assistant", "content": "(canned answer)"},
            usage=None,
        )

    def tool_result_message(self, tool_call_id, content):
        return {"role": "user", "content": content}


class _Host(TerminalHost):
    """Records what the loop pushed to the terminal side."""

    def __init__(self):
        self.context_pcts = []

    def snapshot(self):
        return "(terminal)"

    def run_terminal_tool(self, name, args):
        return "ok"

    def narrate(self, text):
        pass

    def set_activity(self, label):
        pass

    def add_info(self, text):
        pass

    def set_context_usage(self, tokens, pct):
        self.context_pcts.append(pct)


def _make_core(root: Path) -> AgentCore:
    os.environ["LUDVART_SESSIONS_DIR"] = str(root)
    return AgentCore(
        FakeLLM(), _Host(), system_prompt="SYS", session=SessionStore()
    )


def _seed_conversation(core: AgentCore, turns=SAMPLE_CONVERSATION) -> None:
    """Populate history + transcript from ``turns`` the way a live ask would.

    User turns are wrapped in the ``<screenContext>/<userRequest>`` envelope
    ludvart actually sends, and each turn is persisted like a live session.
    """
    for question, answer in turns:
        core.transcript.append(("you", question))
        core.history.append(
            {
                "role": "user",
                "content": (
                    "<screenContext>\n(terminal)\n</screenContext>\n"
                    f"<userRequest>\n{question}\n</userRequest>"
                ),
            }
        )
        core.history.append({"role": "assistant", "content": answer})
        core.transcript.append(("ludvart", answer))
        core._persist()


def test_working_history_slices_to_latest_summary():
    hist = [
        {"role": "user", "content": "old q"},
        {"role": "assistant", "content": "old a"},
        {"role": "user", "content": f"{SUMMARY_MARKER}\nfirst\n</conversationSummary>"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "mid q"},
        {"role": "user", "content": f"{SUMMARY_MARKER}\nsecond\n</conversationSummary>"},
        {"role": "assistant", "content": "ok2"},
        {"role": "user", "content": "new q"},
    ]
    got = working_history(hist)
    assert len(got) == 3, got
    assert got[0]["content"].lstrip().startswith(SUMMARY_MARKER)
    assert "second" in got[0]["content"]
    # No marker -> unchanged.
    plain = [{"role": "user", "content": "hi"}]
    assert working_history(plain) == plain
    print("working_history slices to latest summary: OK")


def test_no_compaction_below_threshold():
    root = Path(tempfile.mkdtemp())
    core = _make_core(root)
    _seed_conversation(core)
    before = len(core.history)
    core.context_pct = 50.0
    assert core.maybe_compact() is False
    assert core.llm.calls == []  # no summary requested
    assert len(core.history) == before
    print("no compaction below threshold: OK")


def test_compaction_above_threshold_reseeds():
    root = Path(tempfile.mkdtemp())
    core = _make_core(root)
    _seed_conversation(core)
    core.context_pct = 85.0
    assert core.maybe_compact() is True
    # One summary request was made.
    assert len(core.llm.calls) == 1
    # History purged to a 2-message summary seed carrying the model's summary.
    assert len(core.history) == 2
    assert core.history[0]["role"] == "user"
    assert core.history[0]["content"].lstrip().startswith(SUMMARY_MARKER)
    assert "test_refund_idempotency" in core.history[0]["content"]
    assert "Next steps:" in core.history[0]["content"]
    assert core.history[1]["role"] == "assistant"
    # The purged detail is gone from the model context...
    assert "DuplicateRefundError" not in json.dumps(core.history)
    # ...but the full transcript is retained, plus a summary entry.
    kinds = [k for (k, _t) in core.transcript]
    assert "summary" in kinds
    assert any("DuplicateRefundError" in t for (_k, t) in core.transcript)
    # The % counter dropped from 85.
    assert core.context_pct is not None and core.context_pct < 85.0
    print("compaction above threshold reseeds: OK")


def test_seed_only_history_not_recompacted():
    root = Path(tempfile.mkdtemp())
    core = _make_core(root)
    core.history = [
        {"role": "user", "content": f"{SUMMARY_MARKER}\nbrief\n</conversationSummary>"},
        {"role": "assistant", "content": "ok"},
    ]
    core.context_pct = 99.0
    assert core.maybe_compact() is False
    assert core.llm.calls == []  # <= 2 messages: left alone
    print("seed-only history not recompacted: OK")


def test_compact_tab_completion():
    from ludvart.session import complete_slash

    assert complete_slash("/comp") == "/compact "
    print("/compact tab completion: OK")


def test_end_to_end_log_and_reload():
    """Full path: seed -> compact -> new turn -> reload in a fresh core.

    Verifies the on-disk log has the right entries and that a session load
    resumes from the latest summary while keeping the full visible transcript.
    """
    root = Path(tempfile.mkdtemp())
    core = _make_core(root)
    _seed_conversation(core)
    session_id = core.session.session_id

    # Before compaction: the stored context is the full conversation.
    before = load_session(session_id, root=root)
    assert len(before["llm_history"]) == 2 * len(SAMPLE_CONVERSATION)
    assert not any(
        isinstance(m.get("content"), str)
        and m["content"].lstrip().startswith(SUMMARY_MARKER)
        for m in before["llm_history"]
    )

    # Compact, then continue with one more turn (persisted).
    core.context_pct = 90.0
    assert core.maybe_compact() is True
    core.transcript.append(("you", "Did the lock fix it?"))
    core.history.append(
        {"role": "user", "content": "<userRequest>\nDid the lock fix it?\n</userRequest>"}
    )
    core.history.append(
        {"role": "assistant", "content": "Yes -- 20/20 runs passed after the lock."}
    )
    core.transcript.append(("ludvart", "Yes -- 20/20 runs passed after the lock."))
    core._persist()

    # On-disk log: context is seed + the post-compaction turn; the pre-summary
    # detail is gone from the model context but kept in the visible transcript.
    after = load_session(session_id, root=root)
    assert after["llm_history"][0]["content"].lstrip().startswith(SUMMARY_MARKER)
    assert "DuplicateRefundError" not in json.dumps(after["llm_history"])
    kinds = [m[0] for m in after["messages"]]
    assert "summary" in kinds and "you" in kinds and "ludvart" in kinds
    assert any("DuplicateRefundError" in text for (_k, text) in after["messages"])

    # Reload into a FRESH core, the way the backend's /sessions load does.
    core2 = _make_core(root)
    data = load_session(session_id, root=root)
    neutral = neutralize_history(
        list(data["llm_history"]),
        int(data.get("version", 1) or 1),
        provider_family(data.get("provider")),
    )
    core2.resume([tuple(m) for m in data["messages"]], working_history(neutral))

    # Model context resumes from the latest summary + the later turn only.
    assert core2.history[0]["content"].lstrip().startswith(SUMMARY_MARKER)
    assert len(core2.history) == 4  # summary seed (2) + one turn (2)
    assert "DuplicateRefundError" not in json.dumps(core2.history)
    assert "Did the lock fix it?" in json.dumps(core2.history)
    # Visible transcript restored in full, including the summary entry.
    kinds2 = [k for (k, _t) in core2.transcript]
    assert "summary" in kinds2
    assert any("DuplicateRefundError" in t for (_k, t) in core2.transcript)
    print("end-to-end log entries + reload from latest summary: OK")


def main():
    test_working_history_slices_to_latest_summary()
    test_no_compaction_below_threshold()
    test_compaction_above_threshold_reseeds()
    test_seed_only_history_not_recompacted()
    test_compact_tab_completion()
    test_end_to_end_log_and_reload()
    print("\nALL context-compaction tests passed.")


if __name__ == "__main__":
    main()
