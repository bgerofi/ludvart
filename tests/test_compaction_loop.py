"""Regression tests for context auto-compaction inside the agent loop.

Two bugs are covered:

  * Compaction used to run only once per user ask (at the top of the ask), so a
    single agentic turn that made many tool round-trips could grow the context
    past the window without ever compacting again. It must now compact before
    every request inside the loop.
  * ``Usage.context_percent`` used to clamp at 100%, hiding a real overshoot.
    It must now report the true value so the badge and the compaction trigger
    see how far over budget the context is.

The agent loop runs in the backend process, so these drive
:class:`ludvart.agent_core.AgentCore` directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart.agent_core import AgentCore  # noqa: E402
from ludvart.llm import ToolCall, ToolSpec, Turn, Usage  # noqa: E402
from ludvart.terminal_host import TerminalHost  # noqa: E402

SUMMARY_TEXT = "COMPACTED-BRIEF: continue the task."


def test_context_percent_reports_overshoot():
    u = Usage(input_tokens=12000, output_tokens=0, total_tokens=12000,
              context_window=8000)
    assert u.context_percent() == 150.0, u.context_percent()
    print("context_percent reports overshoot: OK")


class _Host(TerminalHost):
    """Records what the loop pushed to the terminal side."""

    def __init__(self):
        self.summaries = []
        self.context_pcts = []
        self.activities = []

    def snapshot(self):
        return "SCREEN-SNAPSHOT"

    def run_terminal_tool(self, name, args):
        return "ok"

    def narrate(self, text):
        pass

    def set_activity(self, label):
        self.activities.append(label)

    def add_info(self, text):
        pass

    def add_summary(self, summary):
        self.summaries.append(summary)

    def set_context_usage(self, tokens, pct):
        self.context_pcts.append(pct)


class _ToolLoopLLM:
    """Drives a multi-tool agentic ask whose context keeps growing.

    Each non-summary turn reports a rising context percentage and requests a
    tool call, until the last entry is reached; then it answers. A
    summarization request (the compaction instruction) returns a fixed brief.
    """

    name = "fake"
    model = "m"
    context_window = 1000

    def __init__(self, pcts):
        self.on_retry = None
        self._pcts = list(pcts)
        self.calls = 0
        self.summarize_calls = 0
        self.seen_messages = []

    def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
        self.seen_messages.append(list(messages))
        last = messages[-1]["content"] if messages else ""
        if isinstance(last, str) and "Summarize the ENTIRE" in last:
            self.summarize_calls += 1
            return Turn(
                text=SUMMARY_TEXT,
                assistant_message={"role": "assistant", "content": SUMMARY_TEXT},
                usage=None,
            )
        idx = self.calls
        self.calls += 1
        pct = self._pcts[min(idx, len(self._pcts) - 1)]
        usage = Usage(
            input_tokens=int(self.context_window * pct / 100.0),
            output_tokens=1,
            total_tokens=1,
            context_window=self.context_window,
        )
        if idx < len(self._pcts) - 1:
            return Turn(
                text="working",
                tool_calls=[ToolCall(id=f"t{idx}", name="b64_encode",
                                     input={"text": "x" * 50})],
                assistant_message={"role": "assistant", "content": "working"},
                usage=usage,
            )
        return Turn(
            text="final answer",
            assistant_message={"role": "assistant", "content": "final answer"},
            usage=usage,
        )

    def tool_result_message(self, tool_call_id, content):
        return {"role": "user", "content": content}


def _core(pcts):
    host = _Host()
    llm = _ToolLoopLLM(pcts)
    tools = [ToolSpec(name="b64_encode", description="d",
                      input_schema={"type": "object"})]
    return AgentCore(llm, host, system_prompt="SYS", tools=tools), host, llm


def test_compacts_inside_agent_loop():
    # A long agentic ask: several tool round-trips whose reported context usage
    # climbs above the 80% threshold mid-loop, then a final answer.
    core, host, llm = _core([30.0, 60.0, 92.0, 40.0])

    result = core.run_turn("do a multi-step task", "SCREEN-SNAPSHOT")
    assert result == "final answer", result

    # It must have compacted mid-loop (not zero, not stuck) when usage crossed
    # the threshold during the single ask.
    assert llm.summarize_calls >= 1, llm.summarize_calls
    # The compaction surfaced to the terminal side.
    assert host.summaries, host.summaries
    # The history was reseeded from the summary, so it stays bounded rather than
    # growing with the whole transcript.
    assert len(core.history) <= 12, len(core.history)
    assert "<conversationSummary>" in core.history[0]["content"]
    # The question being answered survived the reseed -- dropping it would leave
    # the model working blind.
    assert any("do a multi-step task" in str(m.get("content")) for m in core.history)
    print("compacts inside agent loop: OK")


def test_compaction_never_leaves_the_model_a_prefilled_assistant_turn():
    """Every request must end with a user/tool message, not the summary seed's
    assistant acknowledgement -- providers like Copilot reject that with a 400.
    """
    core, _host, llm = _core([30.0, 92.0, 40.0])
    core.run_turn("do a multi-step task", "SCREEN-SNAPSHOT")
    assert llm.summarize_calls >= 1, "expected a compaction to have happened"
    for sent in llm.seen_messages:
        assert sent[-1].get("role") != "assistant", sent[-1]
    print("compaction never ends a request with an assistant turn: OK")


def test_no_compaction_when_under_threshold():
    core, host, llm = _core([20.0, 30.0, 25.0])
    result = core.run_turn("short task", "SCREEN-SNAPSHOT")
    assert result == "final answer", result
    assert llm.summarize_calls == 0, llm.summarize_calls
    assert not host.summaries, host.summaries
    print("no compaction when under threshold: OK")


def test_badge_reflects_overshoot():
    # Even with a single turn over budget, the reported badge shows the true >100%.
    core, host, _ = _core([135.0])  # single turn, no tools, 135% usage
    core.run_turn("q", "SCREEN-SNAPSHOT")
    assert host.context_pcts, host.context_pcts
    assert host.context_pcts[-1] == 135.0, host.context_pcts
    assert core.context_pct == 135.0, core.context_pct
    print("badge reflects overshoot: OK")


def main():
    test_context_percent_reports_overshoot()
    test_compacts_inside_agent_loop()
    test_compaction_never_leaves_the_model_a_prefilled_assistant_turn()
    test_no_compaction_when_under_threshold()
    test_badge_reflects_overshoot()
    print("\nALL compaction loop tests passed.")


if __name__ == "__main__":
    main()
