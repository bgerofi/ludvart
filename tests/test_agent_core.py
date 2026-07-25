"""Unit tests for the terminal-decoupled agent loop (AgentCore).

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_agent_core.py
"""

from ludvart.agent_core import (
    AgentCore,
    neutral_assistant,
    neutral_tool_result,
)
from ludvart.llm import LLMClient, ProviderConfig, ToolCall, ToolSpec, Turn
from ludvart.terminal_host import TerminalHost


class RecordingHost(TerminalHost):
    """A fake terminal host that records calls and returns canned values."""

    def __init__(self, snapshot="SCREEN", tool_output="tool-out"):
        self._snapshot = snapshot
        self._tool_output = tool_output
        self.narrations = []
        self.activities = []
        self.infos = []
        self.tool_calls = []
        self.snapshots = 0

    def snapshot(self):
        self.snapshots += 1
        return self._snapshot

    def run_terminal_tool(self, name, args):
        self.tool_calls.append((name, args))
        return f"{self._tool_output}:{name}"

    def narrate(self, text):
        self.narrations.append(text)

    def set_activity(self, label):
        self.activities.append(label)

    def add_info(self, text):
        self.infos.append(text)


class ScriptedLLM(LLMClient):
    """Return a scripted sequence of Turns, recording the messages it received."""

    def __init__(self, turns):
        super().__init__(
            ProviderConfig(name="custom", api_url="x", api_key="k", model="m")
        )
        self._turns = list(turns)
        self.seen_messages = []
        self.calls = 0

    def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
        self.seen_messages.append(list(messages))
        self.calls += 1
        if on_text:
            on_text(f"thinking {self.calls}")
        return self._turns.pop(0)


def _tool(name):
    return ToolSpec(name=name, description="d", input_schema={"type": "object"})


def _text_turn(text):
    return Turn(
        text=text,
        assistant_message={"role": "assistant", "content": text},
        usage=None,
    )


def _tool_turn(text, call):
    return Turn(
        text=text,
        tool_calls=[call],
        assistant_message={"role": "assistant", "content": text or " "},
        usage=None,
    )


def test_plain_answer_turn():
    host = RecordingHost()
    llm = ScriptedLLM([_text_turn("hello there")])
    core = AgentCore(llm, host, system_prompt="SYS", tools=[_tool("inject_input")])

    reply = core.run_turn("hi", "SCREEN-NOW")

    assert reply == "hello there"
    # History has the user turn (with snapshot embedded) and the assistant reply.
    assert core.history[0]["role"] == "user"
    assert "SCREEN-NOW" in core.history[0]["content"]
    assert "<userRequest>\nhi\n</userRequest>" in core.history[0]["content"]
    assert core.history[1] == {"role": "assistant", "content": "hello there"}
    # Streaming narration reached the host.
    assert host.narrations == ["thinking 1"]
    # The system prompt was prepended to the request.
    assert llm.seen_messages[0][0] == {"role": "system", "content": "SYS"}
    print("plain answer turn records history and narration: OK")


def test_client_tool_routes_through_host():
    host = RecordingHost(tool_output="Injected 6 bytes")
    call = ToolCall(id="c1", name="inject_input", input={"text": "ls", "submit": True})
    llm = ScriptedLLM([_tool_turn("running it", call), _text_turn("all done")])
    core = AgentCore(llm, host, system_prompt="SYS", tools=[_tool("inject_input")])

    reply = core.run_turn("list files", "SCREEN")

    assert reply == "all done"
    # The client tool was dispatched to the host, not run in the backend.
    assert host.tool_calls == [("inject_input", {"text": "ls", "submit": True})]
    # The tool result is threaded back into the history for the second call.
    tool_entry = [m for m in core.history if m["role"] == "tool"][0]
    assert tool_entry["content"] == "Injected 6 bytes:inject_input"
    assert tool_entry["tool_call_id"] == "c1"
    # Activity reflected the tool call.
    assert "Calling inject_input" in host.activities
    print("client tool routes through the host and threads results: OK")


def test_backend_tool_runs_in_process():
    host = RecordingHost()
    call = ToolCall(id="c1", name="b64_encode", input={"text": "hi"})
    llm = ScriptedLLM([_tool_turn("encoding", call), _text_turn("encoded")])
    core = AgentCore(llm, host, system_prompt="SYS", tools=[_tool("b64_encode")])

    reply = core.run_turn("encode hi", "SCREEN")

    assert reply == "encoded"
    # b64_encode ran in the backend (no host tool dispatch) and produced base64.
    assert host.tool_calls == []
    tool_entry = [m for m in core.history if m["role"] == "tool"][0]
    assert tool_entry["content"] == "aGk="  # base64("hi")
    print("backend tool runs in-process (b64_encode): OK")


def test_unknown_backend_tool_reports_gracefully():
    host = RecordingHost()
    call = ToolCall(id="c1", name="web_search", input={"query": "x"})
    llm = ScriptedLLM([_tool_turn("searching", call), _text_turn("answer")])
    core = AgentCore(llm, host, system_prompt="SYS", tools=[_tool("web_search")])

    reply = core.run_turn("search", "SCREEN")

    assert reply == "answer"
    tool_entry = [m for m in core.history if m["role"] == "tool"][0]
    assert "not available in split mode" in tool_entry["content"]
    print("unknown backend tool reports gracefully: OK")


def test_transcript_accumulates_for_persistence():
    host = RecordingHost()
    llm = ScriptedLLM([_text_turn("a1"), _text_turn("a2")])
    core = AgentCore(llm, host, system_prompt="SYS")
    core.run_turn("q1", "S1")
    core.run_turn("q2", "S2")
    assert core.transcript == [
        ("you", "q1"),
        ("ludvart", "a1"),
        ("you", "q2"),
        ("ludvart", "a2"),
    ]
    print("transcript accumulates question/answer pairs: OK")


def test_neutral_helpers():
    call = ToolCall(id="c9", name="inject_input", input={"text": "x"})
    turn = _tool_turn("t", call)
    entry = neutral_assistant(turn)
    assert entry["role"] == "assistant"
    assert entry["tool_calls"][0] == {
        "id": "c9",
        "name": "inject_input",
        "input": {"text": "x"},
    }
    res = neutral_tool_result(call, "out")
    assert res == {
        "role": "tool",
        "tool_call_id": "c9",
        "name": "inject_input",
        "content": "out",
    }
    print("neutral_assistant/neutral_tool_result shapes: OK")


def test_reminder_appended_to_last_user_turn_only():
    host = RecordingHost()
    llm = ScriptedLLM([_text_turn("ok")])
    core = AgentCore(llm, host, system_prompt="SYS", tools=[_tool("inject_input")])

    core.run_turn("first question", "SCREEN")

    sent = llm.seen_messages[0]
    users = [m for m in sent if m.get("role") == "user"]
    assert users, "expected a user message in the request"
    assert AgentCore._TOOL_REMINDER in users[-1]["content"]
    # The reminder sits after the user request, at the very end of the prompt.
    assert users[-1]["content"].rstrip().endswith("</reminder>")
    print("reminder is appended to the last user turn: OK")


def test_reminder_not_stored_in_history():
    host = RecordingHost()
    llm = ScriptedLLM([_text_turn("ok")])
    core = AgentCore(llm, host, system_prompt="SYS", tools=[_tool("inject_input")])

    core.run_turn("a question", "SCREEN")

    for msg in core.history:
        content = msg.get("content")
        if isinstance(content, str):
            assert AgentCore._TOOL_REMINDER not in content
    print("reminder never leaks into the persisted history: OK")


def test_reminder_does_not_accumulate_across_turns():
    host = RecordingHost()
    llm = ScriptedLLM([_text_turn("one"), _text_turn("two")])
    core = AgentCore(llm, host, system_prompt="SYS", tools=[_tool("inject_input")])

    core.run_turn("first", "SCREEN")
    core.run_turn("second", "SCREEN")

    sent = llm.seen_messages[1]
    users = [m for m in sent if m.get("role") == "user"]
    assert len(users) == 2
    # Only the newest user turn carries the reminder; the older one is clean.
    assert AgentCore._TOOL_REMINDER not in users[0]["content"]
    assert AgentCore._TOOL_REMINDER in users[1]["content"]
    # And exactly once, not stacked up.
    assert users[1]["content"].count(AgentCore._TOOL_REMINDER) == 1
    print("reminder does not accumulate across turns: OK")


def test_reminder_survives_tool_loop_iterations():
    host = RecordingHost()
    call = ToolCall(id="c1", name="inject_input", input={"text": "ls"})
    llm = ScriptedLLM([_tool_turn("working", call), _text_turn("done")])
    core = AgentCore(llm, host, system_prompt="SYS", tools=[_tool("inject_input")])

    core.run_turn("do it", "SCREEN")

    # Both model calls in the tool loop carry the reminder on the user turn.
    assert llm.calls == 2
    for sent in llm.seen_messages:
        users = [m for m in sent if m.get("role") == "user"]
        assert AgentCore._TOOL_REMINDER in users[-1]["content"]
    print("reminder is present on every tool-loop iteration: OK")


def main():
    test_plain_answer_turn()
    test_client_tool_routes_through_host()
    test_backend_tool_runs_in_process()
    test_unknown_backend_tool_reports_gracefully()
    test_transcript_accumulates_for_persistence()
    test_neutral_helpers()
    test_reminder_appended_to_last_user_turn_only()
    test_reminder_not_stored_in_history()
    test_reminder_does_not_accumulate_across_turns()
    test_reminder_survives_tool_loop_iterations()
    print("\nALL agent core tests passed.")


if __name__ == "__main__":
    main()
