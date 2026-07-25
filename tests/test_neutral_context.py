"""Provider-neutral conversation log + runtime context building.

The conversation is kept in a single provider-neutral log; the exact message
shape each provider expects is rebuilt at every request by the client's own
``LLMClient.build_context`` method (so provider knowledge lives with the client
and multiple clients can coexist). These tests cover:

  * build_context renders a neutral log (including a tool round-trip) into the
    OpenAI / Anthropic / Google message shapes, with tool calls and their
    results paired correctly per provider;
  * neutralize_history passes neutral (v3+) histories through untouched and
    migrates older provider-native histories to the neutral form;
  * the backend agent loop (`AgentCore.run_turn`) stores neutral entries, so
    the same ongoing conversation can be picked up by a different model.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_neutral_context.py
"""

import base64
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart.llm import (  # noqa: E402
    AnthropicClient,
    GoogleClient,
    OpenAIClient,
    ProviderConfig,
    ToolCall,
    Turn,
    Usage,
)
from ludvart.agent_core import AgentCore  # noqa: E402
from ludvart.tools import builtin_tool_specs  # noqa: E402
from ludvart.terminal_host import TerminalHost  # noqa: E402
from ludvart.session import (  # noqa: E402
    NEUTRAL_SESSIONS_VERSION,
    SessionStore,
    neutralize_history,
)


_CLIENT_CLASS = {
    "openai": OpenAIClient,
    "custom": OpenAIClient,
    "anthropic": AnthropicClient,
    "google": GoogleClient,
}


def _client(name):
    """Construct a client for ``name`` (network-free; no request is made)."""
    cfg = ProviderConfig(
        name=name, api_url="http://x", api_key="k", model="m", context_window=100000
    )
    return _CLIENT_CLASS[name](cfg)


def build_context(log, family):
    """Render ``log`` via the given provider family's client build_context."""
    return _client(family or "openai").build_context(log)


# A neutral log with a full tool round-trip: user -> assistant(tool_call) ->
# tool result -> assistant(answer).
def _neutral_log():
    return [
        {"role": "user", "content": "encode hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "name": "b64_encode", "input": {"text": "hi"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "b64_encode",
            "content": "aGk=",
        },
        {"role": "assistant", "content": "It is aGk=."},
    ]


def test_build_context_openai():
    out = build_context(_neutral_log(), "openai")
    assert out[0] == {"role": "user", "content": "encode hi"}
    # Assistant tool call: OpenAI function shape, content kept non-empty.
    asst = out[1]
    assert asst["role"] == "assistant"
    assert asst["content"], "empty assistant content orphans tool results"
    tc = asst["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "b64_encode"
    assert json.loads(tc["function"]["arguments"]) == {"text": "hi"}
    # Tool result: role="tool" paired by id.
    assert out[2] == {"role": "tool", "tool_call_id": "call_1", "content": "aGk="}
    assert out[3] == {"role": "assistant", "content": "It is aGk=."}
    print("build_context openai: OK")


def test_build_context_anthropic():
    out = build_context(_neutral_log(), "anthropic")
    assert out[0] == {"role": "user", "content": "encode hi"}
    # Assistant tool call -> tool_use block (no empty text block).
    blocks = out[1]["content"]
    assert out[1]["role"] == "assistant"
    tool_use = [b for b in blocks if b["type"] == "tool_use"][0]
    assert tool_use["id"] == "call_1"
    assert tool_use["name"] == "b64_encode"
    assert tool_use["input"] == {"text": "hi"}
    assert all(b.get("text", "x").strip() for b in blocks if b["type"] == "text")
    # Tool result -> user message carrying a tool_result block, paired by id.
    tr = out[2]
    assert tr["role"] == "user"
    assert tr["content"][0]["type"] == "tool_result"
    assert tr["content"][0]["tool_use_id"] == "call_1"
    assert tr["content"][0]["content"] == "aGk="
    print("build_context anthropic: OK")


def test_build_context_google():
    out = build_context(_neutral_log(), "google")
    # Assistant tool call -> function_call block (paired by name).
    blocks = out[1]["content"]
    fc = [b for b in blocks if b["type"] == "function_call"][0]
    assert fc["name"] == "b64_encode"
    assert fc["args"] == {"text": "hi"}
    # Tool result -> role="tool" with a function_response keyed by name.
    tr = out[2]
    assert tr["role"] == "tool"
    fr = tr["content"][0]
    assert fr["type"] == "function_response"
    assert fr["name"] == "b64_encode"
    assert fr["response"] == {"result": "aGk="}
    print("build_context google: OK")


def test_build_context_custom_matches_openai():
    # "custom" gateways share the OpenAI wire shape (base LLMClient default).
    assert _client("custom").build_context(_neutral_log()) == _client(
        "openai"
    ).build_context(_neutral_log())
    print("build_context custom == openai: OK")


def test_build_context_guards_empty_text():
    log = [
        {"role": "assistant", "content": "   "},  # whitespace only
        {"role": "user", "content": ""},
    ]
    # Anthropic must never receive an empty/whitespace-only text block/message.
    out = build_context(log, "anthropic")
    assert out[0]["content"][0]["text"].strip() != ""
    assert out[1]["content"].strip() != ""
    print("build_context guards empty text: OK")


def test_neutralize_history_passthrough_for_neutral():
    log = _neutral_log()
    # A current (v3+) session is already neutral: returned unchanged.
    assert neutralize_history(log, NEUTRAL_SESSIONS_VERSION, "openai") == log
    assert neutralize_history(log, NEUTRAL_SESSIONS_VERSION, None) == log
    print("neutralize_history passthrough for neutral: OK")


def test_neutralize_history_migrates_openai_native():
    # An old (v2) OpenAI-native history with a tool round-trip.
    native = [
        {"role": "user", "content": "weather?"},
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
        {"role": "tool", "tool_call_id": "c1", "content": "noon"},
        {"role": "assistant", "content": "it is noon"},
    ]
    out = neutralize_history(native, 2, "openai")
    # Flattened to plain user/assistant strings; no provider-native structure.
    assert all(m["role"] in ("user", "assistant") for m in out), out
    assert all(isinstance(m["content"], str) for m in out), out
    flat = " ".join(m["content"] for m in out)
    assert "weather?" in flat and "noon" in flat
    # And the migrated log rebuilds cleanly for any provider.
    for family in ("openai", "anthropic", "google"):
        build_context(out, family)
    print("neutralize_history migrates openai native: OK")


def test_neutralize_history_migrates_anthropic_native():
    native = [
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
    out = neutralize_history(native, 2, "anthropic")
    assert all(isinstance(m["content"], str) for m in out), out
    flat = " ".join(m["content"] for m in out)
    assert "echo" in flat and "done" in flat
    print("neutralize_history migrates anthropic native: OK")


# -- agent loop stores neutral entries --------------------------------------


class _ToolThenAnswerLLM:
    """Requests one b64_encode tool call, then returns a final answer."""

    name = "fake"
    model = "m"
    context_window = 100000

    def __init__(self):
        self.on_retry = None
        self.calls = 0

    def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
        self.calls += 1
        if self.calls == 1:
            return Turn(
                text="",
                tool_calls=[
                    ToolCall(id="call_1", name="b64_encode", input={"text": "hi"})
                ],
                assistant_message={"role": "assistant", "content": ""},
                usage=Usage(1, 1, 2, self.context_window),
            )
        return Turn(
            text="done",
            assistant_message={"role": "assistant", "content": "done"},
            usage=Usage(1, 1, 2, self.context_window),
        )


class _Host(TerminalHost):
    def snapshot(self):
        return "SCREEN"

    def run_terminal_tool(self, name, args):
        return "ok"

    def narrate(self, text):
        pass

    def set_activity(self, label):
        pass

    def add_info(self, text):
        pass

    def set_context_pct(self, pct):
        pass


def _make_core(root: Path) -> AgentCore:
    os.environ["LUDVART_SESSIONS_DIR"] = str(root)
    return AgentCore(
        _ToolThenAnswerLLM(),
        _Host(),
        system_prompt="SYS",
        tools=builtin_tool_specs(),
        session=SessionStore(),
    )


def test_agent_loop_stores_neutral_log():
    root = Path(tempfile.mkdtemp())
    core = _make_core(root)

    result = core.run_turn("encode hi", "SCREEN")
    assert result == "done", result

    log = core.history
    # user, assistant(tool_call), tool result, assistant(answer)
    roles = [e["role"] for e in log]
    assert roles == ["user", "assistant", "tool", "assistant"], roles

    # The assistant tool call is stored in the NEUTRAL shape (id/name/input),
    # not any provider-native shape.
    call = log[1]["tool_calls"][0]
    assert call == {"id": "call_1", "name": "b64_encode", "input": {"text": "hi"}}

    # The tool result is a neutral tool entry (keeps id + name for replay), and
    # its content is the real tool output.
    tool_entry = log[2]
    assert tool_entry["tool_call_id"] == "call_1"
    assert tool_entry["name"] == "b64_encode"
    assert base64.b64decode(tool_entry["content"].split()[-1] + "==", validate=False)

    # No provider-native artifacts leaked into the stored log.
    assert "function" not in json.dumps(log)
    assert "tool_use" not in json.dumps(log)
    print("agent loop stores neutral log: OK")


def test_ongoing_conversation_picked_up_by_other_model():
    """A conversation started under one model resumes under another.

    After a tool round-trip recorded under the (openai-family) fake, the SAME
    neutral log builds a valid Anthropic context -- assistant tool_use paired
    with a user tool_result -- so a mid-conversation model switch just works.
    """
    root = Path(tempfile.mkdtemp())
    core = _make_core(root)
    core.run_turn("encode hi", "SCREEN")

    # Rebuild the running log as Anthropic would receive it.
    ctx = build_context(core.history, "anthropic")
    assistant = next(m for m in ctx if m["role"] == "assistant")
    tool_use = [b for b in assistant["content"] if b["type"] == "tool_use"][0]
    tool_result_msg = next(
        m
        for m in ctx
        if m["role"] == "user"
        and isinstance(m["content"], list)
        and m["content"]
        and m["content"][0].get("type") == "tool_result"
    )
    # The tool_use id and the tool_result's tool_use_id match -> valid pairing.
    assert tool_use["id"] == tool_result_msg["content"][0]["tool_use_id"]
    print("ongoing conversation picked up by other model: OK")


def main():
    test_build_context_openai()
    test_build_context_anthropic()
    test_build_context_google()
    test_build_context_custom_matches_openai()
    test_build_context_guards_empty_text()
    test_neutralize_history_passthrough_for_neutral()
    test_neutralize_history_migrates_openai_native()
    test_neutralize_history_migrates_anthropic_native()
    test_agent_loop_stores_neutral_log()
    test_ongoing_conversation_picked_up_by_other_model()
    print("\nALL neutral-context tests passed.")


if __name__ == "__main__":
    main()
