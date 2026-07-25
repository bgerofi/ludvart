"""Tests for streamed interim narration.

While the model produces a turn, ludvart streams its text into a transient,
dim "interim" line above the spinner (an indication of what it is doing), then
removes it once the final reply arrives. These tests cover:

  * the LLM ``converse(..., on_text=...)`` streaming hook (base + Anthropic),
  * the panel rendering the interim line while thinking, and
  * ludvart wiring: interim is refreshed per turn and cleared on completion.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart.llm import (  # noqa: E402
    AnthropicClient,
    LLMClient,
    LLMError,
    ProviderConfig,
    ToolCall,
    ToolSpec,
    Turn,
)
from ludvart.agent_core import AgentCore  # noqa: E402
from ludvart.panel import AiPanel  # noqa: E402
from ludvart.ludvart import Ludvart, _ClientTerminalHost  # noqa: E402


# -- base converse feeds on_text -------------------------------------------


class _TextClient(LLMClient):
    def __init__(self):
        super().__init__(
            ProviderConfig(name="custom", api_url="u", api_key="k", model="m")
        )

    def complete(self, messages, max_tokens=1024):
        return "hello world"


def test_base_converse_calls_on_text():
    seen = []
    turn = _TextClient().converse(
        [{"role": "user", "content": "hi"}], on_text=seen.append
    )
    assert seen == ["hello world"], seen
    assert turn.text == "hello world", turn.text
    print("base converse calls on_text: OK")


# -- Anthropic streaming path ----------------------------------------------


class _FakeStream:
    """Mimics the anthropic SDK streaming context manager."""

    def __init__(self, deltas, final):
        self.text_stream = deltas
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


def test_anthropic_converse_streams_and_assembles():
    text_block = SimpleNamespace(type="text", text="Let me check")
    tool_block = SimpleNamespace(
        type="tool_use", id="t1", name="capture_screen_history", input={}
    )
    final = SimpleNamespace(
        content=[text_block, tool_block],
        usage=SimpleNamespace(input_tokens=10, output_tokens=3),
    )

    captured = {}

    def _stream(**kwargs):
        captured["kwargs"] = kwargs
        return _FakeStream(["Let me ", "check"], final)

    client = AnthropicClient(
        ProviderConfig(name="anthropic", api_url="http://x", api_key="k", model="m")
    )
    client._client = SimpleNamespace(messages=SimpleNamespace(stream=_stream))

    seen = []
    turn = client.converse(
        [{"role": "user", "content": "hi"}],
        tools=[ToolSpec(name="capture_screen_history", description="d", input_schema={})],
        on_text=seen.append,
    )

    # on_text receives the accumulated text as each delta arrives.
    assert seen == ["Let me ", "Let me check"], seen
    # The final Turn is assembled from the streamed final message.
    assert turn.text == "Let me check", turn.text
    assert len(turn.tool_calls) == 1, turn.tool_calls
    assert turn.tool_calls[0].name == "capture_screen_history"
    assert turn.usage is not None and turn.usage.input_tokens == 10
    # The tools were forwarded to the streaming request.
    assert "tools" in captured["kwargs"], captured["kwargs"]
    print("anthropic converse streams and assembles: OK")


def test_anthropic_stream_assembly_error_becomes_llm_error():
    """A truncated/out-of-order stream raising IndexError yields a clean error."""

    class _BrokenStream(_FakeStream):
        def get_final_message(self):
            raise IndexError("list index out of range")

    def _stream(**kwargs):
        return _BrokenStream(["partial"], None)

    client = AnthropicClient(
        ProviderConfig(name="anthropic", api_url="http://x", api_key="k", model="m")
    )
    client._client = SimpleNamespace(messages=SimpleNamespace(stream=_stream))

    raised = None
    try:
        client._stream_turn({"model": "m"}, lambda _t: None)
    except LLMError as exc:
        raised = exc
    assert raised is not None, "IndexError should be wrapped in LLMError"
    msg = str(raised)
    assert "anthropic streaming response could not be assembled" in msg, msg
    assert "IndexError" in msg, msg
    print("anthropic stream assembly error becomes a clean LLMError: OK")


def test_anthropic_stream_assembly_retries_before_visible_output():
    """A stream assembly failure with no visible text yet is retried."""
    import ludvart.llm as llm

    final = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="recovered")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    calls = {"n": 0}

    class _BrokenThenOK(_FakeStream):
        def get_final_message(self):
            if self._final is None:
                raise IndexError("list index out of range")
            return self._final

    def _stream(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _BrokenThenOK([], None)  # no visible text, then raises
        return _BrokenThenOK(["recovered"], final)

    client = AnthropicClient(
        ProviderConfig(name="anthropic", api_url="http://x", api_key="k", model="m")
    )
    client._client = SimpleNamespace(messages=SimpleNamespace(stream=_stream))

    saved_sleep = llm.time.sleep
    llm.time.sleep = lambda _s: None
    try:
        seen = []
        turn = client.converse(
            [{"role": "user", "content": "hi"}], on_text=seen.append
        )
    finally:
        llm.time.sleep = saved_sleep

    assert calls["n"] == 2, calls  # retried once, then succeeded
    assert turn.text == "recovered", turn.text
    print("anthropic stream assembly retries before visible output: OK")


def test_anthropic_stream_assembly_not_retried_after_visible_output():
    """Once text has been shown, a later assembly failure is not replayed."""
    import ludvart.llm as llm

    calls = {"n": 0}

    class _VisibleThenBroken(_FakeStream):
        def get_final_message(self):
            raise IndexError("list index out of range")

    def _stream(**kwargs):
        calls["n"] += 1
        return _VisibleThenBroken(["partial"], None)

    client = AnthropicClient(
        ProviderConfig(name="anthropic", api_url="http://x", api_key="k", model="m")
    )
    client._client = SimpleNamespace(messages=SimpleNamespace(stream=_stream))

    saved_sleep = llm.time.sleep
    llm.time.sleep = lambda _s: None
    try:
        raised = None
        try:
            client.converse(
                [{"role": "user", "content": "hi"}], on_text=lambda _t: None
            )
        except LLMError as exc:
            raised = exc
    finally:
        llm.time.sleep = saved_sleep

    assert raised is not None, "assembly failure should surface as LLMError"
    assert calls["n"] == 1, calls  # visible output -> no retry
    print("anthropic stream assembly not retried after visible output: OK")


def test_anthropic_non_stream_unchanged():
    """Without on_text, converse still uses the plain create() path."""
    text_block = SimpleNamespace(type="text", text="plain")
    resp = SimpleNamespace(
        content=[text_block],
        usage=SimpleNamespace(input_tokens=5, output_tokens=1),
    )
    client = AnthropicClient(
        ProviderConfig(name="anthropic", api_url="http://x", api_key="k", model="m")
    )
    used = {}

    def _create(**kwargs):
        used["called"] = True
        return resp

    client._client = SimpleNamespace(messages=SimpleNamespace(create=_create))
    turn = client.converse([{"role": "user", "content": "hi"}])
    assert used.get("called") is True
    assert turn.text == "plain", turn.text
    print("anthropic non-stream path unchanged: OK")


def _anthropic_client():
    return AnthropicClient(
        ProviderConfig(name="anthropic", api_url="http://x", api_key="k", model="m")
    )


def test_anthropic_drops_empty_text_block():
    """A whitespace-only text block must not survive into the replayed message.

    Claude sometimes emits an empty text block before a tool_use; Anthropic then
    rejects the message on replay ("text content blocks must contain
    non-whitespace text"). The empty block is dropped from assistant_message but
    the tool_use is kept.
    """
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="   "),
            SimpleNamespace(type="tool_use", id="t1", name="do_it", input={}),
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    client = _anthropic_client()
    client._client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kw: resp)
    )
    turn = client.converse([{"role": "user", "content": "hi"}])
    blocks = turn.assistant_message["content"]
    types = [b["type"] for b in blocks]
    assert "text" not in types, blocks  # empty text dropped
    assert "tool_use" in types, blocks  # tool call kept
    assert turn.tool_calls and turn.tool_calls[0].name == "do_it"
    print("anthropic drops empty text block: OK")


def test_anthropic_empty_only_turn_gets_filler():
    """A turn with only an empty text block still yields a valid message."""
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    client = _anthropic_client()
    client._client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kw: resp)
    )
    turn = client.converse([{"role": "user", "content": "hi"}])
    blocks = turn.assistant_message["content"]
    # Not empty, and the single block has non-whitespace text.
    assert blocks and blocks[0]["type"] == "text"
    assert blocks[0]["text"].strip() != ""
    print("anthropic empty-only turn gets filler: OK")


def test_anthropic_sanitizes_stored_whitespace_history():
    """Replaying an older session with whitespace-only content must be repaired.

    Sessions captured before empty blocks were scrubbed at capture time can hold
    an assistant turn whose ``content`` is a lone space (as a plain string or a
    whitespace-only text block). ``_prepare_converse`` must scrub these so the
    request Anthropic sees never contains empty/whitespace-only text.
    """
    client = _anthropic_client()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": " "},  # legacy whitespace-only string
        {"role": "user", "content": "again"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "  "}],  # whitespace-only block
        },
        {"role": "user", "content": "now"},
    ]
    request = client._prepare_converse(messages, tools=None, max_tokens=16)

    # System prompt is pulled out separately; no system turns leak into messages.
    assert request["system"] == "sys"
    for msg in request["messages"]:
        assert msg["role"] != "system"
        content = msg["content"]
        if isinstance(content, str):
            assert content.strip() != "", msg
        else:
            for block in content:
                if block.get("type") == "text":
                    assert block["text"].strip() != "", block
    print("anthropic sanitizes stored whitespace history: OK")


# -- panel renders the interim line ----------------------------------------


def test_panel_renders_interim_line():
    panel = AiPanel(cols=40, height=8, provider="fake")
    panel.thinking = True
    panel.interim = "Reading the screen now"
    blob = b"".join(panel.render(8, 40))
    assert b"Reading the screen now" in blob, blob
    # The spinner label is still shown (below the interim line).
    assert b"Thinking" in blob, blob
    # When cleared, the narration is gone.
    panel.interim = ""
    blob = b"".join(panel.render(8, 40))
    assert b"Reading the screen now" not in blob
    print("panel renders interim line: OK")


# -- agent loop wiring: interim refreshed per turn, cleared on finish --------


class _StreamingLLM:
    name = "fake"
    model = "m"
    context_window = 1000

    def __init__(self, ludvart):
        self.on_retry = None
        self._ludvart = ludvart
        self.turn = 0
        self.interim_at_entry = []

    def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
        self.turn += 1
        # Record the interim value the harness left when this turn begins; the
        # agent loop must reset it to "" before every turn.
        self.interim_at_entry.append(self._ludvart._panel.interim)
        if self.turn == 1:
            if on_text:
                on_text("step one")
                on_text("step one narration")
            return Turn(
                text="step one narration",
                tool_calls=[ToolCall(id="t1", name="b64_encode", input={"text": "hi"})],
                assistant_message={"role": "assistant", "content": "..."},
                usage=None,
            )
        if on_text:
            on_text("final")
            on_text("final answer is 42")
        return Turn(
            text="final answer is 42",
            assistant_message={"role": "assistant", "content": "final answer is 42"},
            usage=None,
        )

    def tool_result_message(self, tool_call_id, content):
        return {"role": "user", "content": content}


def _make_ludvart(root: Path) -> Ludvart:
    os.environ["LUDVART_SESSIONS_DIR"] = str(root)
    r = Ludvart(["true"])
    r._panel = AiPanel(cols=80, height=8, provider="fake")
    r._phys_rows, r._phys_cols = 24, 80
    r._render_split = lambda: None
    r.snapshot_text = lambda: "SCREEN"
    return r


def test_streamed_narration_reaches_the_panel_and_is_cleared(tmp_path: Path):
    """Streamed text lands in the panel's interim line and is cleared on finish.

    The loop runs in the backend and narrates through the host; here the real
    :class:`_ClientTerminalHost` is driven directly so the assertions are on the
    live panel without the protocol's asynchrony.
    """
    r = _make_ludvart(tmp_path)
    panel = r._panel
    llm = _StreamingLLM(r)
    core = AgentCore(
        llm,
        _ClientTerminalHost(r),
        system_prompt="SYS",
        tools=[ToolSpec(name="b64_encode", description="d",
                        input_schema={"type": "object"})],
    )

    result = core.run_turn("what is on screen?", "SCREEN")
    assert result == "final answer is 42", result

    # The first turn begins with a cleared interim; the second begins with the
    # running narration from the first turn -- its streamed reasoning AND its
    # tool-call note -- so nothing vanishes between tool round-trips.
    assert llm.interim_at_entry == [
        "",
        "step one narration\n\u2192 b64_encode(text='hi')",
    ], llm.interim_at_entry
    # After the last streamed turn, interim shows the full history above the
    # final streamed narration...
    assert panel.interim == (
        "step one narration\n\u2192 b64_encode(text='hi')\nfinal answer is 42"
    ), panel.interim

    # ...until the turn is delivered, which removes the transient narration.
    r._ask_result = result
    r._deliver = r._deliver_reply
    r._ask_thread = None
    r._finish_ask()
    assert panel.interim == "", panel.interim
    assert panel.messages[-1] == ("ludvart", "final answer is 42"), panel.messages[-1]
    print("streamed narration reaches the panel and is cleared: OK")


def main():
    import tempfile

    test_base_converse_calls_on_text()
    test_anthropic_converse_streams_and_assembles()
    test_anthropic_stream_assembly_error_becomes_llm_error()
    test_anthropic_stream_assembly_retries_before_visible_output()
    test_anthropic_stream_assembly_not_retried_after_visible_output()
    test_anthropic_non_stream_unchanged()
    test_anthropic_drops_empty_text_block()
    test_anthropic_empty_only_turn_gets_filler()
    test_anthropic_sanitizes_stored_whitespace_history()
    test_panel_renders_interim_line()
    with tempfile.TemporaryDirectory() as d:
        test_streamed_narration_reaches_the_panel_and_is_cleared(Path(d))
    print("ALL streaming/interim tests passed.")


if __name__ == "__main__":
    main()
