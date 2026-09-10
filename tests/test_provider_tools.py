"""Tool calling + streaming across OpenAI and Google clients.

The base :class:`LLMClient` drives ``converse`` as a template (prepare ->
stream/send -> record usage); each provider only specializes the hooks. These
tests exercise the OpenAI and Google specializations with fake SDK clients:

  * tools are advertised in the provider-native shape,
  * a tool call is parsed into a ``ToolCall`` and replayed verbatim,
  * the follow-up ``tool_result_message`` uses the provider's role/format,
  * streamed text deltas feed ``on_text`` and streamed tool-call deltas are
    reassembled.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart.llm import (  # noqa: E402
    GoogleClient,
    OpenAIClient,
    ProviderConfig,
    ToolSpec,
)


def _ns(**kw):
    return SimpleNamespace(**kw)


_WEATHER = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


# -- OpenAI -----------------------------------------------------------------


def _openai_client(handler):
    client = OpenAIClient(
        ProviderConfig(name="custom", api_url="http://x", api_key="k", model="m")
    )
    client._client = _ns(
        chat=_ns(completions=_ns(create=lambda **kw: handler(kw)))
    )
    return client


def test_openai_nonstream_tool_call():
    captured = {}

    def handler(kw):
        captured["kw"] = kw
        message = _ns(
            content=None,
            tool_calls=[
                _ns(
                    id="call_1",
                    function=_ns(name="get_weather", arguments='{"city":"Paris"}'),
                )
            ],
        )
        return _ns(
            choices=[_ns(message=message)],
            usage=_ns(prompt_tokens=12, completion_tokens=4, total_tokens=16),
        )

    client = _openai_client(handler)
    turn = client.converse(
        [{"role": "user", "content": "weather in Paris?"}], tools=[_WEATHER]
    )

    # Tools advertised in OpenAI function shape.
    tool = captured["kw"]["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    assert tool["function"]["parameters"]["required"] == ["city"]

    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "get_weather"
    assert call.input == {"city": "Paris"}

    # Assistant turn is replayed verbatim with its tool_calls.
    assert turn.assistant_message["tool_calls"][0]["id"] == "call_1"
    assert (
        turn.assistant_message["tool_calls"][0]["function"]["name"] == "get_weather"
    )
    assert turn.usage is not None and turn.usage.input_tokens == 12

    # The tool result uses OpenAI's role="tool" pairing.
    tr = client.tool_result_message("call_1", "Sunny, 22C")
    assert tr == {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 22C"}
    print("openai non-stream tool call: OK")


def test_openai_falls_back_to_max_completion_tokens_and_remembers_it():
    requests = []

    def handler(kw):
        requests.append(kw)
        if "max_tokens" in kw:
            raise RuntimeError(
                "Unsupported parameter: 'max_tokens' is not supported with this "
                "model. Use 'max_completion_tokens' instead."
            )
        return _ns(
            choices=[_ns(message=_ns(content="ok", tool_calls=None))],
            usage=_ns(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        )

    client = _openai_client(handler)
    assert client.complete([{"role": "user", "content": "ping"}], 16) == "ok"
    turn = client.converse([{"role": "user", "content": "again"}], max_tokens=32)

    assert turn.text == "ok"
    assert len(requests) == 3, requests
    assert requests[0]["max_tokens"] == 16
    assert requests[1]["max_completion_tokens"] == 16
    assert requests[2]["max_completion_tokens"] == 32
    assert all("max_tokens" not in req for req in requests[1:])
    print("openai max_completion_tokens fallback is remembered: OK")


def test_openai_disables_reasoning_when_chat_tools_require_it():
    requests = []

    def handler(kw):
        requests.append(kw)
        if kw.get("reasoning_effort") != "none":
            raise RuntimeError(
                "Function tools with reasoning_effort are not supported for "
                "gpt-6-astra in /v1/chat/completions. To use function tools, "
                "use /v1/responses or set reasoning_effort to 'none'."
            )
        return _ns(
            choices=[_ns(message=_ns(content="ok", tool_calls=None))],
            usage=_ns(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        )

    client = _openai_client(handler)
    turn = client.converse(
        [{"role": "user", "content": "use a tool"}], tools=[_WEATHER]
    )
    again = client.converse(
        [{"role": "user", "content": "again"}], tools=[_WEATHER]
    )

    assert turn.text == again.text == "ok"
    assert len(requests) == 3, requests
    assert "reasoning_effort" not in requests[0]
    assert requests[1]["reasoning_effort"] == "none"
    assert requests[2]["reasoning_effort"] == "none"
    print("openai disables reasoning when chat tools require it: OK")


def test_openai_stream_tool_call_and_text():
    def handler(kw):
        if not kw.get("stream"):
            raise AssertionError("expected a streamed request")
        # Tool-call arguments arrive split across chunks; the id/name land in
        # the first delta only.
        return iter(
            [
                _ns(
                    choices=[
                        _ns(
                            delta=_ns(
                                content="Let me ",
                                tool_calls=None,
                            )
                        )
                    ],
                    usage=None,
                ),
                _ns(
                    choices=[
                        _ns(
                            delta=_ns(
                                content="check.",
                                tool_calls=[
                                    _ns(
                                        index=0,
                                        id="call_9",
                                        function=_ns(
                                            name="get_weather", arguments='{"ci'
                                        ),
                                    )
                                ],
                            )
                        )
                    ],
                    usage=None,
                ),
                _ns(
                    choices=[
                        _ns(
                            delta=_ns(
                                content=None,
                                tool_calls=[
                                    _ns(
                                        index=0,
                                        id=None,
                                        function=_ns(
                                            name=None, arguments='ty":"Paris"}'
                                        ),
                                    )
                                ],
                            )
                        )
                    ],
                    usage=None,
                ),
                _ns(
                    choices=[],
                    usage=_ns(prompt_tokens=20, completion_tokens=6, total_tokens=26),
                ),
            ]
        )

    client = _openai_client(handler)
    seen = []
    turn = client.converse(
        [{"role": "user", "content": "weather?"}],
        tools=[_WEATHER],
        on_text=seen.append,
    )

    # Text deltas are streamed as accumulated snapshots.
    assert seen == ["Let me ", "Let me check."], seen
    assert turn.text == "Let me check."
    # The split tool-call arguments are reassembled and parsed.
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "call_9"
    assert turn.tool_calls[0].input == {"city": "Paris"}
    # Usage arrives on the final chunk.
    assert turn.usage is not None and turn.usage.input_tokens == 20
    print("openai stream tool call + text: OK")


def test_openai_stream_reports_the_tool_before_its_arguments():
    """A tool-only turn is silent for as long as its arguments take to arrive.

    Nothing reaches ``on_text``, so the name -- which lands in the very first
    delta -- is the only progress there is to show.
    """
    log = []

    def handler(kw):
        def chunks():
            yield _ns(
                choices=[
                    _ns(
                        delta=_ns(
                            content=None,
                            tool_calls=[
                                _ns(
                                    index=0,
                                    id="call_1",
                                    function=_ns(name="write_file", arguments=""),
                                )
                            ],
                        )
                    )
                ],
                usage=None,
            )
            for piece in ('{"path":"a.txt"', ',"text":"body"', "}"):
                log.append("args")
                yield _ns(
                    choices=[
                        _ns(
                            delta=_ns(
                                content=None,
                                tool_calls=[
                                    _ns(
                                        index=0,
                                        id=None,
                                        function=_ns(name=None, arguments=piece),
                                    )
                                ],
                            )
                        )
                    ],
                    usage=None,
                )

        return chunks()

    client = _openai_client(handler)
    seen = []
    turn = client.converse(
        [{"role": "user", "content": "write it"}],
        tools=[_WEATHER],
        on_text=seen.append,
        on_tool=lambda name: log.append(f"tool:{name}"),
    )

    # Reported once, up front -- not after the arguments finished arriving.
    assert log == ["tool:write_file", "args", "args", "args"], log
    assert seen == [], seen
    assert turn.tool_calls[0].input == {"path": "a.txt", "text": "body"}
    print("openai stream reports the tool before its arguments: OK")


def test_openai_stream_reasoning_narrated_not_in_answer():
    def handler(kw):
        if not kw.get("stream"):
            raise AssertionError("expected a streamed request")
        return iter(
            [
                _ns(
                    choices=[
                        _ns(delta=_ns(content=None, reasoning_content="Let me ", tool_calls=None))
                    ],
                    usage=None,
                ),
                _ns(
                    choices=[
                        _ns(delta=_ns(content=None, reasoning_content="think...", tool_calls=None))
                    ],
                    usage=None,
                ),
                _ns(
                    choices=[
                        _ns(delta=_ns(content="The answer", reasoning_content=None, tool_calls=None))
                    ],
                    usage=None,
                ),
                _ns(
                    choices=[
                        _ns(delta=_ns(content=" is 42.", reasoning_content=None, tool_calls=None))
                    ],
                    usage=_ns(prompt_tokens=5, completion_tokens=3, total_tokens=8),
                ),
            ]
        )

    client = _openai_client(handler)
    seen = []
    turn = client.converse([{"role": "user", "content": "q"}], on_text=seen.append)

    # Reasoning is narrated first (while no answer text), then the answer text.
    assert seen == ["Let me ", "Let me think...", "The answer", "The answer is 42."], seen
    # The reasoning is NOT part of the answer or the replayed assistant message.
    assert turn.text == "The answer is 42."
    assert turn.assistant_message["content"] == "The answer is 42."
    assert "think" not in turn.assistant_message["content"]
    print("openai stream reasoning narrated (not in answer): OK")



# -- Google -----------------------------------------------------------------


def _google_client(models):
    client = GoogleClient(
        ProviderConfig(name="google", api_url="http://x", api_key="k", model="m")
    )
    client._client = _ns(models=models)
    return client


def test_google_nonstream_tool_call():
    resp = _ns(
        candidates=[
            _ns(
                content=_ns(
                    parts=[
                        _ns(function_call=_ns(name="get_weather", args={"city": "Paris"}), text=None),
                    ]
                )
            )
        ],
        usage_metadata=_ns(
            prompt_token_count=30, candidates_token_count=5, total_token_count=35
        ),
    )
    captured = {}

    def _generate(**kw):
        captured["kw"] = kw
        return resp

    client = _google_client(_ns(generate_content=_generate))
    turn = client.converse(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "weather in Paris?"},
        ],
        tools=[_WEATHER],
    )

    # Tools are advertised as Gemini function declarations.
    cfg = captured["kw"]["config"]
    decl = cfg.tools[0].function_declarations[0]
    assert decl.name == "get_weather"

    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    # Gemini has no ids: the ToolCall id is the function name.
    assert call.id == "get_weather"
    assert call.name == "get_weather"
    assert call.input == {"city": "Paris"}
    # Assistant turn carries a replayable function_call block.
    blocks = turn.assistant_message["content"]
    assert {"type": "function_call", "name": "get_weather", "args": {"city": "Paris"}} in blocks
    assert turn.usage is not None and turn.usage.input_tokens == 30
    print("google non-stream tool call: OK")


def test_google_tool_result_roundtrips_to_content():
    client = _google_client(_ns())

    # tool_result_message keys off the function name (the ToolCall id).
    tr = client.tool_result_message("get_weather", "Sunny, 22C")
    assert tr["role"] == "tool"
    assert tr["content"][0]["name"] == "get_weather"

    # A tool result converts to a user Content with a function_response part.
    content = client._message_to_content(tr)
    assert content.role == "user"
    assert content.parts[0].function_response.name == "get_weather"
    assert content.parts[0].function_response.response == {"result": "Sunny, 22C"}

    # An assistant turn with text + function_call converts to a model Content.
    assistant = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "checking"},
            {"type": "function_call", "name": "get_weather", "args": {"city": "Paris"}},
        ],
    }
    c2 = client._message_to_content(assistant)
    assert c2.role == "model"
    assert c2.parts[0].text == "checking"
    assert c2.parts[1].function_call.name == "get_weather"
    print("google tool result round-trips to content: OK")


def test_google_stream_text_deltas():
    def _stream(**kw):
        return iter(
            [
                _ns(
                    candidates=[_ns(content=_ns(parts=[_ns(function_call=None, text="Hello ")]))],
                    usage_metadata=None,
                ),
                _ns(
                    candidates=[_ns(content=_ns(parts=[_ns(function_call=None, text="world")]))],
                    usage_metadata=_ns(
                        prompt_token_count=8, candidates_token_count=2, total_token_count=10
                    ),
                ),
            ]
        )

    client = _google_client(_ns(generate_content_stream=_stream))
    seen = []
    turn = client.converse(
        [{"role": "user", "content": "hi"}], on_text=seen.append
    )
    assert seen == ["Hello ", "Hello world"], seen
    assert turn.text == "Hello world"
    assert turn.usage is not None and turn.usage.input_tokens == 8
    print("google stream text deltas: OK")


def test_google_thinking_gate_and_thought_streaming():
    from ludvart.llm import _gemini_supports_thinking

    # Thinking (include_thoughts) is only enabled for 2.5+/3.x models.
    assert _gemini_supports_thinking("gemini-2.5-flash")
    assert _gemini_supports_thinking("gemini-3-pro-preview")
    assert not _gemini_supports_thinking("gemini-2.0-flash")
    assert not _gemini_supports_thinking("m")

    # A streamed "thought" part is narrated (as reasoning) but kept out of the
    # answer text; the following real text is the answer.
    def _stream(**kw):
        return iter(
            [
                _ns(
                    candidates=[
                        _ns(content=_ns(parts=[_ns(function_call=None, text="I am pondering", thought=True)]))
                    ],
                    usage_metadata=None,
                ),
                _ns(
                    candidates=[
                        _ns(content=_ns(parts=[_ns(function_call=None, text="42", thought=False)]))
                    ],
                    usage_metadata=_ns(
                        prompt_token_count=4, candidates_token_count=1, total_token_count=5
                    ),
                ),
            ]
        )

    client = _google_client(_ns(generate_content_stream=_stream))
    seen = []
    turn = client.converse([{"role": "user", "content": "q"}], on_text=seen.append)
    assert seen == ["I am pondering", "42"], seen
    assert turn.text == "42", turn.text
    print("google thinking gate + thought streaming: OK")



def main():
    test_openai_nonstream_tool_call()
    test_openai_falls_back_to_max_completion_tokens_and_remembers_it()
    test_openai_disables_reasoning_when_chat_tools_require_it()
    test_openai_stream_tool_call_and_text()
    test_openai_stream_reports_the_tool_before_its_arguments()
    test_openai_stream_reasoning_narrated_not_in_answer()
    test_google_nonstream_tool_call()
    test_google_tool_result_roundtrips_to_content()
    test_google_stream_text_deltas()
    test_google_thinking_gate_and_thought_streaming()
    print("ALL provider tool-calling tests passed.")


if __name__ == "__main__":
    main()
