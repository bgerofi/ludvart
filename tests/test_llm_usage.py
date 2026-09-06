"""Unit tests for LLM context-window / token usage reporting (llm.py).

Covers the provider-agnostic ``Usage`` dataclass and the
``usage_from_response`` normalizer, using lightweight fakes that mimic the
shapes returned by the OpenAI, Anthropic, and Google SDKs -- so no network,
API keys, or provider packages are required.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_llm_usage.py
"""

from types import SimpleNamespace

from ludvart.llm import Usage, usage_from_response


def test_context_percent_basic():
    u = Usage(input_tokens=2000, output_tokens=100, total_tokens=2100,
              context_window=8000)
    assert u.context_percent() == 25.0, u.context_percent()
    print("context_percent basic: OK")


def test_context_percent_unknown_window():
    u = Usage(input_tokens=500, output_tokens=10, total_tokens=510)
    assert u.context_percent() is None
    print("context_percent unknown window: OK")


def test_context_percent_over_100():
    # When the prompt exceeds the window, the true overshoot is reported (not
    # capped at 100) so the badge and auto-compaction see the real pressure.
    u = Usage(input_tokens=9000, output_tokens=0, total_tokens=9000,
              context_window=8000)
    assert u.context_percent() == 112.5, u.context_percent()
    print("context_percent over 100: OK")


def test_known_context_window_claude4_is_1m():
    # The Claude 4 family (Opus 4.x / Sonnet 4.x) has a 1M window; older Claude
    # models remain 200k. Prevents over-estimating usage on endpoints whose
    # models API is not exposed (so ludvart must fall back to this table).
    from ludvart.llm import _known_context_window
    assert _known_context_window("claude-opus-4-8") == 1_000_000
    assert _known_context_window("claude-sonnet-4-5") == 1_000_000
    assert _known_context_window("claude-opus-5") == 500_000
    assert _known_context_window("claude-3-5-sonnet") == 200_000
    assert _known_context_window("claude-3-opus") == 200_000
    print("known context window claude4 is 1M: OK")


def test_openai_shape():
    resp = SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=120, completion_tokens=30, total_tokens=150))
    u = usage_from_response(resp, context_window=128000)
    assert u.input_tokens == 120, u
    assert u.output_tokens == 30, u
    assert u.total_tokens == 150, u
    assert abs(u.context_percent() - (100.0 * 120 / 128000)) < 1e-9
    print("openai shape: OK")


def test_anthropic_shape():
    resp = SimpleNamespace(usage=SimpleNamespace(
        input_tokens=200, output_tokens=45))
    u = usage_from_response(resp, context_window=200000)
    assert u.input_tokens == 200, u
    assert u.output_tokens == 45, u
    # No total field -> derived from sum.
    assert u.total_tokens == 245, u
    print("anthropic shape: OK")


def test_google_shape():
    resp = SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=300, candidates_token_count=60,
        total_token_count=360))
    u = usage_from_response(resp, context_window=1000000)
    assert u.input_tokens == 300, u
    assert u.output_tokens == 60, u
    assert u.total_tokens == 360, u
    print("google shape: OK")


def test_dict_shape():
    resp = {"usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "total_tokens": 15}}
    u = usage_from_response(resp, context_window=4096)
    assert (u.input_tokens, u.output_tokens, u.total_tokens) == (10, 5, 15), u
    print("dict shape: OK")


def test_missing_usage_returns_none():
    resp = SimpleNamespace(choices=[])
    assert usage_from_response(resp) is None
    assert usage_from_response({"choices": []}) is None
    print("missing usage returns None: OK")


def test_partial_and_bad_values():
    # Missing output + non-numeric values are treated as 0.
    resp = SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=42, completion_tokens=None, total_tokens="nope"))
    u = usage_from_response(resp)
    assert u.input_tokens == 42, u
    assert u.output_tokens == 0, u
    assert u.total_tokens == 42, u  # derived sum
    assert u.context_window == 0
    assert u.context_percent() is None
    print("partial and bad values: OK")


# (runner appended at end of file)

# -- panel prompt badge (in front of "ludvart> ") ---------------------------

from ludvart.panel import AiPanel


def _find_input_row(panel):
    rows = panel.render(height=8, cols=80)
    # The input row is the last non-empty row containing the prompt bytes.
    for row in reversed(rows):
        if b"ludvart> " in row:
            return row
    raise AssertionError("no input row with prompt found")


def test_panel_prompt_no_badge_by_default():
    panel = AiPanel(cols=80, height=8, provider="anthropic")
    assert panel._prompt_prefix() == ""
    row = _find_input_row(panel)
    # No badge anywhere on the row.
    import re as _re
    assert _re.search(rb"\[c:", row) is None, row
    print("panel prompt no badge by default: OK")


def test_panel_prompt_shows_badge():
    panel = AiPanel(cols=80, height=8, provider="anthropic")
    panel.context_tokens = 9000
    panel.context_pct = 45.0
    assert panel._prompt_prefix() == "[c:9.0k(45%)] "
    row = _find_input_row(panel)
    assert b"[c:9.0k(45%)] " in row
    # The badge comes before the prompt on the row.
    assert row.index(b"[c:9.0k(45%)] ") < row.index(b"ludvart> ")
    print("panel prompt shows badge: OK")


def test_panel_context_without_a_known_window_shows_only_tokens():
    # An endpoint that never reports its window still reports prompt sizes, and
    # the count is worth showing on its own.
    panel = AiPanel(cols=80, height=8)
    panel.context_tokens = 12500
    panel.context_pct = None
    assert panel._prompt_prefix() == "[c:12k] "
    print("panel context without a known window shows only tokens: OK")


def test_format_tokens_scales_with_a_unit_suffix():
    from ludvart.panel import format_tokens
    assert format_tokens(0) == "0"
    assert format_tokens(940) == "940"
    assert format_tokens(1234) == "1.2k"
    assert format_tokens(18204) == "18k"
    assert format_tokens(412350) == "412k"
    assert format_tokens(3_500_000) == "3.5M"
    assert format_tokens(12_000_000_000) == "12G"
    print("format tokens scales with a unit suffix: OK")


def test_panel_badge_carries_token_totals():
    panel = AiPanel(cols=80, height=8)
    panel.context_tokens = 10240
    panel.context_pct = 6.0
    panel.total_input_tokens = 412350
    panel.total_output_tokens = 1234
    assert panel._prompt_prefix() == "[c:10k(6%) i:412k o:1.2k] "
    # A field with nothing to report is left out rather than shown as zero.
    panel.total_output_tokens = 0
    assert panel._prompt_prefix() == "[c:10k(6%) i:412k] "
    panel.context_tokens = 0
    assert panel._prompt_prefix() == "[i:412k] "
    print("panel badge carries token totals: OK")


def test_panel_cursor_col_accounts_for_badge():
    panel = AiPanel(cols=80, height=8)
    panel.editor.set_text("hi")
    base_col = panel.cursor_col()
    panel.context_tokens = 9000
    panel.context_pct = 45.0  # prefix "[c:9.0k(45%)] " is 14 chars
    assert panel.cursor_col() == base_col + 14, (base_col, panel.cursor_col())
    print("panel cursor col accounts for badge: OK")



# -- base converse plumbs _last_usage into the Turn -----------------------

from ludvart.llm import LLMClient, ProviderConfig, Turn


class _FakeClient(LLMClient):
    """A client that returns fixed text and records usage, no SDK/network."""

    def __init__(self, ctx=8000):
        cfg = ProviderConfig(
            name="custom", api_url="http://x", api_key="k", model="m",
            context_window=ctx,
        )
        super().__init__(cfg)

    def complete(self, messages, max_tokens=1024):
        resp = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=1600, completion_tokens=40, total_tokens=1640))
        self._last_usage = usage_from_response(resp, self.context_window)
        return "hello"


def test_base_converse_carries_usage():
    c = _FakeClient(ctx=8000)
    turn = c.converse([{"role": "user", "content": "hi"}])
    assert isinstance(turn, Turn)
    assert turn.text == "hello"
    assert turn.usage is not None
    assert turn.usage.input_tokens == 1600, turn.usage
    assert turn.usage.context_percent() == 20.0, turn.usage.context_percent()
    print("base converse carries usage: OK")


def test_provider_config_context_window_default():
    cfg = ProviderConfig(name="openai", api_url="u", api_key="k", model="m")
    assert cfg.context_window == 0
    print("provider config context_window default: OK")

if __name__ == "__main__":
    test_context_percent_basic()
    test_context_percent_unknown_window()
    test_context_percent_over_100()
    test_known_context_window_claude4_is_1m()
    test_openai_shape()
    test_anthropic_shape()
    test_google_shape()
    test_dict_shape()
    test_missing_usage_returns_none()
    test_partial_and_bad_values()
    test_panel_prompt_no_badge_by_default()
    test_panel_prompt_shows_badge()
    test_panel_context_without_a_known_window_shows_only_tokens()
    test_format_tokens_scales_with_a_unit_suffix()
    test_panel_badge_carries_token_totals()
    test_panel_cursor_col_accounts_for_badge()
    test_base_converse_carries_usage()
    test_provider_config_context_window_default()
    print("\nALL llm-usage tests passed.")