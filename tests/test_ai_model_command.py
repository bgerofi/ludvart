"""/model panel command: list, use, remove, and the guided add flow.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_ai_model_command.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart import backend  # noqa: E402
from ludvart.panel import AiPanel  # noqa: E402
from ludvart.ludvart import Ludvart  # noqa: E402


class _FakeClient:
    def __init__(self, name="openai", ok=True, context_window=0):
        self.name = name
        self._ok = ok
        self.context_window = context_window

    def verify(self):
        if not self._ok:
            raise RuntimeError("verify failed")


def _install_fakes():
    def fake_build(reg_, *, status=None):
        return backend.Backend(
            _FakeClient(
                reg_["provider"],
                reg_.get("_ok", True),
                reg_.get("context_window", 0),
            )
        )

    backend.build_backend = fake_build
    backend.verify_backend = lambda b: b.client.verify()


def _reg(provider="openai", model="m", active=False, context_window=0):
    return {
        "provider": provider,
        "api_url": "http://x",
        "api_key": "k",
        "model": model,
        "context_window": context_window,
        "active": active,
    }


def _make_ludvart(models):
    os.environ["LUDVART_MODELS_FILE"] = os.path.join(tempfile.mkdtemp(), "m.json")
    mgr = backend.ModelManager(list(models), [True] * len(models), _FakeClient())
    r = Ludvart(["true"], model_manager=mgr)
    r._panel = AiPanel(cols=80, height=10, provider="openai")
    r._render_split = lambda: None
    # Run background actions synchronously and capture their result as a system
    # line, so tests don't depend on threads.
    def sync_action(worker, *, info=None, activity="Working"):
        r._panel.add_system(worker())
    r._start_action = sync_action
    return r


def _systems(r):
    return [t for kind, t in r._panel.messages if kind == "system"]


def test_model_list():
    _install_fakes()
    r = _make_ludvart([_reg(model="a", active=True), _reg("anthropic", "b")])
    r._cmd_model(["list"])
    joined = "\n".join(_systems(r))
    assert "openai:a" in joined and "in use" in joined
    assert "anthropic:b" in joined
    print("/model list: OK")


def test_model_use_switches():
    _install_fakes()
    r = _make_ludvart([_reg(model="a", active=True), _reg("anthropic", "b")])
    r._cmd_model(["use", "2"])
    assert r._models.models[1]["active"]
    assert r.llm is r._models.client
    assert r._panel.provider == "anthropic:b"
    print("/model use switches active + client: OK")


def test_switch_recomputes_badge_for_new_window():
    _install_fakes()
    r = _make_ludvart(
        [
            _reg(model="a", active=True, context_window=100_000),
            _reg("anthropic", "b", context_window=200_000),
        ]
    )
    r._llm_history = [{"role": "user", "content": "hi"}]
    r._last_input_tokens = 20_000  # 20% of the 200k target window
    r._cmd_model(["use", "2"])
    assert r._panel.context_pct == 10.0
    assert r._panel_context_pct == 10.0
    print("switch recomputes badge for new window: OK")


def test_switch_auto_compacts_when_new_window_too_small():
    _install_fakes()
    r = _make_ludvart(
        [
            _reg(model="a", active=True, context_window=200_000),
            _reg("anthropic", "b", context_window=1_000),
        ]
    )
    r._llm_history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    r._last_input_tokens = 900  # 90% of the 1000-token target -> over threshold
    called = []
    r._compact_history = lambda: (called.append(True) or "SUMMARY")
    r._estimate_context_pct = lambda summary: 3.0
    r._cmd_model(["use", "2"])
    assert called, "expected compaction before switching to the smaller model"
    assert r._panel.context_pct == 3.0
    print("switch auto-compacts when new window too small: OK")


def test_switch_no_compaction_when_it_fits():
    _install_fakes()
    r = _make_ludvart(
        [
            _reg(model="a", active=True, context_window=200_000),
            _reg("anthropic", "b", context_window=200_000),
        ]
    )
    r._llm_history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    r._last_input_tokens = 1_000  # 0.5% of the target -> no compaction
    called = []
    r._compact_history = lambda: (called.append(True) or "SUMMARY")
    r._cmd_model(["use", "2"])
    assert not called
    assert r._panel.context_pct == 0.5
    print("switch skips compaction when history fits: OK")


def test_model_use_by_name_and_unknown():
    _install_fakes()
    r = _make_ludvart([_reg(model="gpt", active=True), _reg("anthropic", "claude")])
    r._cmd_model(["use", "claude"])
    assert r._models.models[1]["active"]
    r._cmd_model(["use", "nope"])
    assert "No model matches" in _systems(r)[-1]
    print("/model use by-name + unknown: OK")


def test_model_remove():
    _install_fakes()
    r = _make_ludvart([_reg(model="a", active=True), _reg("anthropic", "b")])
    r._cmd_model(["remove", "1"])  # active -> forbidden
    assert "in use" in _systems(r)[-1]
    r._cmd_model(["remove", "2"])
    assert [m["model"] for m in r._models.models] == ["a"]
    print("/model remove: OK")


def test_guided_add_flow_direct_provider():
    _install_fakes()
    r = _make_ludvart([_reg(model="a", active=True)])
    r._cmd_model(["add"])
    assert r._model_add is not None and r._model_add["step"] == "service"
    # 0) service name (arbitrary label)
    r._feed_model_add("work")
    assert r._model_add["step"] == "provider"
    # 1) provider -> openai
    r._feed_model_add("1")
    assert r._model_add["step"] == "url"
    # 2) URL (accept default by leaving blank)
    r._feed_model_add("")
    assert r._model_add["step"] == "key"
    assert r._panel.masked, "key entry must be masked"
    # 3) key
    r._feed_model_add("sk-secret")
    assert not r._panel.masked
    assert r._model_add["step"] == "model"
    # 4) model -> finishes and registers (sync worker)
    r._feed_model_add("gpt-4o")
    assert r._model_add is None
    assert [m["model"] for m in r._models.models] == ["a", "gpt-4o"]
    assert r._models.models[1]["provider"] == "openai"
    assert r._models.models[1]["service"] == "work"
    assert r._models.models[1]["api_url"] == "https://api.openai.com/v1"
    print("guided add flow (direct provider): OK")


def test_guided_add_copilot_lists_subscription_models():
    _install_fakes()
    from ludvart import gateway

    orig = (
        gateway.litellm_available,
        gateway.copilot_authenticated,
        gateway.list_copilot_models,
    )
    gateway.litellm_available = lambda: True
    gateway.copilot_authenticated = lambda: True
    gateway.list_copilot_models = lambda: ["gpt-4o", "claude-opus-4.8"]
    try:
        r = _make_ludvart([_reg(model="a", active=True)])
        r._cmd_model(["add"])
        r._feed_model_add("work")  # service
        r._feed_model_add("5")  # GitHub Copilot
        assert r._model_add["step"] == "copilot_model"
        joined = "\n".join(_systems(r))
        assert "available to your GitHub Copilot account" in joined
        assert "1) gpt-4o" in joined and "2) claude-opus-4.8" in joined
        # Pick by number -> resolves to the slug and registers as copilot.
        r._feed_model_add("2")
        assert r._model_add is None
        assert r._models.models[-1]["provider"] == "copilot"
        assert r._models.models[-1]["model"] == "claude-opus-4.8"
        assert r._models.models[-1]["service"] == "work"
    finally:
        (
            gateway.litellm_available,
            gateway.copilot_authenticated,
            gateway.list_copilot_models,
        ) = orig
    print("guided add copilot lists subscription models: OK")


def test_guided_add_copilot_typed_slug_fallback():
    _install_fakes()
    from ludvart import gateway

    orig = (
        gateway.litellm_available,
        gateway.copilot_authenticated,
        gateway.list_copilot_models,
    )
    gateway.litellm_available = lambda: True
    gateway.copilot_authenticated = lambda: True
    gateway.list_copilot_models = lambda: []  # listing unavailable -> free text
    try:
        r = _make_ludvart([_reg(model="a", active=True)])
        r._cmd_model(["add"])
        r._feed_model_add("")  # service (blank -> falls back to provider:model)
        r._feed_model_add("copilot")  # by name
        assert r._model_add["step"] == "copilot_model"
        assert "model slug" in _systems(r)[-1]
        r._feed_model_add("gpt-5.3-codex")
        assert r._model_add is None
        assert r._models.models[-1]["provider"] == "copilot"
        assert r._models.models[-1]["model"] == "gpt-5.3-codex"
        assert r._models.models[-1]["service"] == ""
    finally:
        (
            gateway.litellm_available,
            gateway.copilot_authenticated,
            gateway.list_copilot_models,
        ) = orig
    print("guided add copilot typed slug fallback: OK")


def test_guided_add_cancel():
    _install_fakes()
    r = _make_ludvart([_reg(model="a", active=True)])
    r._cmd_model(["add"])
    r._feed_model_add("work")  # service
    r._feed_model_add("1")  # provider -> openai
    r._feed_model_add("cancel")
    assert r._model_add is None
    assert not r._panel.masked
    assert len(r._models.models) == 1
    assert "cancelled" in _systems(r)[-1].lower()
    print("guided add cancel: OK")


def test_guided_add_copilot_backend_mode():
    """In backend mode the Copilot slug is collected and forwarded (not rejected)."""
    _install_fakes()
    r = _make_ludvart([_reg(model="a", active=True)])
    r._backend_client = object()  # truthy -> backend mode
    captured = {}
    r._forward_command_to_backend = lambda line, payload=None: captured.update(
        {"line": line, "payload": payload}
    )
    r._cmd_model(["add"])
    r._feed_model_add("gh-sub")  # service
    r._feed_model_add("5")  # GitHub Copilot -> backend collects slug
    assert r._model_add["step"] == "copilot_model"
    assert "slug" in _systems(r)[-1].lower()
    r._feed_model_add("claude-opus-4.8")
    assert r._model_add is None
    assert captured["line"] == "model add"
    assert captured["payload"]["provider"] == "copilot"
    assert captured["payload"]["model"] == "claude-opus-4.8"
    assert captured["payload"]["service"] == "gh-sub"
    print("guided add copilot backend mode: OK")


def main():
    test_model_list()
    test_model_use_switches()
    test_switch_recomputes_badge_for_new_window()
    test_switch_auto_compacts_when_new_window_too_small()
    test_switch_no_compaction_when_it_fits()
    test_model_use_by_name_and_unknown()
    test_model_remove()
    test_guided_add_flow_direct_provider()
    test_guided_add_copilot_lists_subscription_models()
    test_guided_add_copilot_typed_slug_fallback()
    test_guided_add_copilot_backend_mode()
    test_guided_add_cancel()
    print("\nALL /model command tests passed.")


if __name__ == "__main__":
    main()
