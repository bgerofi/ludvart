"""The guided ``/model add`` flow, which still runs on the client.

The model registry itself lives on the backend (see
``test_backend_model_manager.py`` and ``test_backend_rpc.py``); only ``/model
add`` is driven from the panel, because it is a multi-step prompt. The finished
registration is then forwarded to the backend to be verified and stored, so
these tests assert on the prompts, the masking, and the forwarded payload.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_ai_model_command.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart.panel import AiPanel  # noqa: E402
from ludvart.ludvart import Ludvart  # noqa: E402


class _Backend:
    """Stands in for the backend: records requests, returns a canned model list."""

    def __init__(self, copilot_models=("gpt-4o", "claude-opus-4.8")):
        self.requests = []
        self._copilot_models = list(copilot_models)

    def request(self, line, host, payload=None):
        self.requests.append(line)
        return {"copilot_models": list(self._copilot_models), "ready": True}


def _make_ludvart(copilot_models=("gpt-4o", "claude-opus-4.8")):
    r = Ludvart(["true"])
    r._panel = AiPanel(cols=80, height=10, provider="openai")
    r._render_split = lambda: None
    r._backend_client = _Backend(copilot_models)
    r.forwarded = {}
    r._forward_command_to_backend = lambda line, payload=None: r.forwarded.update(
        {"line": line, "payload": payload}
    )
    return r


def _systems(r):
    return [t for kind, t in r._panel.messages if kind == "system"]


def test_guided_add_flow_direct_provider():
    r = _make_ludvart()
    r._model_add_start()
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
    # 4) model -> finishes and hands the registration to the backend
    r._feed_model_add("gpt-4o")
    assert r._model_add is None
    assert r.forwarded["line"] == "model add"
    reg = r.forwarded["payload"]
    assert reg["provider"] == "openai"
    assert reg["model"] == "gpt-4o"
    assert reg["service"] == "work"
    assert reg["api_url"] == "https://api.openai.com/v1"
    assert reg["api_key"] == "sk-secret"
    print("guided add flow (direct provider): OK")


def test_guided_add_copilot_lists_subscription_models():
    """Picking Copilot asks the backend which models the subscription offers."""
    r = _make_ludvart()
    r._model_add_start()
    r._feed_model_add("gh-sub")  # service
    r._feed_model_add("5")  # GitHub Copilot -> backend supplies the pick-list
    assert r._backend_client.requests == ["model copilot-models"]
    assert r._model_add["step"] == "copilot_model"
    joined = "\n".join(_systems(r))
    assert "1) gpt-4o" in joined and "2) claude-opus-4.8" in joined
    # Pick by number -> resolves to the slug and registers as copilot.
    r._feed_model_add("2")
    assert r._model_add is None
    reg = r.forwarded["payload"]
    assert reg["provider"] == "copilot"
    assert reg["model"] == "claude-opus-4.8"
    assert reg["service"] == "gh-sub"
    print("guided add copilot lists subscription models: OK")


def test_guided_add_copilot_typed_slug_fallback():
    """With no list available the slug is typed free-form instead."""
    r = _make_ludvart(copilot_models=())
    r._model_add_start()
    r._feed_model_add("")  # service (blank -> falls back to provider:model)
    r._feed_model_add("copilot")  # by name
    assert r._model_add["step"] == "copilot_model"
    assert "model slug" in _systems(r)[-1]
    r._feed_model_add("gpt-5.3-codex")
    assert r._model_add is None
    reg = r.forwarded["payload"]
    assert reg["provider"] == "copilot"
    assert reg["model"] == "gpt-5.3-codex"
    assert reg["service"] == ""
    print("guided add copilot typed slug fallback: OK")


def test_guided_add_cancel():
    r = _make_ludvart()
    r._model_add_start()
    r._feed_model_add("work")  # service
    r._feed_model_add("1")  # provider -> openai
    r._feed_model_add("cancel")
    assert r._model_add is None
    assert not r._panel.masked
    assert not r.forwarded, r.forwarded
    assert "cancelled" in _systems(r)[-1].lower()
    print("guided add cancel: OK")


def test_guided_add_needs_a_backend():
    """Without a backend there is no registry to add to, so the flow is a no-op."""
    r = _make_ludvart()
    r._backend_client = None
    r._model_add_start()
    assert r._model_add is None
    print("guided add needs a backend: OK")


def main():
    test_guided_add_flow_direct_provider()
    test_guided_add_copilot_lists_subscription_models()
    test_guided_add_copilot_typed_slug_fallback()
    test_guided_add_cancel()
    test_guided_add_needs_a_backend()
    print("\nALL /model add tests passed.")


if __name__ == "__main__":
    main()
