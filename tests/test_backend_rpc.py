"""Integration tests for the client<->backend RPC over framed channels.

Two levels:

* In-process loopback: run ``serve`` (with a fake LLM) on one thread and a
  ``BackendClient`` on another, connected by a pair of OS pipes. Exercises the
  full RemoteTerminalHost <-> BackendClient request/response + panel path
  without a subprocess.
* Real subprocess: fork ``python -m ludvart serve`` with the fake-LLM env and
  drive it through a ``Transport``, proving the CLI entry and stdio framing.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_backend_rpc.py
"""

import os
import subprocess
import sys
import threading

from ludvart.backend_client import BackendClient
from ludvart.protocol import FrameChannel
from ludvart.server import _FakeBackendLLM, serve
from ludvart.terminal_host import TerminalHost
from ludvart.transport import local_backend


class RecordingHost(TerminalHost):
    def __init__(self):
        self.tool_calls = []
        self.narrations = []
        self.activities = []
        self.infos = []
        self.systems = []
        self.summaries = []
        self.model_label = None
        self.snapshots = 0
        self.context_pcts = []

    def snapshot(self):
        self.snapshots += 1
        return "CLIENT-SCREEN"

    def run_terminal_tool(self, name, args):
        self.tool_calls.append((name, args))
        return f"Injected via {name}"

    def narrate(self, text):
        self.narrations.append(text)

    def set_activity(self, label):
        self.activities.append(label)

    def add_info(self, text):
        self.infos.append(text)

    def add_system(self, text):
        self.systems.append(text)

    def add_summary(self, text):
        self.summaries.append(text)

    def set_model(self, label):
        self.model_label = label

    def set_context_pct(self, pct):
        self.context_pcts.append(pct)


def _pipe_pair():
    """Return (client_channel, backend_channel) connected by two OS pipes."""
    a_r, a_w = os.pipe()  # backend -> client
    b_r, b_w = os.pipe()  # client -> backend
    client = FrameChannel(os.fdopen(a_r, "rb"), os.fdopen(b_w, "wb"))
    backend = FrameChannel(os.fdopen(b_r, "rb"), os.fdopen(a_w, "wb"))
    return client, backend


def test_loopback_turn_with_client_tool():
    client_ch, backend_ch = _pipe_pair()

    def run_backend():
        serve(backend_ch, llm=_FakeBackendLLM())

    t = threading.Thread(target=run_backend, daemon=True)
    t.start()

    client = BackendClient(client_ch)
    host = RecordingHost()

    # The backend sends HELLO first; skip it before submitting.
    hello = client_ch.recv()
    assert hello["type"] == "hello", hello

    reply = client.ask("please echo", "SNAPSHOT-AT-ASK", host)

    # The fake LLM requests an inject_input tool call, then answers echoing it.
    assert host.tool_calls == [
        ("inject_input", {"text": "echo hi", "submit": True})
    ], host.tool_calls
    assert reply.startswith("done ("), reply
    assert "Injected via inject_input" in reply, reply
    # Narration + activity flowed to the client host.
    assert "working on it" in host.narrations
    assert any("Calling inject_input" in a for a in host.activities)

    client_ch.close()
    t.join(timeout=2)
    assert not t.is_alive()
    backend_ch.close()
    print("loopback turn drives a client tool and returns the reply: OK")


def test_loopback_plain_turn_without_tools():
    client_ch, backend_ch = _pipe_pair()

    class NoToolLLM(_FakeBackendLLM):
        def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
            if on_text:
                on_text("hi")
            from ludvart.llm import Turn

            return Turn(
                text="just text",
                assistant_message={"role": "assistant", "content": "just text"},
                usage=None,
            )

    t = threading.Thread(
        target=lambda: serve(backend_ch, llm=NoToolLLM()), daemon=True
    )
    t.start()

    client = BackendClient(client_ch)
    host = RecordingHost()
    assert client_ch.recv()["type"] == "hello"
    reply = client.ask("hi", "S", host)
    assert reply == "just text"
    assert host.tool_calls == []

    client_ch.close()
    t.join(timeout=2)
    backend_ch.close()
    print("loopback plain turn returns text with no tool calls: OK")


def test_subprocess_serve_end_to_end():
    # Fork the real `python -m ludvart serve` with the offline fake LLM.
    env = dict(os.environ)
    env["LUDVART_BACKEND_FAKE_LLM"] = "1"
    transport = local_backend(env=env, stderr=subprocess.DEVNULL)
    try:
        client = BackendClient(transport.channel)
        host = RecordingHost()
        # Backend greets with HELLO.
        assert transport.channel.recv()["type"] == "hello"
        reply = client.ask("do it", "SNAP", host)
        assert host.tool_calls == [
            ("inject_input", {"text": "echo hi", "submit": True})
        ], host.tool_calls
        assert reply.startswith("done ("), reply
    finally:
        transport.close()
    assert transport.poll() is not None  # backend reaped, no leak
    print("forked `ludvart serve` runs a full turn over stdio: OK")


def test_hello_reports_model_label_and_verification():
    client_ch, backend_ch = _pipe_pair()
    t = threading.Thread(
        target=lambda: serve(backend_ch, llm=_FakeBackendLLM()), daemon=True
    )
    t.start()
    hello = client_ch.recv()
    assert hello["type"] == "hello"
    assert hello["active_label"] == "custom:fake", hello
    assert hello["verified"] is True
    assert hello["verify_error"] is None
    client_ch.close()
    t.join(timeout=2)
    backend_ch.close()
    print("HELLO carries the active model label and verification status: OK")


class _FakeManager:
    """A minimal ModelManager stand-in for /model command tests."""

    def __init__(self):
        self.models = [
            {"provider": "openai", "model": "gpt-x", "active": True},
            {"provider": "anthropic", "model": "claude-y", "active": False},
        ]
        self.available = [True, True]
        self.client = _FakeBackendLLM()
        self.used = []
        self.added = []
        self.removed = []

    def active_index(self):
        for i, m in enumerate(self.models):
            if m.get("active"):
                return i
        return None

    def describe(self):
        return [
            "  1) openai:gpt-x  [in use, available]",
            "  2) anthropic:claude-y  [available]",
        ]

    def use(self, index, *, status=None, before_swap=None):
        if status:
            status("starting gateway")
        for i, m in enumerate(self.models):
            m["active"] = i == index
        self.used.append(index)
        self.client = _FakeBackendLLM()
        return True, f"Now using model {index + 1}."

    def add(self, reg, *, status=None):
        if status:
            status("verifying new model")
        self.models.append(dict(reg))
        self.available.append(True)
        self.added.append(dict(reg))
        return True, f"Added {reg['provider']}:{reg['model']} (verified)."

    def remove(self, index):
        reg = self.models.pop(index)
        self.available.pop(index)
        self.removed.append(index)
        return True, f"Removed {reg['provider']}:{reg['model']}."


def _run_command(manager, command_line, payload=None):
    """Drive one /model command over a loopback and collect the host output."""
    client_ch, backend_ch = _pipe_pair()
    t = threading.Thread(
        target=lambda: serve(backend_ch, manager=manager), daemon=True
    )
    t.start()
    client = BackendClient(client_ch)
    host = RecordingHost()
    assert client_ch.recv()["type"] == "hello"
    client.command(command_line, host, payload=payload)
    client_ch.close()
    t.join(timeout=2)
    backend_ch.close()
    return host


def test_model_list_command_over_backend():
    manager = _FakeManager()
    host = _run_command(manager, "model list")
    joined = "\n".join(host.systems)
    assert "Registered models (backend):" in joined
    assert "openai:gpt-x" in joined
    assert "anthropic:claude-y" in joined
    print("/model list is served by the backend registry: OK")


def test_model_use_command_switches_backend_model():
    manager = _FakeManager()
    host = _run_command(manager, "model use 2")
    assert manager.used == [1], manager.used  # 0-based index for "2"
    assert manager.models[1]["active"] is True
    assert any("Now using" in s for s in host.systems), host.systems
    # The client learns the new active label via a set_model notification.
    assert host.model_label == "anthropic:claude-y", host.model_label
    print("/model use switches the backend active model and label: OK")


def test_model_add_command_over_backend():
    manager = _FakeManager()
    reg = {
        "provider": "openai",
        "api_url": "https://api.openai.com/v1",
        "api_key": "sk-secret",
        "model": "gpt-new",
        "context_window": 0,
        "active": False,
    }
    host = _run_command(manager, "model add", payload=reg)
    assert len(manager.added) == 1, manager.added
    assert manager.added[0]["model"] == "gpt-new"
    assert manager.added[0]["api_key"] == "sk-secret"
    assert any("Added" in s for s in host.systems), host.systems
    print("/model add verifies and registers on the backend: OK")


def test_model_remove_command_over_backend():
    manager = _FakeManager()
    host = _run_command(manager, "model remove 2")
    assert manager.removed == [1], manager.removed  # 1-based "2" -> index 1
    assert len(manager.models) == 1
    assert any("Removed" in s for s in host.systems), host.systems
    print("/model remove unregisters on the backend: OK")


def test_model_copilot_models_query_over_backend():
    from ludvart import server

    orig = server._copilot_model_choices
    server._copilot_model_choices = lambda: {
        "copilot_models": ["gpt-4o", "claude-opus-4.8"],
        "ready": True,
    }
    try:
        manager = _FakeManager()
        client_ch, backend_ch = _pipe_pair()
        t = threading.Thread(
            target=lambda: serve(backend_ch, manager=manager), daemon=True
        )
        t.start()
        client = BackendClient(client_ch)
        host = RecordingHost()
        assert client_ch.recv()["type"] == "hello"
        result = client.request("model copilot-models", host)
        client_ch.close()
        t.join(timeout=2)
        backend_ch.close()
    finally:
        server._copilot_model_choices = orig
    assert result == {
        "copilot_models": ["gpt-4o", "claude-opus-4.8"],
        "ready": True,
    }, result
    print("/model copilot-models query returns the backend list: OK")


def test_context_pct_flows_from_backend_to_client_panel():
    """A turn's token usage reaches the client host as a context percentage."""
    from ludvart.llm import Turn, Usage

    class UsageLLM(_FakeBackendLLM):
        def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
            return Turn(
                text="answered",
                assistant_message={"role": "assistant", "content": "answered"},
                usage=Usage(input_tokens=3000, context_window=12000),
            )

    client_ch, backend_ch = _pipe_pair()
    t = threading.Thread(
        target=lambda: serve(backend_ch, llm=UsageLLM()), daemon=True
    )
    t.start()
    client = BackendClient(client_ch)
    host = RecordingHost()
    assert client_ch.recv()["type"] == "hello"

    reply = client.ask("q", "SNAP", host)

    client_ch.close()
    t.join(timeout=2)
    backend_ch.close()
    assert reply == "answered", reply
    assert host.context_pcts == [25.0], host.context_pcts
    print("context usage reaches the client panel over the protocol: OK")


def test_compact_command_compacts_the_backend_conversation():
    """`/compact` is forwarded: the conversation and the model both live here."""
    from ludvart import server
    from ludvart.llm import Turn

    class SummarizingLLM(_FakeBackendLLM):
        def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
            return Turn(
                text="BRIEF",
                assistant_message={"role": "assistant", "content": "BRIEF"},
                usage=None,
            )

    client_ch, backend_ch = _pipe_pair()
    core_box = []
    orig_loop = server._request_loop

    def capture(channel, manager, core):
        core_box.append(core)
        core.history = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        return orig_loop(channel, manager, core)

    server._request_loop = capture
    try:
        t = threading.Thread(
            target=lambda: serve(backend_ch, llm=SummarizingLLM()), daemon=True
        )
        t.start()
        client = BackendClient(client_ch)
        host = RecordingHost()
        assert client_ch.recv()["type"] == "hello"

        client.command("compact", host)

        client_ch.close()
        t.join(timeout=2)
        backend_ch.close()
    finally:
        server._request_loop = orig_loop

    core = core_box[0]
    # The backend history was reseeded from the summary...
    assert len(core.history) == 2, core.history
    assert "BRIEF" in core.history[0]["content"], core.history
    # ...and the client was told what happened, plus shown the summary marker.
    assert any("Compacted 3 messages" in s for s in host.systems), host.systems
    assert host.summaries == ["BRIEF"], host.summaries
    print("/compact compacts the backend conversation: OK")


def test_compact_command_declines_a_short_conversation():
    client_ch, backend_ch = _pipe_pair()
    t = threading.Thread(
        target=lambda: serve(backend_ch, llm=_FakeBackendLLM()), daemon=True
    )
    t.start()
    client = BackendClient(client_ch)
    host = RecordingHost()
    assert client_ch.recv()["type"] == "hello"

    client.command("compact", host)

    client_ch.close()
    t.join(timeout=2)
    backend_ch.close()
    assert any("already compact" in s for s in host.systems), host.systems
    print("/compact declines a conversation that is already compact: OK")


def main():
    test_loopback_turn_with_client_tool()
    test_loopback_plain_turn_without_tools()
    test_subprocess_serve_end_to_end()
    test_hello_reports_model_label_and_verification()
    test_model_list_command_over_backend()
    test_model_use_command_switches_backend_model()
    test_model_add_command_over_backend()
    test_model_remove_command_over_backend()
    test_model_copilot_models_query_over_backend()
    test_context_pct_flows_from_backend_to_client_panel()
    test_compact_command_compacts_the_backend_conversation()
    test_compact_command_declines_a_short_conversation()
    print("\nALL backend RPC tests passed.")


if __name__ == "__main__":
    main()
