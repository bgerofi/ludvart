"""The ludvart backend server: run the agent loop over a framed channel.

``python -m ludvart serve`` starts this on a duplex byte stream (its stdin and
stdout), which the client reaches either by forking it locally or by spawning it
on a remote host over SSH. It reads ``SUBMIT`` frames, runs an
:class:`~ludvart.agent_core.AgentCore` turn against a
:class:`~ludvart.remote_host.RemoteTerminalHost` (so terminal tools execute back
on the client), and returns a ``REPLY``.

State stays under ``~/.ludvart/`` on the backend host. Only protocol frames are
written to stdout; diagnostics go to stderr.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

from .agent_core import DEFAULT_CLIENT_TOOLS, AgentCore
from .llm import LLMClient, ProviderConfig, ToolCall, ToolSpec, Turn
from .protocol import (
    DEFAULT_MAX_FRAME,
    FrameChannel,
    MsgType,
    message,
    msg_type,
)
from .remote_host import RemoteTerminalHost

#: Compact system prompt used by the split backend prototype. The full,
#: helper-aware prompt still lives in the monolithic client path.
_SYSTEM_PROMPT = (
    "You are ludvart, an assistant embedded in a terminal. Each user message "
    "carries a <screenContext> snapshot followed by a <userRequest>. Use the "
    "tools available to you to act in the terminal; keep replies concise and in "
    "plain ASCII."
)


def _prototype_tools() -> list[ToolSpec]:
    """The tool set the split backend advertises (prototype subset)."""
    return [
        ToolSpec(
            name="inject_input",
            description="Type characters into the user's terminal.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "submit": {"type": "boolean"},
                    "interpret_escapes": {"type": "boolean"},
                },
                "required": ["text"],
            },
        ),
        ToolSpec(
            name="capture_screen_history",
            description="Read lines from the terminal scrollback history.",
            input_schema={
                "type": "object",
                "properties": {
                    "offset": {"type": "integer"},
                    "length": {"type": "integer"},
                },
                "required": ["offset", "length"],
            },
        ),
        ToolSpec(
            name="b64_encode",
            description="Base64-encode UTF-8 text.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        ToolSpec(
            name="b64_decode",
            description="Base64-decode to UTF-8 text.",
            input_schema={
                "type": "object",
                "properties": {"b64": {"type": "string"}},
                "required": ["b64"],
            },
        ),
    ]


class _FakeBackendLLM(LLMClient):
    """A deterministic offline LLM for hermetic backend tests.

    First model call of a turn requests one ``inject_input`` tool call; once a
    tool result is present in the replayed history, it returns a final text
    reply that echoes the tool output. No network is used.
    """

    def __init__(self) -> None:
        super().__init__(
            ProviderConfig(name="custom", api_url="x", api_key="k", model="fake")
        )

    def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
        has_tool_result = any(
            isinstance(m, dict) and m.get("role") == "tool" for m in messages
        )
        if on_text:
            on_text("working on it")
        if not has_tool_result and tools:
            call = ToolCall(
                id="c1",
                name="inject_input",
                input={"text": "echo hi", "submit": True},
            )
            return Turn(
                text="working on it",
                tool_calls=[call],
                assistant_message={"role": "assistant", "content": "working on it"},
                usage=None,
            )
        tool_output = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "tool":
                tool_output = str(m.get("content", ""))
        return Turn(
            text=f"done ({tool_output[:40]})",
            assistant_message={"role": "assistant", "content": "done"},
            usage=None,
        )


def _build_manager(status=None):
    """Activate the registered model on the backend, capturing verification.

    Returns ``(manager, verify_error)``: a
    :class:`~ludvart.backend.ModelManager` whose active client is built, and the
    verification error string (or ``None`` on success). ``status`` (optional)
    receives progress notes -- the active model's verification, the Copilot
    gateway launch, and each other model's verification -- so the client can show
    startup progress the way the in-process path prints it to stderr.
    """
    from .backend import ModelManager, build_backend, verify_backend
    from .models import active_index, label, load_registry

    def note(msg: str) -> None:
        if status is not None:
            status(msg)

    models = load_registry()
    idx = active_index(models) if models else None
    if idx is None:
        raise RuntimeError("no active model registered on the backend")
    active = models[idx]
    note(f"verifying {label(active)} (model {active['model']!r})...")
    backend = build_backend(active, status=note)
    verify_error = None
    try:
        verify_backend(backend)
        note(f"{label(active)}: ok")
    except Exception as exc:  # noqa: BLE001 - reported to the client, not fatal
        verify_error = str(exc)
        note(f"{label(active)}: FAILED ({exc})")
    available = _verify_others(models, idx, note)
    available[idx] = True
    manager = ModelManager(models, available, backend.client, backend.gateway)
    return manager, verify_error


def _verify_others(models, active_idx, note) -> list[bool]:
    """Verify every non-active model, reporting each via ``note``.

    Direct providers get a tiny live request; Copilot models are marked available
    when the gateway is installed and authorized (they are only truly started on
    ``/model use``), mirroring the in-process startup check.
    """
    from .llm import build_client
    from .models import is_copilot, label, registration_to_config

    available = [False] * len(models)
    for i, reg in enumerate(models):
        if i == active_idx:
            available[i] = True
            continue
        note(f"verifying {label(reg)}...")
        if is_copilot(reg):
            ok = _copilot_ready()
            available[i] = ok
            note(f"{label(reg)}: {'ok' if ok else 'unavailable'}")
            continue
        try:
            client = build_client(registration_to_config(reg))
            client.verify()
            available[i] = True
            note(f"{label(reg)}: ok")
        except Exception as exc:  # noqa: BLE001 - availability probe
            available[i] = False
            note(f"{label(reg)}: unavailable ({exc})")
    return available


def _copilot_ready() -> bool:
    """Whether a Copilot backend could start (installed + authorized)."""
    from .gateway import copilot_authenticated, litellm_available

    return litellm_available() and copilot_authenticated()


def _manager_active_label(manager) -> str:
    from .models import label

    idx = manager.active_index()
    if idx is None:
        return "backend"
    return label(manager.models[idx])


def _client_label(llm: LLMClient) -> str:
    return f"{getattr(llm, 'name', 'llm')}:{getattr(llm, 'model', 'model')}"


def _handle_command(msg, manager, core, channel: FrameChannel) -> None:
    """Run a forwarded slash command (``/model`` or ``/sessions``) on the backend.

    Emits result lines as ``PANEL_UPDATE`` system frames, applies the effect
    (switch/add/remove model, load/new session), and always sends a terminating
    ``REPLY`` so the client's command call returns. ``msg`` is the raw COMMAND
    frame; its ``payload`` carries structured data (e.g. a new registration).
    """
    def emit(text: str) -> None:
        channel.send(message(MsgType.PANEL_UPDATE, kind="system", text=text))

    line = msg.get("command", "") if isinstance(msg, dict) else str(msg)
    payload = msg.get("payload") if isinstance(msg, dict) else None
    parts = line.split()
    cmd = parts[0] if parts else ""
    if cmd == "model":
        _handle_model(parts[1:], manager, core, channel, emit, payload)
    elif cmd == "sessions":
        _handle_sessions(parts[1:], core, channel, emit)
    else:
        emit(f"[ludvart] command not supported in backend mode: /{cmd}")
    channel.send(message(MsgType.REPLY, text=""))


def _handle_model(args, manager, core, channel: FrameChannel, emit, payload=None) -> None:
    if manager is None:
        emit("Model management is unavailable on this backend.")
        return
    sub = args[0] if args else "list"
    if sub == "list":
        emit("Registered models (backend):")
        for descr in manager.describe():
            emit(descr)
        emit("Use /model use <n>|<model>, add, or remove <n>|<model>.")
    elif sub == "use":
        if len(args) < 2:
            emit("Usage: /model use <n>|<model>")
        else:
            _do_model_use(args[1], manager, core, channel, emit)
    elif sub == "remove":
        if len(args) < 2:
            emit("Usage: /model remove <n>|<model>")
        else:
            _do_model_remove(args[1], manager, emit)
    elif sub == "add":
        if not payload:
            emit("Model add is started from the client's guided prompts.")
        else:
            _do_model_add(payload, manager, channel, emit)
    else:
        emit(f"Supported: list, use, add, remove (got {sub!r}).")


def _do_model_remove(token: str, manager, emit) -> None:
    from .models import find_registration

    idx = find_registration(manager.models, token)
    if idx is None:
        emit(f"No model matches {token!r}. See /model list.")
        return
    _ok, msg = manager.remove(idx)
    emit(msg)


def _do_model_add(reg, manager, channel: FrameChannel, emit) -> None:
    def status(note: str) -> None:
        channel.send(message(MsgType.PANEL_UPDATE, kind="activity", label=note))

    _ok, msg = manager.add(reg, status=status)
    emit(msg)


def _handle_sessions(args, core, channel: FrameChannel, emit) -> None:
    from .session import (
        SessionStore,
        list_sessions,
        parse_rename_args,
        rename_session,
        resolve_session_ref,
    )

    sub = args[0] if args else "list"
    if sub == "list":
        core.session_list = list_sessions()
        if not core.session_list:
            emit("No saved sessions yet.")
            return
        current = core.session.session_id if core.session is not None else None
        for i, s in enumerate(core.session_list, 1):
            marker = "*" if s["id"] == current else " "
            label = s.get("title") or s.get("preview") or "(no messages)"
            if len(label) > 48:
                label = label[:47] + "..."
            emit(f"{marker}{i}. {s['id']}  ({s['count']} msgs)  {label}")
        emit('Use /sessions load <n>|<id>, new, or rename <id> "Title".')
    elif sub == "load":
        if len(args) < 2:
            emit("Usage: /sessions load <n>|<id>")
        else:
            _do_session_load(args[1], core, channel, emit)
    elif sub == "new":
        core.reset()
        core.session = SessionStore.create_new()
        channel.send(message(MsgType.PANEL_UPDATE, kind="transcript", messages=[]))
        channel.send(
            message(
                MsgType.PANEL_UPDATE,
                kind="session",
                session_id=core.session.session_id,
            )
        )
        emit(f"Started new session {core.session.session_id}.")
    elif sub == "rename":
        parsed = parse_rename_args(" ".join(args[1:]))
        if parsed is None:
            emit('Usage: /sessions rename <id> New title')
            return
        ref, title = parsed
        session_id, error = resolve_session_ref(ref, core.session_list)
        if error is not None:
            emit(error)
            return
        if not rename_session(session_id, title):
            emit(f"Could not rename session: {session_id}")
            return
        if core.session is not None and core.session.session_id == session_id:
            core.session.title = title
        if title:
            emit(f'Renamed {session_id} to "{title}".')
        else:
            emit(f"Cleared the title of {session_id}.")
    else:
        emit(f"Unknown subcommand: /sessions {sub}")


def _do_session_load(ref: str, core, channel: FrameChannel, emit) -> None:
    from .session import (
        SessionStore,
        load_session,
        neutralize_history,
        provider_family,
        working_history,
    )

    session_id = ref
    if ref.isdigit():
        idx = int(ref)
        if not (1 <= idx <= len(core.session_list)):
            emit(f"No session #{idx}. Run /sessions list first.")
            return
        session_id = core.session_list[idx - 1]["id"]
    try:
        data = load_session(session_id)
    except (OSError, ValueError):
        emit(f"Could not load session: {session_id}")
        return
    messages = [tuple(m) for m in data.get("messages", [])]
    version = int(data.get("version", 1) or 1)
    stored_family = provider_family(data.get("provider"))
    neutral = neutralize_history(
        list(data.get("llm_history", [])), version, stored_family
    )
    history = working_history(neutral)
    core.resume(messages, history)
    core.session = SessionStore.open_existing(session_id)
    core.session.title = data.get("title", "") or ""
    channel.send(
        message(
            MsgType.PANEL_UPDATE,
            kind="transcript",
            messages=[list(m) for m in messages],
        )
    )
    channel.send(
        message(MsgType.PANEL_UPDATE, kind="session", session_id=session_id)
    )
    emit(f"Loaded session {session_id} ({len(messages)} msgs).")


def _do_model_use(token: str, manager, core, channel: FrameChannel, emit) -> None:
    from .models import find_registration

    idx = find_registration(manager.models, token)
    if idx is None:
        emit(f"No model matches {token!r}. See /model list.")
        return

    def status(note: str) -> None:
        channel.send(message(MsgType.PANEL_UPDATE, kind="activity", label=note))

    ok, msg = manager.use(idx, status=status)
    emit(msg)
    if ok:
        core.llm = manager.client
        channel.send(
            message(MsgType.PANEL_UPDATE, kind="model", label=_manager_active_label(manager))
        )


def serve(
    channel: FrameChannel,
    *,
    llm: LLMClient | None = None,
    manager=None,
    session=None,
) -> None:
    """Run the backend request loop on ``channel`` until the client disconnects.

    One turn at a time: read a ``SUBMIT``, run it, send a ``REPLY``; forwarded
    ``/model`` and ``/sessions`` commands are handled via ``COMMAND``. A ``BYE``
    or a clean end-of-stream ends the loop. With ``llm`` given the model registry
    is bypassed (used by tests); otherwise the active registered model is built.
    ``session`` persists the conversation under ``~/.ludvart`` on the backend;
    it is created automatically only on the real path so tests stay hermetic.
    """
    verify_error = None
    if manager is not None:
        client = manager.client
        active_label = _manager_active_label(manager)
    elif llm is None and os.environ.get("LUDVART_BACKEND_FAKE_LLM"):
        # Hermetic offline path for tests: bypass the model registry entirely.
        client = _FakeBackendLLM()
        active_label = _client_label(client)
    elif llm is not None:
        client = llm
        active_label = _client_label(llm)
    else:
        # Real path: report build/verify progress (gateway launch, per-model
        # verification) as LOG frames so the client can show it at startup.
        def _startup(msg: str) -> None:
            channel.send(message(MsgType.LOG, text=msg))

        manager, verify_error = _build_manager(status=_startup)
        client = manager.client
        active_label = _manager_active_label(manager)
        if session is None:
            from .session import SessionStore

            session = SessionStore()

    host = RemoteTerminalHost(channel)
    core = AgentCore(
        client,
        host,
        system_prompt=_SYSTEM_PROMPT,
        tools=_prototype_tools(),
        client_tools=DEFAULT_CLIENT_TOOLS,
        session=session,
    )
    channel.send(
        message(
            MsgType.HELLO,
            app="ludvart",
            protocol=1,
            active_label=active_label,
            verified=verify_error is None,
            verify_error=verify_error,
            session_id=getattr(session, "session_id", None),
        )
    )
    while True:
        msg = channel.recv()
        if msg is None:
            return
        kind = msg_type(msg)
        if kind == MsgType.BYE:
            return
        if kind == MsgType.SUBMIT:
            text = msg.get("text", "")
            snapshot = msg.get("snapshot", "")
            try:
                reply = core.run_turn(text, snapshot)
            except ConnectionError:
                return  # client vanished mid-turn
            except Exception as exc:  # noqa: BLE001 - report, keep serving
                reply = f"[ludvart] backend error: {exc}"
            channel.send(message(MsgType.REPLY, text=reply))
        elif kind == MsgType.COMMAND:
            try:
                _handle_command(msg, manager, core, channel)
            except ConnectionError:
                return
        # Other client message kinds are ignored in the prototype.



def serve_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry for ``ludvart serve``: bind the framed channel to stdio.

    Reads frames from stdin and writes them to stdout; nothing else may touch
    stdout or the protocol stream is corrupted.
    """
    reader = sys.stdin.buffer
    writer = sys.stdout.buffer
    channel = FrameChannel(reader, writer, max_frame=DEFAULT_MAX_FRAME)
    try:
        serve(channel)
    except (BrokenPipeError, ConnectionError):
        return 0
    return 0
