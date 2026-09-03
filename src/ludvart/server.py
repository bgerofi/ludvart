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
import threading
import time
from typing import BinaryIO, Callable, Sequence

from .agent_core import DEFAULT_CLIENT_TOOLS, AgentCore
from .llm import LLMClient, ProviderConfig, ToolCall, ToolSpec, Turn
from .prompt import system_prompt
from .protocol import (
    DEFAULT_MAX_FRAME,
    FrameChannel,
    MsgType,
    message,
    msg_type,
)
from .remote_host import RemoteTerminalHost
from .tools import builtin_tool_specs


def _start_mcp():
    """Discover external MCP tools on the backend host, or return ``None``.

    MCP servers are configured (and launched) where the agent loop runs, so a
    remote backend uses that host's ``~/.ludvart/mcp.json``. Discovery failures
    are non-fatal: the agent simply runs with its built-in tools.
    """
    from .mcp import McpManager

    mcp = McpManager()
    if not mcp.config_exists():
        return None
    try:
        mcp.refresh()
    except Exception:  # noqa: BLE001 - a broken MCP server must not stop serving
        return None
    return mcp


def _agent_tools(mcp) -> list[ToolSpec]:
    """The full tool set advertised to the model: built-ins plus MCP tools."""
    specs = builtin_tool_specs()
    if mcp is not None:
        specs += mcp.tool_specs()
    return specs


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


def _manager_or_setup(status=None):
    """Activate the registered model, or report that setup is still needed.

    Returns ``(manager, verify_error, needs_setup)``. An empty registry is not
    an error: the backend cannot prompt for anything (its stdin/stdout carry the
    protocol), so it hands back a manager over the empty registry and lets the
    client drive registration. ``/model add`` then works normally and the first
    model added becomes the active one.
    """
    from .backend import ModelManager
    from .models import load_registry

    models = load_registry()
    if not models:
        return ModelManager(models, [], None), None, True
    manager, verify_error = _build_manager(status=status)
    return manager, verify_error, False


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
    result = None
    if cmd == "model":
        result = _handle_model(parts[1:], manager, core, channel, emit, payload)
    elif cmd == "sessions":
        _handle_sessions(parts[1:], core, channel, emit)
    elif cmd == "compact":
        _do_compact(core, channel, emit)
    elif cmd == "mcp_refresh":
        _do_mcp_refresh(core, emit)
    elif cmd == "mcp_login":
        _do_mcp_login(parts[1:], core, emit)
    elif cmd == "mcp_auth":
        _do_mcp_auth(parts[1:], core, emit)
    else:
        emit(f"[ludvart] command not supported in backend mode: /{cmd}")
    channel.send(message(MsgType.REPLY, text="", payload=result))


def _do_compact(core, channel: FrameChannel, emit) -> None:
    """Run ``/compact``: summarize the conversation on demand.

    Same mechanism as the automatic 80%-full compaction, but triggered by the
    user. The conversation and the model both live here, so this is where it has
    to happen; the client only renders the resulting summary marker.
    """
    if core.llm is None:
        emit("No model is registered on the backend; nothing to compact.")
        return
    if len(core.history) <= 2:
        emit("Conversation is already compact.")
        return
    before = len(core.history)
    compacted = core.compact()
    core.host.set_activity("")  # no turn is running; drop the spinner again
    if not compacted:
        emit("Compaction failed; the conversation was left unchanged.")
        return
    pct = core.context_pct
    pct_note = f", context now ~{pct:.0f}%" if pct is not None else ""
    emit(f"Compacted {before} messages into a summary{pct_note}.")


def _do_mcp_refresh(core, emit) -> None:
    """Run ``/mcp_refresh``: re-read mcp.json and rediscover external tools.

    The servers run on the backend host, so both the config and the child
    processes belong here. The refreshed tool list is folded back into the
    system prompt so the model is told what it can actually call.
    """
    from .mcp import McpManager

    if core.mcp is None:
        core.mcp = McpManager()
    if not core.mcp.config_exists():
        emit("No ~/.ludvart/mcp.json on the backend host; nothing to refresh.")
        core.mcp = None
        return
    try:
        report = core.mcp.refresh().report()
    except Exception as exc:  # noqa: BLE001 - reported to the user
        emit(f"MCP refresh failed: {exc}")
        return
    core.tools = _agent_tools(core.mcp)
    core.system_prompt = system_prompt(core.tools)
    emit(report)


def _do_mcp_login(args, core, emit) -> None:
    """Run ``/mcp_login <server>``: print the URL the user must authorize at.

    The browser is on the user's machine, not the backend host, so there is no
    loopback port to catch the redirect. ``/mcp_auth`` brings back what the
    browser landed on.
    """
    from .mcp import McpManager

    if core.mcp is None:
        core.mcp = McpManager()
    if not args:
        names = core.mcp.oauth_servers()
        if not names:
            emit("No OAuth-authenticated MCP servers in ~/.ludvart/mcp.json.")
        else:
            emit("Usage: /mcp_login <server>. Configured for OAuth:")
            for name in names:
                emit(f"  - {name}")
        return
    name = args[0]
    try:
        pending = core.mcp.begin_login(name)
    except Exception as exc:  # noqa: BLE001 - reported to the user
        emit(f"Could not start authorization for {name!r}: {exc}")
        return
    emit(f"Open this URL in your browser and approve access for {name!r}:")
    emit(pending.url)
    if pending.scope:
        emit(f"Scopes requested: {pending.scope}")
    redirect = pending.settings.redirect_uri
    emit(
        f"The browser will then fail to load {redirect} -- that is expected, "
        "nothing listens there. Copy the whole URL from its address bar and run:"
    )
    emit(f"  /mcp_auth {name} <pasted-url>")
    emit(
        f"If the provider rejects the request with 'redirect_uri_mismatch', add "
        f"{redirect} to the client's authorized redirect URIs, or set "
        f"oauth.redirectUri for {name!r} in mcp.json to one it already accepts."
    )


def _do_mcp_auth(args, core, emit) -> None:
    """Run ``/mcp_auth <server> <url>``: redeem the code the browser came back with."""
    if core.mcp is None or len(args) < 2:
        emit("Usage: /mcp_auth <server> <redirected-url>")
        return
    name, pasted = args[0], " ".join(args[1:])
    try:
        core.mcp.complete_login(name, pasted)
    except Exception as exc:  # noqa: BLE001 - reported to the user
        emit(f"Authorization for {name!r} failed: {exc}")
        return
    emit(f"Authorized {name!r}. Rediscovering tools...")
    _do_mcp_refresh(core, emit)


def _handle_model(args, manager, core, channel: FrameChannel, emit, payload=None):
    if manager is None:
        emit("Model management is unavailable on this backend.")
        return None
    sub = args[0] if args else "list"
    if sub == "list":
        emit("Registered models (backend):")
        for descr in manager.describe():
            emit(descr)
        emit("Use /model use <n>|<model>, add, or remove <n>|<model>.")
    elif sub == "copilot-models":
        return _copilot_model_choices()
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
            _do_model_add(payload, manager, core, channel, emit)
    else:
        emit(f"Supported: list, use, add, remove (got {sub!r}).")
    return None


def _copilot_model_choices() -> dict:
    """List the Copilot models available to this backend's authorization.

    The backend host owns the Copilot credentials and gateway, so the client's
    guided ``/model add`` flow asks us for the menu instead of listing locally.
    """
    from .gateway import (
        copilot_authenticated,
        list_copilot_models,
        litellm_available,
    )

    if not (litellm_available() and copilot_authenticated()):
        return {"copilot_models": [], "ready": False}
    return {"copilot_models": list(list_copilot_models()), "ready": True}


def _do_model_remove(token: str, manager, emit) -> None:
    from .models import find_registration

    idx = find_registration(manager.models, token)
    if idx is None:
        emit(f"No model matches {token!r}. See /model list.")
        return
    _ok, msg = manager.remove(idx)
    emit(msg)


def _do_model_add(reg, manager, core, channel: FrameChannel, emit) -> None:
    def status(note: str) -> None:
        channel.send(message(MsgType.PANEL_UPDATE, kind="activity", label=note))

    ok, msg = manager.add(reg, status=status)
    emit(msg)
    if ok and manager.client is None:
        # First model on a fresh backend: adding it is also what puts the
        # backend into service, so activate it instead of leaving it idle.
        _do_model_use(str(len(manager.models)), manager, core, channel, emit)


def _handle_sessions(args, core, channel: FrameChannel, emit) -> None:
    from .session import (
        SessionStore,
        list_sessions,
        parse_rename_args,
        rename_session,
        resolve_session_ref,
    )

    def row(text: str) -> None:
        channel.send(message(MsgType.PANEL_UPDATE, kind="row", text=text))

    sub = args[0] if args else "list"
    if sub == "list":
        core.session_list = list_sessions()
        if not core.session_list:
            emit("No saved sessions yet.")
            return
        current = core.session.session_id if core.session is not None else None
        width = len(str(len(core.session_list)))
        for i, s in enumerate(core.session_list, 1):
            marker = "*" if s["id"] == current else " "
            label = s.get("title") or s.get("preview") or "(no messages)"
            label = " ".join(label.split())  # a preview may span lines
            row(f"{marker}{str(i).rjust(width)}. {s['id']}  "
                f"({s['count']} msgs)  {label}")
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

    def before_swap(new_client) -> None:
        # Runs while the *old* model is still active. If the conversation would
        # overflow the target model's (possibly smaller) window, compact it here
        # -- using the outgoing model, which can still hold the full history --
        # before the swap tears that model down.
        new_cw = getattr(new_client, "context_window", 0) or 0
        if (
            new_cw > 0
            and core.last_input_tokens
            and len(core.history) > 2
            and 100.0 * core.last_input_tokens / new_cw >= core.CONTEXT_COMPACT_PCT
        ):
            core.compact()

    ok, msg = manager.use(idx, status=status, before_swap=before_swap)
    emit(msg)
    if ok:
        core.llm = manager.client
        channel.send(
            message(MsgType.PANEL_UPDATE, kind="model", label=_manager_active_label(manager))
        )
        # The new model may have a different context window, so re-derive the
        # badge now instead of leaving a stale percentage until the next turn.
        channel.send(
            message(
                MsgType.PANEL_UPDATE,
                kind="context",
                pct=_context_pct_for(core, manager.client),
            )
        )


def _context_pct_for(core, client) -> float | None:
    """Percent of ``client``'s window used by the last prompt, or ``None``."""
    tokens = getattr(core, "last_input_tokens", 0) or 0
    window = getattr(client, "context_window", 0) or 0
    if tokens <= 0 or window <= 0:
        return None
    return 100.0 * tokens / window


#: How long the backend keeps serving after the last word from a keepalive-
#: capable client. Generous: missing this many pings means the client is gone,
#: not merely busy or on a slow link.
CLIENT_TIMEOUT = float(os.environ.get("LUDVART_CLIENT_TIMEOUT", "300"))


def _shutdown(manager, core) -> None:
    """Release everything the backend owns outside its own process."""
    for close in (
        lambda: manager.gateway.stop() if getattr(manager, "gateway", None) else None,
        core.close,
    ):
        try:
            close()
        except Exception:
            pass


def watch_client(
    channel: FrameChannel,
    cleanup: Callable[[], None],
    *,
    timeout: float | None = None,
    exit_process: Callable[[int], None] = os._exit,
) -> threading.Event:
    """Exit the backend once a keepalive-capable client stops checking in.

    End-of-stream is the normal signal that the client is gone, but it never
    arrives if something else holds the pipe's write end open -- an SSH channel
    that was not torn down, say -- leaving the backend blocked in read() forever
    with its gateway still running. Pings distinguish a dead client from a
    merely idle one, so a session nobody is using is not killed.

    Arms itself only after the first ping, so an older client that sends none is
    served exactly as before. Returns an event that cancels the watch.
    """
    limit = CLIENT_TIMEOUT if timeout is None else timeout
    step = max(0.01, min(30.0, limit / 10.0))
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(step):
            if not channel.saw_ping:
                continue
            if time.monotonic() - channel.last_recv < limit:
                continue
            print(
                f"[ludvart] no word from the client in {limit:.0f}s; shutting down",
                file=sys.stderr,
                flush=True,
            )
            try:
                cleanup()
            finally:
                # The main thread is parked in a read that will never return, so
                # unwinding it is not an option.
                exit_process(0)
            return

    threading.Thread(target=loop, name="ludvart-client-watchdog", daemon=True).start()
    return stop


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
    needs_setup = False
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

        manager, verify_error, needs_setup = _manager_or_setup(status=_startup)
        if needs_setup:
            # Nothing registered here yet. Serve anyway with no active client:
            # the client is told to run its registration flow, and the first
            # /model add both registers and activates a model.
            client = None
            active_label = ""
        else:
            client = manager.client
            active_label = _manager_active_label(manager)
        if session is None:
            from .session import SessionStore

            session = SessionStore()

    host = RemoteTerminalHost(channel)
    mcp = _start_mcp()
    tools = _agent_tools(mcp)
    core = AgentCore(
        client,
        host,
        system_prompt=system_prompt(tools),
        tools=tools,
        client_tools=DEFAULT_CLIENT_TOOLS,
        session=session,
        mcp=mcp,
    )
    # The client owns no model, so lend it the active one for the one-shot
    # calls it makes while serving our requests (the settle detector).
    host.llm_provider = lambda: core.llm
    channel.send(
        message(
            MsgType.HELLO,
            app="ludvart",
            protocol=1,
            keepalive=True,
            active_label=active_label,
            verified=verify_error is None and not needs_setup,
            verify_error=verify_error,
            needs_setup=needs_setup,
            session_id=getattr(session, "session_id", None),
        )
    )
    watch_client(channel, lambda: _shutdown(manager, core))
    # A CANCEL arrives while the main thread is busy running the turn it wants
    # stopped, so it has to be read off the wire by somebody else.
    channel.start_reader(lambda msg: _consume_cancel(msg, core))
    try:
        _request_loop(channel, manager, core)
    finally:
        core.close()


def _consume_cancel(msg: dict, core: AgentCore) -> bool:
    """Handle a CANCEL frame off the reader thread; True if it was one."""
    if msg_type(msg) != MsgType.CANCEL:
        return False
    core.cancel.set()
    return True


def _request_loop(channel: FrameChannel, manager, core: AgentCore) -> None:
    """Serve requests on ``channel`` until the client disconnects."""
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
            if core.llm is None:
                reply = (
                    "[ludvart] no model is registered on the backend yet. "
                    "Use /model add to register one."
                )
                channel.send(message(MsgType.REPLY, text=reply))
                continue
            try:
                # Frames are queued in arrival order, so a CANCEL meant for the
                # previous turn has already been consumed by the time this
                # SUBMIT is dequeued and can never leak into this one.
                core.cancel.clear()
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
        # Other client message kinds are ignored.



def claim_stdout() -> BinaryIO:
    """Take exclusive ownership of the protocol stream.

    "Nothing else may touch stdout" is not something the backend can ask of the
    libraries it loads -- one stray ``print`` from deep inside a provider SDK
    lands mid-frame and desynchronises the client for the rest of the session.
    So fd 1 is handed over to stderr and the protocol keeps a private dup.
    """
    sys.stdout.flush()
    private = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(private, "wb", buffering=0)


def serve_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry for ``ludvart serve``: bind the framed channel to stdio.

    Reads frames from stdin and writes them to a private dup of stdout; see
    :func:`claim_stdout` for why nothing else can reach it.
    """
    from .llm import ensure_context_windows_file

    writer = claim_stdout()
    # The backend owns every model concern now, including the editable
    # context-window table, so seed it here rather than on the client.
    ensure_context_windows_file()
    reader = sys.stdin.buffer
    channel = FrameChannel(reader, writer, max_frame=DEFAULT_MAX_FRAME)
    try:
        serve(channel)
    except (BrokenPipeError, ConnectionError):
        return 0
    return 0
