"""External MCP (Model Context Protocol) server support.

ludvart reads server definitions from ``~/.ludvart/mcp.json`` -- the same shape as
VS Code's ``mcp.json`` (a top-level ``"servers"`` map; the Claude-Desktop
``"mcpServers"`` key is also accepted) -- and exposes every server's tools to
the LLM as ordinary function-calling tools. Both transports are supported via
the official ``mcp`` SDK:

* **stdio** -- a subprocess launched from ``command`` / ``args`` / ``env``.
* **http / sse** -- a remote endpoint given by ``url`` (or ``serverUrl``;
  streamable HTTP by default, or the legacy SSE transport when
  ``"type": "sse"``).

Authorization
-------------
A remote server that answers with ``401`` and an OAuth challenge is handled by
adding an ``oauth`` block to its entry -- ``true`` to register a client on the
fly, or ``clientId`` / ``clientSecret`` / ``scopes`` when the provider issued
credentials in advance. Where the challenge appears differs between servers:
some reject ``initialize``, while Google's Gmail endpoint serves its whole tool
list in the clear and only refuses ``tools/call``. ``/mcp_login`` therefore
discovers the authorization server from metadata rather than waiting to be
challenged, and a tool called without a usable token answers with an error
saying so. See :mod:`ludvart.mcpauth`; the interactive half runs through the
``/mcp_login`` and ``/mcp_auth`` commands, never during startup.

Discovery
---------
An MCP server does not describe its tools up front: ludvart must connect, run the
``initialize`` handshake and then ``tools/list`` to learn the tool API. That is
the discovery step; it happens automatically the first time the panel opens and
can be forced again with the ``/mcp_refresh`` command.

Threading model
---------------
ludvart's core is synchronous (a PTY select loop, with LLM calls on a worker
thread) while the MCP SDK is asyncio-based, so all MCP I/O runs on a single
background event-loop thread. Synchronous callers submit coroutines to that loop
and block on the result. Each server is driven by its own long-lived worker
coroutine that owns the transport/session context for its entire lifetime and
takes commands (``call`` / ``close``) off an :class:`asyncio.Queue`. Keeping
every ``async with`` entered and exited in the same task is what anyio requires
and avoids "cancel scope in a different task" errors.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import os
import re
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from .llm import ToolSpec
from . import mcpauth

try:  # The SDK is a declared dependency, but degrade gracefully if it is absent.
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.sse import sse_client

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - only when mcp is not installed
    _IMPORT_ERROR = exc


class McpConfigError(RuntimeError):
    """Raised when ``mcp.json`` cannot be read or is malformed."""


def available() -> bool:
    """True if the ``mcp`` SDK imported successfully."""
    return _IMPORT_ERROR is None


def config_path() -> str:
    """Path of the MCP config file (``~/.ludvart/mcp.json``)."""
    return os.path.join(os.path.expanduser("~"), ".ludvart", "mcp.json")


def _stderr_sink():
    """Return a writable text stream for a stdio server's stderr.

    Uses ``LUDVART_MCP_LOG`` (append) when set so server diagnostics can be
    inspected; otherwise discards them. Never the real terminal, which ludvart
    composites and must keep byte-exact.
    """
    log = os.environ.get("LUDVART_MCP_LOG")
    target = log if log else os.devnull
    try:
        return open(target, "a", encoding="utf-8", errors="replace")
    except OSError:
        return open(os.devnull, "a", encoding="utf-8")


_ENV_REF = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: str) -> str:
    """Expand ``${env:VAR}`` references in ``value`` from the environment."""
    if not isinstance(value, str):
        return value
    return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value)


def load_config(path: str | None = None) -> dict[str, dict]:
    """Return the ``{server_name: config}`` map from ``mcp.json``.

    Missing file -> empty map. A present-but-unreadable or invalid file raises
    :class:`McpConfigError` so the caller can report it.
    """
    path = path or config_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise McpConfigError(f"cannot read {path}: {exc}") from exc
    servers = raw.get("servers")
    if servers is None:
        servers = raw.get("mcpServers")
    if not isinstance(servers, dict):
        return {}
    return {
        name: cfg
        for name, cfg in servers.items()
        if isinstance(cfg, dict) and not cfg.get("disabled")
    }


_SAFE = re.compile(r"[^a-zA-Z0-9_-]")


def _describe(exc: BaseException) -> str:
    """A message worth showing the user, not the wrapper it arrived in.

    The SDK's transports run inside anyio task groups, so a failure surfaces as
    ``unhandled errors in a TaskGroup (1 sub-exception)`` -- which says nothing
    about the missing token or refused connection underneath.
    """
    leaves: list[BaseException] = []

    def walk(err: BaseException) -> None:
        nested = getattr(err, "exceptions", None)
        if nested:
            for sub in nested:
                walk(sub)
        else:
            leaves.append(err)

    walk(exc)
    texts = [str(leaf) or leaf.__class__.__name__ for leaf in leaves]
    return "; ".join(dict.fromkeys(texts)) or str(exc) or exc.__class__.__name__


def _public_name(server: str, tool: str, used: set[str]) -> str:
    """Build a unique, schema-legal tool name (``^[A-Za-z0-9_-]{1,64}$``)."""
    name = "mcp_" + _SAFE.sub("_", f"{server}_{tool}")
    if len(name) > 64:
        digest = hashlib.md5(f"{server}\x00{tool}".encode("utf-8")).hexdigest()[:8]
        name = name[:55] + "_" + digest
    base, i = name, 2
    while name in used:
        suffix = f"_{i}"
        name = base[: 64 - len(suffix)] + suffix
        i += 1
    used.add(name)
    return name


def _result_to_text(result: Any) -> str:
    """Flatten an MCP ``CallToolResult`` into plain text for the model."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
            continue
        btype = getattr(block, "type", "content")
        if btype == "resource":
            res = getattr(block, "resource", None)
            rtext = getattr(res, "text", None)
            parts.append(rtext if rtext is not None else f"[{btype}]")
        else:
            parts.append(f"[{btype} content]")
    out = "\n".join(parts).strip()
    if not out:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            try:
                out = json.dumps(structured)
            except (TypeError, ValueError):
                out = str(structured)
    if getattr(result, "isError", False):
        return f"[MCP tool error] {out}" if out else "[MCP tool error]"
    return out or "(no output)"


@dataclass
class _ServerState:
    """Live state for one configured server on the event loop."""

    name: str
    cfg: dict
    queue: "asyncio.Queue | None" = None
    task: "asyncio.Task | None" = None
    ready: "asyncio.Future | None" = None
    tools: list = field(default_factory=list)
    error: str | None = None


@dataclass
class _ToolEntry:
    public: str
    server: str
    tool: str
    spec: ToolSpec


@dataclass
class McpStatus:
    """Outcome of a refresh: per-server tool counts / errors and a total."""

    servers: dict[str, tuple[int, str | None]]
    total_tools: int

    def report(self) -> str:
        """A multi-line, ASCII status summary for the panel."""
        if not self.servers:
            return "No MCP servers configured in ~/.ludvart/mcp.json."
        ok = sum(1 for c, e in self.servers.values() if e is None)
        head = (
            f"MCP: {ok}/{len(self.servers)} server(s) connected, "
            f"{self.total_tools} tool(s) available."
        )
        lines = [head]
        for name, (count, err) in self.servers.items():
            if err is None:
                lines.append(f"  - {name}: {count} tool(s)")
            else:
                lines.append(f"  - {name}: ERROR {err}")
        return "\n".join(lines)


class McpManager:
    """Connects to configured MCP servers and exposes their tools.

    Thread-safe: public methods may be called from ludvart's worker threads; all
    MCP I/O is marshalled onto one private event-loop thread.
    """

    def __init__(
        self,
        config_file: str | None = None,
        connect_timeout: float = 20.0,
        call_timeout: float = 120.0,
    ) -> None:
        self._config_file = config_file or config_path()
        self._connect_timeout = connect_timeout
        self._call_timeout = call_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._servers: dict[str, _ServerState] = {}
        self._tools: dict[str, _ToolEntry] = {}
        self._pending: dict[str, mcpauth.PendingLogin] = {}
        self._lock = threading.Lock()

    # -- public API ----------------------------------------------------------

    def config_exists(self) -> bool:
        return os.path.exists(self._config_file)

    def tool_specs(self) -> list[ToolSpec]:
        """Advertised tools across all connected servers (namespaced)."""
        with self._lock:
            return [e.spec for e in self._tools.values()]

    def is_mcp_tool(self, name: str) -> bool:
        with self._lock:
            return name in self._tools

    def refresh(self) -> McpStatus:
        """(Re)load the config, reconnect every server, and discover tools.

        Blocking; safe to call from a worker thread. Raises
        :class:`McpConfigError` for a bad config and :class:`RuntimeError` if
        the SDK is unavailable.
        """
        if not available():
            raise RuntimeError(f"mcp SDK not available: {_IMPORT_ERROR}")
        configs = load_config(self._config_file)
        self._ensure_loop()
        budget = self._connect_timeout * (len(configs) + 1) + 10.0
        return self._submit(self._reconnect(configs), timeout=budget)

    def oauth_servers(self) -> list[str]:
        """Names of configured servers that authenticate with OAuth."""
        names = []
        for name, cfg in load_config(self._config_file).items():
            try:
                if mcpauth.parse_settings(cfg, _expand) is not None:
                    names.append(name)
            except ValueError:
                continue
        return names

    def begin_login(self, name: str) -> mcpauth.PendingLogin:
        """Start an interactive authorization; the result carries the URL to visit.

        Nothing is connected here. The server is only asked for its metadata,
        because whether it challenges a connection at all varies -- some do it
        at ``initialize``, others not until a tool is called.
        """
        configs = load_config(self._config_file)
        if name not in configs:
            raise McpConfigError(f"no MCP server named {name!r} in mcp.json")
        cfg = configs[name]
        settings = mcpauth.parse_settings(cfg, _expand)
        if settings is None:
            raise McpConfigError(f"server {name!r} is not configured for OAuth")
        url = _expand(str(cfg.get("url") or cfg.get("serverUrl") or ""))
        if not url:
            raise McpConfigError(f"server {name!r} has no url to authorize against")
        try:
            pending = mcpauth.start_login(name, url, settings)
        except Exception as exc:  # noqa: BLE001 - reported to the user
            raise McpConfigError(_describe(exc)) from None
        self._pending[name] = pending
        return pending

    def complete_login(self, name: str, pasted: str) -> None:
        """Hand the pasted redirect back to a login started by :meth:`begin_login`.

        Succeeding means the tokens are on disk; the caller re-runs discovery so
        the server joins the tool list through the normal path.
        """
        pending = self._pending.get(name)
        if pending is None:
            raise McpConfigError(f"no login in progress for {name!r}")
        try:
            mcpauth.finish_login(pending, pasted)
        except Exception as exc:  # noqa: BLE001 - reported to the user
            raise McpConfigError(_describe(exc)) from None
        finally:
            self._pending.pop(name, None)

    def cancel_login(self, name: str) -> None:
        """Abandon a login left half-finished by an earlier attempt."""
        self._pending.pop(name, None)

    def call_tool(self, public_name: str, arguments: dict | None) -> str:
        """Invoke a namespaced MCP tool and return its text result."""
        with self._lock:
            entry = self._tools.get(public_name)
            server = self._servers.get(entry.server) if entry else None
        if entry is None:
            return f"[ludvart] unknown MCP tool: {public_name}"
        if server is None or server.task is None or server.task.done():
            return (
                f"[ludvart] MCP server '{entry.server}' is not connected; "
                "run /mcp_refresh"
            )
        try:
            result = self._submit(
                self._call(server, entry.tool, arguments or {}),
                timeout=self._call_timeout,
            )
        except concurrent.futures.TimeoutError:
            return (
                f"[ludvart] MCP tool '{public_name}' timed out after "
                f"{self._call_timeout:.0f}s"
            )
        except Exception as exc:  # noqa: BLE001 - reported to the model
            return f"[ludvart] MCP tool '{public_name}' failed: {_describe(exc)}"
        return _result_to_text(result)

    def close(self) -> None:
        """Disconnect all servers and stop the event-loop thread."""
        if self._loop is None:
            return
        try:
            self._submit(self._close_all(), timeout=10.0)
        except Exception:  # noqa: BLE001 - best-effort shutdown
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._loop = None
        self._thread = None

    # -- event-loop plumbing -------------------------------------------------

    def _ensure_loop(self) -> None:
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="ludvart-mcp", daemon=True
        )
        self._thread.start()

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro, timeout: float):
        assert self._loop is not None
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout)

    # -- coroutines (run on the loop thread) --------------------------------

    async def _reconnect(self, configs: dict[str, dict]) -> McpStatus:
        await self._close_all()
        servers: dict[str, _ServerState] = {}
        for name, cfg in configs.items():
            st = _ServerState(name=name, cfg=cfg)
            st.queue = asyncio.Queue()
            st.ready = self._loop.create_future()  # type: ignore[union-attr]
            st.task = self._loop.create_task(self._worker(st))  # type: ignore[union-attr]
            servers[name] = st

        results: dict[str, tuple[int, str | None]] = {}
        tools: dict[str, _ToolEntry] = {}
        used: set[str] = set()
        for name, st in servers.items():
            try:
                await asyncio.wait_for(
                    asyncio.shield(st.ready), self._connect_timeout
                )
            except asyncio.TimeoutError:
                st.error = f"timed out after {self._connect_timeout:.0f}s"
            except Exception as exc:  # noqa: BLE001
                st.error = _describe(exc)
            if st.error:
                if st.task is not None and not st.task.done():
                    st.task.cancel()
                results[name] = (0, st.error)
                continue
            for tool in st.tools:
                public = _public_name(name, tool.name, used)
                spec = ToolSpec(
                    name=public,
                    description=self._describe(name, tool),
                    input_schema=self._schema(tool),
                )
                tools[public] = _ToolEntry(public, name, tool.name, spec)
            results[name] = (len(st.tools), None)

        with self._lock:
            self._servers = servers
            self._tools = tools
        return McpStatus(results, len(tools))

    @staticmethod
    def _describe(server: str, tool) -> str:
        desc = (getattr(tool, "description", None) or "").strip()
        if not desc:
            desc = f"MCP tool '{tool.name}'."
        return f"[MCP:{server}] {desc}"

    @staticmethod
    def _schema(tool) -> dict:
        schema = getattr(tool, "inputSchema", None)
        if isinstance(schema, dict) and schema:
            return schema
        return {"type": "object", "properties": {}}

    async def _worker(self, st: _ServerState) -> None:
        """Own one server's connection for its whole lifetime (single task)."""
        try:
            async with AsyncExitStack() as stack:
                read, write = await self._open_transport(stack, st.name, st.cfg)
                session = await stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()
                listed = await session.list_tools()
                st.tools = list(listed.tools)
                if st.ready is not None and not st.ready.done():
                    st.ready.set_result(True)
                while True:
                    cmd = await st.queue.get()  # type: ignore[union-attr]
                    if cmd[0] == "close":
                        cmd[1].set_result(True)
                        return
                    if cmd[0] == "call":
                        _, tool_name, args, fut = cmd
                        try:
                            res = await session.call_tool(tool_name, args)
                            if not fut.done():
                                fut.set_result(res)
                        except Exception as exc:  # noqa: BLE001
                            if not fut.done():
                                fut.set_exception(exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - discovery/transport failure
            st.error = st.error or _describe(exc)
            if st.ready is not None and not st.ready.done():
                st.ready.set_exception(exc)

    async def _open_transport(self, stack: AsyncExitStack, name: str, cfg: dict):
        command = cfg.get("command")
        url = cfg.get("url") or cfg.get("serverUrl")
        if command:
            overrides = {
                str(k): _expand(str(v))
                for k, v in (cfg.get("env") or {}).items()
            }
            params = StdioServerParameters(
                command=_expand(str(command)),
                args=[_expand(str(a)) for a in (cfg.get("args") or [])],
                env={**os.environ, **overrides},
                cwd=_expand(str(cfg["cwd"])) if cfg.get("cwd") else None,
            )
            # A stdio server's stderr must NOT reach the real terminal: ludvart
            # composites the screen from a pyte model, so stray bytes there
            # corrupt the display. Route it to LUDVART_MCP_LOG if set, else drop it.
            errlog = stack.enter_context(_stderr_sink())
            read, write = await stack.enter_async_context(
                stdio_client(params, errlog=errlog)
            )
            return read, write
        if url:
            url = _expand(str(url))
            headers = {
                str(k): _expand(str(v))
                for k, v in (cfg.get("headers") or {}).items()
            } or None
            factory = self._client_factory(name, cfg)
            extra = {"httpx_client_factory": factory} if factory else {}
            if str(cfg.get("type", "")).lower() == "sse":
                read, write = await stack.enter_async_context(
                    sse_client(url, headers=headers, **extra)
                )
            else:
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(url, headers=headers, **extra)
                )
            return read, write
        raise McpConfigError(
            "server entry needs 'command' (stdio) or 'url' (http/sse)"
        )

    def _client_factory(self, name: str, cfg: dict):
        """The HTTP client factory that authorizes this server, or ``None``."""
        settings = mcpauth.parse_settings(cfg, _expand)
        if settings is None:
            return None
        return mcpauth.client_factory(name, settings)

    async def _call(self, st: _ServerState, tool_name: str, args: dict):
        fut = self._loop.create_future()  # type: ignore[union-attr]
        await st.queue.put(("call", tool_name, args, fut))  # type: ignore[union-attr]
        return await fut

    async def _close_all(self) -> None:
        for st in list(self._servers.values()):
            task = st.task
            if task is None or task.done():
                continue
            try:
                fut = self._loop.create_future()  # type: ignore[union-attr]
                await st.queue.put(("close", fut))  # type: ignore[union-attr]
                await asyncio.wait_for(fut, timeout=5.0)
            except Exception:  # noqa: BLE001
                task.cancel()
            try:
                await task
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            self._servers = {}
            self._tools = {}
