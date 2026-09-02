"""Persistent conversation storage and in-panel slash commands.

Every ludvart run that opens the AI panel starts a *session*: a timestamped
directory under ``~/.ludvart/sessions/`` holding a ``conversation.json`` file that
is (re)written as the conversation grows. The layout is::

    ~/.ludvart/sessions/YYYY-MM-DD/HH_MM_SS/conversation.json

where the date and time are UTC. A new session is created the first time the
panel is opened in a process; subsequent opens in the same process keep
extending the same file (unless a different session is loaded).

This module has no terminal/UI dependencies so it can be unit tested directly.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

#: Bumped to 3 when ``llm_history`` became a provider-neutral log (see
#: :func:`build_context` / :func:`neutralize_history`). Files written by older
#: ludvart versions store a provider-native ``llm_history`` and are migrated on
#: load.
SESSIONS_VERSION = 3

#: The first ``SESSIONS_VERSION`` whose ``llm_history`` is stored in the
#: provider-neutral form. Anything older holds a provider-native history.
NEUTRAL_SESSIONS_VERSION = 3

#: Sentinel "family" for the neutral log: it is not a real provider, but passing
#: it as the *target* of :func:`sanitize_history` forces the provider-native ->
#: neutral flattening used to migrate old sessions.
NEUTRAL_FAMILY = "neutral"

_CONV_NAME = "conversation.json"
_DAY_FMT = "%Y-%m-%d"
_TIME_FMT = "%H_%M_%S"

#: Provider names collapsed to the wire *shape* their ``llm_history`` uses.
#: "openai" and "custom" both speak the OpenAI chat shape, so they share a
#: family and can resume each other's histories verbatim; "anthropic" and
#: "google" each have their own incompatible message/tool shapes.
_PROVIDER_FAMILY = {
    "openai": "openai",
    "custom": "openai",
    "anthropic": "anthropic",
    "google": "google",
}


def provider_family(provider: str | None) -> str | None:
    """Map a provider name to its wire-shape family (see ``_PROVIDER_FAMILY``).

    Unknown/absent providers return ``None`` so callers treat them as "family
    not known" and fall back to sanitizing conservatively.
    """
    if not provider:
        return None
    return _PROVIDER_FAMILY.get(provider)

# Message kinds that are part of the real conversation and therefore persisted.
# Slash-command echoes/output use the "system" kind and are never saved. The
# "summary" kind marks an automatic context-compaction point (see the compaction
# logic in :mod:`ludvart`).
_PERSISTED_KINDS = ("you", "ludvart", "info", "summary")

# Text marker wrapping a compaction summary in the model-facing ``llm_history``.
# It is ordinary message content (safe to send to any provider) and lets us find
# the latest summary when resuming a conversation.
SUMMARY_MARKER = "<conversationSummary>"
SUMMARY_MARKER_END = "</conversationSummary>"


def sessions_root() -> Path:
    """Return the root directory that holds all session folders.

    Honours ``LUDVART_SESSIONS_DIR`` (used by tests) and otherwise defaults to
    ``~/.ludvart/sessions``.
    """
    override = os.environ.get("LUDVART_SESSIONS_DIR")
    if override:
        return Path(override)
    return Path(os.path.expanduser("~/.ludvart/sessions"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    """Format a UTC datetime as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def persisted_messages(
    messages: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Drop ephemeral (slash-command) messages so only the conversation saves."""
    return [(k, t) for (k, t) in messages if k in _PERSISTED_KINDS]


class SessionStore:
    """Owns the on-disk file for one conversation and rewrites it on save."""

    def __init__(
        self,
        root: Path | str | None = None,
        started_at: datetime | None = None,
        session_id: str | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else sessions_root()
        if session_id is not None:
            # Bind to an existing session (e.g. after ``/sessions load``).
            self.session_id = session_id
            self.started_at = started_at or _parse_session_id(session_id)
        else:
            self.started_at = started_at or _utc_now()
            day = self.started_at.strftime(_DAY_FMT)
            clock = self.started_at.strftime(_TIME_FMT)
            self.session_id = f"{day}/{clock}"
        self.dir = self.root / self.session_id
        self.path = self.dir / _CONV_NAME
        # Optional, user-assigned display title (empty = fall back to preview).
        self.title = ""

    @classmethod
    def open_existing(
        cls, session_id: str, root: Path | str | None = None
    ) -> "SessionStore":
        """Return a store bound to an already-saved session directory."""
        return cls(root=root, session_id=session_id)

    @classmethod
    def create_new(cls, root: Path | str | None = None) -> "SessionStore":
        """Return a store bound to a directory that does not yet exist.

        Session ids have one-second resolution and no collision suffix, so two
        sessions started within the same second would otherwise share a
        directory and clobber each other on save. Advance ``started_at`` one
        second at a time (no sleeping) until an unused directory is found.
        """
        store = cls(root=root)
        while store.dir.exists():
            store = cls(root=root, started_at=store.started_at + timedelta(seconds=1))
        return store

    def save(
        self,
        messages: list[tuple[str, str]],
        llm_history: list[dict[str, Any]],
        provider: str | None = None,
    ) -> None:
        """Atomically (re)write the conversation file with the current state.

        ``provider`` is the name of the active LLM provider whose native message
        shape ``llm_history`` is stored in. It is recorded so that resuming the
        session under a different provider can detect the mismatch and sanitize
        the history to a provider-neutral form (see :func:`sanitize_history`).
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": SESSIONS_VERSION,
            "session_id": self.session_id,
            "started_at": _iso(self.started_at),
            "updated_at": _iso(_utc_now()),
            "provider": provider,
            "title": getattr(self, "title", "") or "",
            "messages": [list(m) for m in persisted_messages(messages)],
            "llm_history": llm_history,
        }
        tmp = self.dir / (_CONV_NAME + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        os.replace(tmp, self.path)


def _parse_session_id(session_id: str) -> datetime:
    """Best-effort parse of ``YYYY-MM-DD/HH_MM_SS`` into a UTC datetime."""
    try:
        day, clock = session_id.split("/", 1)
        dt = datetime.strptime(f"{day} {clock}", f"{_DAY_FMT} {_TIME_FMT}")
        return dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return _utc_now()


def list_sessions(root: Path | str | None = None) -> list[dict[str, Any]]:
    """Return a summary of every saved session, oldest first.

    Each summary has ``id``, ``started_at``, ``updated_at``, ``count`` (number
    of persisted messages) and ``preview`` (the first user line, if any).
    """
    base = Path(root) if root is not None else sessions_root()
    out: list[dict[str, Any]] = []
    if not base.exists():
        return out
    for day in sorted(p for p in base.iterdir() if p.is_dir()):
        for sess in sorted(p for p in day.iterdir() if p.is_dir()):
            conv = sess / _CONV_NAME
            if not conv.is_file():
                continue
            try:
                data = json.loads(conv.read_text())
            except (OSError, ValueError):
                continue
            messages = data.get("messages", [])
            preview = ""
            for kind, text in messages:
                if kind == "you":
                    preview = text
                    break
            out.append(
                {
                    "id": data.get("session_id", f"{day.name}/{sess.name}"),
                    "started_at": data.get("started_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "title": data.get("title", "") or "",
                    "count": len(messages),
                    "preview": preview,
                }
            )
    return out


def rename_session(
    session_id: str, title: str, root: Path | str | None = None
) -> bool:
    """Set the display ``title`` on a stored session (atomically). 

    Returns ``True`` on success, ``False`` if the session file is missing or
    unreadable. An empty ``title`` clears the title (reverting to the preview).
    """
    base = Path(root) if root is not None else sessions_root()
    conv = base / session_id / _CONV_NAME
    try:
        data = json.loads(conv.read_text())
    except (OSError, ValueError):
        return False
    data["title"] = title
    tmp = conv.parent / (_CONV_NAME + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        os.replace(tmp, conv)
    except OSError:
        return False
    return True


def parse_rename_args(text: str) -> tuple[str, str] | None:
    """Parse ``<session_id_or_index> <title>`` into ``(ref, title)``.

    The title is everything after the first whitespace-separated token, so it may
    contain spaces without quoting (e.g. ``150 TCP timeout issue``). A single
    pair of surrounding quotes is stripped if present (``id "New title"``), and
    an empty quoted title (``id ""``) clears the title. Returns ``None`` when no
    title is given, so callers can print a usage hint.
    """
    text = text.strip()
    if not text:
        return None
    parts = text.split(None, 1)
    if len(parts) < 2:
        return None
    ref, title = parts[0], parts[1].strip()
    if len(title) >= 2 and title[0] == title[-1] and title[0] in ("'", '"'):
        title = title[1:-1]
    return ref, title


def resolve_session_ref(
    ref: str, session_list: list[dict[str, Any]]
) -> tuple[str | None, str | None]:
    """Resolve a 1-based list index or a raw id to a session id.

    ``session_list`` is the most recent :func:`list_sessions` result. Returns
    ``(session_id, None)`` on success or ``(None, error_message)`` when a numeric
    index is out of range. A non-numeric ``ref`` is treated as a session id and
    returned unchanged.
    """
    if ref.isdigit():
        idx = int(ref)
        if not (1 <= idx <= len(session_list)):
            return None, f"No session #{idx}. Run /sessions list first."
        return session_list[idx - 1]["id"], None
    return ref, None


def load_session(
    session_id: str, root: Path | str | None = None
) -> dict[str, Any]:
    """Load and return the full stored data for ``session_id``."""
    base = Path(root) if root is not None else sessions_root()
    conv = base / session_id / _CONV_NAME
    data = json.loads(conv.read_text())
    return data


def working_history(
    llm_history: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return the resumable slice of a stored ``llm_history``.

    Automatic compaction replaces the model-facing context with a summary seed
    whose first message content starts with :data:`SUMMARY_MARKER`. When a
    conversation is resumed we start from the *last* such summary, so it always
    continues from its latest compression point and never replays the purged
    pre-summary turns. Histories without a marker are returned unchanged.
    """
    start = 0
    for i, msg in enumerate(llm_history):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.lstrip().startswith(SUMMARY_MARKER):
            start = i
    return list(llm_history[start:])


def _text_from_content(content: Any) -> str:
    """Flatten any provider's message ``content`` to plain text.

    ``content`` is either a string (OpenAI) or a list of provider-native blocks
    (Anthropic / Google). Tool-call and tool-result blocks are rendered as short
    human-readable summaries so the model keeps the gist of what happened without
    any provider-specific structure surviving.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(str(block.get("text", "")))
        elif btype in ("tool_use", "function_call"):
            name = block.get("name", "?")
            args = block.get("input", block.get("args", {}))
            parts.append(f"[called tool {name} with {args}]")
        elif btype == "tool_result":
            parts.append(f"[tool result] {block.get('content', '')}")
        elif btype == "function_response":
            resp = block.get("response", {})
            result = resp.get("result", resp) if isinstance(resp, dict) else resp
            parts.append(f"[tool result] {result}")
    return "\n".join(p for p in parts if p).strip()


def sanitize_history(
    llm_history: list[dict[str, Any]],
    stored_family: str | None,
    target_family: str | None,
) -> list[dict[str, Any]]:
    """Make a stored ``llm_history`` safe to replay under ``target_family``.

    Each provider serializes tool calls/results into its own message shape
    (e.g. OpenAI uses a ``"tool"`` role; Anthropic uses ``user``/``assistant``
    with ``tool_use``/``tool_result`` blocks). Replaying one provider's shape to
    another's API is rejected (e.g. Anthropic errors on the ``"tool"`` role).

    When the stored and target families match (including both unknown) the
    history is returned unchanged, preserving full tool round-trips. Otherwise
    every message is flattened to a provider-neutral ``user``/``assistant``
    message whose content is a plain string: tool-result turns become ``user``
    notes and tool-call/assistant turns keep their text (with a short summary of
    any calls). This loses the raw tool structure but keeps the conversation
    resumable across providers.
    """
    if stored_family == target_family:
        return list(llm_history)

    neutral: list[dict[str, Any]] = []
    for msg in llm_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = _text_from_content(msg.get("content"))
        # A tool-result turn (OpenAI/Google "tool" role, or Anthropic "user"
        # with tool_result blocks) is folded into a user-visible note so the
        # neutral history only ever uses the universally accepted user/assistant
        # roles.
        if role == "assistant":
            out_role = "assistant"
        else:
            out_role = "user"
        if not text:
            continue
        # Coalesce consecutive same-role messages so we don't emit empty or
        # role-adjacent fragments that some providers reject.
        if neutral and neutral[-1]["role"] == out_role:
            neutral[-1]["content"] += "\n\n" + text
        else:
            neutral.append({"role": out_role, "content": text})
    return neutral


# -- neutral conversation log <-> provider-native context -------------------
#
# The conversation is kept in memory (and persisted) as a provider-neutral log
# whose entries are one of:
#
#   {"role": "user",      "content": <str>}
#   {"role": "assistant", "content": <str>, "tool_calls": [
#         {"id": <str>, "name": <str>, "input": <dict>}, ...]}   # key optional
#   {"role": "tool",      "tool_call_id": <str>, "name": <str>, "content": <str>}
#
# Rendering a log into a provider's native message shape lives on the provider
# itself (``LLMClient.build_context``), so provider-specific knowledge stays
# with the client and multiple clients can coexist. This module only owns the
# *persistence* side: storing the neutral log and migrating older, provider-
# native sessions to it on load.


def neutralize_history(
    llm_history: list[dict[str, Any]],
    version: int,
    stored_family: str | None,
) -> list[dict[str, Any]]:
    """Return a neutral conversation log for a stored ``llm_history``.

    Sessions written at :data:`NEUTRAL_SESSIONS_VERSION` or later already store
    the neutral form and are returned unchanged. Older sessions hold a
    provider-native history; they are flattened to the neutral (text-only) form
    so any model can resume them. The raw tool-call structure of such legacy
    sessions is not reconstructed -- tool calls/results survive as short text
    notes -- but the conversation stays intact and resumable.
    """
    if version >= NEUTRAL_SESSIONS_VERSION:
        return list(llm_history)
    return sanitize_history(llm_history, stored_family, NEUTRAL_FAMILY)


# -- slash commands ---------------------------------------------------------

# Registry of in-panel commands and their subcommands. Used both to dispatch and
# to drive Tab completion. Keep the subcommand lists sorted for stable output.
SLASH_COMMANDS: dict[str, list[str]] = {
    "compact": [],
    "help": [],
    "init_helpers": [],
    "mcp_refresh": [],
    "model": ["add", "list", "remove", "use"],
    "perf": ["dump", "summary"],
    "revoke_approval": [],
    "sessions": ["list", "load", "new", "rename"],
}

# One-line usage + description for each command, shown by ``/help``. Ordered the
# way they should be listed. Keep in sync with :data:`SLASH_COMMANDS`.
SLASH_COMMAND_HELP: list[tuple[str, str]] = [
    ("/help", "Show this list of internal panel commands."),
    (
        "/compact",
        "Summarise the conversation so far and replace the working context "
        "with that summary, freeing up the context window.",
    ),
    (
        "/init_helpers",
        "Install or verify ~/.ludvart/bin/ludvart_helper on the foreground host "
        "(for precise file read/edit/search).",
    ),
    (
        "/mcp_refresh",
        "Reconnect to the MCP servers in ~/.ludvart/mcp.json and refresh their "
        "tool definitions.",
    ),
    (
        "/mcp_login <server>",
        "Start OAuth authorization for an MCP server; prints the URL to open.",
    ),
    (
        "/mcp_auth <server> <url>",
        "Finish /mcp_login by pasting the URL the browser was redirected to.",
    ),
    ("/sessions list", "List saved conversation sessions (current is marked *)."),
    ("/sessions load <n>|<id>", "Load and resume a saved session by number or id."),
    ("/sessions new", "Start a fresh, empty conversation in a new session file."),
    (
        "/sessions rename <id> \"Title\"",
        "Give a saved session a title so it is easy to find in the list.",
    ),
    ("/model list", "List registered models (in-use and available are marked)."),
    ("/model add", "Register a new model endpoint (guided prompts, then verify)."),
    ("/model use <n>|<model>", "Switch to another registered, available model."),
    ("/model remove <n>|<model>", "Unregister a model (not the one in use)."),
    (
        "/perf summary",
        "Report min/avg/max timing per operation type (LLM requests, tool "
        "calls) for this session.",
    ),
    ("/perf dump", "Dump the raw per-operation timing records into the panel."),
    (
        "/revoke_approval",
        "Revoke a previous 'approve everything' choice so future inject_input "
        "tool calls ask for approval again.",
    ),
]


def _common_completion(word: str, matches: list[str]) -> str | None:
    """Return the completion for ``word`` given candidate ``matches``.

    A single match completes to that match; multiple matches complete to their
    longest common prefix (only if it extends ``word``); no match yields None.
    """
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    prefix = os.path.commonprefix(matches)
    if len(prefix) > len(word):
        return prefix
    return None


def complete_slash(text: str) -> str | None:
    """Tab-complete a slash command line.

    Returns the completed line, or ``None`` if there is nothing to complete.
    A uniquely completed command/subcommand gains a trailing space so the next
    token can be typed straight away. Only command names and their immediate
    subcommands are completed (arguments are left untouched).
    """
    if not text.startswith("/"):
        return None
    body = text[1:]
    parts = body.split(" ")

    if len(parts) == 1:
        word = parts[0]
        matches = [c for c in sorted(SLASH_COMMANDS) if c.startswith(word)]
        completion = _common_completion(word, matches)
        if completion is None:
            return None
        new = "/" + completion
        if len(matches) == 1 and completion == matches[0]:
            new += " "
        return new if new != text else None

    if len(parts) == 2:
        cmd, word = parts
        subs = SLASH_COMMANDS.get(cmd)
        if not subs:
            return None
        matches = [s for s in subs if s.startswith(word)]
        completion = _common_completion(word, matches)
        if completion is None:
            return None
        new = "/" + cmd + " " + completion
        if len(matches) == 1 and completion == matches[0]:
            new += " "
        return new if new != text else None

    return None
