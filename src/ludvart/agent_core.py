"""The provider-agnostic agent loop, decoupled from the terminal.

:class:`AgentCore` owns the multi-turn conversation and the converse/tool loop.
It holds the provider-neutral history, calls the LLM, and dispatches tool calls
-- executing *backend* tools in-process and routing *client* tools (the ones
that touch the terminal, e.g. ``inject_input``) through a
:class:`~ludvart.terminal_host.TerminalHost`.

This is the piece that runs on the backend when the client and backend are
split. It has no dependency on the PTY, pyte, or rendering; everything terminal
lives behind the host interface.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Sequence

from . import tools as builtin
from .llm import LLMClient, ToolCall, ToolSpec, Turn
from .session import SUMMARY_MARKER, SUMMARY_MARKER_END
from .terminal_host import TerminalHost

#: Tools that must run where the terminal is (the client). Everything else is a
#: backend tool executed in-process by :meth:`AgentCore._run_tool`.
DEFAULT_CLIENT_TOOLS = builtin.CLIENT_TOOL_NAMES


class TurnCancelled(BaseException):
    """Unwinds an in-flight turn when the user steers or cancels it.

    Derives from ``BaseException`` so it passes through the providers' ``except
    Exception`` retry wrapper untouched, instead of being reported as an API
    failure and replayed.
    """


def neutral_assistant(turn: Turn) -> dict:
    """Neutral-log entry for an assistant turn (text plus any tool calls)."""
    entry: dict = {"role": "assistant", "content": turn.text or ""}
    if turn.tool_calls:
        entry["tool_calls"] = [
            {"id": c.id, "name": c.name, "input": dict(c.input)}
            for c in turn.tool_calls
        ]
    return entry


def neutral_tool_result(call: ToolCall, output: str) -> dict:
    """Neutral-log entry for a tool result (keeps id and name for replay)."""
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": output,
    }


def tool_call_note(call: ToolCall) -> str:
    """One-line summary of a tool invocation for the live narration.

    Appended to the transient narration so the user can see the running history
    of what the agent is doing -- including fast tools like
    ``capture_screen_history`` whose "Calling ..." label would flash by faster
    than a render frame. String arguments are quoted (so control characters
    injected via ``inject_input`` show up as escapes) and long values truncated.
    """
    parts: list[str] = []
    for key, val in call.input.items():
        if isinstance(val, str) and len(val) > 60:
            val = val[:57] + "\u2026"
        parts.append(f"{key}={val!r}")
    return f"\u2192 {call.name}(" + ", ".join(parts) + ")"


class AgentCore:
    """Runs the agent loop for one conversation against a :class:`TerminalHost`.

    ``client_tools`` names the tools that must execute on the client (terminal)
    side; they are dispatched through the host. All other advertised tools are
    backend tools handled by :meth:`_run_tool`.
    """

    #: Compact the conversation once a prompt fills this much of the window.
    CONTEXT_COMPACT_PCT = 80.0

    #: Token budget for the summary request itself.
    SUMMARY_MAX_TOKENS = 2048

    #: Messages a compaction reseeds the history with (summary + acknowledgement).
    _SEED_LEN = 2

    def __init__(
        self,
        llm: LLMClient,
        host: TerminalHost,
        *,
        system_prompt: str,
        tools: Sequence[ToolSpec] | None = None,
        client_tools: frozenset[str] = DEFAULT_CLIENT_TOOLS,
        max_tokens: int = 8192,
        session=None,
        mcp=None,
    ) -> None:
        self.llm = llm
        self.host = host
        self.system_prompt = system_prompt
        self.tools = list(tools) if tools else []
        self.client_tools = client_tools
        self.max_tokens = max_tokens
        #: External MCP servers discovered on this host (None when unused).
        self.mcp = mcp
        #: Scratch space for tools that write files (e.g. ``fetch_url``).
        self.scratch = builtin.ScratchDir()
        #: The running provider-neutral conversation log.
        self.history: list[dict] = []
        #: Human-readable transcript pairs, for session persistence.
        self.transcript: list[tuple[str, str]] = []
        #: Persistent conversation store on the backend (None disables saving).
        self.session = session
        #: Cache of the last `/sessions list`, for index -> id resolution.
        self.session_list: list[dict] = []
        #: Prompt tokens reported by the most recent model call, so the context
        #: badge can be recomputed when the active model (window) changes.
        self.last_input_tokens = 0
        #: Percent of the window used by the most recent prompt (drives both the
        #: badge and the compaction trigger). ``None`` until a turn reports usage.
        self.context_pct: float | None = None
        #: Tokens billed to this conversation across every request it has made.
        #: Not the size of the context: the whole context is resent each turn, so
        #: these grow far past what the conversation currently holds.
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        #: Set from the channel's reader thread to abandon the in-flight turn.
        self.cancel = threading.Event()

    def _raise_if_cancelled(self) -> None:
        if self.cancel.is_set():
            raise TurnCancelled()

    def run_turn(self, question: str, snapshot: str) -> str:
        """Run one user turn to completion and return the assistant's reply.

        Appends the user turn (embedding the ask-time ``snapshot``), then loops:
        call the model, run any requested tools, feed results back, until the
        model returns a plain-text answer.
        """
        # Compact before the turn is recorded, so the summary seed is followed
        # by the user's question. Reseeding after it would leave the assistant
        # acknowledgement as the final message, which providers that reject an
        # assistant prefill (e.g. Copilot) refuse with a 400.
        self.maybe_compact()
        self.transcript.append(("you", question))
        # Stamp each snapshot with a UTC timestamp (nanosecond precision) so it
        # can be addressed later. Snapshots are never sent where they are stored
        # (see :meth:`_collapse_screenshots`); the timestamp survives in the
        # breadcrumb, letting the model fetch the full snapshot back via the
        # ``get_past_snapshot`` tool.
        snapshot_ts = self._utc_ns_timestamp()
        user_content = (
            f'<screenContext ts="{snapshot_ts}">\n'
            f"{snapshot}\n"
            "</screenContext>\n"
            f"<userRequest>\n{question}\n</userRequest>"
        )
        self.history.append({"role": "user", "content": user_content})
        # Where this turn starts, so a mid-loop compaction can carry it over.
        checkpoint = len(self.history) - 1
        system = {"role": "system", "content": self.system_prompt}

        # Running narration for this ask. Streamed commentary and one note per
        # tool call accumulate here so the transient interim line keeps showing
        # what the agent has been doing across tool round-trips, instead of each
        # request's stream overwriting the previous one.
        narration: list[str] = []
        last_stream = ""

        def compose(streamed: str = "") -> str:
            parts = list(narration)
            if streamed:
                parts.append(streamed)
            return "\n".join(parts)

        def on_text(text: str) -> None:
            nonlocal last_stream
            # The stream is the longest uninterruptible stretch of a turn, so a
            # steer that only took effect between steps would be felt as "it
            # ignored me until it finished".
            self._raise_if_cancelled()
            last_stream = text
            self.host.narrate(compose(text))

        while True:
            # Compact before EVERY request, not just once per user turn: one
            # agentic turn can issue many tool round-trips and each re-sends the
            # whole history (snapshots + tool output), so the context grows
            # inside this loop. The in-flight turn is carried over the reseed --
            # dropping it would strand the question being answered and orphan
            # the tool_use/tool_result pairs the model is waiting on.
            if self.maybe_compact(keep_tail=self.history[checkpoint:]):
                checkpoint = self._SEED_LEN  # the tail now follows the seed
            self.host.set_activity("Thinking")
            last_stream = ""
            interrupted_while_streaming = True
            try:
                # A cancel that landed while the previous step's tools were
                # running would otherwise sit unnoticed until this request
                # starts streaming -- and a provider that streams nothing would
                # never notice it at all.
                self._raise_if_cancelled()
                turn = self.llm.converse(
                    [system, *self._build_context()],
                    tools=self.tools or None,
                    max_tokens=self.max_tokens,
                    on_text=on_text,
                )
                interrupted_while_streaming = False
                self.history.append(neutral_assistant(turn))
                self._report_usage(turn)
                if not turn.tool_calls:
                    self.transcript.append(("ludvart", turn.text))
                    self._persist()
                    return turn.text
                # Keep this request's streamed commentary in the narration (above
                # the tool notes) so it stays visible through the tool round-trips.
                if last_stream:
                    narration.append(last_stream)
                for call in turn.tool_calls:
                    self._raise_if_cancelled()
                    narration.append(tool_call_note(call))
                    self.host.narrate(compose())
                    self.host.set_activity(f"Calling {call.name}")
                    output = self._stamp_screenshot(self._run_tool(call))
                    self.history.append(neutral_tool_result(call, output))
            except TurnCancelled:
                return self._end_cancelled_turn(
                    last_stream if interrupted_while_streaming else ""
                )
            except BaseException:
                # Roll the turn back so the history stays well-formed. Failing
                # between a tool call and its result (a dropped backend
                # connection, a cancel, a provider error) would otherwise leave
                # an assistant turn whose tool calls are never answered, and
                # every later request built from it is rejected.
                del self.history[checkpoint:]
                raise

    def _end_cancelled_turn(self, partial: str) -> str:
        """Close out an interrupted turn, keeping the work it already finished.

        Unlike a failed turn this is *not* rolled back. The steering instruction
        arrives as the next user turn, so the model reads its own partial work
        above it rather than a reconstruction of it -- and the tools that already
        ran keep their record, which matters because their effect on the terminal
        is real and cannot be rolled back with the history. A tool call left
        unanswered is dropped when the context is rendered.
        """
        if partial:
            self.history.append({"role": "assistant", "content": partial})
            self.transcript.append(("ludvart", partial))
        self._persist()
        return partial

    def _report_usage(self, turn) -> None:
        """Push the prompt's context usage to the host (drives the ``[NN%]`` badge)."""
        usage = getattr(turn, "usage", None)
        if usage is None:
            return
        self._bank_usage(usage)
        self.last_input_tokens = usage.input_tokens
        self.context_pct = usage.context_percent()
        self.host.set_context_pct(self.context_pct)

    def _bank_usage(self, usage) -> None:
        """Add one request's tokens to the conversation's running total.

        Every request counts, including the tool round-trips within a turn and
        the compaction summaries between them, because every one of them is
        billed. Providers that report no usage block are simply not counted, so
        the totals are a floor rather than an invoice.
        """
        if usage is None:
            return
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.host.set_token_totals(self.total_input_tokens, self.total_output_tokens)

    # -- context compaction --------------------------------------------------

    def maybe_compact(self, keep_tail: list[dict] | None = None) -> bool:
        """Compact the conversation into a summary if the window is nearly full.

        Triggered when the last prompt filled at least
        :attr:`CONTEXT_COMPACT_PCT` of the model's context window. A history
        that is already just a fresh summary seed (<= 2 messages) is left alone.
        ``keep_tail`` is replayed after the seed (see :meth:`_compact_history`).
        Returns ``True`` when it actually compacted.
        """
        if len(self.history) <= self._SEED_LEN:
            return False
        if self.context_pct is None or self.context_pct < self.CONTEXT_COMPACT_PCT:
            return False
        return self._compact_history(keep_tail=keep_tail) is not None

    def compact(self) -> str | None:
        """Compact now regardless of how full the window is (the ``/compact``
        command). Returns the summary, or ``None`` if the request failed.
        """
        return self._compact_history()

    def _compact_history(self, keep_tail: list[dict] | None = None) -> str | None:
        """Summarize the conversation and reseed the context from that summary.

        The model-facing history is replaced by a two-message seed; the visible
        transcript keeps the whole conversation with a compaction marker, so a
        reloaded session shows where it happened. ``keep_tail`` (the messages of
        a turn that is still in flight) is replayed after the seed, so the model
        keeps the request it is answering and no tool call is left unanswered.
        Returns the summary text, or ``None`` when the summary request failed
        (history left unchanged).
        """
        self.host.set_activity("Compacting context")
        summary = self._summarize_history()
        if not summary:
            return None  # failed; keep going with the uncompacted history
        self.history = [
            {
                "role": "user",
                "content": f"{SUMMARY_MARKER}\n{summary}\n{SUMMARY_MARKER_END}",
            },
            {
                "role": "assistant",
                "content": "Understood. I will continue the task from this summary.",
            },
            *(keep_tail or []),
        ]
        self.transcript.append(("summary", summary))
        self.host.add_summary(summary)
        self.context_pct = self._estimate_context_pct(summary)
        self.host.set_context_pct(self.context_pct)
        self.host.set_activity("Thinking")
        self._persist()
        return summary

    def _summarize_history(self) -> str | None:
        """Ask the model to condense the history into a resumable brief."""
        instruction = (
            "You are about to run out of context window. Summarize the ENTIRE "
            "conversation above into concise notes that let you CONTINUE the "
            "task with no loss of essential information: the user's goal(s), the "
            "decisions made, facts, commands and file paths discovered, the "
            "current state of the work and the terminal, and the immediate next "
            "steps. Write it as a compact brief to yourself. Omit greetings, "
            "apologies, and filler."
        )
        messages = [
            {
                "role": "system",
                "content": "You compact your own working memory into a "
                "resumable brief so you can keep working after older turns are "
                "dropped.",
            },
            *self._build_context(),
            {"role": "user", "content": instruction},
        ]
        try:
            turn = self.llm.converse(
                messages, tools=None, max_tokens=self.SUMMARY_MAX_TOKENS
            )
        except Exception as exc:  # never crash a turn; just skip compaction
            self.host.add_info(f"[ludvart] context compaction failed: {exc}")
            return None
        self._bank_usage(getattr(turn, "usage", None))
        return (turn.text or "").strip() or None

    def _estimate_context_pct(self, summary: str) -> float | None:
        """Rough post-compaction usage (~4 chars/token + seed overhead).

        The next real turn replaces this with the provider-reported value; this
        just makes the badge reflect the drop immediately.
        """
        cw = getattr(self.llm, "context_window", 0) or 0
        if cw <= 0:
            return None
        approx_tokens = (len(summary) + 400) // 4
        return max(0.0, 100.0 * approx_tokens / cw)

    def _persist(self) -> None:
        """Save the conversation to the backend session store (best effort)."""
        if self.session is None:
            return
        try:
            self.session.save(
                self.transcript,
                self.history,
                provider=getattr(self.llm, "name", None),
                input_tokens=self.total_input_tokens,
                output_tokens=self.total_output_tokens,
            )
        except Exception:  # noqa: BLE001 - persistence must never break a turn
            pass

    def resume(self, transcript, history, input_tokens=0, output_tokens=0) -> None:
        """Replace the running conversation with a loaded session's state.

        Sessions saved before token totals were recorded carry none, so those
        resume from zero rather than reporting a total that is missing its past.
        """
        self.transcript = [tuple(m) for m in transcript]
        self.history = list(history)
        self.total_input_tokens = int(input_tokens or 0)
        self.total_output_tokens = int(output_tokens or 0)
        self.host.set_token_totals(self.total_input_tokens, self.total_output_tokens)

    def reset(self) -> None:
        """Clear the conversation for a fresh session."""
        self.transcript = []
        self.history = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.host.set_token_totals(0, 0)

    def close(self) -> None:
        """Release resources owned by the loop (scratch files, MCP servers)."""
        self.scratch.cleanup()
        if self.mcp is not None:
            self.mcp.close()
            self.mcp = None

    #: Carried in the trailing live block (see :meth:`_live_block`), never in
    #: ``self.history``, so it does not accumulate across turns and is not
    #: written to the persisted session.
    _TOOL_REMINDER = (
        "<reminder>If you are operating on a console/terminal, remember to use "
        "your ludvart helper tools (read, write, append, replace, replace-range, "
        "structured-patch, search, run) for file and command operations rather "
        "than improvising ad-hoc shell commands. The helper is NOT on PATH: "
        "always invoke it by its full path, ~/.ludvart/bin/ludvart_helper. If "
        "it is genuinely not installed there, ask the user to run the "
        "/init_helpers command in the ludvart panel.</reminder>"
    )

    #: Introduces the live screen inside the trailing block, so the model knows
    #: this one is current and the breadcrumbs above it are not.
    _LIVE_SCREEN_INTRO = (
        "This is the terminal as it looks right now. It is the only live "
        "screen in this conversation; the breadcrumbs above mark where older "
        "snapshots were taken."
    )

    @classmethod
    def _live_block(cls, live: str) -> str:
        """Build the single message that may differ between requests.

        Everything that changes request to request -- the current screen and the
        tool reminder -- is concentrated here, at the very end of the prompt.
        Nothing before it is ever rewritten, so a provider's prompt cache keeps
        matching the whole prefix instead of losing it to an edit in the middle.
        """
        parts = []
        if live:
            parts.append(cls._LIVE_SCREEN_INTRO)
            parts.append(live)
        parts.append(cls._TOOL_REMINDER)
        return "\n".join(parts)

    @staticmethod
    def _drop_unanswered_tool_calls(history: list[dict]) -> list[dict]:
        """Return a copy with every unanswered tool call removed.

        :meth:`run_turn` rolls a failed turn back, but a session persisted by an
        older build (or interrupted between the append and the rollback) can
        still carry an assistant turn whose tool calls never got a result.
        Providers reject such a request outright, so a single dropped connection
        would otherwise poison every later turn of the conversation.
        """
        answered = {
            msg.get("tool_call_id")
            for msg in history
            if isinstance(msg, dict) and msg.get("role") == "tool"
        }

        def unanswered(msg) -> list[dict]:
            calls = msg.get("tool_calls") if isinstance(msg, dict) else None
            if not isinstance(calls, list):
                return []
            return [c for c in calls if c.get("id") not in answered]

        out = list(history)
        # A trailing one has nothing left to say once its calls are gone, and
        # leaving it would end the request on an assistant turn.
        while out and unanswered(out[-1]):
            out.pop()
        for i, msg in enumerate(out):
            stale = unanswered(msg)
            if not stale:
                continue
            patched = dict(msg)
            kept = [c for c in msg["tool_calls"] if c not in stale]
            if kept:
                patched["tool_calls"] = kept
            else:
                patched.pop("tool_calls", None)
            out[i] = patched
        return out

    def _build_context(self) -> list[dict]:
        """Render the neutral history into the active provider's message shape.

        Every stored snapshot is collapsed to a breadcrumb (see
        :meth:`_collapse_screenshots`) and the newest one is re-attached as a
        trailing block (see :meth:`_live_block`), so every message before that
        block renders byte-identically on every later request.
        """
        history = self._drop_unanswered_tool_calls(self.history)
        history, live = self._collapse_screenshots(history)
        history.append({"role": "user", "content": self._live_block(live)})
        build = getattr(self.llm, "build_context", None)
        if build is None:
            return history
        return build(history)

    def _run_tool(self, call: ToolCall) -> str:
        """Execute a tool: client tools via the host, backend tools in-process."""
        if call.name in self.client_tools:
            return self.host.run_terminal_tool(call.name, dict(call.input))
        if call.name == "get_past_snapshot":
            return self._tool_get_past_snapshot(call.input)
        if call.name == "b64_encode":
            return builtin.b64_encode(call.input)
        if call.name == "b64_decode":
            return builtin.b64_decode(call.input)
        if call.name == "web_search":
            return builtin.web_search(call.input)
        if call.name == "fetch_url":
            return builtin.fetch_url(call.input, self.scratch)
        if call.name == "read_local_file":
            return builtin.read_local_file(call.input)
        if call.name == "get_local_file_info":
            return builtin.get_local_file_info(call.input)
        if self.mcp is not None and self.mcp.is_mcp_tool(call.name):
            return self.mcp.call_tool(call.name, dict(call.input))
        return f"[ludvart] unknown tool: {call.name}"

    # -- past screen snapshots ------------------------------------------------

    #: Breadcrumb that replaces the screen snapshot of a stored message in the
    #: model-facing context. No snapshot is sent where it sits: the newest is
    #: re-attached as a trailing block and every stored one collapses to this
    #: line to save context (the live screen is re-fetchable via tools, and the
    #: exact past snapshot via get_past_snapshot(timestamp)). The stored log is
    #: untouched. This bare form is only a fallback for a snapshot missing its
    #: timestamp; the normal breadcrumb carries the ts (see
    #: :meth:`_screen_breadcrumb`).
    _SCREEN_PLACEHOLDER = "[screen omitted; superseded by a newer snapshot]"

    #: Matches a ``<screenContext>`` open tag with any (or no) attributes.
    _SCREEN_OPEN_RE = re.compile(r"<screenContext(?:\s[^>]*)?>")
    #: Extracts the ``ts="..."`` timestamp attribute from an open tag.
    _SCREEN_TS_RE = re.compile(r'ts="([^"]*)"')

    @staticmethod
    def _utc_ns_timestamp() -> str:
        """Return the current UTC time as ``YYYY-MM-DDThh:mm:ss.<nanoseconds>``.

        Used to stamp each screen snapshot with a unique, human-readable key.
        The nanosecond field comes from :func:`time.time_ns` so the value is
        precise enough to be unique within a session, while the date/time
        portion stays readable. The same string is later echoed in the
        breadcrumb and accepted by ``get_past_snapshot``.
        """
        ns = time.time_ns()
        secs, frac_ns = divmod(ns, 1_000_000_000)
        base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(secs))
        return f"{base}.{frac_ns:09d}"

    @classmethod
    def _screen_breadcrumb(cls, ts: str | None) -> str:
        """The line that replaces a stripped snapshot, keyed by its timestamp."""
        if not ts:
            return cls._SCREEN_PLACEHOLDER
        return (
            f"[screen from {ts} omitted; "
            f"queryable by get_past_snapshot({ts})]"
        )

    @classmethod
    def _stamp_screenshot(cls, text: str) -> str:
        """Timestamp an unstamped ``<screenContext>`` in a tool result.

        ``inject_input`` returns the settled screen, which is as big as any user
        turn's snapshot. Stamping it lets :meth:`_collapse_screenshots` collapse
        it and still leave the model a breadcrumb it can expand again with
        ``get_past_snapshot``.
        """
        if not isinstance(text, str):
            return text

        def stamp(m: re.Match) -> str:
            if cls._SCREEN_TS_RE.search(m.group(0)):
                return m.group(0)
            return f'<screenContext ts="{cls._utc_ns_timestamp()}">'

        return cls._SCREEN_OPEN_RE.sub(stamp, text, count=1)

    @classmethod
    def _collapse_screenshots(cls, history: list[dict]) -> tuple[list[dict], str]:
        """Collapse every stored snapshot to a breadcrumb; return the newest one.

        Screen snapshots reach the log from two directions: every user turn
        embeds a ``<screenContext ts="...">...</screenContext>`` block, and tools
        that act on the terminal (``inject_input``) report the settled screen the
        same way. Both are thousands of tokens and both go stale the moment the
        screen changes, so none of them is sent where it sits -- each becomes a
        timestamped breadcrumb (see :meth:`_screen_breadcrumb`) the model can
        expand again with ``get_past_snapshot(timestamp)``, and the newest one is
        handed back for :meth:`_build_context` to re-attach at the end.

        Collapsing *every* snapshot, rather than sparing the newest in place, is
        what makes the rendered history append-only: a message reads the same on
        every later request, so a provider's prompt cache keeps matching it. The
        surrounding text (the ``<userRequest>``, a tool result's prose) is left
        untouched, and :attr:`history` and the persisted session keep the full
        snapshots.
        """
        close_tag = "</screenContext>"

        def screen_span(msg: dict):
            """The open-tag match and end offset of ``msg``'s snapshot, if any."""
            if not (
                isinstance(msg, dict) and isinstance(msg.get("content"), str)
            ):
                return None
            m = cls._SCREEN_OPEN_RE.search(msg["content"])
            if m is None:
                return None
            end = msg["content"].find(close_tag)
            if end < 0:
                return None
            return m, end + len(close_tag)

        out: list[dict] = []
        live = ""
        for msg in history:
            span = screen_span(msg)
            if span is None:
                out.append(msg)
                continue
            m, end = span
            content = msg["content"]
            open_tag = m.group(0)
            ts_m = cls._SCREEN_TS_RE.search(open_tag)
            ts = ts_m.group(1) if ts_m else None
            live = content[m.start():end]
            new_msg = dict(msg)
            new_msg["content"] = (
                content[: m.start()]
                + open_tag
                + "\n"
                + cls._screen_breadcrumb(ts)
                + "\n"
                + close_tag
                + content[end:]
            )
            out.append(new_msg)
        return out, live

    def _snapshot_by_timestamp(self, ts: str) -> str | None:
        """Return the snapshot body stored under timestamp ``ts``, or ``None``.

        Scans the *unstripped* neutral log (:attr:`history`) -- not the
        model-facing context, which may have had this snapshot collapsed to a
        breadcrumb -- for the message whose ``<screenContext ts="...">`` open
        tag carries ``ts`` and returns the text between the open and close tags
        (the raw screenshot). Because the log is what gets persisted, a resumed
        session can still answer for snapshots captured before the restart.
        Returns ``None`` if no snapshot has that timestamp.
        """
        close_tag = "</screenContext>"
        for msg in self.history:
            if not (
                isinstance(msg, dict)
                and isinstance(msg.get("content"), str)
            ):
                continue
            content = msg["content"]
            m = self._SCREEN_OPEN_RE.search(content)
            if m is None or close_tag not in content:
                continue
            ts_m = self._SCREEN_TS_RE.search(m.group(0))
            if ts_m is None or ts_m.group(1) != ts:
                continue
            start = m.end()
            end = content.find(close_tag, start)
            if end < 0:
                continue
            return content[start:end].strip("\n")
        return None

    def _tool_get_past_snapshot(self, args: dict) -> str:
        """Return a stored past screen snapshot addressed by its timestamp."""
        ts = args.get("timestamp")
        if not isinstance(ts, str) or not ts.strip():
            return (
                "[ludvart] get_past_snapshot: 'timestamp' must be a non-empty "
                "string. Provide a valid snapshot timestamp exactly as shown in "
                "a breadcrumb."
            )
        ts = ts.strip()
        snapshot = self._snapshot_by_timestamp(ts)
        if snapshot is None:
            return (
                f"[ludvart] get_past_snapshot: no snapshot found for timestamp "
                f"{ts!r}. A valid snapshot timestamp is needed -- pass one "
                "exactly as it appears in a breadcrumb."
            )
        return (
            f"Terminal screen snapshot captured at {ts}:\n"
            f'<screenContext ts="{ts}">\n'
            f"{snapshot}\n"
            "</screenContext>"
        )
