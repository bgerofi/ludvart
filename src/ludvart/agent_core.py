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

import base64
from typing import Sequence

from .llm import LLMClient, ToolCall, ToolSpec, Turn
from .session import SUMMARY_MARKER, SUMMARY_MARKER_END
from .terminal_host import TerminalHost

#: Tools that must run where the terminal is (the client). Everything else is a
#: backend tool executed in-process by :meth:`AgentCore._run_tool`.
DEFAULT_CLIENT_TOOLS = frozenset({"inject_input", "capture_screen_history"})


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
    ) -> None:
        self.llm = llm
        self.host = host
        self.system_prompt = system_prompt
        self.tools = list(tools) if tools else []
        self.client_tools = client_tools
        self.max_tokens = max_tokens
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

    def run_turn(self, question: str, snapshot: str) -> str:
        """Run one user turn to completion and return the assistant's reply.

        Appends the user turn (embedding the ask-time ``snapshot``), then loops:
        call the model, run any requested tools, feed results back, until the
        model returns a plain-text answer.
        """
        self.transcript.append(("you", question))
        user_content = (
            "<screenContext>\n"
            f"{snapshot}\n"
            "</screenContext>\n"
            f"<userRequest>\n{question}\n</userRequest>"
        )
        self.history.append({"role": "user", "content": user_content})
        system = {"role": "system", "content": self.system_prompt}

        while True:
            # Compact before EVERY request, not just once per user turn: one
            # agentic turn can issue many tool round-trips and each re-sends the
            # whole history (snapshots + tool output), so the context grows
            # inside this loop.
            self.maybe_compact()
            self.host.set_activity("Thinking")
            turn = self.llm.converse(
                [system, *self._build_context()],
                tools=self.tools or None,
                max_tokens=self.max_tokens,
                on_text=self.host.narrate,
            )
            self.history.append(neutral_assistant(turn))
            self._report_usage(turn)
            if not turn.tool_calls:
                self.transcript.append(("ludvart", turn.text))
                self._persist()
                return turn.text
            for call in turn.tool_calls:
                self.host.set_activity(f"Calling {call.name}")
                output = self._run_tool(call)
                self.history.append(neutral_tool_result(call, output))

    def _report_usage(self, turn) -> None:
        """Push the prompt's context usage to the host (drives the ``[NN%]`` badge)."""
        usage = getattr(turn, "usage", None)
        if usage is None:
            return
        self.last_input_tokens = usage.input_tokens
        self.context_pct = usage.context_percent()
        self.host.set_context_pct(self.context_pct)

    # -- context compaction --------------------------------------------------

    def maybe_compact(self) -> bool:
        """Compact the conversation into a summary if the window is nearly full.

        Triggered when the last prompt filled at least
        :attr:`CONTEXT_COMPACT_PCT` of the model's context window. A history
        that is already just a fresh summary seed (<= 2 messages) is left alone.
        Returns ``True`` when it actually compacted.
        """
        if len(self.history) <= 2:
            return False
        if self.context_pct is None or self.context_pct < self.CONTEXT_COMPACT_PCT:
            return False
        return self._compact_history() is not None

    def _compact_history(self) -> str | None:
        """Summarize the conversation and reseed the context from that summary.

        The model-facing history is replaced by a two-message seed; the visible
        transcript keeps the whole conversation with a compaction marker, so a
        reloaded session shows where it happened. Returns the summary text, or
        ``None`` when the summary request failed (history left unchanged).
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
            )
        except Exception:  # noqa: BLE001 - persistence must never break a turn
            pass

    def resume(self, transcript, history) -> None:
        """Replace the running conversation with a loaded session's state."""
        self.transcript = [tuple(m) for m in transcript]
        self.history = list(history)

    def reset(self) -> None:
        """Clear the conversation for a fresh session."""
        self.transcript = []
        self.history = []

    #: Appended to the most recent user turn at send time only. It is never
    #: stored in ``self.history``, so it does not accumulate across turns and
    #: is not written to the persisted session.
    _TOOL_REMINDER = (
        "<reminder>If you are operating on a console/terminal, remember to use "
        "your ludvart helper tools (read, write, append, replace, replace-range, "
        "structured-patch, search, run) for file and command operations rather "
        "than improvising ad-hoc shell commands.</reminder>"
    )

    def _with_reminder(self) -> list[dict]:
        """Return a shallow copy of the history with the reminder appended.

        Only the last ``user`` message is touched, and only when its content is
        plain text. The originals in ``self.history`` are left untouched.
        """
        out = list(self.history)
        for i in range(len(out) - 1, -1, -1):
            msg = out[i]
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                patched = dict(msg)
                patched["content"] = content + "\n" + self._TOOL_REMINDER
                out[i] = patched
            break
        return out

    def _build_context(self) -> list[dict]:
        """Render the neutral history into the active provider's message shape."""
        history = self._with_reminder()
        build = getattr(self.llm, "build_context", None)
        if build is None:
            return history
        return build(history)

    def _run_tool(self, call: ToolCall) -> str:
        """Execute a tool: client tools via the host, backend tools in-process."""
        if call.name in self.client_tools:
            return self.host.run_terminal_tool(call.name, dict(call.input))
        if call.name == "b64_encode":
            return self._tool_b64_encode(call.input)
        if call.name == "b64_decode":
            return self._tool_b64_decode(call.input)
        return f"[ludvart] backend tool not available in split mode: {call.name}"

    @staticmethod
    def _tool_b64_encode(args: dict) -> str:
        text = args.get("text")
        if not isinstance(text, str):
            return "[ludvart] b64_encode: 'text' must be a string"
        return base64.b64encode(text.encode("utf-8")).decode("ascii")

    @staticmethod
    def _tool_b64_decode(args: dict) -> str:
        data = args.get("b64")
        if not isinstance(data, str):
            return "[ludvart] b64_decode: 'b64' must be a string"
        try:
            return base64.b64decode(data, validate=True).decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001 - reported to the model
            return f"[ludvart] b64_decode: invalid base64: {exc}"
