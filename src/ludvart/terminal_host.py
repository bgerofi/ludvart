"""The terminal-side capabilities the agent loop depends on.

:class:`TerminalHost` is the seam between the agent loop (the "backend": LLM
calls, session state, the converse/tool loop) and the terminal (the "client":
PTY, screen model, rendering, the injection-approval gate). The backend only
ever touches the terminal through this interface, so the same
:class:`~ludvart.agent_core.AgentCore` runs whether the host is:

* in-process (the client object implements these methods directly), or
* remote (a :class:`~ludvart.remote_host.RemoteTerminalHost` proxy that turns
  each call into a protocol frame to the attached client).

Value-returning methods (:meth:`snapshot`, :meth:`run_terminal_tool`) are
request/response; the rest are one-way UI notifications.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TerminalHost(ABC):
    """Terminal capabilities the agent loop needs from the client side."""

    @abstractmethod
    def snapshot(self) -> str:
        """Return the current terminal screen as text."""

    @abstractmethod
    def run_terminal_tool(self, name: str, args: dict) -> str:
        """Execute a client-side tool (e.g. ``inject_input``) and return its text.

        The host owns any user interaction the tool requires -- notably the
        injection-approval gate -- so approval decisions never cross the wire;
        only the tool's final result string comes back.
        """

    @abstractmethod
    def narrate(self, text: str) -> None:
        """Show the model's live, transient narration for the current turn."""

    @abstractmethod
    def set_activity(self, label: str) -> None:
        """Update the spinner/activity label (e.g. ``"Calling inject_input"``)."""

    @abstractmethod
    def add_info(self, text: str) -> None:
        """Add a dim, non-fatal info/diagnostic line to the transcript."""

    # -- optional capabilities (concrete no-op defaults so existing hosts and
    #    test doubles keep working without implementing them) -----------------

    def add_system(self, text: str) -> None:
        """Add an ephemeral system line (e.g. command output). Defaults to info."""
        self.add_info(text)

    def set_model(self, label: str) -> None:
        """Update the displayed active-model label (e.g. after ``/model use``)."""
        # Default: nothing to display.

    def set_context_pct(self, pct: float | None) -> None:
        """Update the context-usage badge (percent of the window the prompt used).

        ``None`` hides the badge (window size unknown).
        """
        # Default: nothing to display.

    def set_transcript(self, messages: list) -> None:
        """Replace the visible transcript (e.g. after ``/sessions load|new``).

        ``messages`` is a list of ``[kind, text]`` pairs.
        """
        # Default: nothing to render.
