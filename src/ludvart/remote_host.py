"""Backend-side proxy that drives a client's terminal over the wire.

When the agent loop runs on the backend, its :class:`~ludvart.agent_core.AgentCore`
talks to a :class:`RemoteTerminalHost` instead of a real terminal. Each
value-returning host call becomes a ``REQUEST`` frame the client answers with a
``RESPONSE``; UI notifications (narration/activity/info) are one-way
``PANEL_UPDATE`` frames.

The matching client-side dispatch lives in
:func:`ludvart.backend_client.handle_backend_message`.
"""

from __future__ import annotations

from .protocol import FrameChannel, MsgType, message, msg_type
from .terminal_host import TerminalHost


class RemoteTerminalHost(TerminalHost):
    """A :class:`TerminalHost` whose calls are served by the attached client.

    Runs on the backend. Value-returning calls block reading the channel until
    the matching ``RESPONSE`` arrives, so the backend turn is driven
    synchronously: the client executes each request (including any approval
    prompt) and replies before the loop proceeds.
    """

    def __init__(self, channel: FrameChannel) -> None:
        self._channel = channel
        self._counter = 0

    # -- value-returning calls (request/response) ---------------------------

    def snapshot(self) -> str:
        result = self._request("snapshot", {})
        return result if isinstance(result, str) else ""

    def run_terminal_tool(self, name: str, args: dict) -> str:
        result = self._request("tool", {"name": name, "args": args})
        return result if isinstance(result, str) else ""

    def _request(self, method: str, params: dict):
        self._counter += 1
        call_id = f"r{self._counter}"
        self._channel.send(
            message(MsgType.REQUEST, call_id=call_id, method=method, params=params)
        )
        while True:
            msg = self._channel.recv()
            if msg is None:
                raise ConnectionError(
                    f"client disconnected awaiting response to {method!r}"
                )
            if msg_type(msg) == MsgType.RESPONSE and msg.get("call_id") == call_id:
                return msg.get("result")
            # The client answers requests in order and sends nothing else while a
            # request is outstanding, so anything else here is a protocol error.
            raise ConnectionError(
                f"expected response {call_id!r}, got {msg_type(msg)!r}"
            )

    # -- one-way UI notifications -------------------------------------------

    def narrate(self, text: str) -> None:
        self._notify("interim", text=text)

    def set_activity(self, label: str) -> None:
        self._notify("activity", label=label)

    def set_context_pct(self, pct: float | None) -> None:
        self._notify("context", pct=pct)

    def add_summary(self, text: str) -> None:
        self._notify("summary", text=text)

    def add_info(self, text: str) -> None:
        self._notify("info", text=text)

    def _notify(self, kind: str, **fields) -> None:
        self._channel.send(message(MsgType.PANEL_UPDATE, kind=kind, **fields))
