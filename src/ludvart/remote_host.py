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

    def __init__(self, channel: FrameChannel, llm_provider=None) -> None:
        self._channel = channel
        self._counter = 0
        #: Resolves the backend's active model, lent to the client for one-shot
        #: calls it cannot make itself (it owns no LLM). Called per request so a
        #: mid-session ``/model use`` is picked up. ``None`` disables them.
        self.llm_provider = llm_provider

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
            if msg_type(msg) == MsgType.BACKEND_REQUEST:
                # A nested call from the client while it serves our request
                # (e.g. the settle detector borrowing our model). Answer it and
                # keep waiting for the response we are actually blocked on.
                self._serve_backend_request(msg)
                continue
            # The client answers requests in order and sends nothing else while a
            # request is outstanding, so anything else here is a protocol error.
            raise ConnectionError(
                f"expected response {call_id!r}, got {msg_type(msg)!r}"
            )

    def _serve_backend_request(self, msg: dict) -> None:
        """Run one client-initiated call on the backend and reply."""
        method = msg.get("method")
        params = msg.get("params") or {}
        result = None
        llm = self.llm_provider() if self.llm_provider is not None else None
        if method == "complete" and llm is not None:
            # A caller may ask for fewer retries than a conversational turn gets
            # (the settle detector does: it is only asking whether to keep
            # waiting, so a failed attempt is an answer, not something to sit
            # through again). Nothing else can be using the client -- we are
            # blocked serving the request this call is nested inside.
            prior = llm.max_retries
            if params.get("max_retries") is not None:
                llm.max_retries = max(0, int(params["max_retries"]))
            try:
                result = llm.complete(
                    list(params.get("messages") or []),
                    max_tokens=int(params.get("max_tokens") or 64),
                )
            except Exception:  # noqa: BLE001 - the caller decides what to do
                result = None
            finally:
                llm.max_retries = prior
        self._channel.send(
            message(
                MsgType.BACKEND_RESPONSE,
                call_id=msg.get("call_id"),
                result=result,
            )
        )

    # -- one-way UI notifications -------------------------------------------

    def narrate(self, text: str) -> None:
        self._notify("interim", text=text)

    def set_activity(self, label: str) -> None:
        self._notify("activity", label=label)

    def set_context_pct(self, pct: float | None) -> None:
        self._notify("context", pct=pct)

    def set_token_totals(self, inp: int, out: int) -> None:
        self._notify("tokens", inp=inp, out=out)

    def add_summary(self, text: str) -> None:
        self._notify("summary", text=text)

    def add_info(self, text: str) -> None:
        self._notify("info", text=text)

    def _notify(self, kind: str, **fields) -> None:
        self._channel.send(message(MsgType.PANEL_UPDATE, kind=kind, **fields))
