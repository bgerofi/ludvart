"""Behavioural parity between the local and backend (split) agent loops.

``Ludvart._ask_llm`` and ``AgentCore.run_turn`` are two implementations of the
same contract. Until now each was tested only against itself, so features added
to one could silently go missing in the other -- which is exactly how the
context badge, the thinking timer and auto-compaction all ended up broken in
backend mode while every test stayed green.

These tests run the SAME scenario down BOTH paths and assert on the SAME
observable: a real :class:`AiPanel`. The backend leg goes through the whole
chain (AgentCore -> RemoteTerminalHost -> framed protocol -> BackendClient ->
_ClientTerminalHost -> panel), so a gap anywhere in it fails here.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_local_backend_parity.py
"""

import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from ludvart.backend_client import BackendClient  # noqa: E402
from ludvart.llm import ToolCall, Turn, Usage  # noqa: E402
from ludvart.ludvart import Ludvart, _ClientTerminalHost  # noqa: E402
from ludvart.panel import AiPanel  # noqa: E402
from ludvart.protocol import FrameChannel  # noqa: E402
from ludvart.server import serve  # noqa: E402
from ludvart.session import SUMMARY_MARKER  # noqa: E402

SUMMARY_TEXT = "COMPACTED-BRIEF: continue the task."


class ScriptedLLM:
    """Reports a scripted context usage per call, with one tool round-trip.

    Duck-typed to satisfy both loops. A compaction request (recognised by the
    summarize instruction) returns a fixed brief and is counted.
    """

    name = "fake"
    model = "m"
    context_window = 1000

    def __init__(self, pcts):
        self.on_retry = None
        self._pcts = list(pcts)
        self.calls = 0
        self.summarize_calls = 0

    def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
        last = messages[-1].get("content") if messages else ""
        if isinstance(last, str) and "Summarize the ENTIRE" in last:
            self.summarize_calls += 1
            return Turn(
                text=SUMMARY_TEXT,
                assistant_message={"role": "assistant", "content": SUMMARY_TEXT},
                usage=None,
            )
        idx = self.calls
        self.calls += 1
        pct = self._pcts[min(idx, len(self._pcts) - 1)]
        usage = Usage(
            input_tokens=int(self.context_window * pct / 100.0),
            output_tokens=1,
            total_tokens=1,
            context_window=self.context_window,
        )
        if idx < len(self._pcts) - 1:
            return Turn(
                text="working",
                tool_calls=[
                    ToolCall(
                        id=f"t{idx}", name="b64_encode", input={"text": "x" * 50}
                    )
                ],
                assistant_message={"role": "assistant", "content": "working"},
                usage=usage,
            )
        return Turn(
            text="final answer",
            assistant_message={"role": "assistant", "content": "final answer"},
            usage=usage,
        )

    def tool_result_message(self, tool_call_id, content):
        return {"role": "user", "content": content}


def _make_client(root: Path) -> Ludvart:
    """A Ludvart with a real panel, used as the observable in both legs."""
    os.environ["LUDVART_SESSIONS_DIR"] = str(root)
    r = Ludvart(["true"])
    r._panel = AiPanel(cols=80, height=8, provider="fake")
    r._phys_rows, r._phys_cols = 24, 80
    r._render_split = lambda: None
    r.snapshot_text = lambda: "SCREEN-SNAPSHOT"
    return r


def _pipe_pair():
    a_r, a_w = os.pipe()  # backend -> client
    b_r, b_w = os.pipe()  # client -> backend
    client = FrameChannel(os.fdopen(a_r, "rb"), os.fdopen(b_w, "wb"))
    backend = FrameChannel(os.fdopen(b_r, "rb"), os.fdopen(a_w, "wb"))
    return client, backend


def run_local(root: Path, pcts) -> tuple[Ludvart, ScriptedLLM, str]:
    """Drive one ask through the in-process loop; return the client and reply."""
    r = _make_client(root)
    llm = ScriptedLLM(pcts)
    r.llm = llm
    reply = r._ask_llm("do a multi-step task")
    return r, llm, reply


def run_backend(root: Path, pcts) -> tuple[Ludvart, ScriptedLLM, str]:
    """Drive the same ask across the real protocol; return the client and reply."""
    r = _make_client(root)
    llm = ScriptedLLM(pcts)
    client_ch, backend_ch = _pipe_pair()
    t = threading.Thread(target=lambda: serve(backend_ch, llm=llm), daemon=True)
    t.start()
    try:
        assert client_ch.recv()["type"] == "hello"
        backend = BackendClient(client_ch)
        host = _ClientTerminalHost(r)
        reply = backend.ask("do a multi-step task", r.snapshot_text(), host)
    finally:
        client_ch.close()
        t.join(timeout=2)
        backend_ch.close()
    return r, llm, reply


#: The two implementations of the agent-loop contract, run by every test below.
MODES = [("local", run_local), ("backend", run_backend)]


@pytest.mark.parametrize("mode,run", MODES, ids=[m for m, _ in MODES])
def test_parity_context_badge_is_updated(tmp_path: Path, mode, run):
    """A turn's reported usage must reach the panel badge in both modes."""
    r, _llm, reply = run(tmp_path, [42.0])
    assert reply == "final answer", reply
    assert r._panel.context_pct == 42.0, (mode, r._panel.context_pct)
    assert r._panel._prompt_prefix() == "[42%] ", (mode, r._panel._prompt_prefix())


@pytest.mark.parametrize("mode,run", MODES, ids=[m for m, _ in MODES])
def test_parity_compacts_when_over_threshold(tmp_path: Path, mode, run):
    """Crossing the threshold mid-ask must compact and mark it, in both modes."""
    r, llm, reply = run(tmp_path, [30.0, 60.0, 92.0, 40.0])
    assert reply == "final answer", reply
    assert llm.summarize_calls >= 1, (mode, llm.summarize_calls)
    kinds = [k for k, _ in r._panel.messages]
    assert "summary" in kinds, (mode, kinds)


@pytest.mark.parametrize("mode,run", MODES, ids=[m for m, _ in MODES])
def test_parity_no_compaction_under_threshold(tmp_path: Path, mode, run):
    """A comfortable conversation must not be compacted in either mode."""
    r, llm, reply = run(tmp_path, [20.0, 30.0, 25.0])
    assert reply == "final answer", reply
    assert llm.summarize_calls == 0, (mode, llm.summarize_calls)
    kinds = [k for k, _ in r._panel.messages]
    assert "summary" not in kinds, (mode, kinds)


@pytest.mark.parametrize("mode,run", MODES, ids=[m for m, _ in MODES])
def test_parity_activity_labels_reach_the_panel(tmp_path: Path, mode, run):
    """The spinner label must track the loop's phases in both modes.

    The elapsed counter itself is timed on the client, so what matters here is
    that each phase (re)starts a waiting period the client can time.
    """
    r, _llm, _reply = run(tmp_path, [30.0, 40.0])
    assert r._panel.activity, mode
    # A tool step was announced, so the client saw a fresh waiting phase.
    assert r._wait_since is not None or r._panel.activity_elapsed is None, mode


def test_backend_compaction_reseeds_the_model_context(tmp_path: Path):
    """After compacting, the backend's model-facing history is a small seed."""
    from ludvart.agent_core import AgentCore
    from ludvart.terminal_host import TerminalHost

    class _Host(TerminalHost):
        def snapshot(self):
            return "SCREEN"

        def run_terminal_tool(self, name, args):
            return "out"

        def narrate(self, text):
            pass

        def set_activity(self, label):
            pass

        def add_info(self, text):
            pass

    llm = ScriptedLLM([30.0, 92.0, 40.0])
    core = AgentCore(llm, _Host(), system_prompt="SYS")
    core.run_turn("task", "SCREEN")

    assert llm.summarize_calls >= 1, llm.summarize_calls
    # The history was purged and reseeded from the summary, not left unbounded.
    seed = core.history[0]
    assert SUMMARY_MARKER in seed["content"], seed
    assert ("summary", SUMMARY_TEXT) in core.transcript, core.transcript
    print("backend compaction reseeds the model context: OK")


def main():
    for name, run in MODES:
        for test in (
            test_parity_context_badge_is_updated,
            test_parity_compacts_when_over_threshold,
            test_parity_no_compaction_under_threshold,
            test_parity_activity_labels_reach_the_panel,
        ):
            with tempfile.TemporaryDirectory() as d:
                test(Path(d), name, run)
            print(f"{test.__name__} [{name}]: OK")
    with tempfile.TemporaryDirectory() as d:
        test_backend_compaction_reseeds_the_model_context(Path(d))
    print("\nALL local/backend parity tests passed.")


if __name__ == "__main__":
    main()
