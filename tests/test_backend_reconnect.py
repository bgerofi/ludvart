"""Backend reconnection: respawn a dropped connection and restore the session.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_backend_reconnect.py
"""

import contextlib
import os
import shutil
import tempfile
import threading

from ludvart.backend_client import BackendClient, BackendReconnector
from ludvart.protocol import FrameChannel
from ludvart.server import _FakeBackendLLM, serve
from ludvart.session import SessionStore
from ludvart.terminal_host import TerminalHost


@contextlib.contextmanager
def _tmp_sessions():
    old = os.environ.get("LUDVART_SESSIONS_DIR")
    root = tempfile.mkdtemp(prefix="ludvart_sess_")
    os.environ["LUDVART_SESSIONS_DIR"] = root
    try:
        yield root
    finally:
        if old is None:
            os.environ.pop("LUDVART_SESSIONS_DIR", None)
        else:
            os.environ["LUDVART_SESSIONS_DIR"] = old
        shutil.rmtree(root, ignore_errors=True)


class RecordingHost(TerminalHost):
    def __init__(self):
        self.tool_calls = []
        self.activities = []
        self.infos = []
        self.transcripts = []

    def snapshot(self):
        return "CLIENT-SCREEN"

    def run_terminal_tool(self, name, args):
        self.tool_calls.append((name, args))
        return f"Injected via {name}"

    def narrate(self, text):
        pass

    def set_activity(self, label):
        self.activities.append(label)

    def add_info(self, text):
        self.infos.append(text)

    def add_system(self, text):
        self.infos.append(text)

    def set_transcript(self, messages):
        self.transcripts.append(messages)


def _pipe_pair():
    a_r, a_w = os.pipe()
    b_r, b_w = os.pipe()
    client = FrameChannel(os.fdopen(a_r, "rb"), os.fdopen(b_w, "wb"))
    backend = FrameChannel(os.fdopen(b_r, "rb"), os.fdopen(a_w, "wb"))
    return client, backend


class _LoopbackBackend:
    """A live in-process backend (serve on a thread) exposed like a transport."""

    def __init__(self, session=None):
        self.channel, self._backend_ch = _pipe_pair()
        self._session = session
        self._thread = threading.Thread(
            target=lambda: serve(
                self._backend_ch, llm=_FakeBackendLLM(), session=session
            ),
            daemon=True,
        )
        self._thread.start()

    def start_keepalive(self):
        # In-process: the backend cannot outlive us, so nothing to ping.
        pass

    def close(self):
        try:
            self.channel.close()
        except Exception:
            pass
        self._thread.join(timeout=2)
        try:
            self._backend_ch.close()
        except Exception:
            pass


class _DeadChannel:
    """Accepts a send, then reports the connection dropped on the next recv."""

    def __init__(self):
        self.sent = []

    def send(self, obj):
        self.sent.append(obj)

    def recv(self):
        return None  # simulate a dropped connection


class _StubReconnector:
    def __init__(self, new_channel):
        self._new = new_channel
        self.calls = 0
        self.session_id = None

    def reconnect(self, notify, host):
        self.calls += 1
        notify("stub reconnecting...")
        return self._new


def test_ask_retries_after_a_dropped_connection():
    # A working backend to reconnect to.
    good = _LoopbackBackend()
    assert good.channel.recv()["type"] == "hello"  # consume HELLO
    stub = _StubReconnector(good.channel)

    client = BackendClient(_DeadChannel(), reconnector=stub)
    host = RecordingHost()
    reply = client.ask("do it", "SNAP", host)

    assert stub.calls == 1, "should reconnect exactly once"
    assert reply.startswith("done ("), reply
    assert host.tool_calls == [
        ("inject_input", {"text": "echo hi", "submit": True})
    ], host.tool_calls
    assert "Reconnecting" in host.activities
    assert any("reconnecting" in i for i in host.infos), host.infos

    good.close()
    print("ask reconnects and retries after a dropped connection: OK")


def test_no_reconnector_propagates_the_drop():
    client = BackendClient(_DeadChannel())  # no reconnector
    host = RecordingHost()
    try:
        client.ask("do it", "SNAP", host)
    except ConnectionError:
        pass
    else:
        raise AssertionError("a drop without a reconnector must raise")
    print("without a reconnector a dropped connection raises: OK")


def test_reconnector_respawns_and_serves_next_turn():
    spawned = []

    def spawn():
        b = _LoopbackBackend()
        spawned.append(b)
        return b

    reconnector = BackendReconnector(spawn)
    hello = reconnector.connect()
    assert hello["type"] == "hello"

    host = RecordingHost()
    notes = []
    channel = reconnector.reconnect(notify=notes.append, host=host)

    assert len(spawned) == 2, "reconnect should spawn a new backend"
    assert any("reconnected to backend" in n for n in notes), notes

    # The freshly reconnected channel serves a turn normally.
    reply = BackendClient(channel).ask("go", "SNAP", host)
    assert reply.startswith("done ("), reply

    reconnector.close()
    for b in spawned:
        b.close()
    print("reconnector respawns the backend and serves the next turn: OK")


def test_reconnect_restores_the_previous_session():
    with _tmp_sessions():
        # A saved session to restore after the drop.
        saved = SessionStore.create_new()
        saved.save(
            [("you", "earlier question"), ("ludvart", "earlier answer")],
            [{"role": "user", "content": "earlier question"}],
            provider="custom",
        )

        def spawn():
            return _LoopbackBackend(session=SessionStore.create_new())

        reconnector = BackendReconnector(spawn)
        reconnector.connect()
        # Pretend the client had switched to the saved session before the drop.
        reconnector.session_id = saved.session_id

        host = RecordingHost()
        reconnector.reconnect(notify=lambda _m: None, host=host)

        # The restore reloaded the saved session and pushed its transcript.
        assert host.transcripts, "reconnect should restore + push a transcript"
        assert host.transcripts[-1] == [
            ["you", "earlier question"],
            ["ludvart", "earlier answer"],
        ], host.transcripts[-1]
        assert reconnector.session_id == saved.session_id
        reconnector.close()
    print("reconnect restores the previous session's transcript: OK")


def main():
    test_ask_retries_after_a_dropped_connection()
    test_no_reconnector_propagates_the_drop()
    test_reconnector_respawns_and_serves_next_turn()
    test_reconnect_restores_the_previous_session()
    print("\nALL backend reconnect tests passed.")


if __name__ == "__main__":
    main()
