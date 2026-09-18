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
        self.users = []

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

    def add_user(self, text):
        self.users.append(text)


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
        self.closed = False
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
        self.closed = True
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
        self.pending = []

    def reconnect(self, notify, host, *, reason="backend connection lost", pending=None):
        self.calls += 1
        self.pending.append(pending)
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


def test_restart_kills_the_backend_and_reports_the_verified_model():
    spawned = []

    def spawn():
        b = _LoopbackBackend()
        spawned.append(b)
        return b

    reconnector = BackendReconnector(spawn)
    reconnector.connect()
    old = spawned[0]

    host = RecordingHost()
    client = BackendClient(reconnector.channel, reconnector=reconnector)
    result = client.restart(host)

    assert len(spawned) == 2, "restart should spawn a fresh backend"
    assert old.closed, "restart should shut the previous backend down"
    assert "Reconnecting" in host.activities
    assert any("restarting the backend" in i for i in host.infos), host.infos
    assert "verified" in result, result
    # The client talks to the new backend, not the corpse of the old one.
    assert BackendClient(client._channel).ask("go", "SNAP", host).startswith("done (")

    reconnector.close()
    for b in spawned:
        b.close()
    print("restart respawns the backend and reports its model: OK")


def test_restart_reports_a_model_that_failed_verification():
    class _Reconnector:
        label = "gpt-5 (copilot)"
        verified = False
        verify_error = "401 unauthorized"
        needs_setup = False

        def reconnect(self, notify, host, *, reason="backend connection lost"):
            notify(f"{reason}; reconnecting...")
            return object()

    host = RecordingHost()
    result = BackendClient(object(), reconnector=_Reconnector()).restart(host)
    assert "failed verification" in result and "401 unauthorized" in result, result
    print("restart surfaces a failed model verification: OK")


def test_restart_without_a_backend_says_so():
    host = RecordingHost()
    result = BackendClient(object()).restart(host)  # no reconnector
    assert "No backend process" in result, result
    print("restart without a backend explains itself: OK")


def test_the_in_flight_question_survives_the_session_restore():
    with _tmp_sessions():
        # The saved session predates the question that was in flight.
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
        reconnector.session_id = saved.session_id

        host = RecordingHost()
        reconnector.reconnect(
            notify=lambda _m: None, host=host, pending="what I just asked"
        )

        assert host.transcripts, "reconnect should restore + push a transcript"
        assert [m[1] for m in host.transcripts[-1]] == [
            "earlier question",
            "earlier answer",
        ], host.transcripts[-1]
        assert host.users == ["what I just asked"], host.users
        reconnector.close()
    print("the in-flight question survives the session restore: OK")


def test_ask_puts_the_dropped_question_back_on_the_panel():
    good = _LoopbackBackend()
    assert good.channel.recv()["type"] == "hello"
    stub = _StubReconnector(good.channel)

    host = RecordingHost()
    BackendClient(_DeadChannel(), reconnector=stub).ask("do it", "SNAP", host)

    assert stub.pending == ["do it"], stub.pending
    good.close()
    print("ask hands the dropped question to the reconnect: OK")


def main():
    test_ask_retries_after_a_dropped_connection()
    test_no_reconnector_propagates_the_drop()
    test_reconnector_respawns_and_serves_next_turn()
    test_reconnect_restores_the_previous_session()
    test_restart_kills_the_backend_and_reports_the_verified_model()
    test_restart_reports_a_model_that_failed_verification()
    test_restart_without_a_backend_says_so()
    test_the_in_flight_question_survives_the_session_restore()
    test_ask_puts_the_dropped_question_back_on_the_panel()
    print("\nALL backend reconnect tests passed.")


if __name__ == "__main__":
    main()
