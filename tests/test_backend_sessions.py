"""Backend-side session persistence and /sessions command round-trips.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_backend_sessions.py
"""

import contextlib
import os
import shutil
import tempfile
import threading

from ludvart.agent_core import AgentCore
from ludvart.backend_client import BackendClient
from ludvart.llm import LLMClient, ProviderConfig, Turn
from ludvart.protocol import FrameChannel
from ludvart.server import _FakeBackendLLM, serve
from ludvart.session import SessionStore, list_sessions, load_session
from ludvart.terminal_host import TerminalHost


@contextlib.contextmanager
def _tmp_sessions():
    """Point the session store at a throwaway directory for the duration."""
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


class _TextLLM(LLMClient):
    def __init__(self):
        super().__init__(ProviderConfig("custom", "x", "k", "m"))

    def converse(self, messages, tools=None, max_tokens=1024, on_text=None):
        if on_text:
            on_text("thinking")
        return Turn(
            text="reply",
            assistant_message={"role": "assistant", "content": "reply"},
            usage=None,
        )


class RecordingHost(TerminalHost):
    def __init__(self):
        self.systems = []
        self.rows = []
        self.transcripts = []
        self.model_label = None

    def snapshot(self):
        return "SCREEN"

    def run_terminal_tool(self, name, args):
        return "x"

    def narrate(self, text):
        pass

    def set_activity(self, label):
        pass

    def add_info(self, text):
        pass

    def add_system(self, text):
        self.systems.append(text)

    def add_system_row(self, text):
        self.rows.append(text)
        self.systems.append(text)

    def set_model(self, label):
        self.model_label = label

    def set_transcript(self, messages):
        self.transcripts.append(messages)


def _pipe_pair():
    a_r, a_w = os.pipe()
    b_r, b_w = os.pipe()
    client = FrameChannel(os.fdopen(a_r, "rb"), os.fdopen(b_w, "wb"))
    backend = FrameChannel(os.fdopen(b_r, "rb"), os.fdopen(a_w, "wb"))
    return client, backend


def _run_command(command_line, session):
    client_ch, backend_ch = _pipe_pair()
    t = threading.Thread(
        target=lambda: serve(backend_ch, llm=_FakeBackendLLM(), session=session),
        daemon=True,
    )
    t.start()
    client = BackendClient(client_ch)
    host = RecordingHost()
    assert client_ch.recv()["type"] == "hello"
    client.command(command_line, host)
    client_ch.close()
    t.join(timeout=2)
    backend_ch.close()
    return host


def test_agent_core_persists_to_backend_session():
    with _tmp_sessions():
        host = RecordingHost()
        session = SessionStore.create_new()
        core = AgentCore(_TextLLM(), host, system_prompt="SYS", session=session)
        core.run_turn("hello there", "SCREEN")

        sessions = list_sessions()
        assert len(sessions) == 1, sessions
        data = load_session(sessions[0]["id"])
        msgs = [tuple(m) for m in data["messages"]]
        assert ("you", "hello there") in msgs, msgs
        assert ("ludvart", "reply") in msgs, msgs
        assert data["llm_history"], data["llm_history"]
    print("AgentCore persists the conversation to the backend session: OK")


def test_sessions_list_over_backend():
    with _tmp_sessions():
        saved = SessionStore.create_new()
        saved.save(
            [("you", "old question"), ("ludvart", "old answer")],
            [{"role": "user", "content": "old question"}],
            provider="custom",
        )
        current = SessionStore.create_new()
        host = _run_command("sessions list", current)
        joined = "\n".join(host.systems)
        assert "old question" in joined, joined
        assert saved.session_id in joined, joined
    print("/sessions list is served by the backend store: OK")


def test_sessions_list_sends_whole_titles_as_clippable_rows():
    """The backend must not pre-truncate: only the panel knows the width.

    A title cut to a fixed 48 columns loses exactly the part that tells two
    similar conversations apart, and leaves the rest of the row blank.
    """
    long_title = (
        "Rework the steering path so a correction interrupts the in-flight "
        "turn instead of queueing behind it"
    )
    with _tmp_sessions():
        saved = SessionStore.create_new()
        saved.title = long_title
        saved.save(
            [("you", "q")], [{"role": "user", "content": "q"}], provider="custom"
        )
        current = SessionStore.create_new()
        host = _run_command("sessions list", current)
        assert len(host.rows) == 1, host.rows
        assert long_title in host.rows[0], host.rows
        assert "..." not in host.rows[0], host.rows
    print("/sessions list sends whole titles as clippable rows: OK")


def test_sessions_new_over_backend():
    with _tmp_sessions():
        current = SessionStore.create_new()
        host = _run_command("sessions new", current)
        assert host.transcripts and host.transcripts[-1] == [], host.transcripts
        assert any("Started new session" in s for s in host.systems), host.systems
    print("/sessions new clears the transcript on the client: OK")


def test_sessions_load_over_backend():
    with _tmp_sessions():
        saved = SessionStore.create_new()
        saved.save(
            [("you", "loaded q"), ("ludvart", "loaded a")],
            [{"role": "user", "content": "loaded q"}],
            provider="custom",
        )
        current = SessionStore.create_new()
        host = _run_command(f"sessions load {saved.session_id}", current)
        assert host.transcripts, "load should push a transcript"
        assert host.transcripts[-1] == [
            ["you", "loaded q"],
            ["ludvart", "loaded a"],
        ], host.transcripts[-1]
        assert any("Loaded session" in s for s in host.systems), host.systems
    print("/sessions load restores and pushes the transcript: OK")


def test_sessions_load_by_index_over_backend():
    with _tmp_sessions():
        saved = SessionStore.create_new()
        saved.save(
            [("you", "indexed q"), ("ludvart", "indexed a")],
            [{"role": "user", "content": "indexed q"}],
            provider="custom",
        )
        current = SessionStore.create_new()
        # Populate the backend's session_list, then load by 1-based index.
        client_ch, backend_ch = _pipe_pair()
        t = threading.Thread(
            target=lambda: serve(
                backend_ch, llm=_FakeBackendLLM(), session=current
            ),
            daemon=True,
        )
        t.start()
        client = BackendClient(client_ch)
        host = RecordingHost()
        assert client_ch.recv()["type"] == "hello"
        client.command("sessions list", host)
        client.command("sessions load 1", host)
        client_ch.close()
        t.join(timeout=2)
        backend_ch.close()
        assert host.transcripts[-1] == [
            ["you", "indexed q"],
            ["ludvart", "indexed a"],
        ], host.transcripts[-1]
    print("/sessions load <n> resolves the index on the backend: OK")


def test_sessions_rename_over_backend():
    from ludvart.session import load_session

    with _tmp_sessions():
        saved = SessionStore.create_new()
        saved.save(
            [("you", "rename me"), ("ludvart", "ok")],
            [{"role": "user", "content": "rename me"}],
            provider="custom",
        )
        sid = saved.session_id
        current = SessionStore.create_new()
        host = _run_command(f'sessions rename {sid} "Renamed title"', current)

        assert any("Renamed" in s for s in host.systems), host.systems
        # The title is persisted on the backend store.
        assert load_session(sid)["title"] == "Renamed title"
    print("/sessions rename sets the title on the backend store: OK")


def test_sessions_list_shows_title_over_backend():
    with _tmp_sessions():
        saved = SessionStore.create_new()
        saved.title = "Nice title"
        saved.save(
            [("you", "the first line preview")],
            [{"role": "user", "content": "x"}],
            provider="custom",
        )
        current = SessionStore.create_new()
        host = _run_command("sessions list", current)
        joined = "\n".join(host.systems)
        assert "Nice title" in joined, joined
        # The title takes precedence over the message preview.
        assert "the first line preview" not in joined, joined
    print("/sessions list shows the title instead of the preview: OK")


def test_sessions_rename_by_index_and_unquoted_title():
    from ludvart.session import load_session

    with _tmp_sessions():
        saved = SessionStore.create_new()
        saved.save(
            [("you", "find me")],
            [{"role": "user", "content": "find me"}],
            provider="custom",
        )
        current = SessionStore.create_new()
        # list populates the backend index, then rename by 1-based index with a
        # multi-word, unquoted title (the case the user hit).
        client_ch, backend_ch = _pipe_pair()
        t = threading.Thread(
            target=lambda: serve(
                backend_ch, llm=_FakeBackendLLM(), session=current
            ),
            daemon=True,
        )
        t.start()
        client = BackendClient(client_ch)
        host = RecordingHost()
        assert client_ch.recv()["type"] == "hello"
        client.command("sessions list", host)
        client.command("sessions rename 1 PythonSV-CDO TCP timeout issue", host)
        client_ch.close()
        t.join(timeout=2)
        backend_ch.close()

        assert (
            load_session(saved.session_id)["title"]
            == "PythonSV-CDO TCP timeout issue"
        )
        assert any("Renamed" in s for s in host.systems), host.systems
    print("/sessions rename <n> with an unquoted multi-word title: OK")


def main():
    test_agent_core_persists_to_backend_session()
    test_sessions_list_over_backend()
    test_sessions_list_sends_whole_titles_as_clippable_rows()
    test_sessions_new_over_backend()
    test_sessions_load_over_backend()
    test_sessions_load_by_index_over_backend()
    test_sessions_rename_over_backend()
    test_sessions_rename_by_index_and_unquoted_title()
    test_sessions_list_shows_title_over_backend()
    print("\nALL backend session tests passed.")


if __name__ == "__main__":
    main()
