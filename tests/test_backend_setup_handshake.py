"""A backend with nothing registered must ask the client to run setup.

The backend's stdin/stdout carry the framed protocol, so it can never prompt for
a model itself. Instead of failing to start it serves anyway, advertises
``needs_setup`` in its HELLO, and refuses turns until a model exists -- the
client owns the interactive part, and ``/model add`` both registers and
activates the first model.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_backend_setup_handshake.py
"""

import os
import threading
from pathlib import Path

from ludvart.ludvart import Ludvart
from ludvart.panel import AiPanel
from ludvart.protocol import FrameChannel
from ludvart.server import serve


def _pipe_pair():
    a_r, a_w = os.pipe()  # backend -> client
    b_r, b_w = os.pipe()  # client -> backend
    client = FrameChannel(os.fdopen(a_r, "rb"), os.fdopen(b_w, "wb"))
    backend = FrameChannel(os.fdopen(b_r, "rb"), os.fdopen(a_w, "wb"))
    return client, backend


def _serve_empty_registry(root: Path):
    """Start a real backend whose model registry is empty; return the channel."""
    models_file = root / "models.json"
    models_file.write_text("[]")
    saved = {
        name: os.environ.get(name)
        for name in (
            "LUDVART_MODELS_FILE",
            "LUDVART_SESSIONS_DIR",
            "LUDVART_BACKEND_FAKE_LLM",
        )
    }
    os.environ["LUDVART_MODELS_FILE"] = str(models_file)
    os.environ["LUDVART_SESSIONS_DIR"] = str(root)
    os.environ.pop("LUDVART_BACKEND_FAKE_LLM", None)
    client_ch, backend_ch = _pipe_pair()
    thread = threading.Thread(target=lambda: serve(backend_ch), daemon=True)
    thread.start()
    return client_ch, backend_ch, thread, saved


def _shutdown(client_ch, backend_ch, thread, saved):
    client_ch.close()
    thread.join(timeout=2)
    backend_ch.close()
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_hello_requests_setup_when_no_model_is_registered(tmp_path: Path):
    client_ch, backend_ch, thread, saved = _serve_empty_registry(tmp_path)
    try:
        hello = client_ch.recv()
    finally:
        _shutdown(client_ch, backend_ch, thread, saved)
    assert hello["type"] == "hello", hello
    assert hello["needs_setup"] is True, hello
    assert hello["verified"] is False, hello
    assert not hello["active_label"], hello
    print("empty registry -> HELLO asks the client to run setup: OK")


def test_a_turn_is_refused_until_a_model_is_registered(tmp_path: Path):
    client_ch, backend_ch, thread, saved = _serve_empty_registry(tmp_path)
    try:
        assert client_ch.recv()["type"] == "hello"
        client_ch.send({"type": "submit", "text": "hi", "snapshot": "SNAP"})
        reply = client_ch.recv()
    finally:
        _shutdown(client_ch, backend_ch, thread, saved)
    assert reply["type"] == "reply", reply
    assert "no model is registered" in reply["text"], reply
    assert "/model add" in reply["text"], reply
    print("a turn before setup is refused with an actionable message: OK")


def test_startup_opens_the_panel_when_setup_is_needed():
    """Setup must start on its own, not wait for the user to find the summon key."""
    r = Ludvart(["true"], backend_needs_setup=True)
    opened = []

    class _Stop(Exception):
        pass

    def _open():
        opened.append(True)
        raise _Stop  # stop before the select loop; we only care that it opened

    r._open_panel = _open
    r._master_fd, r._stdin_fd = -1, -1
    try:
        r._loop()
    except _Stop:
        pass
    assert opened, "the panel was never opened at startup"
    print("an empty registry opens the panel at startup: OK")


def test_guided_registration_starts_once_and_explains_why():
    r = Ludvart(["true"], backend_needs_setup=True)
    r._panel = AiPanel(cols=80, height=8, provider="backend")
    r._render_split = lambda: None
    r._backend_client = object()  # non-None: the registration goes to the backend

    r._maybe_start_backend_setup()
    assert r._model_add is not None and r._model_add["step"] == "service", r._model_add
    text = " ".join(m[1] for m in r._panel.messages if len(m) > 1)
    assert "No model is registered on the backend yet." in text, text

    # Re-opening the panel later must not restart the flow.
    r._model_add = None
    r._maybe_start_backend_setup()
    assert r._model_add is None, "registration restarted on a later panel open"
    print("registration starts once, with an explanation: OK")


def main():
    import tempfile

    for fn in (
        test_hello_requests_setup_when_no_model_is_registered,
        test_a_turn_is_refused_until_a_model_is_registered,
    ):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    test_startup_opens_the_panel_when_setup_is_needed()
    test_guided_registration_starts_once_and_explains_why()
    print("\nALL OK")


if __name__ == "__main__":
    main()
