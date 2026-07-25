"""Client-side backend startup handshake: stream progress, read HELLO.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_backend_startup.py
"""

from ludvart.backend_client import read_hello


class _FakeChannel:
    def __init__(self, frames):
        self._frames = list(frames)

    def recv(self):
        return self._frames.pop(0) if self._frames else None


def test_read_hello_streams_logs_then_returns_hello():
    frames = [
        {"type": "log", "text": "verifying copilot:opus (model 'x')..."},
        {"type": "log", "text": "starting the GitHub Copilot gateway (model 'x')..."},
        {"type": "log", "text": "copilot:opus: ok"},
        {"type": "log", "text": "verifying anthropic:claude..."},
        {"type": "log", "text": "anthropic:claude: ok"},
        {
            "type": "hello",
            "active_label": "copilot:opus",
            "verified": True,
            "verify_error": None,
            "session_id": "2026-07-24/00_00_00",
        },
    ]
    logs = []
    hello = read_hello(_FakeChannel(frames), logs.append)
    assert hello["active_label"] == "copilot:opus", hello
    assert hello["session_id"] == "2026-07-24/00_00_00", hello
    assert "starting the GitHub Copilot gateway (model 'x')..." in logs, logs
    assert "verifying anthropic:claude..." in logs, logs
    print("read_hello streams gateway + verification logs, returns hello: OK")


def test_read_hello_reports_verification_failure_in_dict():
    frames = [
        {"type": "log", "text": "copilot:opus: FAILED (boom)"},
        {
            "type": "hello",
            "active_label": "copilot:opus",
            "verified": False,
            "verify_error": "boom",
        },
    ]
    logs = []
    hello = read_hello(_FakeChannel(frames), logs.append)
    assert hello["verified"] is False
    assert hello["verify_error"] == "boom"
    assert "copilot:opus: FAILED (boom)" in logs
    print("read_hello surfaces a verification failure in the HELLO dict: OK")


def test_read_hello_raises_on_early_close():
    try:
        read_hello(_FakeChannel([]), lambda _m: None)
    except ConnectionError:
        pass
    else:
        raise AssertionError("expected ConnectionError when HELLO never arrives")
    print("read_hello raises when the backend closes before HELLO: OK")


def main():
    test_read_hello_streams_logs_then_returns_hello()
    test_read_hello_reports_verification_failure_in_dict()
    test_read_hello_raises_on_early_close()
    print("\nALL backend startup tests passed.")


if __name__ == "__main__":
    main()

