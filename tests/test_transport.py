"""Unit tests for the client<->backend process transports.

These spawn a small echo "backend" (a python -c that speaks the same framed
protocol) to exercise the real subprocess stdio path and the cleanup logic,
without needing the full ludvart backend.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_transport.py
"""

import subprocess
import sys
import time

from ludvart.protocol import MsgType, message
from ludvart.transport import (
    Transport,
    local_backend_argv,
    parse_backend_spec,
    spawn_transport,
    ssh_backend_argv,
)


# A minimal "backend": echo each received frame back until stdin hits EOF.
_ECHO_BACKEND = (
    "import sys\n"
    "from ludvart.protocol import FrameChannel, message\n"
    "ch = FrameChannel(sys.stdin.buffer, sys.stdout.buffer)\n"
    "while True:\n"
    "    m = ch.recv()\n"
    "    if m is None:\n"
    "        break\n"
    "    ch.send(message('echo', got=m))\n"
)

# A backend that ignores stdin and never exits on its own, to test forced stop.
_HANG_BACKEND = (
    "import sys, time\n"
    "sys.stderr.write('up\\n'); sys.stderr.flush()\n"
    "while True:\n"
    "    time.sleep(0.2)\n"
)


def _echo_transport() -> Transport:
    return spawn_transport(
        [sys.executable, "-c", _ECHO_BACKEND], stderr=subprocess.DEVNULL
    )


def test_parse_backend_spec():
    assert parse_backend_spec("host:/opt/ludvart") == ("host", "/opt/ludvart")
    assert parse_backend_spec("user@host:~/ludvart") == ("user@host", "~/ludvart")
    # Folder may contain a colon; only the first colon splits.
    assert parse_backend_spec("h:/a:b") == ("h", "/a:b")
    print("parse_backend_spec splits host/folder: OK")


def test_parse_backend_spec_rejects_bad():
    for bad in ("nofolder", "host:", ":folder", "  :  "):
        try:
            parse_backend_spec(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")
    print("parse_backend_spec rejects malformed specs: OK")


def test_local_backend_argv():
    argv = local_backend_argv()
    assert argv == [sys.executable, "-m", "ludvart", "serve"], argv
    argv2 = local_backend_argv(python="/custom/python")
    assert argv2[0] == "/custom/python", argv2
    print("local_backend_argv targets 'ludvart serve': OK")


def test_ssh_backend_argv():
    argv = ssh_backend_argv("me@box", "/opt/ludvart")
    assert argv[0] == "ssh"
    assert "-T" in argv and "BatchMode=yes" in argv
    assert "me@box" in argv
    remote = argv[-1]
    assert "cd /opt/ludvart" in remote
    assert ".venv/bin/python -m ludvart serve" in remote
    print("ssh_backend_argv builds the remote command: OK")


def test_ssh_backend_argv_forwards_ports():
    argv = ssh_backend_argv("me@box", "/opt/ludvart", forward_ports=[33419, 8080])
    assert "ExitOnForwardFailure=yes" in argv
    assert argv[argv.index("-L") + 1] == "33419:127.0.0.1:33419"
    assert argv.count("-L") == 2
    assert "8080:127.0.0.1:8080" in argv
    assert argv[-2] == "me@box"
    print("ssh_backend_argv adds local port forwards: OK")


def test_ssh_backend_argv_quotes_folder():
    argv = ssh_backend_argv("h", "/weird path/with space")
    remote = argv[-1]
    # The folder must be shell-quoted so spaces do not split the cd argument.
    assert "'/weird path/with space'" in remote, remote
    print("ssh_backend_argv shell-quotes the folder: OK")


def test_ssh_backend_argv_injects_remote_env():
    argv = ssh_backend_argv(
        "h", "/opt/ludvart", remote_env={"LUDVART_BACKEND_FAKE_LLM": "1"}
    )
    remote = argv[-1]
    assert "env LUDVART_BACKEND_FAKE_LLM=1 .venv/bin/python -m ludvart serve" in remote, remote
    # Env values are quoted too.
    argv2 = ssh_backend_argv("h", "/o", remote_env={"K": "a b"})
    assert "env K='a b' " in argv2[-1], argv2[-1]
    print("ssh_backend_argv injects and quotes remote env: OK")


def test_transport_roundtrip_and_cleanup():
    t = _echo_transport()
    try:
        assert t.poll() is None  # running
        t.channel.send(message(MsgType.SUBMIT, text="hi"))
        reply = t.channel.recv()
        assert reply == {
            "type": "echo",
            "got": {"type": "submit", "text": "hi"},
        }, reply
    finally:
        t.close()
    # After close the backend has exited and been reaped.
    assert t.poll() is not None
    print("transport round-trips a frame and reaps the backend: OK")


def test_transport_close_signals_eof():
    # Closing the transport closes the backend's stdin; the echo backend's recv()
    # returns None on EOF and it exits with status 0.
    t = _echo_transport()
    t.channel.send(message("m"))
    assert t.channel.recv()["type"] == "echo"
    t.close()
    assert t.poll() == 0, t.poll()
    print("closing the transport lets the backend exit cleanly (EOF): OK")


def test_transport_force_kills_a_hung_backend():
    # A backend that ignores EOF must still be torn down (terminate/kill) rather
    # than leaking. Shrink the grace windows so the test is fast.
    import ludvart.transport as transport

    saved_stop, saved_term = transport._STOP_GRACE, transport._TERM_GRACE
    transport._STOP_GRACE, transport._TERM_GRACE = 0.3, 0.3
    try:
        t = spawn_transport(
            [sys.executable, "-c", _HANG_BACKEND], stderr=subprocess.DEVNULL
        )
        assert t.poll() is None
        t.close()
        assert t.poll() is not None  # was terminated/killed, not leaked
    finally:
        transport._STOP_GRACE, transport._TERM_GRACE = saved_stop, saved_term
    print("a hung backend is force-stopped on close: OK")


def test_transport_close_is_idempotent():
    t = _echo_transport()
    t.close()
    first = t.poll()
    t.close()  # must not raise or change the outcome
    assert t.poll() == first
    print("transport close is idempotent: OK")


def test_transport_context_manager():
    with _echo_transport() as t:
        t.channel.send(message("m"))
        assert t.channel.recv()["type"] == "echo"
        pid_alive = t.poll() is None
    assert pid_alive
    assert t.poll() is not None  # exited on context exit
    print("transport context manager cleans up on exit: OK")


def main():
    test_parse_backend_spec()
    test_parse_backend_spec_rejects_bad()
    test_local_backend_argv()
    test_ssh_backend_argv()
    test_ssh_backend_argv_forwards_ports()
    test_ssh_backend_argv_quotes_folder()
    test_ssh_backend_argv_injects_remote_env()
    test_transport_roundtrip_and_cleanup()
    test_transport_close_signals_eof()
    test_transport_force_kills_a_hung_backend()
    test_transport_close_is_idempotent()
    test_transport_context_manager()
    print("\nALL transport tests passed.")


if __name__ == "__main__":
    main()
