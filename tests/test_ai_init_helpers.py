"""Unit test: the /init_helpers panel command deterministically installs the helper.

/init_helpers no longer asks the LLM to generate the helper. Instead the harness
injects a self-contained shell command (built from the embedded golden source)
and reports the parsed result. Also asserts the old first-open
auto-initialization has been removed.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_ai_init_helpers.py
"""

import time

from ludvart.ludvart import Ludvart
from ludvart.panel import AiPanel
from ludvart.helper_src import (
    LUDVART_HELPER_MD5,
    LUDVART_HELPER_SOURCE,
    LUDVART_HELPER_VERSION,
    helper_install_command,
    helper_install_payload_b64,
    helper_probe_command,
)


def make_ludvart(with_llm: bool = True):
    r = Ludvart(["true"])
    r.llm = object() if with_llm else None  # truthy stub; only presence matters
    r._panel = AiPanel(cols=80, height=8, provider="test")
    r._phys_rows, r._phys_cols = 24, 80
    r._render_split = lambda: None
    # Capture deterministic actions instead of spawning a thread; run the worker
    # synchronously with stubbed injection so we can inspect the result.
    actions: list = []

    def fake_start_action(worker, **kw):
        actions.append(kw)
        r._panel.add_system(worker())

    r._start_action = fake_start_action
    # Also capture _start_ask to prove the LLM path is NOT used anymore.
    asks: list = []
    r._start_ask = lambda q, **kw: asks.append((q, kw))
    return r, actions, asks


def submit(r, text: str) -> None:
    for ch in text:
        r._panel_key(ch.encode())
    r._panel_key(b"\r")


def test_init_helpers_is_deterministic():
    r, actions, asks = make_ludvart(with_llm=True)
    # Stub the injection so the "screen" contains a realistic install result.
    written: list = []
    r._write_all = lambda fd, data: written.append(data)
    r.INJECT_CHUNK_PAUSE = 0
    r._helper_is_current = lambda: False
    r._wait_for_helper_init = (
        lambda: "$ ...\n"
        f"LUDVART_HELPER_INIT status=installed version={LUDVART_HELPER_VERSION} "
        "ok=1 reason=missing\n$ "
    )
    submit(r, "/init_helpers")

    assert asks == [], "the LLM must NOT be involved in /init_helpers anymore"
    assert len(actions) == 1, actions
    # The injected bytes are the golden install command + Enter, fed in small
    # bursts so the tty is never outrun.
    assert written and written[-1].endswith(b"\r")
    assert max(len(b) for b in written) <= r.INJECT_CHUNK_BYTES
    sent = b"".join(written).decode()
    assert sent == helper_install_command().replace("\n", "\r") + "\r"
    assert "\n" not in sent, "each command must be submitted with a carriage return"
    # The parsed status is shown as a system line.
    msgs = [t for t in r._panel.messages]
    assert msgs[0][0] == "system" and msgs[0][1] == "> /init_helpers"
    assert any("installed" in t and LUDVART_HELPER_VERSION in t
               for k, t in msgs if k == "system"), msgs
    print("/init_helpers is deterministic (no LLM): OK")


def test_init_helpers_works_without_llm():
    # No provider configured must NOT block a deterministic install.
    r, actions, asks = make_ludvart(with_llm=False)
    r._write_all = lambda fd, data: None
    r.INJECT_CHUNK_PAUSE = 0
    r._helper_is_current = lambda: False
    r._wait_for_helper_init = (
        lambda: f"LUDVART_HELPER_INIT status=current "
        f"version={LUDVART_HELPER_VERSION} ok=1 reason=match"
    )
    submit(r, "/init_helpers")
    assert asks == []
    assert len(actions) == 1
    assert any("up to date" in t for k, t in r._panel.messages if k == "system")
    print("/init_helpers works without an LLM: OK")


def test_an_up_to_date_helper_is_not_re_sent():
    """The checksum must be checked *before* the payload is typed, not after.

    The install can only reach the foreground host as base64 typed at its
    shell, one paced chunk at a time. Deciding on disk -- as the install
    command does on its own -- still pays that whole cost, so a host that
    already has the right helper has to be asked first.
    """
    r, _, _ = make_ludvart(with_llm=True)
    written: list = []
    r._write_all = lambda fd, data: written.append(data)
    r.INJECT_CHUNK_PAUSE = 0
    r.SETTLE_POLL = 0.01
    # The screen carries the command's own echo above the answer, and that echo
    # contains the literal "md5=%s" template.
    r._safe_snapshot = (
        lambda: f"$ {helper_probe_command()}\n"
        f"LUDVART_HELPER_HAVE md5={LUDVART_HELPER_MD5}\n$ "
    )

    submit(r, "/init_helpers")

    sent = b"".join(written).decode()
    assert sent == helper_probe_command() + "\r", sent
    assert helper_install_payload_b64()[:64] not in sent, "the payload was re-sent"
    assert any("up to date" in t for k, t in r._panel.messages if k == "system")
    print("an up-to-date helper is not re-sent: OK")


def test_a_stale_helper_is_still_installed():
    r, _, _ = make_ludvart(with_llm=True)
    written: list = []
    r._write_all = lambda fd, data: written.append(data)
    r.INJECT_CHUNK_PAUSE = 0
    r.SETTLE_POLL = 0.01
    r._safe_snapshot = lambda: "LUDVART_HELPER_HAVE md5=" + "0" * 32
    r._wait_for_helper_init = (
        lambda: f"LUDVART_HELPER_INIT status=installed "
        f"version={LUDVART_HELPER_VERSION} ok=1 reason=stale_or_modified"
    )

    submit(r, "/init_helpers")

    sent = b"".join(written).decode()
    assert sent.startswith(helper_probe_command() + "\r"), sent
    assert helper_install_payload_b64()[:64] in sent, "the payload was not sent"
    assert any("reinstalled" in t for k, t in r._panel.messages if k == "system")
    print("a stale helper is still installed: OK")


def test_an_unanswered_probe_falls_back_to_installing():
    """A host we cannot question must still get the helper.

    Nothing answers the probe when python3 is missing or the foreground is not
    a shell at all; treating that silence as "up to date" would quietly leave
    the machine without a helper.
    """
    r, _, _ = make_ludvart(with_llm=True)
    written: list = []
    r._write_all = lambda fd, data: written.append(data)
    r.INJECT_CHUNK_PAUSE = 0
    r.SETTLE_POLL = 0.01
    r.HELPER_PROBE_MAX_WAIT = 0.2
    r._safe_snapshot = lambda: "$ "
    r._wait_for_helper_init = (
        lambda: f"LUDVART_HELPER_INIT status=installed "
        f"version={LUDVART_HELPER_VERSION} ok=1 reason=missing"
    )

    submit(r, "/init_helpers")

    sent = b"".join(written).decode()
    assert helper_install_payload_b64()[:64] in sent, "a silent probe skipped the install"
    print("an unanswered probe falls back to installing: OK")


def test_the_wait_gives_up_instead_of_hanging():
    """The result must be waited for by name, and the wait must be bounded.

    The install is many short commands, so the shell prompt returns between
    them; the generic settle heuristic would declare victory after the first
    one, and its LLM fallback could stall for as long as it liked.
    """
    r, _, _ = make_ludvart(with_llm=True)
    r.HELPER_INIT_MAX_WAIT = 0.2
    r.SETTLE_POLL = 0.01
    r._safe_snapshot = lambda: "$ mkdir -p ~/.ludvart/bin\n$ "
    started = time.time()
    assert "Could not confirm" in r._parse_helper_init(r._wait_for_helper_init())
    assert time.time() - started < 5, "the wait is not bounded"

    seen = ["$ ", "$ ", f"LUDVART_HELPER_INIT status=installed "
            f"version={LUDVART_HELPER_VERSION} ok=1 reason=missing"]
    r.HELPER_INIT_MAX_WAIT = 5
    r._safe_snapshot = lambda: seen.pop(0) if len(seen) > 1 else seen[0]
    assert "installed" in r._parse_helper_init(r._wait_for_helper_init())
    print("the install wait is by-result and bounded: OK")


def test_parse_helper_init_cases():
    p = Ludvart._parse_helper_init
    v = LUDVART_HELPER_VERSION
    assert "installed" in p(f"LUDVART_HELPER_INIT status=installed version={v} ok=1 reason=missing")
    assert "up to date" in p(f"LUDVART_HELPER_INIT status=current version={v} ok=1 reason=match")
    assert "reinstalled" in p(f"LUDVART_HELPER_INIT status=installed version={v} ok=1 reason=stale_or_modified")
    assert "FAILED" in p(f"LUDVART_HELPER_INIT status=installed version={v} ok=0 reason=stale_or_modified")
    # A corrupted install must say what actually arrived, not just "mismatch".
    corrupt = p(
        f"LUDVART_HELPER_INIT status=installed version={v} ok=0 "
        "reason=stale_or_modified bytes=9000 got=deadbeefdeadbeef"
    )
    assert "9000 bytes" in corrupt and "deadbeef" in corrupt, corrupt
    assert str(len(LUDVART_HELPER_SOURCE)) in corrupt, corrupt
    assert "Could not confirm" in p("nothing relevant here")
    # Echo-safe: the command echo contains 'status=%s' but the real line wins.
    echoed = (
        'print("...status=%s version=%s ok=%s...")\n'
        f"LUDVART_HELPER_INIT status=installed version={v} ok=1 reason=missing"
    )
    assert "installed" in p(echoed)
    print("_parse_helper_init cases: OK")


def test_autoinit_removed():
    r, _, _ = make_ludvart()
    # The first-open auto-initialization machinery must be gone.
    assert not hasattr(r, "_helpers_init_attempted"), "auto-init flag should be gone"
    assert not hasattr(r, "_maybe_init_helpers"), "_maybe_init_helpers should be gone"
    assert not hasattr(r, "_looks_like_shell"), "_looks_like_shell should be gone"
    print("auto-init removed: OK")


def test_tab_completion():
    r, _, _ = make_ludvart()
    for ch in "/init":
        r._panel_key(ch.encode())
    r._panel_key(b"\t")
    assert r._panel.editor.text == "/init_helpers ", repr(r._panel.editor.text)
    print("/init tab completion: OK")


def test_tab_completion_completes_a_subcommand():
    """Completion continues into a command's subcommands, not just its name."""
    r, _, _ = make_ludvart()
    for ch in "/sess":
        r._panel_key(ch.encode())
    r._panel_key(b"\t")
    assert r._panel.editor.text == "/sessions ", repr(r._panel.editor.text)

    for ch in "li":
        r._panel_key(ch.encode())
    r._panel_key(b"\t")
    assert r._panel.editor.text == "/sessions list ", repr(r._panel.editor.text)
    print("/sessions list tab completion: OK")


if __name__ == "__main__":
    test_init_helpers_is_deterministic()
    test_init_helpers_works_without_llm()
    test_an_up_to_date_helper_is_not_re_sent()
    test_a_stale_helper_is_still_installed()
    test_an_unanswered_probe_falls_back_to_installing()
    test_the_wait_gives_up_instead_of_hanging()
    test_parse_helper_init_cases()
    test_autoinit_removed()
    test_tab_completion()
    test_tab_completion_completes_a_subcommand()
    print("all init-helpers tests passed")
