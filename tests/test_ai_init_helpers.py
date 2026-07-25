"""Unit test: the /init_helpers panel command deterministically installs the helper.

/init_helpers no longer asks the LLM to generate the helper. Instead the harness
injects a self-contained shell command (built from the embedded golden source)
and reports the parsed result. Also asserts the old first-open
auto-initialization has been removed.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_ai_init_helpers.py
"""

from ludvart.ludvart import Ludvart
from ludvart.panel import AiPanel
from ludvart.helper_src import LUDVART_HELPER_VERSION, helper_install_command


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
    r._current_prompt_prefix = lambda: "$ "
    r._wait_for_injection_to_settle = (
        lambda cmd, prefix: "$ ...\n"
        f"LUDVART_HELPER_INIT status=installed version={LUDVART_HELPER_VERSION} "
        "ok=1 reason=missing\n$ "
    )
    submit(r, "/init_helpers")

    assert asks == [], "the LLM must NOT be involved in /init_helpers anymore"
    assert len(actions) == 1, actions
    # The injected bytes are the golden install command + Enter.
    assert written and written[0].endswith(b"\r")
    assert written[0][:-1].decode() == helper_install_command()
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
    r._current_prompt_prefix = lambda: "$ "
    r._wait_for_injection_to_settle = (
        lambda cmd, prefix: f"LUDVART_HELPER_INIT status=current "
        f"version={LUDVART_HELPER_VERSION} ok=1 reason=match"
    )
    submit(r, "/init_helpers")
    assert asks == []
    assert len(actions) == 1
    assert any("up to date" in t for k, t in r._panel.messages if k == "system")
    print("/init_helpers works without an LLM: OK")


def test_parse_helper_init_cases():
    p = Ludvart._parse_helper_init
    v = LUDVART_HELPER_VERSION
    assert "installed" in p(f"LUDVART_HELPER_INIT status=installed version={v} ok=1 reason=missing")
    assert "up to date" in p(f"LUDVART_HELPER_INIT status=current version={v} ok=1 reason=match")
    assert "reinstalled" in p(f"LUDVART_HELPER_INIT status=installed version={v} ok=1 reason=stale_or_modified")
    assert "FAILED" in p(f"LUDVART_HELPER_INIT status=installed version={v} ok=0 reason=stale_or_modified")
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
    test_parse_helper_init_cases()
    test_autoinit_removed()
    test_tab_completion()
    test_tab_completion_completes_a_subcommand()
    print("all init-helpers tests passed")
