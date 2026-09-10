"""Approval gate for LLM-driven inject_input tool calls.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_ai_inject_approval.py
"""

import threading
import time

from ludvart.ludvart import Ludvart
from ludvart.panel import AiPanel


def _make_ludvart():
    runner = Ludvart(["true"])
    runner._panel = AiPanel(cols=120, height=10, provider="test")
    runner._render_split = lambda: None
    writes = []
    runner._write_all = lambda fd, data: writes.append(data)
    runner._current_prompt_prefix = lambda: "$ "
    runner._wait_for_injection_to_settle = lambda injected, prefix: "SCREEN"
    return runner, writes


def _run_inject_tool(runner, args):
    out = {}

    def worker():
        out["result"] = runner._tool_inject_input(args)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t, out


def _wait_pending(runner, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if runner._inject_approval_pending:
            return True
        time.sleep(0.01)
    return False


def test_inject_prompt_yes_executes_tool():
    runner, writes = _make_ludvart()
    t, out = _run_inject_tool(runner, {"text": "ls -la", "submit": True})
    assert _wait_pending(runner), "approval prompt did not appear"
    prompt = runner._panel.confirm_prompt
    assert "WARNING: Approve terminal input: " in prompt
    assert '"ls -la"' in prompt

    runner._panel_input(b"y")
    t.join(timeout=2)

    assert out["result"].startswith("Injected ")
    assert writes, "inject_input should write to the PTY when approved"
    assert runner._inject_approval_pending is False
    assert runner._panel.confirm_prompt == ""
    print("inject_input y approval executes tool: OK")


def test_inject_prompt_no_declines_tool():
    runner, writes = _make_ludvart()
    t, out = _run_inject_tool(runner, {"text": "rm -rf /tmp/nope", "submit": True})
    assert _wait_pending(runner), "approval prompt did not appear"

    runner._panel_input(b"n")
    t.join(timeout=2)

    assert out["result"] == "[ludvart] inject_input declined by user approval gate."
    assert writes == []
    assert runner._inject_approval_pending is False
    assert runner._panel.confirm_prompt == ""
    print("inject_input n approval declines tool: OK")


def test_inject_prompt_a_approves_all_for_session():
    runner, writes = _make_ludvart()

    first, out1 = _run_inject_tool(runner, {"text": "whoami", "submit": True})
    assert _wait_pending(runner), "approval prompt did not appear"
    runner._panel_input(b"a")
    first.join(timeout=2)

    assert out1["result"].startswith("Injected ")
    assert runner._inject_approval_all is True

    # Subsequent inject_input calls in this process should not prompt again.
    out2 = runner._tool_inject_input({"text": "pwd", "submit": True})
    assert out2.startswith("Injected ")
    assert runner._inject_approval_pending is False
    assert len(writes) == 2
    print("inject_input a approval enables session-wide auto-approve: OK")


def test_cancel_unblocks_pending_approval_as_declined():
    runner, writes = _make_ludvart()
    t, out = _run_inject_tool(runner, {"text": "echo hi", "submit": True})
    assert _wait_pending(runner), "approval prompt did not appear"

    runner._cancel_ask()
    t.join(timeout=2)

    assert out["result"] == "[ludvart] inject_input declined by user approval gate."
    assert writes == []
    assert runner._inject_approval_pending is False
    assert runner._panel.confirm_prompt == ""
    print("cancelling ask unblocks pending inject approval safely: OK")


def test_revoke_approval_reprompts_future_injections():
    runner, writes = _make_ludvart()

    # Grant session-wide approval, then revoke it.
    first, out1 = _run_inject_tool(runner, {"text": "whoami", "submit": True})
    assert _wait_pending(runner), "approval prompt did not appear"
    runner._panel_input(b"a")
    first.join(timeout=2)
    assert runner._inject_approval_all is True

    runner._handle_slash_command("/revoke_approval")
    assert runner._inject_approval_all is False
    assert any(
        "revoked" in t for k, t in runner._panel.messages if k == "system"
    )

    # A later inject_input must prompt again rather than auto-approve.
    t, out = _run_inject_tool(runner, {"text": "pwd", "submit": True})
    assert _wait_pending(runner), "revoked approval should prompt again"
    runner._panel_input(b"n")
    t.join(timeout=2)
    assert out["result"] == "[ludvart] inject_input declined by user approval gate."
    print("/revoke_approval makes future injections ask again: OK")


def test_revoke_approval_without_grant_is_noop_message():
    runner, _writes = _make_ludvart()
    assert runner._inject_approval_all is False
    runner._handle_slash_command("/revoke_approval")
    assert runner._inject_approval_all is False
    assert any(
        "nothing to revoke" in t for k, t in runner._panel.messages if k == "system"
    )
    print("/revoke_approval with no grant reports nothing to revoke: OK")


def test_revoke_approval_is_completable():
    from ludvart.session import complete_slash

    assert complete_slash("/revoke") == "/revoke_approval "
    print("/revoke_approval tab-completes: OK")


def main():
    test_inject_prompt_yes_executes_tool()
    test_inject_prompt_no_declines_tool()
    test_inject_prompt_a_approves_all_for_session()
    test_cancel_unblocks_pending_approval_as_declined()
    test_helper_run_preview_decodes_command()
    test_helper_run_preview_keeps_invalid_payload()
    test_run_command_tool_line_is_previewed_decoded()
    test_file_edit_tool_line_is_previewed_decoded()
    test_revoke_approval_reprompts_future_injections()
    test_revoke_approval_without_grant_is_noop_message()
    test_revoke_approval_is_completable()
    print("\nALL inject approval tests passed.")



def test_helper_run_preview_decodes_command():
    runner, _writes = _make_ludvart()
    encoded = "Z2l0IHN0YXR1cyAtLXNob3J0"
    prompt = runner._inject_approval_prompt(
        f"~/.ludvart/bin/ludvart_helper run --b64 {encoded}"
    )

    assert '"git status --short"' in prompt
    assert encoded not in prompt
    print("helper run approval preview decodes command: OK")


def test_helper_run_preview_keeps_invalid_payload():
    runner, _writes = _make_ludvart()
    text = "ludvart_helper run --b64 not-base64!"

    assert runner._inject_approval_preview(text) == text
    print("helper run approval preview keeps invalid payload: OK")


def test_run_command_tool_line_is_previewed_decoded():
    """run_shell_command must not hide the command behind base64."""
    from ludvart.tools import helper_run_line

    runner, _writes = _make_ludvart()
    command = "rm -rf build && make install"

    prompt = runner._inject_approval_prompt(helper_run_line(command))

    assert f'"{command}"' in prompt, prompt
    print("run_command tool line is previewed decoded: OK")


def test_file_edit_tool_line_is_previewed_decoded():
    """An approval gate showing only a base64 blob would be worthless."""
    from ludvart.tools import helper_edit_lines

    runner, _writes = _make_ludvart()
    line = helper_edit_lines(
        "replace_in_file",
        {"path": "app.py", "old": "DEBUG", "new": "PRODUCTION"},
    )[0]

    prompt = runner._inject_approval_prompt(line)

    assert "app.py" in prompt, prompt
    assert "DEBUG" in prompt and "PRODUCTION" in prompt, prompt
    assert "b64" not in prompt, prompt
    print("file edit tool line is previewed decoded: OK")


if __name__ == "__main__":
    main()
