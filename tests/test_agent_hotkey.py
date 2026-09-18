"""The key that summons the AI panel is configurable via --agent-hotkey.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_agent_hotkey.py
"""

import argparse
import contextlib
import io

from ludvart.__main__ import _parse_agent_hotkey, main as cli_main
from ludvart.ludvart import AGENT_HOTKEYS, DEFAULT_SUMMON, Ludvart, hotkey_label
from ludvart.panel import AiPanel


def test_the_spec_is_accepted_in_the_forms_people_type():
    for spec in ("ctrl-t", "CTRL-T", "C-t", "^t", " ctrl-t "):
        assert _parse_agent_hotkey(spec) == b"\x14", spec
    assert _parse_agent_hotkey("ctrl-o") == DEFAULT_SUMMON
    assert _parse_agent_hotkey("^g") == b"\x07"
    assert _parse_agent_hotkey("ctrl-]") == b"\x1d"
    print("the spec is accepted in the forms people type: OK")


def test_a_key_the_terminal_needs_is_refused():
    # ^R is reverse-i-search, ^C is SIGINT: taking either would break the shell
    # the panel sits under, so only the vetted list is accepted.
    for spec in ("ctrl-r", "ctrl-c", "ctrl-a", "nonsense", "^"):
        try:
            _parse_agent_hotkey(spec)
        except argparse.ArgumentTypeError as exc:
            assert "choose one of" in str(exc), exc
        else:
            raise AssertionError(f"{spec!r} should have been refused")
    print("a key the terminal needs is refused: OK")


def test_every_offered_key_has_a_label():
    assert hotkey_label(AGENT_HOTKEYS["ctrl-o"]) == "^O"
    assert hotkey_label(AGENT_HOTKEYS["ctrl-t"]) == "^T"
    assert hotkey_label(AGENT_HOTKEYS["ctrl-g"]) == "^G"
    assert hotkey_label(AGENT_HOTKEYS["ctrl-]"]) == "^]"
    print("every offered key has a label: OK")


def test_the_panel_advertises_the_key_that_closes_it():
    panel = AiPanel(cols=80, height=8, provider="test", summon_label="^T")
    header = panel._header(0)
    assert b"^T/Esc:close" in header, header
    assert b"^O/Esc:close" not in header, header
    print("the panel advertises the key that closes it: OK")


def test_the_chosen_key_reaches_the_relay_and_the_panel():
    r = Ludvart(["true"], summon=AGENT_HOTKEYS["ctrl-t"])
    assert r.summon == b"\x14"
    r._phys_rows, r._phys_cols = 24, 80
    r._panel = AiPanel(80, 8, "test", hotkey_label(r.summon))
    assert b"^T/Esc:close" in r._panel._header(0)
    print("the chosen key reaches the relay and the panel: OK")


def test_the_hotkey_may_not_shadow_the_prefix():
    # --prefix defaults to Ctrl-G, so asking for Ctrl-G without moving it would
    # leave one of the two keys unreachable.
    try:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            cli_main(["--agent-hotkey", "ctrl-g", "--no-llm"])
    except SystemExit as exc:
        assert exc.code == 2, exc.code
        assert "cannot be the same key" in err.getvalue(), err.getvalue()
    else:
        raise AssertionError("a hotkey equal to the prefix must be refused")
    print("the hotkey may not shadow the prefix: OK")


def main():
    test_the_spec_is_accepted_in_the_forms_people_type()
    test_a_key_the_terminal_needs_is_refused()
    test_every_offered_key_has_a_label()
    test_the_panel_advertises_the_key_that_closes_it()
    test_the_chosen_key_reaches_the_relay_and_the_panel()
    test_the_hotkey_may_not_shadow_the_prefix()
    print("\nALL agent-hotkey tests passed.")


if __name__ == "__main__":
    main()
