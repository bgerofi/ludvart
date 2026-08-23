"""Unit coverage for the extracted system-prompt builder.

``ludvart.prompt.system_prompt`` is what tells the model which tools really
exist. Its only previous coverage was through the client's in-process
``_llm_system_prompt``, which is going away, so the guarantees are pinned here
directly against the module the backend actually calls.
"""

from ludvart.helper_src import LUDVART_HELPER_SPEC, LUDVART_HELPER_VERSION
from ludvart.prompt import LUDVART_HELPERS_DOC, SELF_MD_MAX_CHARS, system_prompt
from ludvart.tools import builtin_tool_specs


def _advertised(prompt: str) -> set[str]:
    """Return the tool names from the prompt's generated bullet list.

    Only the contiguous run of bullets following the tool-calling anchor is
    read. The prompt also embeds the helper doc and the machine's ``self.md``,
    both of which contain unrelated ``- `` bullets, so an unanchored scan would
    pick up prose.
    """
    anchor = "tool/function-calling mechanism):"
    _, sep, rest = prompt.partition(anchor)
    assert sep, "system prompt no longer contains the tool-list anchor"
    names = set()
    for line in rest.splitlines():
        if not line.strip():
            if names:
                break
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and ": " in stripped:
            names.add(stripped[2:].split(": ", 1)[0])
        elif names:
            break
    return names


def test_every_tool_is_advertised_by_name_and_description():
    tools = builtin_tool_specs()
    assert tools, "expected built-in tool specs to be non-empty"
    prompt = system_prompt(tools)
    for spec in tools:
        assert spec.name in prompt, f"tool {spec.name} missing from system prompt"
        assert spec.description in prompt, f"description of {spec.name} missing"


def test_tool_list_reflects_the_tools_passed_in():
    """The advertised bullet list must contain exactly the tools passed in.

    Only the generated ``  - name: description`` lines are inspected: the
    surrounding prose legitimately names a few tools (the base64 helpers) in
    fixed wording, so scanning the whole prompt would be a false positive.
    """
    tools = builtin_tool_specs()
    subset = [t for t in tools if t.name == "inject_input"]
    assert subset, "inject_input should be a built-in tool"
    bullets = _advertised(system_prompt(subset))
    assert bullets == {"inject_input"}, bullets
    assert _advertised(system_prompt(tools)) == {t.name for t in tools}


def test_prompt_requires_helper_over_raw_injected_shell():
    prompt = system_prompt(builtin_tool_specs())
    assert "MUST use ludvart_helper instead of injecting raw shell input" in prompt
    assert "Use raw injected shell input only for interactive terminal work" in prompt


def test_prompt_describes_the_screen_context_and_user_request_blocks():
    prompt = system_prompt(builtin_tool_specs())
    assert "<screenContext>" in prompt
    assert "<userRequest>" in prompt


def test_helpers_doc_is_ascii_and_documents_the_helper():
    LUDVART_HELPERS_DOC.encode("ascii")  # raises if a stray unicode dash returns
    assert "ludvart_helper" in LUDVART_HELPERS_DOC


def test_the_helper_interface_comes_from_the_helper_itself():
    """The prompt must quote the shipped helper's spec, not a hand-kept copy.

    The two drifted before: the prompt still described v0.1.0 and omitted
    replace-range, structured-patch, --dry-run and --expect-count long after the
    helper grew them, so the model kept calling an interface that did not exist.
    """
    prompt = system_prompt(builtin_tool_specs())
    assert LUDVART_HELPER_SPEC in prompt
    assert LUDVART_HELPER_VERSION in LUDVART_HELPERS_DOC
    # One statement of the interface, not two that can disagree.
    assert prompt.count("Subcommands") == 1


def test_prompt_never_invokes_the_helper_by_a_bare_name():
    """The helper is not on PATH, so every example must spell out its path.

    Models kept running a bare ``ludvart_helper``, getting "command not found",
    and concluding the helper was missing -- because the spec named the path
    once but then used the bare name in its examples.
    """
    prompt = system_prompt(builtin_tool_specs())
    assert "NOT on PATH" in prompt
    for line in prompt.splitlines():
        for bad in ("`ludvart_helper ", "'ludvart_helper ", "$ ludvart_helper "):
            assert bad not in line, f"bare helper invocation: {line!r}"


def test_the_prompt_says_a_run_reports_its_real_exit_code():
    """The model believed it had to guess whether an injected command worked.

    The spec did give run an exit code, but the framing section opened with
    "every subcommand except run", which reads as "run is unframed" -- so the
    model stopped looking and fell back on judging output by eye.
    """
    prompt = system_prompt(builtin_tool_specs())
    assert "except run" not in prompt
    assert "run included" in prompt
    run_section = prompt.split("\nrun --b64 CMD\n", 1)[1].split("\ninfo\n", 1)[0]
    assert "exit=" in run_section


def test_the_prompt_says_how_to_get_more_than_one_call_across():
    """The 2 KB cap was stated for inject_input but not for the helper.

    The helper is reached by typing a command line, so the cap applies to its
    base64 arguments too -- but the spec never said so, and never named
    write-then-append as the way past it. The model filled the gap by calling a
    documented limit a discipline problem of its own.
    """
    prompt = system_prompt(builtin_tool_specs())
    assert "Size limit:" in LUDVART_HELPER_SPEC
    assert "2 KB" in LUDVART_HELPER_SPEC
    write_section = prompt.split("\nwrite PATH --b64 DATA", 1)[1]
    write_section = write_section.split("\nreplace PATH", 1)[0]
    assert "append" in write_section, "nothing points from a big write to append"


def test_self_md_limit_is_a_sane_positive_bound():
    assert isinstance(SELF_MD_MAX_CHARS, int)
    assert SELF_MD_MAX_CHARS > 0


def main() -> None:
    test_every_tool_is_advertised_by_name_and_description()
    test_tool_list_reflects_the_tools_passed_in()
    test_prompt_requires_helper_over_raw_injected_shell()
    test_prompt_describes_the_screen_context_and_user_request_blocks()
    test_helpers_doc_is_ascii_and_documents_the_helper()
    test_the_helper_interface_comes_from_the_helper_itself()
    test_prompt_never_invokes_the_helper_by_a_bare_name()
    test_the_prompt_says_a_run_reports_its_real_exit_code()
    test_the_prompt_says_how_to_get_more_than_one_call_across()
    test_self_md_limit_is_a_sane_positive_bound()
    print("prompt: OK")


if __name__ == "__main__":
    main()
