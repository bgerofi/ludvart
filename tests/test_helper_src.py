"""Unit tests for the embedded golden ludvart_helper (helper_src.py).

Covers: the asset loads and matches its pinned checksum; the install command is
well-formed, single-quote-safe, and self-contained; and running it against a
throwaway HOME installs, then reports "current", then repairs a tampered copy.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_helper_src.py
"""

import base64
import hashlib
import os
import re
import shutil
import subprocess
import tempfile

from ludvart.helper_src import (
    LUDVART_HELPER_MD5,
    LUDVART_HELPER_MD5_EXPECTED,
    LUDVART_HELPER_SOURCE,
    LUDVART_HELPER_SPEC,
    LUDVART_HELPER_VERSION,
    helper_install_command,
    helper_install_payload_b64,
)


def test_asset_integrity():
    assert LUDVART_HELPER_MD5 == LUDVART_HELPER_MD5_EXPECTED
    assert LUDVART_HELPER_MD5 == hashlib.md5(LUDVART_HELPER_SOURCE).hexdigest()
    # Version derived from the source matches the source's VER line.
    assert re.search(rb'^VER\s*=\s*"%s"' % LUDVART_HELPER_VERSION.encode(),
                     LUDVART_HELPER_SOURCE, re.MULTILINE)
    assert LUDVART_HELPER_SOURCE.startswith(b"#!")
    print("asset integrity + version: OK")


def test_source_is_py36_compatible():
    """Guarantee the helper avoids APIs/syntax newer than Python 3.6.

    Some hosts still ship Python 3.6, so the helper must not use 3.7+-only
    constructs. This scans for the known offenders and rejects any syntax newer
    than 3.6 via the compiler's feature_version gate.
    """
    import ast

    src = LUDVART_HELPER_SOURCE
    # subprocess.run(..., capture_output=True) is 3.7+; we use Popen instead.
    assert b"capture_output" not in src, "capture_output is Python 3.7+"
    # add_subparsers(required=...) keyword is 3.7+; must be enforced manually.
    assert re.search(rb"add_subparsers\([^)]*required", src) is None, \
        "add_subparsers(required=...) is Python 3.7+"
    # No f-strings (fine in 3.6, but keep it % formatting for older readers).
    assert re.search(rb"(?<![A-Za-z0-9_])[fF][\"']", src) is None, "no f-strings"
    # Reject any syntax newer than 3.6 (walrus, positional-only params, ...).
    ast.parse(src, feature_version=(3, 6))
    print("source is Python 3.6-compatible: OK")


def test_runs_under_old_python_if_available():
    """If a real 3.5/3.6 interpreter is present, run the helper under it."""
    old = None
    for name in ("python3.5", "python3.6"):
        path = shutil.which(name)
        if path:
            old = path
            break
    if old is None:
        print("no old python found; skipped (scan test covers compatibility)")
        return
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as fh:
        fh.write(LUDVART_HELPER_SOURCE)
        script = fh.name
    try:
        r = subprocess.run([old, script, "info"], capture_output=True, text=True)
        assert r.returncode == 0 and "LUDVART:BEGIN op=info" in r.stdout, (old, r.stderr)
    finally:
        os.unlink(script)
    print("runs under %s: OK" % old)


def test_spec_is_the_helpers_own_documentation():
    """The spec must be extractable, complete, and what the helper itself prints.

    It is folded verbatim into the model's system prompt, so a capability the
    helper grew but never documented would silently go unused -- and an example
    on a bare name would send the model back to "command not found".
    """
    assert LUDVART_HELPER_SPEC.strip(), "SPEC missing from the golden asset"
    LUDVART_HELPER_SPEC.encode("ascii")
    assert "~/.ludvart/bin/ludvart_helper" in LUDVART_HELPER_SPEC

    caps = re.search(rb'^CAPS\s*=\s*"([^"]+)"', LUDVART_HELPER_SOURCE, re.MULTILINE)
    assert caps, "helper no longer declares CAPS"
    for cap in caps.group(1).decode().split(","):
        assert re.search(r"^%s\b" % re.escape(cap), LUDVART_HELPER_SPEC, re.M), \
            "subcommand %r is undocumented in SPEC" % cap

    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as fh:
        fh.write(LUDVART_HELPER_SOURCE)
        script = fh.name
    try:
        r = subprocess.run(["python3", script, "spec"], capture_output=True, text=True)
    finally:
        os.unlink(script)
    assert r.returncode == 0, r.stderr
    payload = base64.b64decode(r.stdout.splitlines()[1]).decode()
    assert payload == LUDVART_HELPER_SPEC
    print("spec is extracted, complete, and self-consistent: OK")


def run_helper(*args):
    """Run the golden asset with ``args`` and return the completed process."""
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as fh:
        fh.write(LUDVART_HELPER_SOURCE)
        script = fh.name
    try:
        return subprocess.run(["python3", script] + list(args),
                              capture_output=True, text=True)
    finally:
        os.unlink(script)


def test_a_wrong_argument_answers_with_the_right_one():
    """A misspelled flag must come back framed, carrying that subcommand's spec.

    Models do invent plausible flags (a "--cmd" for run). A bare argparse usage
    message is not machine-readable and exits 2, which the helper already uses
    for "old text not found", so the mistake used to dead-end.
    """
    r = run_helper("run", "--cmd", "ls")
    assert r.returncode == 7, (r.returncode, r.stderr)
    lines = r.stdout.splitlines()
    assert lines[0] == "<<<LUDVART:BEGIN op=usage>>>"
    assert "exit=7" in lines[-1] and "subcommand=run" in lines[-1]
    payload = "\n".join(lines[1:-1])
    assert "run --b64 CMD" in payload, payload
    assert "read PATH" not in payload, "dumped the whole spec, not the section"
    print("a wrong argument answers with the right one: OK")


def test_the_correction_is_readable_where_it_lands():
    """The usage payload is the one that is not base64.

    Every other payload is data, which base64 protects. This one is help, and
    encoding it hides the answer from the reader who just got it wrong -- who
    must then spend a decode step to learn what a single line of text on the
    screen could have told them.
    """
    r = run_helper("run", "--cmd", "ls")
    payload = "\n".join(r.stdout.splitlines()[1:-1])
    assert payload.startswith("error: "), payload
    try:
        base64.b64decode(payload.splitlines()[0], validate=True)
        raise AssertionError("the usage payload is still base64")
    except base64.binascii.Error:
        pass
    # Printing it raw is only safe while it cannot quote the framing itself.
    assert "<<<LUDVART:" not in payload, payload
    print("the usage error reads as plain text: OK")


def test_an_unknown_subcommand_does_not_spill_the_whole_spec():
    """There is no section to name, and the spec quotes the sentinel itself."""
    r = run_helper("grep", "foo")
    assert r.returncode == 7, (r.returncode, r.stderr)
    payload = "\n".join(r.stdout.splitlines()[1:-1])
    assert "<<<LUDVART:" not in payload, payload
    assert "subcommands: read, write" in payload, payload
    assert len(payload.splitlines()) < 8, payload
    print("an unknown subcommand gets a short pointer: OK")


def test_search_takes_its_pattern_the_way_grep_would():
    """Every other payload here arrives through a --flag, so the model reaches
    for one; and it reaches for grep's spelling, -n and all. Both forms name the
    same search unambiguously, so refusing them only cost a round-trip.
    """
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "bashrc")
        with open(target, "w") as fh:
            fh.write("PS1=x\n# interactive\nreturn\n")

        flags = run_helper("search", "--path", target, "--pattern", "PS1", "-n")
        assert flags.returncode == 0, flags.stdout
        assert "matches=1" in flags.stdout.splitlines()[-1], flags.stdout

        positional = run_helper("search", "PS1", "--path", target)
        assert positional.stdout == flags.stdout, (positional.stdout, flags.stdout)

        # A flag that would change the answer must still be refused.
        both = run_helper("search", "PS1", "--pattern", "PS1", "--path", target)
        assert both.returncode == 7, both.stdout
        assert "twice" in both.stdout, both.stdout
        ignore_case = run_helper("search", "ps1", "--path", target, "-i")
        assert ignore_case.returncode == 7, ignore_case.stdout

        missing = run_helper("search", "--path", target)
        assert missing.returncode == 7, missing.stdout
        assert "pattern is required" in missing.stdout, missing.stdout
    print("search accepts the pattern as a flag: OK")


def test_search_survives_its_arguments_arriving_backwards():
    """search is the only subcommand whose first argument is not a path.

    Six of the seven that take one put it first, so the model learns that shape
    from the interface itself and writes "search FILE REGEX". Both readings of
    two bare arguments name the same search whenever exactly one of them exists,
    which is the whole of the ambiguity -- so resolve it instead of refusing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "bashrc")
        with open(target, "w") as fh:
            fh.write("PS1=x\n# interactive\nreturn\n")

        forwards = run_helper("search", "PS1|return", "--path", target)
        assert forwards.returncode == 0, forwards.stdout
        for spelling in (
            ["search", target, "PS1|return"],          # the natural slip
            ["search", "PS1|return", target],          # pattern first, path bare
            ["search", "--path", target, "--pattern", "PS1|return"],
        ):
            r = run_helper(*spelling)
            assert r.stdout == forwards.stdout, (spelling, r.stdout)

        # Nothing to go on: two paths, or two non-paths.
        other = os.path.join(tmp, "sub")
        os.mkdir(other)
        for pair in ((target, other), ("nope1", "nope2")):
            r = run_helper("search", pair[0], pair[1])
            assert r.returncode == 7, (pair, r.stdout)
            assert "cannot tell which" in r.stdout, r.stdout

        # An unquoted pattern with spaces looks like three arguments.
        r = run_helper("search", "def", "main", target)
        assert r.returncode == 7 and "quoted" in r.stdout, r.stdout
    print("search accepts its arguments in either order: OK")


def test_every_subcommand_takes_its_path_the_same_two_ways():
    """--path worked on search alone, so callers used it everywhere and lost.

    A convention that holds for one subcommand out of seven is not a convention,
    it is a trap: "replace --path FILE ..." reads as obviously correct and was
    rejected outright. Accepting both spellings on all of them means there is no
    longer a choice to get wrong.
    """
    payload = base64.b64encode(b"x").decode()
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "f.txt")
        cases = [
            ["read"],
            ["write", "--b64", payload],
            ["append", "--b64", payload],
            ["replace", "--old-b64", payload, "--new-b64", payload],
            ["replace-range", "--start", "1", "--end", "1", "--b64", payload],
            ["structured-patch", "--b64", base64.b64encode(
                b'{"edits":[{"old_b64":"eA==","new_b64":"eA==",'
                b'"expect_count":null}]}').decode()],
        ]
        for case in cases:
            with open(target, "w") as fh:
                fh.write("x\n")
            positional = run_helper(case[0], target, *case[1:])
            with open(target, "w") as fh:
                fh.write("x\n")
            flagged = run_helper(case[0], "--path", target, *case[1:])
            assert flagged.stdout == positional.stdout, (case, flagged.stdout)
            assert flagged.returncode == 0, (case, flagged.stdout)

        # Giving it both ways at once is the one thing that cannot be resolved.
        clash = run_helper("read", target, "--path", os.path.join(tmp, "other"))
        assert clash.returncode == 7 and "twice" in clash.stdout, clash.stdout
        none = run_helper("read")
        assert none.returncode == 7, none.stdout
        assert "a path is required" in none.stdout, none.stdout
    print("every subcommand takes --path as well as the positional: OK")


def test_a_near_miss_option_name_is_understood_not_refused():
    """--start-line means --start and nothing else; refusing it buys nothing.

    Each refusal costs a whole round-trip to say something the call already made
    unambiguous. Only names whose value means exactly the same thing are folded,
    so no alias can change how a value is interpreted.
    """
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "f.txt")
        with open(target, "w") as fh:
            fh.write("aaa\nbbb\nccc\n")

        canonical = run_helper("read", target, "--start", "2", "--end", "3")
        assert canonical.returncode == 0, canonical.stdout
        for spelling in (
            ["read", target, "--start-line", "2", "--end-line", "3"],
            ["read", target, "--start_line", "2", "--end_line", "3"],
            ["read", "--file", target, "--begin", "2", "--stop", "3"],
            ["read", "--path", target, "--START", "2", "--end", "3"],
        ):
            r = run_helper(*spelling)
            assert r.stdout == canonical.stdout, (spelling, r.stdout)

        hits = run_helper("search", "--regex", "bbb", "--in", target)
        assert hits.returncode == 0 and "matches=1" in hits.stdout, hits.stdout

        # An alias table is not a spell-checker: a real unknown still fails.
        unknown = run_helper("read", target, "--lines", "2")
        assert unknown.returncode == 7, unknown.stdout
    print("near-miss option names are understood: OK")


def test_a_b64_option_refuses_anything_that_is_not_base64():
    """b64decode drops non-alphabet characters, so plain text decoded to junk.

    That junk was then written to the file, or searched for and not found -- a
    silent corruption where an error was wanted. The one forgiveness is padding,
    which is a clerical detail rather than evidence the value was never base64.
    """
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "f.txt")
        with open(target, "w") as fh:
            fh.write("hello\n")

        bad = run_helper("write", target, "--b64", "this is not base64!")
        assert bad.returncode == 7, bad.stdout
        assert "not base64" in bad.stdout and "b64_encode" in bad.stdout, bad.stdout
        with open(target) as fh:
            assert fh.read() == "hello\n", "a refused call still wrote the file"

        # The dangerous shape: prose whose alphabet characters happen to number
        # a multiple of four, which the lenient decoder turns into bytes.
        assert len("".join(c for c in "a b c d" if c.isalnum())) % 4 == 0
        prose = run_helper("write", target, "--b64", "a b c d")
        assert prose.returncode == 7, prose.stdout
        assert "not base64" in prose.stdout, prose.stdout
        with open(target) as fh:
            assert fh.read() == "hello\n", "plain text was decoded and written"

        # Padding is the one clerical detail worth forgiving.
        unpadded = base64.b64encode(b"by").decode()
        assert unpadded.endswith("="), unpadded
        ok = run_helper("write", target, "--b64", unpadded.rstrip("="))
        assert ok.returncode == 0, ok.stdout
        with open(target) as fh:
            assert fh.read() == "by"
    print("a -b64 option refuses plain text: OK")


def test_a_missing_file_still_answers_with_a_frame():
    """A traceback carries no END sentinel, so exit= never arrives.

    The caller is told to decide success from exit= alone; an unframed traceback
    leaves it waiting for a line that is never printed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        r = run_helper("read", os.path.join(tmp, "nosuch.txt"))
        assert r.returncode == 8, r.stdout
        last = r.stdout.splitlines()[-1]
        assert last.startswith("<<<LUDVART:END op=read exit=8"), r.stdout
        assert "error=io" in last, last
        assert "Traceback" not in r.stdout, r.stdout
    print("a missing file answers with a frame: OK")


def test_the_spec_names_the_one_subcommand_that_breaks_the_pattern():
    """Six subcommands demonstrate "path first" and one contradicts it.

    Stating the exception forty lines below where the pattern was set loses to
    the pattern: a rule shown six times beats a rule said once. The synopsis
    puts them side by side in a fixed column, so the odd one out is seen rather
    than remembered.
    """
    body = LUDVART_HELPER_SPEC.split("At a glance", 1)[1].split("\n\n", 1)[0]
    columns = {}
    for line in body.splitlines():
        m = re.match(r"^    (\S+)\s+(PATH|PATTERN)\b", line)
        if m:
            columns[m.group(1)] = (m.group(2), line.index(m.group(2)))
    assert columns.get("search", (None,))[0] == "PATTERN", columns
    others = [v for k, v in columns.items() if k != "search"]
    assert len(others) >= 6, columns
    assert {v[0] for v in others} == {"PATH"}, columns
    assert len({v[1] for v in columns.values()}) == 1, \
        "the synopsis only shows the exception if the column lines up"
    print("the synopsis shows the argument-order exception: OK")


def test_command_is_quote_safe():
    cmd = helper_install_command()
    lines = cmd.split("\n")
    # Every line must be a complete command on its own: a line lost in transit
    # can then only shorten the payload (which the md5 gate catches), never
    # strand the shell waiting for a continuation.
    assert lines[0].startswith("mkdir -p ")
    assert lines[-1].startswith("python3 -c '") and lines[-1].endswith("'")
    for line in lines[1:-1]:
        assert line.startswith("printf %s '") and "' >> " in line
    # No line may be big enough to outrun a tty's input buffer.
    assert max(len(line) for line in lines) < 1024, max(len(l) for l in lines)
    inner = lines[-1][len("python3 -c '"):-1]
    assert "'" not in inner, "inner program must not contain a single quote"
    # The staged chunks reassemble into exactly the golden source's base64.
    staged = "".join(line.split("'")[1] for line in lines[1:-1])
    assert staged == helper_install_payload_b64()
    print("command is quote-safe + carries golden payload: OK")


def _run(cmd, home):
    env = dict(os.environ, HOME=home)
    return subprocess.run(["bash", "-c", cmd], env=env,
                          capture_output=True, text=True)


def test_probe_reports_the_installed_checksum():
    """The probe is what lets an up-to-date host skip the 21 KB transfer."""
    from ludvart.helper_src import helper_probe_command

    probe = helper_probe_command()
    assert "\n" not in probe, "the probe must be one short line"
    assert probe.startswith("python3 -c '") and probe.endswith("'")
    assert "'" not in probe[len("python3 -c '"):-1]
    with tempfile.TemporaryDirectory() as home:
        out = _run(probe, home).stdout
        assert "LUDVART_HELPER_HAVE md5=-" in out, out

        _run(helper_install_command(), home)
        out = _run(probe, home).stdout
        assert f"LUDVART_HELPER_HAVE md5={LUDVART_HELPER_MD5}" in out, out

        dest = os.path.join(home, ".ludvart", "bin", "ludvart_helper")
        with open(dest, "ab") as fh:
            fh.write(b"\n# sneaky change\n")
        out = _run(probe, home).stdout
        assert "LUDVART_HELPER_HAVE md5=" in out, out
        assert LUDVART_HELPER_MD5 not in out, out
    print("the probe reports the installed checksum: OK")


def test_install_current_and_repair():
    cmd = helper_install_command()
    with tempfile.TemporaryDirectory() as home:
        dest = os.path.join(home, ".ludvart", "bin", "ludvart_helper")

        # 1. Fresh install.
        out = _run(cmd, home).stdout
        assert "status=installed" in out and "ok=1" in out and "reason=missing" in out, out
        assert os.path.isfile(dest)
        assert os.access(dest, os.X_OK), "helper must be executable"
        assert hashlib.md5(open(dest, "rb").read()).hexdigest() == LUDVART_HELPER_MD5

        # 2. Already current -> no rewrite.
        out = _run(cmd, home).stdout
        assert "status=current" in out and "reason=match" in out, out

        # 3. Tamper, then repair.
        with open(dest, "ab") as fh:
            fh.write(b"\n# sneaky change\n")
        assert hashlib.md5(open(dest, "rb").read()).hexdigest() != LUDVART_HELPER_MD5
        out = _run(cmd, home).stdout
        assert "status=installed" in out and "reason=stale_or_modified" in out, out
        assert hashlib.md5(open(dest, "rb").read()).hexdigest() == LUDVART_HELPER_MD5

        # 4. The repaired helper actually runs (info + a run subcommand).
        info = subprocess.run([dest, "info"], capture_output=True, text=True)
        assert info.returncode == 0 and "LUDVART:BEGIN op=info" in info.stdout, info.stdout
        import base64 as _b64
        payload = _b64.b64encode(b"echo hi").decode()
        run = subprocess.run([dest, "run", "--b64", payload],
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stdout
        assert run.stdout == "hi\n<<<LUDVART:END op=run exit=0>>>\n", run.stdout
    print("install / current / repair round-trip: OK")


def test_a_mangled_transfer_never_overwrites_a_good_helper():
    """Drop a chunk mid-transfer: the install must refuse, not write garbage.

    The payload can only reach the host by being typed at its shell, and a tty
    drops characters silently when it is outrun. base64 decoding is happy to
    swallow that damage, so the md5 has to be checked before the write.
    """
    cmd = helper_install_command()
    with tempfile.TemporaryDirectory() as home:
        dest = os.path.join(home, ".ludvart", "bin", "ludvart_helper")
        _run(cmd, home)
        with open(dest, "ab") as fh:
            fh.write(b"\n# an older build\n")
        stale = open(dest, "rb").read()

        lines = cmd.split("\n")
        del lines[2]  # a chunk never made it to the shell
        out = _run("\n".join(lines), home).stdout
        assert "ok=0" in out, out
        assert open(dest, "rb").read() == stale, "clobbered the helper with garbage"
        arrived = re.search(r"bytes=(\d+) got=(\w+)", out)
        assert arrived, out
        assert int(arrived.group(1)) < len(LUDVART_HELPER_SOURCE)
        assert arrived.group(2) != LUDVART_HELPER_MD5
        stage = os.path.join(home, ".ludvart", "bin", ".ludvart_helper.b64")
        assert not os.path.exists(stage), "staging file left behind"
    print("a mangled transfer never overwrites a good helper: OK")


if __name__ == "__main__":
    test_asset_integrity()
    test_source_is_py36_compatible()
    test_runs_under_old_python_if_available()
    test_spec_is_the_helpers_own_documentation()
    test_a_wrong_argument_answers_with_the_right_one()
    test_the_correction_is_readable_where_it_lands()
    test_an_unknown_subcommand_does_not_spill_the_whole_spec()
    test_search_takes_its_pattern_the_way_grep_would()
    test_every_subcommand_takes_its_path_the_same_two_ways()
    test_a_near_miss_option_name_is_understood_not_refused()
    test_a_b64_option_refuses_anything_that_is_not_base64()
    test_a_missing_file_still_answers_with_a_frame()
    test_search_survives_its_arguments_arriving_backwards()
    test_the_spec_names_the_one_subcommand_that_breaks_the_pattern()
    test_command_is_quote_safe()
    test_install_current_and_repair()
    test_a_mangled_transfer_never_overwrites_a_good_helper()
    print("all helper_src tests passed")
