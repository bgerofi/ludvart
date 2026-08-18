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


def test_a_wrong_argument_answers_with_the_right_one():
    """A misspelled flag must come back framed, carrying that subcommand's spec.

    Models do invent plausible flags (a "--cmd" for run). A bare argparse usage
    message is not machine-readable and exits 2, which the helper already uses
    for "old text not found", so the mistake used to dead-end.
    """
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as fh:
        fh.write(LUDVART_HELPER_SOURCE)
        script = fh.name
    try:
        r = subprocess.run(["python3", script, "run", "--cmd", "ls"],
                           capture_output=True, text=True)
    finally:
        os.unlink(script)
    assert r.returncode == 7, (r.returncode, r.stderr)
    lines = r.stdout.splitlines()
    assert lines[0] == "<<<LUDVART:BEGIN op=usage>>>"
    assert "exit=7" in lines[-1] and "subcommand=run" in lines[-1]
    payload = base64.b64decode(lines[1]).decode()
    assert "run --b64 CMD" in payload, payload
    assert "read PATH" not in payload, "dumped the whole spec, not the section"
    print("a wrong argument answers with the right one: OK")


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
    test_command_is_quote_safe()
    test_install_current_and_repair()
    test_a_mangled_transfer_never_overwrites_a_good_helper()
    print("all helper_src tests passed")
