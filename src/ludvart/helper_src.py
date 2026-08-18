"""Golden copy of ``ludvart_helper``, its version, and an integrity checksum.

The canonical helper script lives beside this module as the data file
``assets/ludvart_helper`` so it can be edited and updated directly (and version
managed in git) instead of being generated on the fly by the model. This module
loads that file, derives its version and md5, and builds a deterministic,
self-contained shell command that installs or repairs the helper on whatever
machine is hosting the foreground shell.

That command must NOT depend on the harness's own environment: the foreground
shell may be a remote host reached over ssh, so the install logic runs entirely
in the remote's own ``python3`` and resolves ``~`` via the remote's ``HOME``.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import os
import re

# Path to the golden helper script shipped inside the package.
_ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets", "ludvart_helper")


def _load_source() -> bytes:
    with open(_ASSET_PATH, "rb") as fh:
        return fh.read()


def _parse_version(src: bytes) -> str:
    m = re.search(rb'^VER\s*=\s*"([^"]+)"', src, re.MULTILINE)
    return m.group(1).decode("ascii") if m else "0.0.0"


def _parse_spec(src: bytes) -> str:
    """Pull the helper's own ``SPEC`` text out of the source without running it.

    The helper documents itself so the model's instructions cannot drift from
    the code they describe; the prompt embeds whatever this returns.
    """
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "SPEC":
                value = ast.literal_eval(node.value)
                return value if isinstance(value, str) else ""
    return ""


#: Raw bytes of the golden helper script.
LUDVART_HELPER_SOURCE: bytes = _load_source()

#: Version string declared inside the helper (its ``VER = "..."`` line).
LUDVART_HELPER_VERSION: str = _parse_version(LUDVART_HELPER_SOURCE)

#: The helper's self-description, ready to be dropped into the system prompt.
LUDVART_HELPER_SPEC: str = _parse_spec(LUDVART_HELPER_SOURCE)

#: md5 of the golden source, used to detect a missing/outdated/tampered copy.
LUDVART_HELPER_MD5: str = hashlib.md5(LUDVART_HELPER_SOURCE).hexdigest()

# Trust anchor: the md5 the golden asset is expected to have, pinned here in
# source. If ``assets/ludvart_helper`` is ever changed, this constant must be
# updated to match -- so a silent swap of the asset is caught at import time,
# and the harness only ever installs a helper whose checksum it vouches for.
LUDVART_HELPER_MD5_EXPECTED = "dc93ea7b406bf34822e885524bf69184"

if LUDVART_HELPER_MD5 != LUDVART_HELPER_MD5_EXPECTED:  # pragma: no cover - guard
    raise RuntimeError(
        "ludvart_helper asset checksum mismatch: expected "
        f"{LUDVART_HELPER_MD5_EXPECTED} but assets/ludvart_helper is "
        f"{LUDVART_HELPER_MD5}. Update LUDVART_HELPER_MD5_EXPECTED in helper_src.py "
        "when you intentionally change the helper."
    )


def helper_install_payload_b64() -> str:
    """Return the golden source as a single-line base64 string."""
    return base64.b64encode(LUDVART_HELPER_SOURCE).decode("ascii")


def helper_install_command() -> str:
    """Build a one-line shell command that installs/repairs the helper.

    The command runs the remote ``python3`` (which the helper itself requires)
    to compare the on-disk md5 against the pinned golden md5 *without executing*
    the existing file, and rewrites it from an embedded base64 payload only when
    it is missing, outdated, or modified. It prints a single machine-parseable
    line the harness reads back::

        LUDVART_HELPER_INIT status=<installed|current> version=<v> ok=<0|1> reason=<r>

    Only the remote's own ``python3`` and ``HOME`` are used, so the exact same
    command works whether the foreground shell is local or an ssh session on
    another host.
    """
    payload = helper_install_payload_b64()
    # Note: the runtime ``%s``/``%(`` below are LITERAL parts of the python the
    # remote executes -- this string is assembled by concatenation (no % / format
    # applied here), so they need no escaping.
    py = (
        "import base64,hashlib,os;"
        'p=os.path.expanduser("~/.ludvart/bin/ludvart_helper");'
        'want="' + LUDVART_HELPER_MD5 + '";'
        'ver="' + LUDVART_HELPER_VERSION + '";'
        'src=base64.b64decode("' + payload + '");'
        'cur=(hashlib.md5(open(p,"rb").read()).hexdigest() '
        'if os.path.isfile(p) else "");'
        "_=(cur==want) or (os.makedirs(os.path.dirname(p),exist_ok=True),"
        'open(p,"wb").write(src),os.chmod(p,0o755));'
        'new=hashlib.md5(open(p,"rb").read()).hexdigest();'
        'print("LUDVART_HELPER_INIT status=%s version=%s ok=%s reason=%s"%('
        '"current" if cur==want else "installed",ver,'
        '"1" if new==want else "0",'
        '"match" if cur==want else ("missing" if cur=="" else "stale_or_modified")))'
    )
    # The payload is base64 (no single quotes) and the program uses only double
    # quotes internally, so wrapping the whole thing in single quotes is safe.
    return "python3 -c '" + py + "'"
