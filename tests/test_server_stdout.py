"""Unit test: the backend's protocol stream is unreachable by stray output.

The framed channel lives on the backend's stdout, so one ``print`` from inside a
provider SDK lands mid-frame and desynchronises the client for the rest of the
session (seen as "frame length 2065855609 exceeds limit ..." -- the reader had
drifted onto the ``{"ty`` of a payload). ``claim_stdout`` makes that impossible
rather than merely forbidden.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_server_stdout.py
"""

import io
import os
import subprocess
import sys

from ludvart.protocol import read_frame

CHILD = r"""
import os, sys, ctypes
sys.path.insert(0, {src!r})
from ludvart.server import claim_stdout
from ludvart.protocol import write_frame

writer = claim_stdout()
print("a library said something")
sys.stdout.write("and something else")
sys.stdout.flush()
os.write(1, b"even a raw fd-1 write")
write_frame(writer, {{"type": "hello", "n": 1}})
os.write(1, b"noise between frames")
write_frame(writer, {{"type": "hello", "n": 2}})
"""


def _run_child():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    return subprocess.run(
        [sys.executable, "-c", CHILD.format(src=src)],
        capture_output=True,
    )


def test_stray_output_cannot_corrupt_the_protocol():
    r = _run_child()
    assert r.returncode == 0, r.stderr.decode()
    stream = io.BytesIO(r.stdout)
    assert read_frame(stream) == {"type": "hello", "n": 1}
    assert read_frame(stream) == {"type": "hello", "n": 2}
    assert read_frame(stream) is None, "unexpected trailing bytes on the channel"
    print("stray output cannot corrupt the protocol: OK")


def test_the_stray_output_is_not_lost_just_diverted():
    # It has to end up somewhere a person can read, or debugging gets harder.
    err = _run_child().stderr.decode()
    for expected in ("a library said something", "and something else",
                     "even a raw fd-1 write", "noise between frames"):
        assert expected in err, (expected, err)
    print("stray output is diverted to stderr, not lost: OK")


if __name__ == "__main__":
    test_stray_output_cannot_corrupt_the_protocol()
    test_the_stray_output_is_not_lost_just_diverted()
    print("all server stdout tests passed")
