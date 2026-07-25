"""Unit tests for the built-in helper tools: web_search, fetch_url,
read_local_file and get_local_file_info.

These live in :mod:`ludvart.tools` and are called by the backend agent loop,
so they are exercised directly as module functions rather than through a
terminal instance.

The two network tools are driven against a fake ``urllib.request.urlopen`` so
the tests are hermetic (no real DNS / HTTP). The two filesystem tools run
against real temp files created by pytest's ``tmp_path`` fixture.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python -m pytest tests/test_ai_local_tools.py
"""

import os
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart import tools  # noqa: E402


@pytest.fixture
def scratch():
    """A per-test scratch dir, always cleaned up even if the test fails."""
    d = tools.ScratchDir()
    try:
        yield d
    finally:
        d.cleanup()


# -- fake HTTP plumbing ------------------------------------------------------


class _FakeHeaders:
    def __init__(self, charset):
        self._charset = charset

    def get_content_charset(self):
        return self._charset


class _FakeResponse:
    """Minimal stand-in for the object urllib.request.urlopen yields."""

    def __init__(self, body: bytes, charset="utf-8"):
        self._body = body
        self.headers = _FakeHeaders(charset)

    def read(self, amt=-1):
        if amt is None or amt < 0:
            data, self._body = self._body, b""
            return data
        data, self._body = self._body[:amt], self._body[amt:]
        return data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, response=None, exc=None, capture=None):
    def fake_urlopen(req, timeout=None):
        if capture is not None:
            capture["url"] = req.full_url
            capture["headers"] = dict(req.headers)
            capture["timeout"] = timeout
        if exc is not None:
            raise exc
        return response

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def _saved_path(out: str) -> str:
    for line in out.splitlines():
        if line.startswith("PATH: "):
            return line[len("PATH: ") :]
    raise AssertionError(f"no PATH in output:\n{out}")


# -- fetch_url ---------------------------------------------------------------


def test_fetch_url_rejects_non_string(scratch):
    assert "must be a string" in tools.fetch_url({"url": 123}, scratch)


def test_fetch_url_rejects_empty(scratch):
    assert "is empty" in tools.fetch_url({"url": "   "}, scratch)


def test_fetch_url_rejects_non_http_scheme(scratch):
    out = tools.fetch_url({"url": "file:///etc/passwd"}, scratch)
    assert "unsupported URL scheme" in out
    assert "file" in out


def test_fetch_url_writes_temp_file(scratch, monkeypatch):
    capture = {}
    _patch_urlopen(
        monkeypatch, response=_FakeResponse(b"<html>hi</html>"), capture=capture
    )
    out = tools.fetch_url({"url": "https://example.com/x"}, scratch)
    assert capture["url"] == "https://example.com/x"
    # A User-Agent must be sent (some servers 403 without one).
    assert any(k.lower() == "user-agent" for k in capture["headers"])
    # Extract the saved path and verify the content landed on disk.
    path = _saved_path(out)
    assert os.path.isfile(path)
    assert Path(path).read_text() == "<html>hi</html>"


def test_fetch_url_caps_download_size(scratch, monkeypatch):
    monkeypatch.setattr(tools, "FETCH_URL_MAX_BYTES", 100)
    big = b"a" * 5000
    _patch_urlopen(monkeypatch, response=_FakeResponse(big))
    out = tools.fetch_url({"url": "https://example.com/big"}, scratch)
    assert "truncated" in out
    path = _saved_path(out)
    assert os.path.getsize(path) == 100


def test_fetch_url_reports_http_error(scratch, monkeypatch):
    err = urllib.error.HTTPError("https://x", 404, "Not Found", {}, None)
    _patch_urlopen(monkeypatch, exc=err)
    out = tools.fetch_url({"url": "https://example.com/missing"}, scratch)
    assert "status code: 404" in out


def test_fetch_url_reports_generic_error(scratch, monkeypatch):
    _patch_urlopen(monkeypatch, exc=OSError("connection refused"))
    out = tools.fetch_url({"url": "https://example.com/x"}, scratch)
    assert "fetch_url failed" in out
    assert "connection refused" in out


# -- fetch scratch dir lifecycle ---------------------------------------------


def test_fetch_dir_created_lazily(scratch, monkeypatch):
    # No fetch yet -> no directory allocated.
    assert scratch._path is None
    _patch_urlopen(monkeypatch, response=_FakeResponse(b"x"))
    out = tools.fetch_url({"url": "https://example.com/x"}, scratch)
    assert scratch._path is not None and os.path.isdir(scratch._path)
    # The saved file lives inside the private per-run directory.
    assert _saved_path(out).startswith(scratch._path + os.sep)
    # The directory is private (0700), so other users cannot read fetched
    # content or collide with our files.
    assert (os.stat(scratch._path).st_mode & 0o777) == 0o700


def test_cleanup_removes_everything(scratch, monkeypatch):
    _patch_urlopen(monkeypatch, response=_FakeResponse(b"x"))
    out = tools.fetch_url({"url": "https://example.com/x"}, scratch)
    fetch_dir = scratch._path
    path = _saved_path(out)
    assert os.path.isfile(path)
    scratch.cleanup()
    assert not os.path.exists(fetch_dir)
    assert not os.path.exists(path)
    # Idempotent and resets state so a later fetch makes a fresh directory.
    assert scratch._path is None
    scratch.cleanup()  # no error on second call


def test_cleanup_noop_when_never_fetched(scratch):
    assert scratch._path is None
    scratch.cleanup()  # must not raise
    assert scratch._path is None


# -- web_search --------------------------------------------------------------


_DDG_HTML = """
<html><body>
<div class="links_main links_deep result__body">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&amp;rut=abc">Example &amp; Co</a>
  <a class="result__snippet" href="//x">A snippet with &#x27;quotes&#x27; &amp; more</a>
</div>
<div class="links_main links_deep result__body">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ffoo.org%2F">Foo <b>Site</b></a>
</div>
</body></html>
"""


def test_web_search_rejects_empty():
    assert "nothing to search" in tools.web_search({"query": "  "})


def test_web_search_parses_and_unescapes(monkeypatch):
    capture = {}
    _patch_urlopen(
        monkeypatch,
        response=_FakeResponse(_DDG_HTML.encode("utf-8")),
        capture=capture,
    )
    out = tools.web_search({"query": "hello world"})
    assert "q=hello%20world" in capture["url"]
    # First result: entities decoded in both title and snippet, real URL extracted.
    assert "TITLE: Example & Co" in out
    assert "URL: https://example.com/page" in out
    assert "A snippet with 'quotes' & more" in out
    # Second result: tags stripped, missing snippet tolerated.
    assert "TITLE: Foo Site" in out
    assert "URL: https://foo.org/" in out


def test_web_search_no_results(monkeypatch):
    _patch_urlopen(monkeypatch, response=_FakeResponse(b"<html>nothing</html>"))
    assert "No results found" in tools.web_search({"query": "zzz"})


def test_web_search_reports_http_error(monkeypatch):
    err = urllib.error.HTTPError("https://x", 503, "busy", {}, None)
    _patch_urlopen(monkeypatch, exc=err)
    assert "status code: 503" in tools.web_search({"query": "x"})


# -- read_local_file ---------------------------------------------------------


def test_read_local_file_missing(tmp_path):
    out = tools.read_local_file({"path": str(tmp_path / "nope.txt")})
    assert "is not a file" in out


def test_read_local_file_full(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("l1\nl2\nl3\n")
    out = tools.read_local_file({"path": str(f)})
    assert "l1\nl2\nl3" in out


def test_read_local_file_line_range(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("l1\nl2\nl3\nl4\nl5\n")
    out = tools.read_local_file({"path": str(f), "start_line": 2, "end_line": 4})
    assert "l2\nl3\nl4" in out
    assert "l1" not in out.split("--------")[-2]
    assert "l5" not in out


def test_read_local_file_reports_range_header(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("l1\nl2\nl3\nl4\nl5\n")
    out = tools.read_local_file({"path": str(f), "start_line": 2, "end_line": 3})
    assert "lines 2-3" in out


def test_read_local_file_caps_lines_per_call(tmp_path, monkeypatch):
    # A single call returns at most READ_MAX_LINES lines, regardless of the
    # requested (or default) end_line, and reports where to continue.
    monkeypatch.setattr(tools, "READ_MAX_LINES", 2)
    f = tmp_path / "data.txt"
    f.write_text("".join(f"l{i}\n" for i in range(1, 6)))  # l1..l5
    out = tools.read_local_file({"path": str(f)})  # no end_line -> default
    assert "lines 1-2" in out
    assert "l1\nl2" in out
    assert "l3" not in out.split("--------")[-2]
    assert "start_line=3" in out


def test_read_local_file_pages_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "READ_MAX_LINES", 2)
    f = tmp_path / "data.txt"
    f.write_text("".join(f"l{i}\n" for i in range(1, 6)))  # l1..l5

    pages = []
    start = 1
    for _ in range(10):  # safety bound
        out = tools.read_local_file({"path": str(f), "start_line": start})
        pages.append(out)
        marker = "start_line="
        if marker not in out:
            break
        start = int(out.split(marker)[1].split(".")[0])

    combined = "".join(pages)
    for i in range(1, 6):
        assert f"l{i}" in combined
    # Last page reached EOF -> no further continuation prompt.
    assert "start_line=" not in pages[-1]


def test_read_local_file_end_before_start(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("l1\nl2\n")
    out = tools.read_local_file({"path": str(f), "start_line": 5, "end_line": 3})
    assert "'end_line' must be >= 'start_line'" in out


def test_read_local_file_past_eof(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("l1\nl2\n")
    out = tools.read_local_file({"path": str(f), "start_line": 100})
    assert "past the end of the file" in out


def test_read_local_file_bad_start_line(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("x\n")
    out = tools.read_local_file({"path": str(f), "start_line": "2"})
    assert "'start_line' must be an integer" in out


def test_read_local_file_truncates(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("x" * 200_000)
    out = tools.read_local_file({"path": str(f)})
    assert "truncated" in out


# -- get_local_file_info -----------------------------------------------------


def test_get_local_file_info_missing(tmp_path):
    out = tools.get_local_file_info({"path": str(tmp_path / "nope")})
    assert "is not a file" in out


def test_get_local_file_info_reports_size_lines_mtime(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("a\nb\nc\n")
    out = tools.get_local_file_info({"path": str(f)})
    assert "SIZE: 6 bytes" in out
    assert "LINES: 3 lines" in out
    assert "MODIFIED:" in out
