"""ludvart's built-in tools: what the agent advertises, and how they run.

The tool set is split by where a call has to execute:

* **Client tools** (:data:`CLIENT_TOOL_NAMES`) touch the terminal, so they run
  in the process that owns the PTY and are reached through a
  :class:`~ludvart.terminal_host.TerminalHost`.
* **Backend tools** -- everything else here -- are pure functions of their
  arguments plus the filesystem/network of the host running the agent loop, so
  they execute in-process wherever that loop lives.

The specs are shared by both sides: the backend advertises them to the model,
and the client renders their names in tool-call notes.
"""

from __future__ import annotations

import base64
import os

from .llm import ToolSpec

#: Tools that must run where the terminal is. Everything else in this module is
#: executed by the agent loop itself.
CLIENT_TOOL_NAMES = frozenset({"inject_input", "capture_screen_history"})

#: Cap on how much fetch_url writes to /tmp, so a hostile or accidental huge
#: response cannot fill the disk on the host running ludvart.
FETCH_URL_MAX_BYTES = 10 * 1024 * 1024

#: Maximum number of lines a single read_local_file call returns. Larger files
#: are paged through with repeated calls (like an editor's reader).
READ_MAX_LINES = 2000

#: Secondary cap on raw characters per read (e.g. pathologically long lines).
READ_MAX_CHARS = 150_000


def builtin_tool_specs() -> list[ToolSpec]:
    """ludvart's own, always-available tools."""
    return [
        ToolSpec(
            name="inject_input",
            description=(
                "Type characters into the user's terminal, exactly as if the "
                "user pressed the keys on their keyboard. The characters go to "
                "whatever program is currently in the foreground. Use it to "
                "(1) run a shell command on the user's behalf -- e.g. list or "
                "display files with 'ls' / 'cat', check status, install "
                "packages, etc. (set submit=true to press Enter and execute); "
                "or (2) send keystrokes (including control characters) to an "
                "interactive program such as vim, less, a REPL or a TUI. This "
                "is the way to actually DO things in the terminal; prefer it "
                "over merely telling the user what to type. "
                "IMPORTANT: keep each call's 'text' small -- at most about "
                "2 KB. Larger payloads (e.g. a long base64 blob or a big file "
                "body) often fail to be generated and arrive EMPTY, which "
                "wastes a call; split long content into several sequential "
                "inject_input calls of <=2 KB each (submit=false on the "
                "intermediate parts, then submit=true -- or a trailing "
                "newline -- on the final one to execute)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "The characters to type. For a shell command "
                            "this is the command line, e.g. 'ls -la'. "
                            "Backslash escapes are interpreted (unless "
                            "interpret_escapes=false) so you CAN send "
                            "control keys: use \\xHH for a raw byte (e.g. "
                            "\\x06 = Ctrl-F, \\x1b = Esc), \\cX for a control "
                            "key (e.g. \\cf = Ctrl-F), plus \\e (Esc), \\t "
                            "(Tab), \\r (Enter), \\n (newline). Write \\\\ for "
                            "a literal backslash. Raw control BYTES do not "
                            "survive here -- always express control keys with "
                            "these escapes. A trailing newline (or "
                            "submit=true) is needed to run a shell command. "
                            "Keep this under ~2 KB per call; for longer "
                            "content, split it across several sequential "
                            "inject_input calls (each <=2 KB) instead of one "
                            "large one, which may otherwise arrive empty."
                        ),
                    },
                    "submit": {
                        "type": "boolean",
                        "description": (
                            "If true, press Enter (send a carriage return) "
                            "after the text to execute it. Defaults to false."
                        ),
                    },
                    "interpret_escapes": {
                        "type": "boolean",
                        "description": (
                            "Whether to decode backslash escapes in 'text' "
                            "(\\xHH, \\cX, \\e, \\t, \\r, \\n, \\\\). Defaults "
                            "to true. Set false to send 'text' verbatim, "
                            "e.g. when typing literal backslashes."
                        ),
                    },
                },
                "required": ["text"],
            },
        ),
        ToolSpec(
            name="capture_screen_history",
            description=(
                "Read lines from the terminal's scrollback history -- output "
                "that has scrolled above the currently visible screen. Use "
                "this when a command's output (for example the result of an "
                "inject_input call) is longer than what fits on the visible "
                "screen and you need to see the earlier lines. The history is "
                "the full logical output: everything that scrolled off the "
                "top, followed by the current viewport. 'offset' is a number "
                "of lines measured from the current position (the latest "
                "line) and must be NEGATIVE to look upward -- e.g. "
                "offset=-100 starts 100 lines above the current position. "
                "'length' is how many lines to return starting at that "
                "offset. If the range extends past the top it is clamped, and "
                "the result reports how many lines exist in total so you can "
                "adjust the offset and try again."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "offset": {
                        "type": "integer",
                        "description": (
                            "Lines from the current position to start at. "
                            "Negative goes back into history, e.g. -100 = "
                            "100 lines above the current position."
                        ),
                    },
                    "length": {
                        "type": "integer",
                        "description": (
                            "How many lines to return, starting at 'offset'."
                        ),
                    },
                },
                "required": ["offset", "length"],
            },
        ),
        ToolSpec(
            name="get_past_snapshot",
            description=(
                "Return the exact terminal screen snapshot that was captured "
                "at a previous turn, identified by its UTC timestamp. Each "
                "user turn embeds a <screenContext ts=\"...\"> snapshot; once "
                "superseded, older snapshots are removed from your context "
                "and replaced by a breadcrumb line that keeps the timestamp "
                "(e.g. '[screen from <TS> omitted; queryable by "
                "get_past_snapshot(<TS>)]'). Call this with that <TS> to get "
                "the full snapshot back. Unlike capture_screen_history (which "
                "reads flattened scrollback), this returns a consistent, "
                "point-in-time rectangular screenshot -- useful for full-screen "
                "TUI applications whose past state cannot be reconstructed "
                "from scrollback. Pass the timestamp exactly as shown in the "
                "breadcrumb."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "description": (
                            "The UTC timestamp of the desired snapshot, "
                            "exactly as it appears in the breadcrumb, e.g. "
                            "'2026-07-06T17:28:54.123456789'."
                        ),
                    },
                },
                "required": ["timestamp"],
            },
        ),
        ToolSpec(
            name="b64_encode",
            description=(
                "Encode UTF-8 text to base64 natively (no shell, no "
                "terminal round-trip). Use this to build the base64 "
                "payloads that ludvart_helper subcommands expect (e.g. "
                "--b64 / --old-b64 / --new-b64), avoiding fragile "
                "'printf | base64' shell quoting. Returns the base64 string."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The literal text to base64-encode.",
                    },
                },
                "required": ["text"],
            },
        ),
        ToolSpec(
            name="b64_decode",
            description=(
                "Decode a base64 string to UTF-8 text natively (no shell). "
                "Use this to read base64 payloads returned inside "
                "ludvart_helper's LUDVART:BEGIN/END result frames without piping "
                "through 'base64 -d' on screen. Returns the decoded text."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "b64": {
                        "type": "string",
                        "description": "The base64 string to decode.",
                    },
                },
                "required": ["b64"],
            },
        ),
        ToolSpec(
            name="web_search",
            description=(
                "Perform a DuckDuckGo web search to retrieve the most up-to-date information "
                "on any query. Returns a list of titles, target URLs, and descriptive snippets."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to lookup.",
                    },
                },
                "required": ["query"],
            },
        ),
        ToolSpec(
            name="fetch_url",
            description=(
                "Download the contents of a URL and save it to a temporary file under /tmp on the "
                "remote host where ludvart is running (which might be different from the host the user "
                "sees in the terminal). Returns the path to the saved file and a brief summary of the download."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The absolute HTTP or HTTPS URL to fetch/download.",
                    },
                },
                "required": ["url"],
            },
        ),
        ToolSpec(
            name="read_local_file",
            description=(
                "Read a window of lines from a local file on the host where ludvart runs. "
                "Like a code editor's file reader, this returns at most a bounded number of lines "
                "per call; large files are read by calling repeatedly, paging forward. The typical "
                "pattern is to call get_local_file_info first to learn the line count, then call "
                "read_local_file for successive line ranges. When a read does not reach the end of "
                "the file, the result reports the start_line to pass next to continue. This is useful "
                "to inspect files saved by other tools like fetch_url. The file path refers to the host "
                "running ludvart, which is not necessarily the same host shown in the terminal."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read (e.g. /tmp/...)",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "The 1-based line number to start reading from (inclusive). Defaults to 1.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": (
                            "The 1-based line number to stop reading at (inclusive). If omitted, a "
                            "default window of lines from start_line is returned. The number of lines "
                            "returned in a single call is capped regardless of this value; read again "
                            "with a later start_line to continue."
                        ),
                    },
                },
                "required": ["path"],
            },
        ),
        ToolSpec(
            name="get_local_file_info",
            description=(
                "Retrieve information about a local file on the remote host running ludvart (e.g. size, number of lines, "
                "and modified time). This is helpful to plan chunked reading using read_local_file."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file on the host running ludvart.",
                    },
                },
                "required": ["path"],
            },
        ),
    ]


class ScratchDir:
    """A private, per-run temp directory for tool output (e.g. fetch_url).

    ``tempfile.mkdtemp`` makes a uniquely named 0700 directory owned by the
    current user, so files saved here are never confused with those of other
    users or concurrent ludvart processes. Created on first use and removed by
    :meth:`cleanup` when the agent loop shuts down.
    """

    def __init__(self) -> None:
        self._path: str | None = None

    def path(self) -> str:
        if self._path is None:
            import tempfile

            self._path = tempfile.mkdtemp(prefix="ludvart_")
        return self._path

    def cleanup(self) -> None:
        if self._path is None:
            return
        import shutil

        shutil.rmtree(self._path, ignore_errors=True)
        self._path = None


def b64_encode(args: dict) -> str:
    """Base64-encode text natively (no shell/PTY round-trip)."""
    text = args.get("text")
    if not isinstance(text, str):
        return "[ludvart] b64_encode: 'text' must be a string"
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def b64_decode(args: dict) -> str:
    """Base64-decode a string to UTF-8 text natively (no shell)."""
    data = args.get("b64")
    if not isinstance(data, str):
        return "[ludvart] b64_decode: 'b64' must be a string"
    try:
        return base64.b64decode(data, validate=True).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - reported to the model
        return f"[ludvart] b64_decode: invalid base64: {exc}"


def fetch_url(args: dict, scratch: ScratchDir) -> str:
    """Fetch a URL and save it to a temp file on the host running ludvart."""
    url = args.get("url")
    if not isinstance(url, str):
        return "[ludvart] fetch_url: 'url' must be a string."
    url = url.strip()
    if not url:
        return "[ludvart] fetch_url: 'url' is empty."
    import urllib.request, urllib.error, urllib.parse, tempfile

    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        return (
            f"[ludvart] fetch_url: unsupported URL scheme "
            f"{scheme or '(none)'!r} (only http/https are allowed)."
        )
    max_bytes = FETCH_URL_MAX_BYTES
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            # Read one byte past the cap so we can report truncation without
            # ever holding more than the cap plus one byte in memory.
            raw = resp.read(max_bytes + 1)
            charset = resp.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:
        return f"[ludvart] fetch_url failed with status code: {exc.code}"
    except Exception as exc:  # noqa: BLE001 - reported to the model
        return f"[ludvart] fetch_url failed: {exc}"

    truncated = len(raw) > max_bytes
    if truncated:
        raw = raw[:max_bytes]
    try:
        text = raw.decode(charset, errors="replace")
    except LookupError:
        # Unknown charset advertised by the server -> fall back to utf-8.
        text = raw.decode("utf-8", errors="replace")

    fd, path = tempfile.mkstemp(
        prefix="ludvart_", suffix=".html", dir=scratch.path()
    )
    with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as f:
        f.write(text)

    note = " (truncated at cap)" if truncated else ""
    return (
        f"[ludvart] Successfully fetched {url}\n"
        f"Saved content to a temporary file on the host running ludvart:\n"
        f"PATH: {path}\n"
        f"SIZE: {len(text)} characters{note}\n"
        f"To read this file, use the 'read_local_file' tool."
    )


def get_local_file_info(args: dict) -> str:
    """Get details (size, lines count) of a local file on the host running ludvart."""
    path = args.get("path")
    if not isinstance(path, str):
        return "[ludvart] get_local_file_info: 'path' must be a string."
    import datetime

    if not os.path.isfile(path):
        return f"[ludvart] get_local_file_info: '{path}' is not a file."
    try:
        st = os.stat(path)
        line_count = 0
        with open(path, "rb") as f:
            for _line in f:
                line_count += 1
        mtime = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(
            timespec="seconds"
        )
        return (
            f"[ludvart] File info for {path}:\n"
            f"SIZE: {st.st_size} bytes\n"
            f"LINES: {line_count} lines\n"
            f"MODIFIED: {mtime}\n"
        )
    except Exception as exc:  # noqa: BLE001 - reported to the model
        return f"[ludvart] get_local_file_info failed: {exc}"


def read_local_file(args: dict) -> str:
    """Read a bounded window of lines from a local file on ludvart's host.

    Behaves like a code editor's file reader: a single call returns at most
    ``READ_MAX_LINES`` lines. When the window does not reach end of file the
    result reports the ``start_line`` to pass on the next call, so the model
    pages through large files with successive reads.
    """
    path = args.get("path")
    if not isinstance(path, str):
        return "[ludvart] read_local_file: 'path' must be a string."
    if not os.path.isfile(path):
        return f"[ludvart] read_local_file: '{path}' is not a file."

    start_line = args.get("start_line", 1)
    end_line = args.get("end_line")
    # bool is a subclass of int; reject it explicitly so True/False are not
    # silently treated as line 1/0.
    if isinstance(start_line, bool) or not isinstance(start_line, int):
        return "[ludvart] read_local_file: 'start_line' must be an integer."
    if end_line is not None and (
        isinstance(end_line, bool) or not isinstance(end_line, int)
    ):
        return "[ludvart] read_local_file: 'end_line' must be an integer."
    if start_line < 1:
        start_line = 1
    max_lines = READ_MAX_LINES
    if end_line is None:
        end_line = start_line + max_lines - 1
    if end_line < start_line:
        return "[ludvart] read_local_file: 'end_line' must be >= 'start_line'."
    # Cap the window so one call never returns more than max_lines lines.
    window_end = min(end_line, start_line + max_lines - 1)

    try:
        selected: list[str] = []
        has_more = False
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f, 1):
                if idx < start_line:
                    continue
                if idx > window_end:
                    # There is at least one line beyond the window.
                    has_more = True
                    break
                selected.append(line)
    except Exception as exc:  # noqa: BLE001 - reported to the model
        return f"[ludvart] read_local_file failed: {exc}"

    if not selected:
        return (
            f"[ludvart] read_local_file: {path} has no lines at or after "
            f"line {start_line} (start_line is past the end of the file)."
        )

    last_line = start_line + len(selected) - 1
    content = "".join(selected)
    char_limit = READ_MAX_CHARS
    char_truncated = len(content) > char_limit
    if char_truncated:
        content = content[:char_limit]

    notes: list[str] = []
    if has_more:
        notes.append(
            f"More lines follow; continue with start_line={last_line + 1}."
        )
    if char_truncated:
        notes.append(f"Output truncated to {char_limit} characters.")

    body = content if content.endswith("\n") else content + "\n"
    result = (
        f"[ludvart] {path} lines {start_line}-{last_line}:\n"
        f"--------------------------------------------------\n"
        f"{body}"
        f"--------------------------------------------------\n"
    )
    if notes:
        result += "\n".join(notes) + "\n"
    return result


def web_search(args: dict) -> str:
    """Perform a DuckDuckGo web search to retrieve up-to-date information."""
    query = args.get("query")
    if not isinstance(query, str):
        return "[ludvart] web_search: 'query' must be a string."
    if not query.strip():
        return "[ludvart] web_search: nothing to search (empty 'query')."
    import urllib.request, urllib.error, urllib.parse, re, html

    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read(2 * 1024 * 1024)
            charset = resp.headers.get_content_charset() or "utf-8"
        page = body.decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        return f"[ludvart] web_search failed with status code: {exc.code}"
    except Exception as exc:  # noqa: BLE001 - reported to the model
        return f"[ludvart] web_search failed: {exc}"

    def _clean(fragment: str) -> str:
        # Strip tags, then decode HTML entities (&amp;, &#x27;, ...).
        return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()

    blocks = page.split('<div class="links_main links_deep result__body">')[1:]
    outputs = []
    for block in blocks[:10]:
        title_match = re.search(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            re.DOTALL,
        )
        if not title_match:
            continue
        snippet_match = re.search(
            r'class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL
        )
        raw_url = html.unescape(title_match.group(1))
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
        actual_url = qs.get("uddg", [raw_url])[0]
        title = _clean(title_match.group(2))
        snippet = _clean(snippet_match.group(1)) if snippet_match else ""
        outputs.append(
            f"TITLE: {title}\nURL: {actual_url}\nSNIPPET: {snippet}\n"
        )

    if not outputs:
        return "[ludvart] web_search: No results found."
    return "\n---\n".join(outputs)
