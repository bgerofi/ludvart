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
import json
import os
import shlex

from .llm import ToolSpec

#: Tools that must run where the terminal is. Everything else in this module is
#: executed by the agent loop itself.
CLIENT_TOOL_NAMES = frozenset({"inject_input", "capture_screen_history"})

#: The helper is deliberately not on PATH, so every call spells out its path.
HELPER_PATH = "~/.ludvart/bin/ludvart_helper"

#: The helper is reached by typing a command line, and that channel carries only
#: about 2 KB before it truncates -- which would run a mangled command rather
#: than fail cleanly, so an oversized call is refused here instead.
HELPER_MAX_LINE = 2000

#: File-editing tools, and the ``ludvart_helper`` subcommand each one wraps.
HELPER_EDIT_SUBCOMMANDS = {
    "write_file": "write",
    "append_to_file": "append",
    "replace_in_file": "replace",
    "replace_file_lines": "replace-range",
    "apply_file_edits": "structured-patch",
}

#: ``append`` is the one edit subcommand with nothing to preview.
HELPER_DRY_RUN_TOOLS = frozenset(HELPER_EDIT_SUBCOMMANDS) - {"append_to_file"}

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
                "newline -- on the final one to execute). To put a large file "
                "on the machine, do not split a command line at all: use "
                "ludvart_helper write for the first chunk and append for each "
                "of the rest. For a non-interactive shell command typed at a "
                "shell prompt, prefer run_shell_command, which does the "
                "encoding and the injection in a single call and reports the "
                "command's real exit status."
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
            name="run_shell_command",
            description=(
                "Run a non-interactive shell command on the user's machine in "
                "ONE call. The command is base64-encoded here and injected as "
                "a single '" + HELPER_PATH + " run --b64 ...' line, so you do "
                "NOT need a separate b64_encode call and there is no shell "
                "quoting to get wrong. TWO PRECONDITIONS, both on you to "
                "check: ludvart_helper must be installed (confirm once with '"
                + HELPER_PATH
                + " info'), and the terminal must be sitting at a SHELL "
                "PROMPT. This types a command line, so it does nothing useful "
                "while the screen is held by a full-screen or interactive "
                "program -- vim, less, top, a pager, an ssh session into "
                "another host, or a Python/Node REPL will simply swallow the "
                "text as keystrokes. Look at the current screen first; if a "
                "program owns it, either leave it via inject_input or use "
                "inject_input for the whole job. When those hold, prefer this "
                "over pairing b64_encode with inject_input for every "
                "non-interactive command. Pass the command exactly as you "
                "would type it at a shell prompt -- pipes, redirections, "
                "quotes, '&&' and environment prefixes all work, since it is "
                "run through the shell. Its stdout and stderr go to the "
                "terminal, and the result carries the screen plus the "
                "helper's END sentinel, whose exit= is the command's real "
                "status: read it before judging whether the command worked. "
                "Use inject_input instead for interactive programs and for "
                "sending keystrokes."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "The shell command line to run, exactly as you "
                            "would type it, e.g. 'make test 2>&1 | tail -n 40'. "
                            "Do not base64-encode it yourself and do not wrap "
                            "it in a ludvart_helper invocation -- both are done "
                            "for you."
                        ),
                    },
                },
                "required": ["command"],
            },
        ),
        ToolSpec(
            name="write_file",
            description=(
                "Create or overwrite a file on the user's machine (the one in "
                "the terminal, not the host ludvart runs on) in ONE call: the "
                "content is base64-encoded here and typed as a single helper "
                "'write' line, so no b64_encode call and no shell quoting are "
                "needed. Needs the helper installed and a shell prompt on "
                "screen, exactly like run_shell_command. Parent directories "
                "are created, the previous contents are kept beside the file "
                "as PATH.ludvart.bak, and a .py file is compile-checked after "
                "writing. About 2 KB of content fits in a call: for more, "
                "write the first chunk here and add the rest with "
                "append_to_file."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to write, on the terminal's machine.",
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "The exact text to write. Do not base64-encode it "
                            "yourself; that is done for you."
                        ),
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": (
                            "If true, write nothing and return a unified diff "
                            "of the change instead. Defaults to false."
                        ),
                    },
                },
                "required": ["path", "content"],
            },
        ),
        ToolSpec(
            name="append_to_file",
            description=(
                "Append text to a file on the user's machine (the terminal's, "
                "not ludvart's host) in ONE call, base64-encoded for you, "
                "creating the file if it is absent. This is how a file too "
                "big for a single ~2 KB call is built: write_file the first "
                "chunk, then one append_to_file per remaining chunk, checking "
                "the reported bytes= each time so a short chunk is caught at "
                "once."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to append to, on the terminal's machine.",
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "The exact text to append. Do not base64-encode "
                            "it yourself."
                        ),
                    },
                },
                "required": ["path", "content"],
            },
        ),
        ToolSpec(
            name="replace_in_file",
            description=(
                "Replace exact literal text in a file on the user's machine "
                "in ONE call: the old and new text are both base64-encoded "
                "for you. The match is literal, never a regex. Every "
                "occurrence is replaced unless you limit it; set expect_count "
                "to refuse the edit unless the old text occurs exactly that "
                "many times, which is the safe way to change one known site. "
                "Nothing is written if the old text is not found or the count "
                "does not match, so a failed edit cannot half-apply."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to edit, on the terminal's machine.",
                    },
                    "old": {
                        "type": "string",
                        "description": (
                            "The exact literal text to find, reproduced "
                            "character for character."
                        ),
                    },
                    "new": {
                        "type": "string",
                        "description": "The text to put in its place.",
                    },
                    "expect_count": {
                        "type": "integer",
                        "description": (
                            "Refuse the edit unless the old text occurs "
                            "exactly this many times."
                        ),
                    },
                    "count": {
                        "type": "integer",
                        "description": "Replace at most this many occurrences.",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": (
                            "If true, change nothing and return a unified "
                            "diff instead. Defaults to false."
                        ),
                    },
                },
                "required": ["path", "old", "new"],
            },
        ),
        ToolSpec(
            name="replace_file_lines",
            description=(
                "Replace a 1-indexed, inclusive line range of a file on the "
                "user's machine in ONE call, with the new content "
                "base64-encoded for you. Use it when the old text is awkward "
                "to reproduce exactly; otherwise prefer replace_in_file, "
                "which cannot silently hit the wrong lines if the file has "
                "shifted since you read it."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to edit, on the terminal's machine.",
                    },
                    "start": {
                        "type": "integer",
                        "description": "First line to replace (1-indexed, inclusive).",
                    },
                    "end": {
                        "type": "integer",
                        "description": "Last line to replace (1-indexed, inclusive).",
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "The text to put in place of those lines. Do not "
                            "base64-encode it yourself."
                        ),
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": (
                            "If true, change nothing and return a unified "
                            "diff instead. Defaults to false."
                        ),
                    },
                },
                "required": ["path", "start", "end", "content"],
            },
        ),
        ToolSpec(
            name="apply_file_edits",
            description=(
                "Apply several exact literal edits to one file on the user's "
                "machine in a single call, writing only if ALL of them apply "
                "-- the all-or-nothing alternative to a run of "
                "replace_in_file calls, and one round trip instead of many. "
                "Each edit is base64-encoded and packed into the helper's "
                "structured patch for you. Every edit requires exactly one "
                "occurrence of its old text unless you say otherwise; if any "
                "edit fails, nothing is written and the result names the "
                "failing edit by its 1-based index."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to edit, on the terminal's machine.",
                    },
                    "edits": {
                        "type": "array",
                        "description": "The edits to apply, in order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old": {
                                    "type": "string",
                                    "description": "Exact literal text to find.",
                                },
                                "new": {
                                    "type": "string",
                                    "description": "Text to put in its place.",
                                },
                                "expect_count": {
                                    "type": "integer",
                                    "description": (
                                        "Require exactly this many "
                                        "occurrences. Defaults to 1."
                                    ),
                                },
                                "count": {
                                    "type": "integer",
                                    "description": (
                                        "Replace at most this many occurrences."
                                    ),
                                },
                            },
                            "required": ["old", "new"],
                        },
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": (
                            "If true, change nothing and return a unified "
                            "diff instead. Defaults to false."
                        ),
                    },
                },
                "required": ["path", "edits"],
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
                "at a previous turn, identified by its UTC timestamp. Only the "
                "live screen is shown in full, in a <screenContext ts=\"...\"> "
                "block at the end of the conversation; every earlier snapshot "
                "is replaced by a breadcrumb line that keeps its timestamp "
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
                "terminal round-trip). Use it for the helper payloads the "
                "dedicated tools do not build for you -- driving a subcommand "
                "by hand -- rather than for a write or an edit, which "
                "write_file, append_to_file, replace_in_file, "
                "replace_file_lines and apply_file_edits already encode. "
                "Avoids fragile 'printf | base64' shell quoting. Returns the "
                "base64 string."
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


def helper_run_line(command: str) -> str:
    """The ``ludvart_helper run`` line that executes ``command`` on the terminal.

    Base64 keeps the command opaque to the shell that types it, so quoting,
    pipes and newlines survive injection untouched.
    """
    blob = base64.b64encode(command.encode("utf-8")).decode("ascii")
    return f"{HELPER_PATH} run --b64 {blob}"


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _need_str(args: dict, key: str) -> str:
    val = args.get(key)
    if not isinstance(val, str):
        raise ValueError(f"'{key}' must be a string")
    return val


def _need_int(args: dict, key: str) -> int:
    """Coerce an integer argument, tolerating the digit strings models send."""
    val = args.get(key)
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    if isinstance(val, str) and val.strip().lstrip("-").isdigit():
        return int(val)
    raise ValueError(f"'{key}' must be an integer")


def _structured_patch_json(args: dict) -> str:
    """Pack the ``apply_file_edits`` edit list into the helper's patch JSON."""
    edits = args.get("edits")
    if not isinstance(edits, list) or not edits:
        raise ValueError("'edits' must be a non-empty list")
    packed = []
    for i, edit in enumerate(edits, 1):
        if not isinstance(edit, dict):
            raise ValueError(f"edit {i} must be an object")
        try:
            item: dict = {
                "old_b64": _b64(_need_str(edit, "old")),
                "new_b64": _b64(_need_str(edit, "new")),
            }
            if "expect_count" in edit:
                expect = edit["expect_count"]
                item["expect_count"] = (
                    None if expect is None else _need_int(edit, "expect_count")
                )
            if edit.get("count") is not None:
                item["count"] = _need_int(edit, "count")
        except ValueError as exc:
            raise ValueError(f"edit {i}: {exc}") from None
        packed.append(item)
    return json.dumps({"edits": packed}, separators=(",", ":"))


def helper_edit_line(name: str, args: dict) -> str:
    """Build the ``ludvart_helper`` line for one of the file-editing tools.

    Raises :class:`ValueError` with a model-readable reason, so a malformed
    call is handed back for correction instead of being typed at the terminal.
    """
    sub = HELPER_EDIT_SUBCOMMANDS[name]
    path = _need_str(args, "path")
    if not path.strip():
        raise ValueError("'path' must not be empty")
    parts = [HELPER_PATH, sub, shlex.quote(path)]
    if sub in ("write", "append"):
        parts += ["--b64", _b64(_need_str(args, "content"))]
    elif sub == "replace":
        parts += ["--old-b64", _b64(_need_str(args, "old"))]
        parts += ["--new-b64", _b64(_need_str(args, "new"))]
        if args.get("expect_count") is not None:
            parts += ["--expect-count", str(_need_int(args, "expect_count"))]
        if args.get("count") is not None:
            parts += ["--count", str(_need_int(args, "count"))]
    elif sub == "replace-range":
        parts += ["--start", str(_need_int(args, "start"))]
        parts += ["--end", str(_need_int(args, "end"))]
        parts += ["--b64", _b64(_need_str(args, "content"))]
    else:
        parts += ["--b64", _b64(_structured_patch_json(args))]
    if args.get("dry_run") and name in HELPER_DRY_RUN_TOOLS:
        parts.append("--dry-run")
    line = " ".join(parts)
    if len(line) > HELPER_MAX_LINE:
        raise ValueError(
            f"the command line would be {len(line)} bytes, past the "
            f"~{HELPER_MAX_LINE} the terminal channel carries; send less per "
            "call -- write_file the first chunk and append_to_file the rest, "
            "or change part of the file with replace_in_file / "
            "replace_file_lines instead of rewriting all of it"
        )
    return line


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
