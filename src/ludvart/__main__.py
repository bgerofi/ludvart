"""Command-line entry point for ludvart."""

from __future__ import annotations

import argparse
import os
import sys

from .ludvart import DEFAULT_PREFIX, Ludvart


def _default_shell() -> str:
    return os.environ.get("SHELL") or "/bin/sh"


def _parse_prefix(spec: str) -> bytes:
    """Parse a prefix spec like 'C-g', 'ctrl-g', '^g', or '\\x07' into a byte."""
    s = spec.strip().lower()
    if s.startswith(("c-", "ctrl-", "^")):
        letter = s.split("-", 1)[-1] if "-" in s else s[1:]
        if len(letter) == 1 and letter.isalpha():
            # Control character: clear the top three bits (A -> 0x01, G -> 0x07).
            return bytes([ord(letter.upper()) & 0x1F])
    if s.startswith("\\x") and len(s) == 4:
        return bytes([int(s[2:], 16)])
    raise argparse.ArgumentTypeError(
        f"invalid prefix {spec!r}; use e.g. 'C-g', 'ctrl-g', '^g', or '\\x07'"
    )


def main(argv: list[str] | None = None) -> int:
    # Backend mode: `ludvart serve` runs the agent-loop server on stdin/stdout
    # (spawned by a client locally or over SSH). It speaks only the framed
    # protocol on stdout, so it must be dispatched before any normal output.
    raw = sys.argv[1:] if argv is None else argv
    if raw and raw[0] == "serve":
        from .server import serve_main

        return serve_main(raw[1:])

    parser = argparse.ArgumentParser(
        prog="ludvart",
        description=(
            "PTY-level relay: spawn a command and interact with it transparently. "
            "With no command, spawns your $SHELL."
        ),
        epilog=(
            "Everything after '--' is the command to run, e.g.  ludvart -- htop. "
            "Inside a session, press the prefix key (default Ctrl-G) then 's' to "
            "open the scrollback viewer; press the prefix twice to send it literally."
        ),
    )
    parser.add_argument(
        "--prefix",
        type=_parse_prefix,
        default=DEFAULT_PREFIX,
        metavar="KEY",
        help="Prefix key for ludvart commands, e.g. 'C-g' (default), 'ctrl-o', '^b'.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run as a plain relay without any LLM (skip provider setup/check).",
    )
    parser.add_argument(
        "--backend",
        metavar="SPEC",
        default="local",
        help=(
            "Where the agent loop runs. 'local' (the default) forks it on this "
            "host and talks to it over a pipe; 'host:folder' runs it on an "
            "SSH-reachable host from the ludvart checkout at 'folder' (which "
            "has a .venv). The terminal always stays local; only structured "
            "messages cross the link."
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command (and args) to run. Prefix with '--' to pass flags through.",
    )
    args = parser.parse_args(argv)

    command = args.command
    # argparse.REMAINDER keeps a leading '--' if the user wrote 'ludvart -- cmd'.
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        command = [_default_shell()]

    # The agent loop always runs in a backend process; this process only ever
    # owns the terminal. A forked backend over a pipe and an SSH backend differ
    # only in how they are spawned, so there is a single code path for both.
    if args.no_llm:
        # Plain relay: no agent loop at all, so there is no backend to place.
        try:
            return Ludvart(command, prefix=args.prefix).run()
        except KeyboardInterrupt:
            return 130
    return _run_with_backend(args, command)


def _run_with_backend(args, command: list[str]) -> int:
    """Run a client session whose agent loop lives in a backend process.

    The backend is either forked locally (``--backend local``) or spawned on an
    SSH-reachable host (``--backend host:folder``). A :class:`BackendReconnector`
    owns the process so a dropped connection (e.g. a flaky SSH link) respawns it
    and restores the session, with progress shown on the panel. The transport is
    always closed on exit so the backend process is never leaked.
    """
    from .backend_client import BackendReconnector

    spawn = _backend_spawn(args.backend)

    def _startup_log(text: str) -> None:
        sys.stderr.write(f"ludvart: {text}\n")
        sys.stderr.flush()

    reconnector = BackendReconnector(spawn, on_log=_startup_log)
    try:
        reconnector.connect()
    except Exception as exc:  # noqa: BLE001 - report and exit
        sys.stderr.write(f"ludvart: could not start backend: {exc}\n")
        return 2
    label = reconnector.label or "backend"
    if reconnector.needs_setup:
        sys.stderr.write(
            "ludvart: no model registered yet; starting setup...\n"
        )
    elif reconnector.verified:
        sys.stderr.write(f"ludvart: backend model {label}... ok\n")
    else:
        err = reconnector.verify_error or "unknown error"
        sys.stderr.write(f"ludvart: backend model {label}... FAILED ({err})\n")
    try:
        return Ludvart(
            command,
            prefix=args.prefix,
            backend_channel=reconnector.channel,
            backend_label=label,
            backend_reconnector=reconnector,
            backend_needs_setup=reconnector.needs_setup,
        ).run()
    except KeyboardInterrupt:
        return 130
    finally:
        reconnector.close()


def _backend_spawn(spec: str):
    """Return a zero-arg factory that spawns a fresh backend transport."""
    from .transport import local_backend, parse_backend_spec, ssh_backend

    if spec == "local":
        return lambda: local_backend()
    host, folder = parse_backend_spec(spec)
    return lambda: ssh_backend(host, folder)


if __name__ == "__main__":
    sys.exit(main())
