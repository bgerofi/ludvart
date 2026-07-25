"""Command-line entry point for ludvart."""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from .models import (
    PROVIDER_MENU,
    Registration,
    SERVICE_PROMPT,
    add_registration,
    label,
    load_registry,
    save_models,
)
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


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt for a yes/no answer; Enter takes ``default``."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    try:
        ans = input(prompt + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if not ans:
        return default
    return ans in ("y", "yes")


def _run_setup_wizard() -> bool:
    """Interactively collect a model's settings and register it in models.json.

    Returns ``True`` when a model was registered, ``False`` if the session is
    non-interactive or the user aborted (Ctrl-C / EOF).
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False

    sys.stderr.write(
        "\nludvart: no LLM model is registered yet -- let's set one up.\n"
        "Answers are saved to ~/.ludvart/models.json (press Ctrl-C to skip).\n\n"
    )
    sys.stderr.flush()

    try:
        # 0) Service name -- an arbitrary label for who provides access.
        service = input(SERVICE_PROMPT + " ").strip()

        # 1) Endpoint type.
        sys.stderr.write("Select the API endpoint type:\n")
        for i, (_name, menu_label, _url) in enumerate(PROVIDER_MENU, 1):
            sys.stderr.write(f"  {i}) {menu_label}\n")
        sys.stderr.flush()
        provider = default_url = ""
        while not provider:
            choice = input(f"Choice [1-{len(PROVIDER_MENU)}]: ").strip().lower()
            picked = None
            if choice.isdigit() and 1 <= int(choice) <= len(PROVIDER_MENU):
                picked = PROVIDER_MENU[int(choice) - 1]
            else:
                picked = next((p for p in PROVIDER_MENU if p[0] == choice), None)
            if picked is None:
                sys.stderr.write(f"Please enter a number 1-{len(PROVIDER_MENU)}.\n")
                continue
            provider, _label, default_url = picked

        # GitHub Copilot has its own flow (device auth + local gateway).
        if provider == "copilot":
            return _setup_copilot(service)

        # 2) Endpoint URL (Enter accepts the default when there is one).
        url = ""
        while not url:
            prompt = (
                f"Endpoint URL [{default_url}]: " if default_url else "Endpoint URL: "
            )
            url = input(prompt).strip() or default_url
            if not url:
                sys.stderr.write("An endpoint URL is required.\n")

        # 3) API key (hidden input).
        key = ""
        while not key:
            key = getpass.getpass("API key (input hidden): ").strip()
            if not key:
                sys.stderr.write("An API key is required.\n")

        # 4) Model name.
        model = ""
        while not model:
            model = input(
                "Model name (e.g. gpt-4o, claude-..., gemini-...): "
            ).strip()
            if not model:
                sys.stderr.write("A model name is required.\n")
    except (EOFError, KeyboardInterrupt):
        sys.stderr.write("\nludvart: setup skipped.\n")
        return False

    reg: Registration = {
        "provider": provider,
        "service": service,
        "api_url": url,
        "api_key": key,
        "model": model,
        "context_window": 0,
        "active": True,
    }
    path = _register_model(reg)
    sys.stderr.write(f"ludvart: registered {label(reg)} in {path}\n")
    return True


def _register_model(reg: Registration) -> str:
    """Append ``reg`` to the registry (as the new active model) and save it."""
    models = add_registration(load_registry(), reg, make_active=True)
    return save_models(models)


def _setup_copilot(service: str = "") -> bool:
    """Set up the GitHub Copilot backend: device-flow auth + registered model.

    Walks the user through GitHub's OAuth device flow (paid Copilot subscription
    required), caches the credentials via LiteLLM, and registers the chosen model
    in ``~/.ludvart/models.json``. The gateway itself is spawned later at
    startup / on ``/model use``. Returns ``True`` on success, ``False`` if
    unavailable or aborted.
    """
    from .gateway import GatewayError, authenticate_copilot, litellm_available

    if not litellm_available():
        sys.stderr.write(
            "ludvart: the LiteLLM gateway isn't installed, so the GitHub Copilot\n"
            "       option is unavailable. Re-run ./setup.sh, or:\n"
            "       uv pip install 'litellm[proxy]'\n"
        )
        return False

    sys.stderr.write(
        "\nGitHub Copilot requires an active paid GitHub Copilot subscription.\n"
        "You'll authorize ludvart through GitHub's device flow: a URL and one-time\n"
        "code appear below -- open the URL, enter the code, and approve access.\n"
        "Credentials are cached under ~/.config/litellm and reused afterwards.\n"
        "(This uses GitHub's own OAuth, not ~/.netrc.)\n\n"
    )
    sys.stderr.flush()

    sys.stderr.write("ludvart: starting GitHub authentication...\n")
    sys.stderr.flush()
    try:
        authenticate_copilot()
    except GatewayError as exc:
        sys.stderr.write(f"\nludvart: {exc}\n")
        return False

    model = _choose_copilot_model()
    if not model:
        sys.stderr.write("\nludvart: setup skipped.\n")
        return False

    reg: Registration = {
        "provider": "copilot",
        "service": service,
        "api_url": "",
        "api_key": "",
        "model": model,
        "context_window": 0,
        "active": True,
    }
    path = _register_model(reg)
    sys.stderr.write(
        f"ludvart: GitHub Copilot authorized; registered model {model!r} in {path}\n"
    )
    return True


def _choose_copilot_model(default: str = "gpt-4o") -> str:
    """Prompt for a Copilot model, listing the account's available slugs.

    Returns the chosen slug (e.g. ``claude-opus-4.8``), or ``""`` if aborted.
    Copilot uses its own model ids, so we list what the account can actually use
    rather than making the user guess.
    """
    from .gateway import list_copilot_models

    models = list_copilot_models()
    try:
        if not models:
            # Listing failed; fall back to a free-text prompt.
            return input(f"Copilot model [{default}]: ").strip() or default

        sys.stderr.write("\nModels available to your GitHub Copilot account:\n")
        for i, name in enumerate(models, 1):
            sys.stderr.write(f"  {i:2}) {name}\n")
        sys.stderr.flush()
        hint = default if default in models else models[0]
        raw = input(f"Choose a model [number or name, default {hint}]: ").strip()
        if not raw:
            return hint
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        # Accept a typed slug even if it's not listed (verify() catches a
        # genuinely unsupported one).
        return raw
    except (EOFError, KeyboardInterrupt):
        return ""


if __name__ == "__main__":
    sys.exit(main())
