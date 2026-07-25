"""Command-line entry point for ludvart."""

from __future__ import annotations

import argparse
import getpass
import os
import signal
import sys

from .backend import Backend, ModelManager, build_backend, verify_backend
from .llm import (
    build_client,
    ensure_context_windows_file,
)
from .models import (
    PROVIDER_MENU,
    Registration,
    active_index,
    add_registration,
    is_copilot,
    label,
    load_registry,
    registration_to_config,
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


def _install_gateway_shutdown_handlers(get_gateway):
    """Install best effort signal handlers that stop a running gateway."""
    handlers = []

    def _handler(signum, _frame):
        gw = get_gateway()
        if gw is not None:
            gw.stop()
        raise SystemExit(128 + int(signum))

    for sig_name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            prev = signal.getsignal(sig)
            signal.signal(sig, _handler)
            handlers.append((sig, prev))
        except Exception:
            pass
    return handlers


def _restore_handlers(handlers) -> None:
    for sig, prev in handlers:
        try:
            signal.signal(sig, prev)
        except Exception:
            pass


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
        default=None,
        help=(
            "Run the agent loop in a separate backend process. 'local' forks it "
            "on this host; 'host:folder' runs it on an SSH-reachable host from "
            "the ludvart checkout at 'folder' (which has a .venv). The terminal "
            "stays local; only structured messages cross the link."
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

    # Split mode: the agent loop runs in a backend process; the client keeps the
    # terminal. No local model activation is needed (the backend owns the LLM).
    if args.backend:
        return _run_with_backend(args, command)

    manager = None
    if not args.no_llm:
        manager = _setup_llm()

    llm = manager.client if manager is not None else None
    handlers = _install_gateway_shutdown_handlers(
        lambda: manager.gateway if manager is not None else None
    )
    try:
        return Ludvart(
            command, prefix=args.prefix, llm=llm, model_manager=manager
        ).run()
    except KeyboardInterrupt:
        return 130
    finally:
        if manager is not None and manager.gateway is not None:
            manager.gateway.stop()
        _restore_handlers(handlers)


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
    if reconnector.verified:
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


def _setup_llm() -> ModelManager | None:
    """Load the model registry, run first-time setup if empty, and activate it.

    Models live in ``~/.ludvart/models.json`` (seeded once from any legacy
    ``llm.conf`` / environment config). When the registry is empty and the
    session is interactive, the setup wizard collects and registers the first
    model. The active model is then built and verified; every other registered
    model is verified too so ``/model list`` can show which are available.

    Returns a :class:`~ludvart.backend.ModelManager` (its ``gateway`` must be
    stopped on shutdown), or ``None`` when nothing is configured -- meaning
    ludvart runs as a plain relay.
    """
    # Seed the editable context-window table on first run so users can tune it.
    ensure_context_windows_file()

    models = load_registry()
    if not models:
        if not _run_setup_wizard():
            _report_plain_relay()
            return None
        models = load_registry()
        if not models:
            _report_plain_relay()
            return None

    return _activate_registry(models)


def _activate_registry(models: list[Registration]) -> ModelManager | None:
    """Build+verify the active model (verifying the rest for availability).

    On a failed active-model check, the interactive session may add or re-enter
    a model and retry; otherwise ludvart exits with an error.
    """
    while True:
        idx = active_index(models)
        assert idx is not None
        active = models[idx]
        saved_api_mode = active.get("api_mode")
        backend: Backend | None = None
        checking = False
        # Whether a "starting the ... gateway..." progress line is still open
        # and needs an "ok"/"FAILED" appended to close it.
        gw_line = {"open": False}

        def _status(m: str) -> None:
            sys.stderr.write("ludvart: " + m + " ")
            sys.stderr.flush()
            gw_line["open"] = True

        try:
            # Build first (this is where the Copilot gateway is launched, if
            # any) so its progress is reported before the verification step.
            backend = build_backend(active, status=_status)
            if gw_line["open"]:
                sys.stderr.write("ok\n")
                gw_line["open"] = False
            sys.stderr.write(
                f"ludvart: verifying {label(active)} (model {active['model']!r})... "
            )
            sys.stderr.flush()
            checking = True
            verify_backend(backend)
        except Exception as exc:
            if gw_line["open"] or checking:
                sys.stderr.write("FAILED\n")
            sys.stderr.write(f"ludvart: LLM check failed: {exc}\n")
            if backend is not None:
                backend.stop()
            if sys.stdin.isatty() and _ask_yes_no(
                "ludvart: add or re-enter an LLM model now?", default=True
            ):
                if _run_setup_wizard():
                    models = load_registry()
                    continue
            sys.stderr.write("ludvart: fix the configuration or pass --no-llm.\n")
            sys.exit(2)
        sys.stderr.write("ok\n")
        if active.get("api_mode") != saved_api_mode:
            save_models(models)
        available = _verify_others(models, idx)
        available[idx] = True
        return ModelManager(models, available, backend.client, backend.gateway)


def _verify_others(models: list[Registration], active_idx: int) -> list[bool]:
    """Verify every non-active model sequentially; return an availability list.

    Direct providers are checked with a tiny request. Copilot models aren't
    started here (that needs the gateway); they are marked available when the
    gateway is installed and authorized, and truly verified on ``/model use``.
    """
    available = [False] * len(models)
    for i, reg in enumerate(models):
        if i == active_idx:
            available[i] = True
            continue
        sys.stderr.write(f"ludvart: verifying {label(reg)}... ")
        sys.stderr.flush()
        if is_copilot(reg):
            ok = _copilot_ready()
            available[i] = ok
            sys.stderr.write("ok\n" if ok else "unavailable\n")
            continue
        try:
            client = build_client(registration_to_config(reg))
            client.verify()
            available[i] = True
            sys.stderr.write("ok\n")
        except Exception as exc:
            available[i] = False
            sys.stderr.write(f"unavailable ({exc})\n")
    return available


def _copilot_ready() -> bool:
    """Whether a Copilot backend could start (installed + authorized)."""
    from .gateway import copilot_authenticated, litellm_available

    return litellm_available() and copilot_authenticated()


def _report_plain_relay() -> None:
    sys.stderr.write(
        "ludvart: no LLM model registered. Running as a plain relay.\n"
        "       Register one interactively (re-run in a terminal), or set\n"
        "       {OPENAI,ANTHROPIC,GOOGLE,CUSTOM}_API_URL/_API_KEY/_MODEL, or\n"
        "       pass --no-llm.\n"
    )


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
            return _setup_copilot()

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


def _setup_copilot() -> bool:
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
