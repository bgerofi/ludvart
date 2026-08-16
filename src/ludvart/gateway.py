"""GitHub Copilot support via a local LiteLLM gateway.

ludvart can use GitHub Copilot as an OpenAI-compatible backend by spawning a local
``litellm`` proxy that fronts LiteLLM's ``github_copilot/`` provider. ludvart then
talks to that proxy with its normal OpenAI-compatible client.

Authentication uses GitHub's OAuth device flow (implemented by LiteLLM): on first
setup the user visits a URL and enters a one-time code. The resulting
credentials are cached by LiteLLM under ``~/.config/litellm/github_copilot`` and
reused on later runs, so the gateway starts non-interactively from then on.

Note: GitHub Copilot has its own OAuth flow with a dedicated client id; it does
*not* use ``~/.netrc`` (which holds a git/HTTPS password or PAT), so that cannot
be reused here.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

#: The gateway only ever listens on loopback.
GATEWAY_HOST = "127.0.0.1"
#: The local proxy is started without a master key, so any non-empty API key is
#: accepted. ludvart passes this placeholder to its OpenAI-compatible client.
GATEWAY_API_KEY = "sk-ludvart-local"

#: Base URL of an already-running gateway to reuse instead of spawning one, and
#: the single model it serves. Both must be set for the reuse to apply.
#:
#: A gateway takes tens of seconds to boot and is entirely stateless, so a
#: process that starts many short-lived ludvart instances back to back (the e2e
#: suite) can start one gateway up front and hand it to all of them. Whoever
#: sets these owns the gateway's lifetime; ludvart will not stop it.
SHARED_GATEWAY_URL_ENV = "LUDVART_GATEWAY_URL"
SHARED_GATEWAY_MODEL_ENV = "LUDVART_GATEWAY_MODEL"


def shared_gateway(model: str) -> str | None:
    """Base URL of a reusable gateway serving ``model``, or ``None``.

    A gateway proxies exactly one model, so a shared one is only usable by a
    process that wants that same model; anyone else spawns their own.
    """
    url = os.environ.get(SHARED_GATEWAY_URL_ENV, "").strip()
    served = os.environ.get(SHARED_GATEWAY_MODEL_ENV, "").strip()
    if url and served == model:
        return url
    return None


class GatewayError(RuntimeError):
    """Raised when the LiteLLM gateway can't be authenticated or started."""


def _litellm_cli() -> str | None:
    """Locate the ``litellm`` console script (proxy launcher), or ``None``."""
    candidate = os.path.join(os.path.dirname(sys.executable), "litellm")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return shutil.which("litellm")


def litellm_available() -> bool:
    """True if the LiteLLM library and its proxy launcher are both installed."""
    if _litellm_cli() is None:
        return False
    try:
        import importlib.util

        return importlib.util.find_spec("litellm") is not None
    except Exception:
        return False


def _token_dir() -> str:
    return os.getenv(
        "GITHUB_COPILOT_TOKEN_DIR",
        os.path.expanduser("~/.config/litellm/github_copilot"),
    )


def copilot_authenticated() -> bool:
    """True if a cached GitHub Copilot access token already exists."""
    path = os.path.join(
        _token_dir(),
        os.getenv("GITHUB_COPILOT_ACCESS_TOKEN_FILE", "access-token"),
    )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return bool(fh.read().strip())
    except OSError:
        return False


def authenticate_copilot() -> None:
    """Run GitHub's OAuth device flow so the gateway can serve requests.

    LiteLLM prints the verification URL and one-time code to stdout, then polls
    until the user authorizes (or it times out). On success the access token and
    a short-lived Copilot API key are cached under ``~/.config/litellm``.

    Raises :class:`GatewayError` if litellm is missing or authentication fails
    (e.g. the account has no active Copilot subscription).
    """
    try:
        from litellm.llms.github_copilot.authenticator import Authenticator
    except Exception as exc:  # pragma: no cover - dependency guard
        raise GatewayError(
            "the LiteLLM gateway isn't installed (uv pip install 'litellm[proxy]')"
        ) from exc

    auth = Authenticator()
    try:
        # Triggers the device flow and prints the code when no token is cached.
        auth.get_access_token()
        # Confirms the account actually has GitHub Copilot access.
        auth.get_api_key()
    except Exception as exc:
        raise GatewayError(f"GitHub Copilot authentication failed: {exc}") from exc


def drop_cached_copilot_key() -> bool:
    """Discard the cached Copilot API key so the next call mints a fresh one.

    GitHub hands out a short-lived key alongside a ``refresh_in`` that is
    shorter than its own ``expires_at``, and it invalidates outstanding keys
    early whenever the seat state rotates. LiteLLM caches on ``expires_at``
    alone, so a key it still considers valid can start coming back as a bare
    ``403 forbidden``. Deleting the cache is enough -- the OAuth access token
    is untouched, so no re-login is needed.

    Returns True if a cached key was removed.
    """
    path = os.path.join(
        _token_dir(), os.getenv("GITHUB_COPILOT_API_KEY_FILE", "api-key.json")
    )
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def is_forbidden(exc: BaseException) -> bool:
    """True if ``exc`` looks like Copilot's 403, whatever layer wrapped it."""
    text = str(exc).lower()
    return "403" in text and "forbidden" in text


#: Headers GitHub Copilot expects (simulating the VS Code client), reused for
#: the ``/models`` listing call below.
_COPILOT_HEADERS = {
    "Content-Type": "application/json",
    "Copilot-Integration-Id": "vscode-chat",
    "editor-version": "vscode/1.85.1",
    "editor-plugin-version": "copilot/1.155.0",
    "user-agent": "GithubCopilot/1.155.0",
}


def list_copilot_models() -> list[str]:
    """Return the model ids this account can use through GitHub Copilot.

    Best-effort: returns an empty list if litellm is missing, the account isn't
    authenticated yet, or the request fails. The slugs (e.g. ``claude-opus-4.8``,
    ``gpt-4o``) are what ludvart stores as ``COPILOT_MODEL``.
    """
    try:
        import httpx
        from litellm.llms.github_copilot.authenticator import Authenticator
    except Exception:
        return []

    def fetch() -> list[str]:
        auth = Authenticator()
        api_key = auth.get_api_key()
        api_base = auth.get_api_base().rstrip("/")
        resp = httpx.get(
            f"{api_base}/models",
            headers={"Authorization": f"Bearer {api_key}", **_COPILOT_HEADERS},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        ids = {m.get("id") for m in data.get("data", []) if m.get("id")}
        return sorted(ids)

    try:
        return fetch()
    except Exception as exc:
        if not (is_forbidden(exc) and drop_cached_copilot_key()):
            return []
    try:
        return fetch()
    except Exception:
        return []



def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _linux_set_pdeathsig() -> None:
    """Best effort: ask Linux to signal this child if parent dies."""
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        prctl.restype = ctypes.c_int
        PR_SET_PDEATHSIG = 1
        if prctl(PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        pass


# The gateway subprocess uses PR_SET_PDEATHSIG (see ``_linux_set_pdeathsig``) so
# it is cleaned up if ludvart dies. On Linux that signal fires when the *thread*
# that forked the child terminates -- not the parent process. If we forked from
# a short-lived worker thread (e.g. the ``/model use`` action thread) the gateway
# would be killed the instant that thread finished, right after verification. To
# avoid that, all gateway forks are funnelled through one persistent daemon
# thread that lives for the whole process, so the death signal only fires at
# interpreter shutdown (the intended orphan cleanup).
_spawn_lock = threading.Lock()
_spawn_thread: "threading.Thread | None" = None
_spawn_requests: "queue.Queue" = queue.Queue()


def _spawner_loop() -> None:
    while True:
        func, box, done = _spawn_requests.get()
        try:
            box[0] = func()
        except BaseException as exc:  # relayed to the caller
            box[1] = exc
        finally:
            done.set()


def _spawn_process(func):
    """Run ``func`` (which forks a subprocess) on the persistent spawner thread.

    Returns ``func``'s result or re-raises its exception on the caller's thread.
    """
    global _spawn_thread
    with _spawn_lock:
        if _spawn_thread is None:
            _spawn_thread = threading.Thread(
                target=_spawner_loop, name="ludvart-gateway-spawner", daemon=True
            )
            _spawn_thread.start()
    box: list = [None, None]
    done = threading.Event()
    _spawn_requests.put((func, box, done))
    done.wait()
    if box[1] is not None:
        raise box[1]
    return box[0]


class CopilotGateway:
    """A local ``litellm`` proxy fronting GitHub Copilot for ludvart.

    Spawn with :meth:`start`, point an OpenAI-compatible client at
    :attr:`base_url` using model :attr:`litellm_model`, and call :meth:`stop`
    (idempotent) on shutdown.
    """

    def __init__(
        self,
        model: str,
        *,
        api_mode: str = "chat",
        host: str = GATEWAY_HOST,
        port: int | None = None,
        log_path: str | None = None,
    ) -> None:
        self.model = model  # e.g. "gpt-4o" (no provider prefix)
        self.api_mode = api_mode
        self.host = host
        self.port = port or _free_port(host)
        self.log_path = log_path or os.path.join(
            os.path.expanduser("~"), ".ludvart", "copilot-gateway.log"
        )
        self._proc: subprocess.Popen | None = None
        self._config_path: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def litellm_model(self) -> str:
        """The model id the proxy exposes (and clients must request)."""
        return f"github_copilot/{self.model}"

    def start(self, *, timeout: float = 90.0) -> None:
        """Spawn the proxy and block until it is serving (or raise)."""
        cli = _litellm_cli()
        if cli is None:
            raise GatewayError(
                "the 'litellm' command is not installed "
                "(uv pip install 'litellm[proxy]')"
            )
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        cmd = [
            cli,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.api_mode == "responses":
            # Copilot's Responses-only models require this explicit proxy mode;
            # ``--model`` alone configures a Chat Completions deployment.
            self._config_path = os.path.join(
                os.path.dirname(self.log_path), f"copilot-responses-{self.port}.json"
            )
            config = {
                "model_list": [
                    {
                        "model_name": self.litellm_model,
                        "model_info": {"mode": "responses"},
                        "litellm_params": {"model": self.litellm_model},
                    }
                ]
            }
            with open(self._config_path, "w", encoding="utf-8") as config_file:
                json.dump(config, config_file)
            cmd += ["--config", self._config_path]
        else:
            cmd += ["--model", self.litellm_model]
        # Detach into its own process group so we can tear down the whole tree,
        # and route its noisy output to a log file (never the terminal, which
        # ludvart composites at runtime). The fork itself happens on the shared
        # spawner thread so PR_SET_PDEATHSIG is anchored to a long-lived task
        # (see ``_spawn_process``), not a transient worker thread.
        with open(self.log_path, "ab", buffering=0) as logf:
            self._proc = _spawn_process(
                lambda: subprocess.Popen(
                    cmd,
                    stdout=logf,
                    stderr=logf,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    preexec_fn=(
                        _linux_set_pdeathsig
                        if sys.platform.startswith("linux")
                        else None
                    ),
                )
            )
        self._await_ready(timeout)

    def _await_ready(self, timeout: float) -> None:
        assert self._proc is not None
        url = f"{self.base_url}/health/liveliness"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise GatewayError(
                    f"the LiteLLM gateway exited early "
                    f"(code {self._proc.returncode}); see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except urllib.error.HTTPError:
                # The app answered (any HTTP status means it is serving).
                return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        self.stop()
        raise GatewayError(
            f"the LiteLLM gateway did not become ready within "
            f"{timeout:.0f}s; see {self.log_path}"
        )

    def stop(self) -> None:
        """Terminate the proxy (and its workers). Safe to call more than once."""
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except OSError:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                with contextlib.suppress(Exception):
                    proc.wait(timeout=5)
        if self._config_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(self._config_path)
            self._config_path = None
