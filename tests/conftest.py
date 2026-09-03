"""Pytest configuration and shared fixtures for the ludvart test suite.

Two kinds of test files live under ``tests/``:

* **Unit tests** expose ``test_*`` functions and are collected by pytest in the
  usual way. A couple of them expect a ``tmp`` directory or a ``monkeypatch_cli``
  helper, which are provided as fixtures below.
* **End-to-end tests** expose a single ``main()`` that forks a real ``ludvart``
  process over a PTY. These are collected here as one item each and marked
  ``e2e``. Because a keyless ``ludvart`` drops into an interactive setup wizard
  (which would hang the fork), they are skipped automatically unless an LLM
  provider is configured.

End-to-end tests talk to a real model and cost real money, so they do *not*
inherit whatever expensive model happens to be active in the registry; see
:data:`DEFAULT_E2E_MODEL`.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import pty
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

from e2e_util import isolate_sessions

# Applies to every test, not just the e2e ones: whatever a test saves must not
# end up in the developer's real ~/.ludvart/sessions.
isolate_sessions()


#: Model the e2e suite asks for by default. These tests exercise ludvart's own
#: plumbing (tool calls, injection, settling), not the model's intelligence, so
#: a cheap fast model is the right default -- running them on a frontier model
#: costs a lot for no extra signal. Overridden with ``LUDVART_E2E_MODEL``; set
#: that to an empty string to just use whichever model is active.
DEFAULT_E2E_MODEL = "gpt-5.6-terra"

#: Chooses the e2e model. Accepts anything ``/model use`` accepts: a 1-based
#: position in the registry or a unique substring of the model id.
E2E_MODEL_ENV = "LUDVART_E2E_MODEL"


# --------------------------------------------------------------------------- #
# Fixtures used by the unit tests.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _real_std_streams(monkeypatch: pytest.MonkeyPatch):
    """Give tests real ``sys.stdin``/``stdout`` objects that expose ``fileno()``.

    ``Ludvart.__init__`` records ``sys.stdin.fileno()`` and
    ``sys.stdout.fileno()``. Under pytest's output capture those are replaced by
    pseudo-files without a ``fileno()``. Restoring the original stream objects
    (``sys.__stdin__`` etc.) keeps ``fileno()`` working while pytest still
    captures the output at the file-descriptor level.
    """
    for name in ("stdin", "stdout", "stderr"):
        original = getattr(sys, f"__{name}__", None)
        if original is not None and hasattr(original, "fileno"):
            try:
                original.fileno()
            except (OSError, ValueError):
                continue
            monkeypatch.setattr(sys, name, original)
    yield


@pytest.fixture(autouse=True)
def _isolate_registry_env():
    """Undo any ``LUDVART_MODELS_FILE`` a test points at its own temp registry.

    Several tests redirect the registry so they never touch the developer's real
    ``~/.ludvart/models.json``, but they set the variable in place. Restoring it
    afterwards keeps that redirect from leaking into unrelated tests (whose
    outcome would then depend on the run order).
    """
    sentinel = object()
    previous = os.environ.get("LUDVART_MODELS_FILE", sentinel)
    yield
    if previous is sentinel:
        os.environ.pop("LUDVART_MODELS_FILE", None)
    else:
        os.environ["LUDVART_MODELS_FILE"] = previous


@pytest.fixture
def tmp(tmp_path: Path) -> str:
    """A throwaway directory as a plain string path.

    Several config/gateway tests were written against a ``tempfile`` directory
    and join paths onto it, so they want a ``str`` rather than a ``Path``.
    """
    return str(tmp_path)


@pytest.fixture
def monkeypatch_cli(monkeypatch: pytest.MonkeyPatch):
    """Return a helper that points the gateway at a fake ``litellm`` CLI.

    Passing ``None`` simulates the CLI being missing. The original resolver is
    restored automatically at the end of the test by ``monkeypatch``.
    """
    from ludvart import gateway

    def _set(path: str | None) -> None:
        monkeypatch.setattr(gateway, "_litellm_cli", lambda: path)

    return _set


# --------------------------------------------------------------------------- #
# Collection of the ``main()``-style end-to-end scripts.
# --------------------------------------------------------------------------- #
def _llm_configured() -> bool:
    """True when a model is registered (or a legacy env/llm.conf provider is).

    The registry in ``~/.ludvart/models.json`` is what a backend actually loads,
    so it is checked first; without this the whole e2e suite silently skips on a
    machine that is perfectly well configured.
    """
    try:
        from ludvart.models import load_registry
    except Exception:
        return False
    try:
        if load_registry():
            return True
    except Exception:
        pass
    try:
        from ludvart.llm import copilot_model, create_client
    except Exception:
        return False
    try:
        create_client()
        return True
    except Exception:
        pass
    try:
        return bool(copilot_model())
    except Exception:
        return False


def _e2e_registry(token: str) -> tuple[str | None, str]:
    """Write a one-model registry for the e2e run; return (path, note).

    The e2e children are separate processes, so the model is selected the only
    way that reaches them: a private ``models.json`` pointed at by
    ``LUDVART_MODELS_FILE``. The real registry is never written to.

    Only the selected registration is copied over. Every ludvart startup
    verifies the models it does not activate, which for a direct provider means
    a live request -- pointless here, and paid for once per e2e test.
    """
    from ludvart.models import find_registration, label, load_registry

    models = load_registry()
    if not models:
        return None, ""
    index = find_registration(models, token)
    if index is None:
        return None, f"no registered model matches {token!r}; using the active one"
    chosen = dict(models[index])
    chosen["active"] = True
    path = os.path.join(tempfile.mkdtemp(prefix="ludvart_e2e_"), "models.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([chosen], fh)
    return path, f"e2e model: {label(chosen)}"


class _E2ESession:
    """Process-wide state shared by every e2e test: the model and the gateway.

    Each e2e test forks its own short-lived ``ludvart``. Left alone, every one
    of them boots its own LiteLLM gateway, which costs tens of seconds and
    dominates the runtime of the suite. The gateway is stateless and proxies a
    single model, and all the e2e tests use the same model, so exactly one is
    started here and handed to the children through the environment.
    """

    def __init__(self) -> None:
        self._gateway = None
        self._env: dict[str, str | None] = {}
        self._applied: dict[str, str] = {}

    def _setenv(self, name: str, value: str) -> None:
        self._env.setdefault(name, os.environ.get(name))
        self._applied[name] = value
        os.environ[name] = value

    def apply(self) -> None:
        """Re-export the session's environment before each e2e test.

        Several unit tests repoint ``LUDVART_MODELS_FILE`` at a throwaway
        registry and never put it back. Without this, every e2e test collected
        after one of them would decide no provider is configured and skip
        itself -- silently, and only in a full-suite run.
        """
        for name, value in self._applied.items():
            os.environ[name] = value

    def start(self) -> None:
        from ludvart.models import active_registration, is_copilot, load_registry

        _restore_pristine_registry()
        token = os.environ.get(E2E_MODEL_ENV, DEFAULT_E2E_MODEL).strip()
        if token:
            path, note = _e2e_registry(token)
            if note:
                print(note)
            if path is not None:
                self._setenv("LUDVART_MODELS_FILE", path)
        reg = active_registration(load_registry())
        if reg is None or not is_copilot(reg):
            return  # only Copilot models go through a gateway
        from ludvart.gateway import (
            SHARED_GATEWAY_MODEL_ENV,
            SHARED_GATEWAY_URL_ENV,
            CopilotGateway,
        )

        gateway = CopilotGateway(
            reg["model"], api_mode=str(reg.get("api_mode") or "chat")
        )
        print(f"starting one shared gateway for {reg['model']}...")
        gateway.start()
        self._gateway = gateway
        self._setenv(SHARED_GATEWAY_URL_ENV, gateway.base_url)
        self._setenv(SHARED_GATEWAY_MODEL_ENV, reg["model"])

    def stop(self) -> None:
        if self._gateway is not None:
            self._gateway.stop()
            self._gateway = None
        for name, previous in self._env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        self._env.clear()
        self._applied.clear()


#: ``LUDVART_MODELS_FILE`` as it stood before any test ran, so the e2e model can
#: always be resolved against the developer's real registry.
_PRISTINE_MODELS_FILE = os.environ.get("LUDVART_MODELS_FILE")


def _restore_pristine_registry() -> None:
    if _PRISTINE_MODELS_FILE is None:
        os.environ.pop("LUDVART_MODELS_FILE", None)
    else:
        os.environ["LUDVART_MODELS_FILE"] = _PRISTINE_MODELS_FILE


#: Created on the first e2e test, torn down at the end of the session.
_e2e_session: _E2ESession | None = None


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    global _e2e_session
    if _e2e_session is not None:
        _e2e_session.stop()
        _e2e_session = None


def _reap(pid: int) -> None:
    """Make sure a forked e2e child (and its process group) is really gone.

    The e2e scripts drive a ``pty.fork``ed ludvart and return without killing
    it, so without this each finished test leaves a live ludvart behind.
    """
    import signal

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            break
        for _ in range(20):
            try:
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
                    return
            except ChildProcessError:
                return
            time.sleep(0.05)


def _top_level_functions(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def pytest_collect_file(file_path: Path, parent):
    """Collect ``test_*.py`` files that only expose ``main()`` as e2e items."""
    if file_path.suffix != ".py" or not file_path.name.startswith("test_"):
        return None
    funcs = _top_level_functions(file_path)
    has_unit_tests = any(name.startswith("test_") for name in funcs)
    if has_unit_tests or "main" not in funcs:
        # Let pytest's normal Python collection handle real ``test_*`` funcs.
        return None
    return _MainScriptFile.from_parent(parent, path=file_path)


class _MainScriptFile(pytest.File):
    def collect(self):
        yield _MainScriptItem.from_parent(self, name="main")


class _MainScriptItem(pytest.Item):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_marker(pytest.mark.e2e)

    def setup(self) -> None:
        global _e2e_session
        if _e2e_session is not None:
            _e2e_session.apply()
            return
        _restore_pristine_registry()
        if not _llm_configured():
            pytest.skip(
                "no LLM provider configured; e2e forks a real ludvart which "
                "would otherwise block on the interactive setup wizard"
            )
        _e2e_session = _E2ESession()
        _e2e_session.start()

    def runtest(self) -> None:
        spec = importlib.util.spec_from_file_location(self.path.stem, self.path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        children: list[int] = []
        real_fork = pty.fork

        def recording_fork():
            pid, fd = real_fork()
            if pid != 0:
                children.append(pid)
            return pid, fd

        # The forked ludvart runs a shell that inherits this directory, and the
        # scripts make it create files. Give each one its own so they cannot
        # collide with each other, with a parallel worker, or with the repo.
        origin = os.getcwd()
        workdir = tempfile.mkdtemp(prefix="ludvart_e2e_cwd_")
        os.chdir(workdir)
        pty.fork = recording_fork
        try:
            module.main()
        finally:
            pty.fork = real_fork
            for pid in children:
                _reap(pid)
            os.chdir(origin)
            shutil.rmtree(workdir, ignore_errors=True)

    def reportinfo(self):
        return self.path, 0, f"e2e: {self.path.stem}"
