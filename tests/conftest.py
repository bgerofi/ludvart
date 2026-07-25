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
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest


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
        if not _llm_configured():
            pytest.skip(
                "no LLM provider configured; e2e forks a real ludvart which "
                "would otherwise block on the interactive setup wizard"
            )

    def runtest(self) -> None:
        spec = importlib.util.spec_from_file_location(self.path.stem, self.path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()

    def reportinfo(self):
        return self.path, 0, f"e2e: {self.path.stem}"
