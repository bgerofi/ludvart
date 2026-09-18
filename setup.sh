#!/usr/bin/env bash
# Set up the ludvart development environment using uv.
# Creates a local .venv and installs ludvart (editable) plus its dependencies,
# then verifies the install so a half-configured environment fails loudly.
set -euo pipefail

cd "$(dirname "$0")"

# The Python ludvart is built and run on. Override to test another version.
PYTHON_VERSION="${LUDVART_PYTHON_VERSION:-3.12}"

# Ensure uv is available. If it isn't on PATH, bootstrap a project-local copy
# under ./.uv using the official installer instead of asking the user to do it.
if ! command -v uv >/dev/null 2>&1; then
    echo "==> 'uv' not found; installing a project-local copy into ./.uv"
    export UV_INSTALL_DIR="$PWD/.uv"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
    else
        echo "error: need 'curl' or 'wget' to download uv." >&2
        exit 1
    fi
    export PATH="$UV_INSTALL_DIR:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is still not available after attempting to install it." >&2
    exit 1
fi

echo "==> Installing CPython $PYTHON_VERSION with uv"
# Ludvart runs on an interpreter uv manages, never the host's: a system python
# can be too old, patched oddly, or replaced under us by a distro update.
# --managed-python makes uv download one instead of adopting what is on PATH.
uv python install --managed-python "$PYTHON_VERSION"
# --system --no-project so an existing .venv is not what gets reported back:
# we are asking where the managed interpreter lives, not what is in use here.
MANAGED_PY="$(uv python find --managed-python --system --no-project "$PYTHON_VERSION")"
echo "    using $MANAGED_PY"

echo "==> Creating virtual environment (.venv)"
# Reuse an existing .venv (uv would otherwise prompt interactively, which hangs
# non-interactive runs), but only if it is built on the interpreter above; one
# left over from a host python would defeat the point.
if [[ -d .venv ]]; then
    if [[ "$(readlink -f .venv/bin/python 2>/dev/null || true)" \
          == "$(readlink -f "$MANAGED_PY")" ]]; then
        echo "    .venv already uses the managed interpreter; reusing it."
    else
        echo "    .venv is built on a different interpreter; recreating it."
        rm -rf .venv
    fi
fi
if [[ ! -d .venv ]]; then
    uv venv --managed-python --python "$MANAGED_PY"
fi

# Point uv (and our verification below) unambiguously at this .venv, regardless
# of any other environment that happens to be active. Without this, an inactive
# venv can cause 'uv pip install' to target the wrong interpreter, leaving the
# ludvart module and launcher missing (the "No module named ludvart" symptom).
export VIRTUAL_ENV="$PWD/.venv"
VENV_PY="$VIRTUAL_ENV/bin/python"

echo "==> Installing ludvart (editable) with dev tools and the Copilot gateway"
uv pip install --python "$VENV_PY" -e ".[dev,copilot]"

echo "==> Verifying installation"
if ! "$VENV_PY" -c "import ludvart, ludvart.__main__" >/dev/null 2>&1; then
    echo "error: ludvart did not install correctly (cannot import 'ludvart')." >&2
    echo "Try re-running ./setup.sh; if it persists, check the output of:" >&2
    echo "    uv pip install --python \"$VENV_PY\" -e ." >&2
    exit 1
fi
if [[ ! -x "$VIRTUAL_ENV/bin/ludvart" ]]; then
    echo "error: the 'ludvart' launcher was not created in .venv/bin." >&2
    echo "Confirm pyproject.toml has a [project.scripts] entry for ludvart." >&2
    exit 1
fi

echo
echo "Done. Activate with:"
echo "    source .venv/bin/activate"
echo "Then run:"
echo "    ludvart            # spawns your \$SHELL"
echo "    ludvart -- htop    # spawns any command"
echo "Run the tests with:"
echo "    pytest -m 'not e2e'   # fast unit tests only"
echo "    pytest                # also runs e2e (needs a configured LLM; real tokens)"
