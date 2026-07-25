#!/usr/bin/env bash
#
# Run the ludvart test suite from the project's virtualenv.
#
# Activates the local ".venv" (created by ./setup.sh) and invokes pytest. By
# default every test runs; pass --no-e2e to skip the end-to-end tests that fork
# a real ludvart and require a configured LLM provider (live LLM interaction).
#
# The e2e tests cost real money, so they run on a cheap model rather than on
# whatever is active in the registry; see LUDVART_E2E_MODEL below.
#
# Usage:
#   tests/run.sh [--no-e2e] [extra pytest args...]
#
# Examples:
#   tests/run.sh                     # run everything
#   tests/run.sh --no-e2e            # skip the live-LLM e2e tests
#   tests/run.sh --no-e2e -q         # ... quietly
#   tests/run.sh -k test_session     # forward any pytest args
#
set -euo pipefail

usage() {
    cat <<'EOF'
Run the ludvart test suite from the project's virtualenv.

Usage:
  tests/run.sh [--no-e2e] [extra pytest args...]

Options:
  --no-e2e, --exclude-e2e, --skip-e2e
                    Skip the end-to-end tests that fork a real ludvart and
                    require a configured LLM provider (live LLM interaction).
  -h, --help        Show this help and exit.

Any other arguments are forwarded to pytest unchanged.

Environment:
  LUDVART_E2E_MODEL
                    Which registered model the e2e tests should run on, given
                    as a 1-based /model list position or a unique substring of
                    the model id. Defaults to a cheap model (these tests check
                    ludvart's plumbing, not the model's intelligence, so paying
                    for a frontier model buys no extra signal). Set it to the
                    empty string to use whichever model is currently active.

Examples:
  tests/run.sh                                 # run everything
  tests/run.sh --no-e2e                        # skip the live-LLM e2e tests
  tests/run.sh --no-e2e -q                     # ... quietly

  # Run individual tests (standard pytest selection):
  tests/run.sh tests/test_session_store.py     # a whole file
  tests/run.sh tests/test_session_store.py::test_roundtrip   # one function
  tests/run.sh -k session                      # every test named *session*
  tests/run.sh -k "use and copilot"            # boolean -k expression
  tests/run.sh --no-e2e -k model -v            # skip e2e, only "model", verbose
  tests/run.sh tests/test_ai_paste_e2e.py::main  # a single e2e script (needs LLM)

  tests/run.sh --co -q                         # list test node ids (collect-only)
EOF
}

# Resolve the project root (the parent of this script's directory) so the
# script works regardless of the current working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Parse our own flags in a single pass; forward everything else to pytest.
PYTEST_ARGS=()
EXCLUDE_E2E=0
for arg in "$@"; do
    case "${arg}" in
        -h|--help)
            usage
            exit 0
            ;;
        --no-e2e|--exclude-e2e|--skip-e2e)
            EXCLUDE_E2E=1
            ;;
        *)
            PYTEST_ARGS+=("${arg}")
            ;;
    esac
done

if [[ "${EXCLUDE_E2E}" -eq 1 ]]; then
    PYTEST_ARGS+=(-m "not e2e")
fi

VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
    echo "error: virtualenv not found at ${PROJECT_ROOT}/.venv" >&2
    echo "       run ./setup.sh first to create it." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"

cd -- "${PROJECT_ROOT}"
exec python -m pytest "${PYTEST_ARGS[@]}"
