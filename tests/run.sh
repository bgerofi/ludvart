#!/usr/bin/env bash
#
# Run the ludvart test suite from the project's virtualenv.
#
# Activates the local ".venv" (created by ./setup.sh) and invokes pytest. By
# default every test runs, spread over several worker processes; pass --no-e2e
# to skip the end-to-end tests that fork a real ludvart and require a configured
# LLM provider (live LLM interaction), or --serial to run in one process.
#
# The e2e tests cost real money, so they run on a cheap model rather than on
# whatever is active in the registry; see LUDVART_E2E_MODEL below.
#
# Usage:
#   tests/run.sh [--no-e2e] [--serial] [extra pytest args...]
#
# Examples:
#   tests/run.sh                     # run everything
#   tests/run.sh --no-e2e            # skip the live-LLM e2e tests
#   tests/run.sh --no-e2e -q         # ... quietly
#   tests/run.sh --serial            # one process, in order
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
  --serial, --no-parallel
                    Run in one process. The suite is otherwise spread over
                    several workers, which is worth roughly 5x on wall clock;
                    use this when a failure is easier to read in order, or when
                    debugging with -s (which the workers cannot support).
  -h, --help        Show this help and exit.

Any other arguments are forwarded to pytest unchanged. Passing your own -n
wins over the default parallelism.

Environment:
  LUDVART_TEST_JOBS
                    How many workers to spread the suite over (default 8).
                    Past that the wall clock is bounded by the slowest single
                    e2e script, so more workers buy little.
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
SERIAL=0
HAVE_N=0
for arg in "$@"; do
    case "${arg}" in
        -h|--help)
            usage
            exit 0
            ;;
        --no-e2e|--exclude-e2e|--skip-e2e)
            EXCLUDE_E2E=1
            ;;
        --serial|--no-parallel)
            SERIAL=1
            ;;
        -n|--numprocesses|-n*|--numprocesses=*)
            HAVE_N=1
            PYTEST_ARGS+=("${arg}")
            ;;
        *)
            PYTEST_ARGS+=("${arg}")
            ;;
    esac
done

if [[ "${EXCLUDE_E2E}" -eq 1 ]]; then
    PYTEST_ARGS+=(-m "not e2e")
fi

# Most of the wall clock is e2e tests waiting on a live model, so they overlap
# almost perfectly. "loadfile" keeps a file's tests on one worker, which is what
# the e2e scripts want: each forks a ludvart and owns the process-wide gateway.
if [[ "${SERIAL}" -eq 0 && "${HAVE_N}" -eq 0 ]]; then
    PYTEST_ARGS+=(-n "${LUDVART_TEST_JOBS:-8}" --dist loadfile)
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
