"""Shared helpers for the ``main()``-style end-to-end scripts.

The e2e scripts fork a real ``ludvart`` over a PTY. The agent loop always runs
in a backend process, and by default ``ludvart`` forks one over a plain
stdin/stdout pipe -- no SSH, no network, no assumption that localhost is
reachable. That pipe carries exactly the same framed protocol an SSH backend
would, so the scripts need no flag at all for the common case.

Set ``LUDVART_E2E_BACKEND=host:folder`` to run the same scripts against an SSH
backend instead; the only difference is how the backend is spawned.
"""

import os

E2E_BACKEND_ENV = "LUDVART_E2E_BACKEND"

DEFAULT_COMMAND = ["bash", "--norc", "-i"]


def e2e_backend() -> str:
    """An explicit backend spec for this run, or '' to use ludvart's default."""
    return os.environ.get(E2E_BACKEND_ENV, "").strip()


def ludvart_argv(*flags: str, command=None) -> list[str]:
    """Argv for forking ``ludvart``, honouring an explicit e2e backend spec.

    ``flags`` are ludvart's own options; ``command`` (default an interactive
    bash) is what ludvart should run. ``--no-llm`` wins over the backend spec:
    that relay-only mode has no agent loop to place anywhere.
    """
    argv = ["ludvart"]
    spec = e2e_backend()
    if spec and "--no-llm" not in flags:
        argv += ["--backend", spec]
    argv += [*flags, "--", *(DEFAULT_COMMAND if command is None else command)]
    return argv
