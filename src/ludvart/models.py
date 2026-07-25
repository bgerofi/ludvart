"""Registry of user-registered model endpoints (``~/.ludvart/models.json``).

ludvart can juggle several models at once and let the user switch between them
at runtime. Every registered endpoint lives in ``~/.ludvart/models.json`` as a
JSON *array* of registrations; each registration carries the same essentials the
old single-provider ``llm.conf`` did, plus which one is currently active:

    [
      {
        "provider": "openai",         # openai | anthropic | google | custom | copilot
        "api_url": "https://api.openai.com/v1",
        "api_key": "sk-...",          # empty for copilot (the gateway supplies it)
        "model": "gpt-4o",
        "context_window": 0,           # 0 = auto-detect / use the fallback table
        "api_mode": "chat",           # chat | responses (Copilot gateway wire API)
        "active": true                 # exactly one entry should be active
      },
      ...
    ]

The file holds API keys, so it is created (with ``~/.ludvart``) using owner-only
``0600`` permissions.

``models.json`` fully supersedes ``llm.conf``: the first time it is needed it is
seeded once from any existing ``llm.conf`` / environment configuration (see
:func:`load_registry`), and from then on it is the sole source of truth for which
models exist. Request tuning (``LUDVART_LLM_TIMEOUT`` / ``LUDVART_LLM_MAX_RETRIES``)
is still read from the environment / ``llm.conf`` by the client layer.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .llm import ProviderConfig, copilot_model, resolve_config

#: A single model registration (see the module docstring for the fields).
Registration = dict[str, Any]

#: Providers that may appear in a registration. ``copilot`` is special: it has no
#: stored URL/key and is served through the local LiteLLM gateway at use time.
VALID_PROVIDERS = ("openai", "anthropic", "google", "custom", "copilot")

#: Provider metadata for the registration menu, shared by the CLI setup wizard
#: and the in-panel ``/model add`` flow: ``(name, label, default_url)``. An empty
#: default URL means one must be entered; ``copilot`` has no URL (the local
#: gateway supplies it).
PROVIDER_MENU: tuple[tuple[str, str, str], ...] = (
    ("openai", "OpenAI", "https://api.openai.com/v1"),
    ("anthropic", "Anthropic", "https://api.anthropic.com"),
    ("google", "Google", "https://generativelanguage.googleapis.com"),
    ("custom", "Custom (OpenAI-compatible)", ""),
    ("copilot", "GitHub Copilot (via local LiteLLM gateway)", ""),
)


def models_path() -> str:
    """Path of the model registry file.

    Honors ``LUDVART_MODELS_FILE`` (used by tests and power users), otherwise
    ``~/.ludvart/models.json``.
    """
    override = os.environ.get("LUDVART_MODELS_FILE")
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".ludvart", "models.json")


def is_copilot(reg: Registration) -> bool:
    """Whether ``reg`` is a GitHub Copilot registration (needs the gateway)."""
    return reg.get("provider") == "copilot"


def label(reg: Registration) -> str:
    """A short human-readable id for a registration.

    ``service:provider:model`` when a service name is set (the arbitrary access
    provider entered at registration), else ``provider:model`` for backward
    compatibility with registrations saved before the field existed.
    """
    provider = reg.get("provider", "?")
    model = reg.get("model", "?")
    service = reg.get("service") or ""
    if service:
        return f"{service}:{provider}:{model}"
    return f"{provider}:{model}"


def _coerce(raw: Any) -> Registration | None:
    """Normalize one loaded entry into a registration, or ``None`` if unusable."""
    if not isinstance(raw, dict):
        return None
    provider = raw.get("provider")
    model = raw.get("model")
    if provider not in VALID_PROVIDERS or not isinstance(model, str) or not model:
        return None
    try:
        ctx = int(raw.get("context_window") or 0)
    except (TypeError, ValueError):
        ctx = 0
    api_mode = raw.get("api_mode")
    if api_mode not in ("chat", "responses"):
        api_mode = "chat"
    return {
        "provider": provider,
        "service": str(raw.get("service") or ""),
        "api_url": str(raw.get("api_url") or ""),
        "api_key": str(raw.get("api_key") or ""),
        "model": model,
        "context_window": ctx,
        "api_mode": api_mode,
        "active": bool(raw.get("active")),
    }


def load_models(path: str | None = None) -> list[Registration]:
    """Read and validate the registry file (empty list when absent/invalid)."""
    if path is None:
        path = models_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out: list[Registration] = []
    for raw in data:
        reg = _coerce(raw)
        if reg is not None:
            out.append(reg)
    return _normalize_active(out)


def save_models(models: list[Registration], path: str | None = None) -> str:
    """Write the registry to ``path`` (default ``models.json``) at ``0600``.

    Exactly one entry is kept active (see :func:`_normalize_active`). Returns the
    path written.
    """
    if path is None:
        path = models_path()
    normalized = _normalize_active(models)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(normalized, fh, indent=2)
        fh.write("\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def _normalize_active(models: list[Registration]) -> list[Registration]:
    """Return ``models`` with exactly one active entry (the first if none/many).

    Keeps the first entry already marked active; if none is marked and the list
    is non-empty, the first entry becomes active. Does not mutate the inputs.
    """
    out = [dict(m) for m in models]
    active_idx = next((i for i, m in enumerate(out) if m.get("active")), None)
    if active_idx is None and out:
        active_idx = 0
    for i, m in enumerate(out):
        m["active"] = i == active_idx
    return out


def active_index(models: list[Registration]) -> int | None:
    """Index of the active registration, or ``None`` when the list is empty."""
    if not models:
        return None
    for i, m in enumerate(models):
        if m.get("active"):
            return i
    return 0


def active_registration(models: list[Registration]) -> Registration | None:
    """The active registration, or ``None`` when the list is empty."""
    idx = active_index(models)
    return models[idx] if idx is not None else None


def set_active(models: list[Registration], index: int) -> list[Registration]:
    """Return a copy of ``models`` with entry ``index`` marked active."""
    if not 0 <= index < len(models):
        raise IndexError(index)
    out = [dict(m) for m in models]
    for i, m in enumerate(out):
        m["active"] = i == index
    return out


def add_registration(
    models: list[Registration], reg: Registration, make_active: bool = True
) -> list[Registration]:
    """Append ``reg`` to ``models``; make it active when ``make_active`` (or first)."""
    out = [dict(m) for m in models]
    new = _coerce(reg)
    if new is None:
        raise ValueError(f"invalid registration: {reg!r}")
    out.append(new)
    if make_active or len(out) == 1:
        out = set_active(out, len(out) - 1)
    return _normalize_active(out)


def remove_registration(models: list[Registration], index: int) -> list[Registration]:
    """Return a copy of ``models`` without entry ``index`` (active repointed)."""
    if not 0 <= index < len(models):
        raise IndexError(index)
    out = [dict(m) for i, m in enumerate(models) if i != index]
    return _normalize_active(out)


def find_registration(models: list[Registration], token: str) -> int | None:
    """Resolve a user-supplied ``token`` to a registration index, or ``None``.

    A token is either a 1-based position as shown by ``/model list``, or a
    case-insensitive substring of the model id (unique match required).
    """
    token = token.strip()
    if token.isdigit():
        pos = int(token) - 1
        return pos if 0 <= pos < len(models) else None
    needle = token.lower()
    matches = [i for i, m in enumerate(models) if needle in m.get("model", "").lower()]
    return matches[0] if len(matches) == 1 else None


def registration_to_config(reg: Registration) -> ProviderConfig:
    """Build a :class:`ProviderConfig` for a direct-provider registration.

    Not valid for GitHub Copilot registrations, whose endpoint/key are supplied
    by the local gateway at use time; check :func:`is_copilot` first.
    """
    if is_copilot(reg):
        raise ValueError("copilot registrations are configured via the gateway")
    return ProviderConfig(
        name=reg["provider"],
        api_url=str(reg.get("api_url") or "").rstrip("/"),
        api_key=str(reg.get("api_key") or ""),
        model=reg["model"],
        context_window=int(reg.get("context_window") or 0),
        api_mode=str(reg.get("api_mode") or "chat"),
    )


def migrate_from_conf() -> list[Registration]:
    """Build an initial registry from existing ``llm.conf`` / env configuration.

    Returns a (possibly empty) list of registrations: the direct provider
    selected by :func:`ludvart.llm.resolve_config` (if any) and a GitHub Copilot
    entry when ``COPILOT_MODEL`` is set. The direct provider is preferred as the
    active one, matching the pre-registry startup precedence.
    """
    out: list[Registration] = []
    cfg = resolve_config()
    if cfg is not None:
        out.append(
            {
                "provider": cfg.name,
                "api_url": cfg.api_url,
                "api_key": cfg.api_key,
                "model": cfg.model,
                "context_window": cfg.context_window,
                "active": True,
            }
        )
    cop = copilot_model()
    if cop:
        out.append(
            {
                "provider": "copilot",
                "api_url": "",
                "api_key": "",
                "model": cop,
                "context_window": 0,
                "active": not out,
            }
        )
    return _normalize_active(out)


def load_registry(path: str | None = None) -> list[Registration]:
    """Load the registry, seeding it once from ``llm.conf`` / env if absent.

    If ``models.json`` exists it is the sole source of truth. Otherwise the
    legacy ``llm.conf`` / environment configuration is migrated into a fresh
    ``models.json`` (written only when it yields at least one model) and
    returned. From then on ``llm.conf`` is ignored for model selection.
    """
    if path is None:
        path = models_path()
    if os.path.exists(path):
        return load_models(path)
    migrated = migrate_from_conf()
    if migrated:
        save_models(migrated, path)
    return migrated
