"""Agent profiles: background the model is given up front for a kind of work.

A profile is a markdown file under ``~/.ludvart/profiles/`` describing the
domain the agent is about to work in -- how to investigate a particular system,
what the conventions are, where things live. The active profile's text is read
fresh for every model request, so editing the file is felt immediately, in the
middle of a running conversation.

Which profiles are registered, and which one is active, live in
``~/.ludvart/profiles/config.json``::

    {"version": 1, "profiles": [{"name": "...", "file": "x.md", "active": true}]}

Unlike the model registry, "no profile" is a legal state: a profile is opt-in
and can be switched off again.

This module has no terminal/UI dependencies so it can be unit tested directly.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

CONFIG_NAME = "config.json"
CONFIG_VERSION = 1

#: A profile file is a bare markdown filename inside the profiles directory.
#: Anything with a path separator, a ``..`` or another suffix is refused, so a
#: registration can never name a file outside the directory.
_FILENAME_RE = re.compile(r"[^/\\]+\.md")

#: Rough characters-per-token, the same estimate the compaction trigger uses.
#: Good enough to tell a 2k profile from a 200k one, which is what it is for.
_CHARS_PER_TOKEN = 4

#: A profile above this costs real money and context on every single request,
#: so the listing says so instead of letting it go unnoticed.
LARGE_PROFILE_TOKENS = 50_000

Profile = dict[str, Any]


def profiles_dir() -> Path:
    """Directory holding the profile files and their config.

    Honours ``LUDVART_PROFILES_DIR`` (used by tests) and otherwise defaults to
    ``~/.ludvart/profiles``.
    """
    override = os.environ.get("LUDVART_PROFILES_DIR")
    if override:
        return Path(override)
    return Path(os.path.expanduser("~/.ludvart/profiles"))


def config_path() -> Path:
    """Path of the profile registry file."""
    return profiles_dir() / CONFIG_NAME


def valid_filename(filename: str) -> bool:
    """Whether ``filename`` is a bare ``*.md`` name inside the profiles dir."""
    return bool(filename) and bool(_FILENAME_RE.fullmatch(filename))


def profile_path(filename: str) -> Path | None:
    """Resolve a profile filename to its path, or ``None`` if it is not one."""
    if not valid_filename(filename):
        return None
    return profiles_dir() / filename


def _coerce(raw: Any) -> Profile | None:
    """Normalize one loaded entry into a profile, or ``None`` if unusable."""
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    filename = str(raw.get("file") or "").strip()
    if not name or not valid_filename(filename):
        return None
    return {"name": name, "file": filename, "active": bool(raw.get("active"))}


def _normalize(profiles: list[Profile]) -> list[Profile]:
    """Return a copy of ``profiles`` with at most one entry active."""
    out = [dict(p) for p in profiles]
    active = next((i for i, p in enumerate(out) if p.get("active")), None)
    for i, p in enumerate(out):
        p["active"] = i == active
    return out


def load_profiles(path: Path | str | None = None) -> list[Profile]:
    """Read the registry (empty list when absent, unreadable or malformed)."""
    conf = Path(path) if path is not None else config_path()
    try:
        data = json.loads(conf.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    raw = data.get("profiles")
    if not isinstance(raw, list):
        return []
    return _normalize([p for p in (_coerce(r) for r in raw) if p is not None])


def save_profiles(profiles: list[Profile], path: Path | str | None = None) -> Path:
    """Write the registry atomically, keeping at most one entry active."""
    conf = Path(path) if path is not None else config_path()
    conf.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": CONFIG_VERSION, "profiles": _normalize(profiles)}
    tmp = conf.parent / (conf.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, conf)
    return conf


def find_profile(profiles: list[Profile], token: str) -> int | None:
    """Resolve a 1-based index or a name to a position in ``profiles``."""
    token = (token or "").strip()
    if not token:
        return None
    if token.isdigit():
        idx = int(token) - 1
        return idx if 0 <= idx < len(profiles) else None
    for i, p in enumerate(profiles):
        if p["name"] == token:
            return i
    return None


def add_profile(profiles: list[Profile], name: str, filename: str) -> list[Profile]:
    """Return ``profiles`` plus a new, inactive entry.

    Raises ``ValueError`` for a bad name/filename or a name already in use.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("a profile needs a name")
    if not valid_filename(filename):
        raise ValueError(f"not a profile filename: {filename!r}")
    if any(p["name"] == name for p in profiles):
        raise ValueError(f"a profile named {name!r} already exists")
    out = [dict(p) for p in profiles]
    out.append({"name": name, "file": filename, "active": False})
    return out


def remove_profile(profiles: list[Profile], index: int) -> list[Profile]:
    """Return ``profiles`` without entry ``index`` (the file is left on disk)."""
    if not 0 <= index < len(profiles):
        raise IndexError(index)
    out = [dict(p) for p in profiles]
    del out[index]
    return out


def set_active(profiles: list[Profile], index: int | None) -> list[Profile]:
    """Return a copy with entry ``index`` active, or none active for ``None``."""
    if index is not None and not 0 <= index < len(profiles):
        raise IndexError(index)
    out = [dict(p) for p in profiles]
    for i, p in enumerate(out):
        p["active"] = i == index
    return out


def active_profile(profiles: list[Profile]) -> Profile | None:
    """The active profile, or ``None`` when none is selected."""
    return next((p for p in profiles if p.get("active")), None)


def read_profile_text(filename: str) -> str:
    """Return a profile file's content, or ``""`` if it cannot be read.

    Deliberately uncapped: a profile is the user's own context budget to spend,
    and silently truncating it would give the model a half-read briefing. The
    listing reports the cost instead (see :func:`estimate_tokens`).
    """
    path = profile_path(filename)
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def estimate_tokens(text: str) -> int:
    """Approximate the tokens ``text`` costs in a request."""
    return len(text) // _CHARS_PER_TOKEN


def profile_section(entry: Profile | None) -> str:
    """The system-prompt section for ``entry``, read fresh from disk.

    Empty when there is no active profile or its file has gone missing, so a
    profile that is deleted mid-conversation degrades to no profile rather than
    breaking the turn.
    """
    if not entry:
        return ""
    text = read_profile_text(entry["file"])
    if not text.strip():
        return ""
    return (
        f"\n\n## Agent profile: {entry['name']} "
        f"(from ~/.ludvart/profiles/{entry['file']})\n"
        "Background for the kind of work you are doing here. Treat it as "
        "standing instructions from the user.\n\n" + text
    )


class ActiveProfile:
    """The active profile as the agent loop sees it: always re-read from disk.

    Both the registry and the file are read on every call, so ``/profile use``
    and edits to the markdown take effect on the next request instead of at the
    next restart.
    """

    def entry(self) -> Profile | None:
        return active_profile(load_profiles())

    def name(self) -> str:
        entry = self.entry()
        return entry["name"] if entry else ""

    def section(self) -> str:
        return profile_section(self.entry())
