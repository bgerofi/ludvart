"""Agent profiles: background the model is given up front for a kind of work.

A profile is a directory under ``~/.ludvart/profiles/`` describing the domain
the agent is about to work in. It holds two markdown files:

* ``self.md`` -- the briefing: how to investigate a particular system, what the
  conventions are, where things live. Written by the user, read for every
  request, so editing it is felt immediately.
* ``memory.md`` -- optional long-term memory: what the agent has learned and
  wants to keep across conversations.

Which profiles are registered, and which one is active, live in
``~/.ludvart/profiles/config.json``::

    {"version": 1, "profiles": [{"name": "...", "dir": "xyz", "active": true}]}

Unlike the model registry, "no profile" is a legal state: a profile is opt-in
and can be switched off again.

The two files go to different ends of the prompt on purpose. ``self.md`` is
stable, so it is appended to the system prompt where a provider's cache can
keep it. ``memory.md`` is written *during* conversations, so it rides in the
trailing live block instead: changing it there costs nothing, whereas changing
the system prompt would invalidate the cached prefix of the whole conversation.

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

#: The briefing, and the optional long-term memory, inside a profile directory.
SELF_NAME = "self.md"
MEMORY_NAME = "memory.md"

#: A profile directory is a bare name inside the profiles directory. Anything
#: with a path separator or a leading dot (which covers ``.`` and ``..``) is
#: refused, so a registration can never name a directory outside it.
_DIRNAME_RE = re.compile(r"[^/\\.][^/\\]*")

#: Rough characters-per-token, the same estimate the compaction trigger uses.
#: Good enough to tell a 2k profile from a 200k one, which is what it is for.
_CHARS_PER_TOKEN = 4

#: A profile above this costs real money and context on every single request,
#: so the listing says so instead of letting it go unnoticed.
LARGE_PROFILE_TOKENS = 50_000

Profile = dict[str, Any]


def profiles_dir() -> Path:
    """Directory holding the profile directories and their config.

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


def valid_dirname(dirname: str) -> bool:
    """Whether ``dirname`` names a directory directly inside the profiles dir."""
    return bool(dirname) and bool(_DIRNAME_RE.fullmatch(dirname))


def profile_dir(dirname: str) -> Path | None:
    """Resolve a profile directory name, or ``None`` if it is not one."""
    if not valid_dirname(dirname):
        return None
    return profiles_dir() / dirname


def self_path(dirname: str) -> Path | None:
    """Path of a profile's briefing file, or ``None`` for a bad directory."""
    root = profile_dir(dirname)
    return None if root is None else root / SELF_NAME


def memory_path(dirname: str) -> Path | None:
    """Path of a profile's memory file, or ``None`` for a bad directory."""
    root = profile_dir(dirname)
    return None if root is None else root / MEMORY_NAME


def _coerce(raw: Any) -> Profile | None:
    """Normalize one loaded entry into a profile, or ``None`` if unusable."""
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    dirname = str(raw.get("dir") or "").strip()
    if not name or not valid_dirname(dirname):
        return None
    return {"name": name, "dir": dirname, "active": bool(raw.get("active"))}


def _normalize(profiles: list[Profile]) -> list[Profile]:
    """Return a copy of ``profiles`` with at most one entry active."""
    out = [dict(p) for p in profiles]
    active = next((i for i, p in enumerate(out) if p.get("active")), None)
    for i, p in enumerate(out):
        p["active"] = i == active
    return out


#: Private mode: this process picks its own profile and leaves the shared
#: registry's choice alone. ``None`` means "nothing picked yet, follow the file".
_private = False
_private_active: str | None = None


def set_private(enabled: bool) -> None:
    """Keep this process's active-profile choice out of the shared registry."""
    global _private, _private_active
    _private = enabled
    _private_active = None


def _read_config(conf: Path) -> list[Profile]:
    """The registry exactly as the file has it, private mode notwithstanding."""
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


def load_profiles(path: Path | str | None = None) -> list[Profile]:
    """Read the registry (empty list when absent, unreadable or malformed)."""
    shared = path is None
    profiles = _read_config(config_path() if shared else Path(path))
    if shared and _private and _private_active is not None:
        for profile in profiles:
            profile["active"] = profile["dir"] == _private_active
    return profiles


def save_profiles(profiles: list[Profile], path: Path | str | None = None) -> Path:
    """Write the registry atomically, keeping at most one entry active.

    Registrations are shared even in private mode; only the choice of which
    profile is active stays here, so the file keeps whichever one it already had.
    """
    global _private_active
    shared = path is None
    conf = config_path() if shared else Path(path)
    entries = _normalize(profiles)
    if shared and _private:
        picked = next((p for p in entries if p["active"]), None)
        _private_active = picked["dir"] if picked else ""
        on_disk = {p["dir"] for p in _read_config(conf) if p["active"]}
        for entry in entries:
            entry["active"] = entry["dir"] in on_disk
    conf.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": CONFIG_VERSION, "profiles": entries}
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


def add_profile(profiles: list[Profile], name: str, dirname: str) -> list[Profile]:
    """Return ``profiles`` plus a new, inactive entry.

    Raises ``ValueError`` for a bad name/directory or a name already in use.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("a profile needs a name")
    if not valid_dirname(dirname):
        raise ValueError(f"not a profile directory: {dirname!r}")
    if any(p["name"] == name for p in profiles):
        raise ValueError(f"a profile named {name!r} already exists")
    out = [dict(p) for p in profiles]
    out.append({"name": name, "dir": dirname, "active": False})
    return out


def remove_profile(profiles: list[Profile], index: int) -> list[Profile]:
    """Return ``profiles`` without entry ``index`` (the files stay on disk)."""
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


def _read(path: Path | None) -> str:
    """Read a profile file, or ``""`` if it is absent or unreadable.

    Deliberately uncapped: a profile is the user's own context budget to spend,
    and silently truncating it would give the model a half-read briefing. The
    listing reports the cost instead (see :func:`estimate_tokens`).
    """
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def read_profile_text(dirname: str) -> str:
    """Return a profile's ``self.md``, or ``""`` if it cannot be read."""
    return _read(self_path(dirname))


def read_memory_text(dirname: str) -> str:
    """Return a profile's ``memory.md``, or ``""`` when it does not exist."""
    return _read(memory_path(dirname))


def estimate_tokens(text: str) -> int:
    """Approximate the tokens ``text`` costs in a request."""
    return len(text) // _CHARS_PER_TOKEN


def profile_tokens(dirname: str) -> int:
    """What a profile adds to every request, briefing plus memory."""
    return estimate_tokens(read_profile_text(dirname) + read_memory_text(dirname))


def profile_section(entry: Profile | None) -> str:
    """The system-prompt section for ``entry``, read fresh from disk.

    Empty when there is no active profile or its briefing has gone missing, so
    a profile deleted mid-conversation degrades to no profile rather than
    breaking the turn.
    """
    if not entry:
        return ""
    text = read_profile_text(entry["dir"])
    if not text.strip():
        return ""
    return (
        f"\n\n## Agent profile: {entry['name']} "
        f"(from ~/.ludvart/profiles/{entry['dir']}/{SELF_NAME})\n"
        "Background for the kind of work you are doing here. Treat it as "
        "standing instructions from the user.\n\n" + text
    )


def memory_block(entry: Profile | None) -> str:
    """``entry``'s memory as a block for the trailing message, read from disk.

    Kept out of the system prompt because the agent writes to this file during
    a conversation, and an edit to the prompt's first message would cost the
    provider's cached prefix for everything after it.
    """
    if not entry:
        return ""
    text = read_memory_text(entry["dir"])
    if not text.strip():
        return ""
    return (
        f'<memory profile="{entry["name"]}" '
        f'file="~/.ludvart/profiles/{entry["dir"]}/{MEMORY_NAME}">\n'
        "Long-term notes you have kept across conversations for this profile. "
        "Append to that file when you learn something worth keeping.\n\n"
        f"{text.rstrip()}\n</memory>"
    )


class ActiveProfile:
    """The active profile as the agent loop sees it: always re-read from disk.

    Both the registry and the files are read on every call, so ``/profile use``
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

    def memory(self) -> str:
        return memory_block(self.entry())
