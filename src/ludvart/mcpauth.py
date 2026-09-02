"""OAuth 2.1 authorization for remote MCP servers.

Some MCP servers (hosted ones especially) answer an unauthenticated request
with ``401`` and an OAuth challenge instead of serving tools. The ``mcp`` SDK
ships the protocol half of the answer -- :class:`~mcp.client.auth.OAuthClientProvider`
is an :class:`httpx.Auth` that discovers the authorization server, runs PKCE,
validates ``state`` and refreshes expired tokens -- but it delegates the two
steps that need a human: showing the authorization URL and obtaining the code
the browser is redirected with.

ludvart's backend usually runs on a different host from the browser, so a
loopback listener would be listening on the wrong machine. Instead the
authorization is split across two panel commands: ``/mcp_login <server>`` prints
the URL, and ``/mcp_auth <redirected-url>`` hands back what the browser ended up
on. The redirect target deliberately points at a port nothing is listening on --
the browser shows a connection error, and the address bar holds the ``code`` and
``state`` to copy.

Nothing here is interactive during startup: a server whose tokens are missing or
unusable fails discovery with :class:`NeedsAuthorization` so the backend keeps
serving with its other tools, rather than blocking on a prompt no one can see.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlparse

#: Where the authorization server is told to send the browser. Nothing listens
#: there: the flow is completed by pasting the URL the browser landed on, and a
#: loopback address is the one redirect target public providers still allow for
#: native apps.
DEFAULT_REDIRECT_URI = "http://127.0.0.1:33418/ludvart/callback"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")


class NeedsAuthorization(RuntimeError):
    """Raised when a server needs an interactive login that cannot run now."""


def auth_dir() -> str:
    """Directory holding one token file per authorized server."""
    return os.path.join(os.path.expanduser("~"), ".ludvart", "mcp-auth")


@dataclass
class OAuthSettings:
    """The ``oauth`` block of a server entry in ``mcp.json``."""

    client_id: str = ""
    client_secret: str = ""
    scopes: str = ""
    redirect_uri: str = DEFAULT_REDIRECT_URI
    client_name: str = "ludvart"

    @property
    def dynamic(self) -> bool:
        """True when the client is registered on the fly (no ``clientId``)."""
        return not self.client_id


def parse_settings(cfg: dict, expand) -> OAuthSettings | None:
    """Read a server entry's ``oauth`` block, or ``None`` when it has none.

    ``oauth: true`` (or an empty object) means "this server needs OAuth, work
    the rest out by discovery and dynamic client registration".
    """
    raw = cfg.get("oauth")
    if raw is None or raw is False:
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("'oauth' must be an object or true")

    def text(*keys: str) -> str:
        for key in keys:
            value = raw.get(key)
            if value:
                return expand(str(value)).strip()
        return ""

    scopes: Any = raw.get("scopes") or raw.get("scope") or ""
    if isinstance(scopes, (list, tuple)):
        scopes = " ".join(str(s) for s in scopes)
    return OAuthSettings(
        client_id=text("clientId", "client_id"),
        client_secret=text("clientSecret", "client_secret"),
        scopes=expand(str(scopes)).strip(),
        redirect_uri=text("redirectUri", "redirect_uri") or DEFAULT_REDIRECT_URI,
        client_name=text("clientName", "client_name") or "ludvart",
    )


def parse_redirect(pasted: str) -> tuple[str, str | None]:
    """Extract ``(code, state)`` from the URL the browser was redirected to.

    Accepts the whole URL, a bare query string, or just the code, since what
    lands in a paste buffer depends on where the user copied it from.
    """
    text = (pasted or "").strip().strip("'\"")
    if not text:
        raise ValueError("nothing pasted")
    query = urlparse(text).query if "://" in text else text
    if "=" not in query:
        return text, None  # a bare code, pasted without its query string
    params = parse_qs(query, keep_blank_values=True)
    error = (params.get("error") or [""])[0]
    if error:
        detail = (params.get("error_description") or [""])[0]
        raise ValueError(f"{error}: {detail}" if detail else error)
    code = (params.get("code") or [""])[0]
    if not code:
        raise ValueError("no 'code' in the pasted URL")
    state = (params.get("state") or [None])[0]
    return code, state


class FileTokenStorage:
    """Persist one server's tokens and client registration under ``~/.ludvart``.

    Implements the SDK's ``TokenStorage`` protocol. Pre-registered credentials
    from ``mcp.json`` are handed over as if they had been stored, which is what
    stops the SDK from attempting dynamic client registration.
    """

    def __init__(self, server: str, settings: OAuthSettings) -> None:
        self._settings = settings
        safe = _SAFE_NAME.sub("_", server) or "server"
        self._path = os.path.join(auth_dir(), f"{safe}.json")

    @property
    def path(self) -> str:
        return self._path

    def has_tokens(self) -> bool:
        return bool(self._read().get("tokens"))

    def forget(self) -> None:
        try:
            os.remove(self._path)
        except FileNotFoundError:
            pass

    # -- TokenStorage protocol ---------------------------------------------

    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken

        data = self._read()
        stored = data.get("tokens")
        if not stored:
            return None
        tokens = OAuthToken.model_validate(stored)
        # The SDK only tracks expiry for tokens it fetched itself: one loaded
        # from disk looks valid forever. Handing back a stale access token would
        # earn a 401, which the SDK reads as "never authorized" and answers with
        # a whole new login -- so ludvart would demand /mcp_login every time an
        # access token aged out. Withhold it instead and let the refresh token,
        # which is the reason we store anything at all, do its job.
        if tokens.refresh_token and self._expired(data):
            return tokens.model_copy(update={"access_token": ""})
        return tokens

    async def set_tokens(self, tokens) -> None:
        self._write(
            tokens=json.loads(tokens.model_dump_json(exclude_none=True)),
            token_expires_at=(
                time.time() + tokens.expires_in - 60 if tokens.expires_in else 0
            ),
        )

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull

        configured = self._configured_client_info()
        if configured is not None:
            return configured
        data = self._read().get("client_info")
        if not data:
            return None
        try:
            return OAuthClientInformationFull.model_validate(data)
        except ValueError:
            return None

    async def set_client_info(self, client_info) -> None:
        if not self._settings.dynamic:
            return  # configured credentials are the source of truth
        self._write(
            client_info=json.loads(client_info.model_dump_json(exclude_none=True)),
        )

    # -- file handling ------------------------------------------------------

    def _configured_client_info(self):
        if self._settings.dynamic:
            return None
        from mcp.shared.auth import OAuthClientInformationFull

        return OAuthClientInformationFull(
            client_id=self._settings.client_id,
            client_secret=self._settings.client_secret or None,
            redirect_uris=[self._settings.redirect_uri],
            token_endpoint_auth_method=(
                "client_secret_post" if self._settings.client_secret else "none"
            ),
            scope=self._settings.scopes or None,
            client_name=self._settings.client_name,
        )

    def _read(self) -> dict:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (FileNotFoundError, ValueError, OSError):
            return {}
        if not isinstance(data, dict):
            return {}
        # Credentials issued to a different client are useless and, if the
        # config was repointed at another provider, misleading.
        if data.get("client_id_hint") not in (None, self._settings.client_id):
            return {}
        return data

    @staticmethod
    def _expired(data: dict) -> bool:
        expires_at = data.get("token_expires_at")
        # A server that never says how long its tokens last is taken at its
        # word; only a deadline we were actually given can pass.
        return bool(expires_at) and time.time() >= float(expires_at)

    def _write(self, **fields: Any) -> None:
        data = self._read()
        data.update(fields)
        data["client_id_hint"] = self._settings.client_id
        data["updated"] = int(time.time())
        os.makedirs(auth_dir(), mode=0o700, exist_ok=True)
        tmp = f"{self._path}.tmp"
        # Create the file unreadable to anyone else *before* the refresh token
        # is in it, rather than fixing the mode afterwards.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except BaseException:
            os.unlink(tmp)
            raise
        os.replace(tmp, self._path)


@dataclass
class PendingLogin:
    """One interactive authorization in progress, shared across two commands."""

    server: str
    url: str = ""
    code: str = ""
    state: str | None = None
    url_ready: Any = field(default=None)
    code_ready: Any = field(default=None)
    #: Resolves when the parked connection attempt finishes, with its error.
    connect: Any = field(default=None)
    #: The server state whose task is parked in the flow, so it can be cancelled.
    state_holder: Any = field(default=None)


def build_auth(server: str, url: str, settings: OAuthSettings, pending=None):
    """Return an ``httpx.Auth`` that authorizes requests to ``url``.

    With ``pending`` set, an authorization the SDK decides it needs is carried
    out interactively through that record; without it, needing one raises
    :class:`NeedsAuthorization` instead of waiting for a human who is not there.
    """
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    storage = FileTokenStorage(server, settings)

    async def redirect_handler(auth_url: str) -> None:
        if pending is None:
            raise NeedsAuthorization(
                f"'{server}' needs authorization; run /mcp_login {server}"
            )
        pending.url = auth_url
        pending.url_ready.set()

    async def callback_handler() -> tuple[str, str | None]:
        if pending is None:  # unreachable: redirect_handler runs first
            raise NeedsAuthorization(f"'{server}' needs authorization")
        import anyio

        with anyio.move_on_after(600) as scope:
            await anyio.to_thread.run_sync(pending.code_ready.wait)
        if scope.cancelled_caught or not pending.code:
            raise NeedsAuthorization(
                f"timed out waiting for /mcp_auth for '{server}'"
            )
        return pending.code, pending.state

    return OAuthClientProvider(
        server_url=url,
        client_metadata=OAuthClientMetadata(
            redirect_uris=[settings.redirect_uri],
            client_name=settings.client_name,
            scope=settings.scopes or None,
            token_endpoint_auth_method=(
                "client_secret_post" if settings.client_secret else None
            ),
        ),
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
