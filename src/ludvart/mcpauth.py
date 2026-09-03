"""OAuth 2.1 authorization for remote MCP servers.

Some MCP servers refuse unauthenticated work with ``401`` and an OAuth
challenge. Where that challenge appears varies: a strict server rejects the
``initialize`` handshake, while Google's Gmail MCP endpoint answers discovery in
the clear and only demands a token once a tool is actually called. ludvart
therefore does not wait to be challenged -- ``/mcp_login`` starts from the
server's own metadata (RFC 9728 protected-resource, then RFC 8414 authorization
server) and runs the authorization code grant with PKCE itself.

ludvart's backend usually runs on a different host from the browser, so a
loopback listener would be listening on the wrong machine. The authorization is
split across two panel commands instead: ``/mcp_login <server>`` prints the URL,
and ``/mcp_auth <server> <url>`` hands back what the browser ended up on. The
redirect target deliberately points at a port nothing is listening on -- the
browser shows a connection error, and the address bar holds the ``code`` and
``state`` to copy.

Requests then carry the stored token via :class:`AuthTransport`, which renews it
from the refresh token when it expires. Nothing here is interactive outside
those two commands: a request with no usable token still goes out bare, so
discovery keeps working on a server that allows it, and only a ``401`` turns
into a JSON-RPC error naming the command to run.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

#: Where the authorization server is told to send the browser. Nothing listens
#: there: the flow is completed by pasting the URL the browser landed on, and a
#: loopback address is the one redirect target public providers still allow for
#: native apps. Kept bare, with no path, because providers match the redirect
#: against the client's registered list exactly -- Google's desktop clients
#: accept any loopback port but not an invented path, and its console only
#: accepts the ``localhost`` spelling when registering one by hand. Override it
#: with ``oauth.redirectUri`` to match whatever the client is registered with.
DEFAULT_REDIRECT_URI = "http://localhost:33418"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")
_HTTP_TIMEOUT = 30.0


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
    auth_params: dict = field(default_factory=dict)

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
    extra = raw.get("authorizationParams") or raw.get("authorization_params") or {}
    return OAuthSettings(
        client_id=text("clientId", "client_id"),
        client_secret=text("clientSecret", "client_secret"),
        scopes=expand(str(scopes)).strip(),
        redirect_uri=text("redirectUri", "redirect_uri") or DEFAULT_REDIRECT_URI,
        client_name=text("clientName", "client_name") or "ludvart",
        auth_params={str(k): expand(str(v)) for k, v in dict(extra).items()},
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
        raise ValueError(
            "no 'code' in the pasted URL -- that is still a page on the "
            "provider's own site, so the redirect has not happened yet"
        )
    state = (params.get("state") or [None])[0]
    return code, state


# -- token storage ----------------------------------------------------------


class TokenStore:
    """One server's tokens and dynamic registration, under ``~/.ludvart``."""

    def __init__(self, server: str, settings: OAuthSettings) -> None:
        self._settings = settings
        safe = _SAFE_NAME.sub("_", server) or "server"
        self._path = os.path.join(auth_dir(), f"{safe}.json")

    @property
    def path(self) -> str:
        return self._path

    def read(self) -> dict:
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

    def has_tokens(self) -> bool:
        return bool(self.read().get("access_token"))

    def forget(self) -> None:
        try:
            os.remove(self._path)
        except FileNotFoundError:
            pass

    def client(self) -> tuple[str, str]:
        """The client credentials to use: configured ones, else registered."""
        if not self._settings.dynamic:
            return self._settings.client_id, self._settings.client_secret
        data = self.read()
        return data.get("registered_id", ""), data.get("registered_secret", "")

    def save_client(self, client_id: str, client_secret: str) -> None:
        self.write(registered_id=client_id, registered_secret=client_secret)

    def save_tokens(self, payload: dict, token_endpoint: str, previous=None) -> dict:
        expires_in = payload.get("expires_in")
        return self.write(
            access_token=str(payload.get("access_token") or ""),
            # Providers may omit the refresh token when renewing, which does not
            # mean the one we already hold has stopped working.
            refresh_token=str(
                payload.get("refresh_token")
                or (previous or {}).get("refresh_token")
                or ""
            ),
            scope=str(payload.get("scope") or ""),
            token_endpoint=token_endpoint,
            expires_at=time.time() + float(expires_in) - 60 if expires_in else 0,
        )

    def write(self, **fields: Any) -> dict:
        data = self.read()
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
        return data


def _stale(data: dict) -> bool:
    expires_at = data.get("expires_at")
    # A server that never says how long its token lasts is taken at its word;
    # only a deadline we were actually given can pass.
    return bool(expires_at) and time.time() >= float(expires_at)


# -- discovery --------------------------------------------------------------


@dataclass
class Endpoints:
    """Where to send the user, and where to redeem the code they bring back."""

    authorization: str
    token: str
    registration: str = ""
    scopes: str = ""


def _fetch_json(client: httpx.Client, urls: list[str]) -> dict | None:
    for url in urls:
        try:
            response = client.get(url, headers={"Accept": "application/json"})
        except httpx.HTTPError:
            continue
        if response.status_code != 200:
            continue
        try:
            data = response.json()
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def discover(url: str) -> Endpoints:
    """Find the authorization server for an MCP endpoint and read its metadata."""
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")
    with httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
        # RFC 9728 puts the well-known segment before the resource path, so the
        # document for /mcp/v1 lives at /.well-known/...-resource/mcp/v1.
        prm = _fetch_json(client, [
            f"{origin}/.well-known/oauth-protected-resource{path}",
            f"{origin}/.well-known/oauth-protected-resource",
        ]) or {}
        servers = prm.get("authorization_servers") or []
        issuer = str(servers[0]).rstrip("/") if servers else origin
        meta = _fetch_json(client, [
            f"{issuer}/.well-known/oauth-authorization-server",
            f"{issuer}/.well-known/openid-configuration",
        ])
    if not meta or not meta.get("authorization_endpoint"):
        raise RuntimeError(
            f"no OAuth metadata at {issuer} (is {url} an OAuth-protected server?)"
        )
    return Endpoints(
        authorization=str(meta["authorization_endpoint"]),
        token=str(meta.get("token_endpoint") or ""),
        registration=str(meta.get("registration_endpoint") or ""),
        scopes=" ".join(str(s) for s in prm.get("scopes_supported") or []),
    )


def _register(endpoints: Endpoints, settings: OAuthSettings, scope: str):
    """Register a client on the fly (RFC 7591) with a server that allows it."""
    if not endpoints.registration:
        raise RuntimeError(
            "no clientId configured and the server offers no dynamic "
            "client registration"
        )
    body: dict[str, Any] = {
        "client_name": settings.client_name,
        "redirect_uris": [settings.redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    if scope:
        body["scope"] = scope
    response = httpx.post(endpoints.registration, json=body, timeout=_HTTP_TIMEOUT)
    if response.status_code >= 400:
        raise RuntimeError(f"client registration failed: {_error_detail(response)}")
    data = response.json()
    return str(data.get("client_id") or ""), str(data.get("client_secret") or "")


# -- the authorization code grant -------------------------------------------


class RedirectCatcher:
    """Listens on the redirect target, for when the browser can reach it.

    The backend often runs on another host, but an SSH or editor port forward
    can still carry ``localhost`` from the browser to here. When it does, the
    code is redeemed the moment it arrives and the browser is told how it went,
    which is the only report anyone gets -- the panel is not waiting on it. When
    it does not, the browser fails to load the address and the paste flow takes
    over.
    """

    def __init__(self, redirect_uri: str) -> None:
        parsed = urlparse(redirect_uri)
        catcher = self

        class Handler(BaseHTTPRequestHandler):
            # Browsers open speculative connections and send nothing on them; a
            # handler left blocked reading one would never be shut down.
            timeout = 10

            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                query = urlparse(self.path).query
                if query:
                    catcher._caught = catcher._caught or query
                    message = catcher._redeem(query)
                else:
                    message = "ludvart is waiting for the authorization redirect."
                body = message.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._caught = ""
        self._done = False
        self._error = ""
        self._redeemer = None
        self._server = ThreadingHTTPServer(
            (parsed.hostname or "127.0.0.1", parsed.port or 80), Handler
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def arm(self, redeemer) -> None:
        """Redeem arriving codes with ``redeemer``, which may raise."""
        self._redeemer = redeemer

    def _redeem(self, query: str) -> str:
        if self._redeemer is None:
            return "ludvart received the authorization. You can close this tab."
        if self._done:
            return "ludvart is already authorized for this server."
        try:
            self._redeemer(query)
        except Exception as exc:  # noqa: BLE001 - shown in the browser
            self._error = str(exc)
            return f"ludvart could not complete the authorization: {exc}"
        self._done = True
        return "ludvart is authorized. You can close this tab."

    @property
    def caught(self) -> str:
        """The query string the browser arrived with, or ``""``."""
        return self._caught

    @property
    def done(self) -> bool:
        """True once a caught code has been exchanged for tokens."""
        return self._done

    @property
    def error(self) -> str:
        """Why redeeming a caught code failed, if it did."""
        return self._error

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _catch(redirect_uri: str) -> RedirectCatcher | None:
    """Listen on the redirect target, or give up quietly if we cannot."""
    try:
        return RedirectCatcher(redirect_uri)
    except OSError:
        return None


@dataclass
class PendingLogin:
    """One authorization in progress, carried between the two panel commands."""

    server: str
    settings: OAuthSettings
    endpoints: Endpoints
    client_id: str
    client_secret: str
    verifier: str
    state: str
    scope: str
    url: str = ""
    catcher: RedirectCatcher | None = None


def _default_auth_params(endpoint: str) -> dict:
    host = urlparse(endpoint).hostname or ""
    # Google issues a refresh token only when offline access is asked for, and
    # re-consents only when told to; without both, /mcp_login would have to be
    # repeated every time the access token aged out.
    if host == "accounts.google.com":
        return {"access_type": "offline", "prompt": "consent"}
    return {}


def start_login(
    server: str, url: str, settings: OAuthSettings, on_authorized=None
) -> PendingLogin:
    """Discover the authorization server and build the URL the user must visit.

    ``on_authorized`` runs on the listener's thread once a caught code has been
    redeemed, for whatever has to happen before the tokens are of any use.
    """
    endpoints = discover(url)
    scope = settings.scopes or endpoints.scopes
    store = TokenStore(server, settings)
    # A token still on disk would keep working; an explicit login means the user
    # wants a new one (access revoked, different scopes, another account).
    store.forget()
    client_id, client_secret = settings.client_id, settings.client_secret
    if settings.dynamic:
        client_id, client_secret = _register(endpoints, settings, scope)
        store.save_client(client_id, client_secret)

    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).decode("ascii").rstrip("=")
    state = secrets.token_urlsafe(24)
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": settings.redirect_uri,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    if scope:
        params["scope"] = scope
    params.update(settings.auth_params or _default_auth_params(endpoints.authorization))
    joiner = "&" if "?" in endpoints.authorization else "?"
    pending = PendingLogin(
        server=server,
        settings=settings,
        endpoints=endpoints,
        client_id=client_id,
        client_secret=client_secret,
        verifier=verifier,
        state=state,
        scope=scope,
        url=endpoints.authorization + joiner + urlencode(params),
        catcher=_catch(settings.redirect_uri),
    )
    if pending.catcher is not None:
        pending.catcher.arm(lambda query: _redeem(pending, query, on_authorized))
    return pending


def _redeem(pending: PendingLogin, query: str, on_authorized=None) -> None:
    """Exchange a caught code, then let the caller act on the new tokens."""
    _exchange(pending, query)
    if on_authorized is not None:
        on_authorized()


def finish_login(pending: PendingLogin, pasted: str = "") -> None:
    """Redeem the code the browser was redirected with, and store the tokens."""
    if not (pasted or "").strip():
        catcher = pending.catcher
        if catcher is not None and catcher.done:
            return  # the redirect arrived here and was redeemed already
        if catcher is not None and catcher.error:
            raise RuntimeError(catcher.error)
        if catcher is None or not catcher.caught:
            raise RuntimeError(
                "the browser has not reached the redirect address yet -- "
                "approve the request first, or paste the URL it ended on"
            )
        pasted = catcher.caught
    _exchange(pending, pasted)


def _exchange(pending: PendingLogin, pasted: str) -> None:
    code, state = parse_redirect(pasted)
    if state is not None and not secrets.compare_digest(state, pending.state):
        raise RuntimeError("that URL is from a different login attempt (state mismatch)")
    if not pending.endpoints.token:
        raise RuntimeError("the authorization server advertises no token endpoint")
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": pending.settings.redirect_uri,
        "client_id": pending.client_id,
        "code_verifier": pending.verifier,
    }
    if pending.client_secret:
        form["client_secret"] = pending.client_secret
    response = httpx.post(
        pending.endpoints.token,
        data=form,
        headers={"Accept": "application/json"},
        timeout=_HTTP_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"token exchange failed: {_error_detail(response)}")
    payload = response.json()
    if not payload.get("access_token"):
        raise RuntimeError("the token endpoint returned no access token")
    TokenStore(pending.server, pending.settings).save_tokens(
        payload, pending.endpoints.token
    )


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and body.get("error"):
        detail = body.get("error_description") or ""
        return f"{body['error']}: {detail}" if detail else str(body["error"])
    return f"HTTP {response.status_code}: {response.text[:200]}"


# -- attaching the token to MCP traffic -------------------------------------


class AuthTransport(httpx.AsyncBaseTransport):
    """Carry the stored token on MCP traffic, renewing it when it expires.

    This is an HTTP *transport* rather than an :class:`httpx.Auth` because of
    how the MCP SDK sends requests: each one runs in its own task inside the
    transport's task group, so an exception raised while sending never reaches
    the caller -- ``session.call_tool`` simply waits forever. Down here a
    failure can be turned into a JSON-RPC error instead, which the session
    delivers as an ordinary tool error saying which command would fix it.

    A request with no usable token is still sent bare: servers differ on how
    much they allow unauthenticated -- Google serves its whole tool list -- and
    letting it through is what makes those tools discoverable before a login.
    """

    def __init__(self, server: str, settings: OAuthSettings, inner=None) -> None:
        self._server = server
        self._store = TokenStore(server, settings)
        # A client, not a bare transport: httpx only applies the environment's
        # proxy settings to clients it configures itself, and handing one an
        # explicit transport= skips that, so every request would go direct.
        self._inner = inner or httpx.AsyncClient(
            follow_redirects=True, timeout=_HTTP_TIMEOUT
        )

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def handle_async_request(self, request: httpx.Request):
        body = await request.aread()
        data = self._store.read()
        if _stale(data) and data.get("refresh_token"):
            data = await self._refresh(data)
        response = await self._send(request, body, data)
        if response.status_code == 401 and data.get("refresh_token"):
            await response.aclose()
            data = await self._refresh(data)
            if data.get("access_token"):
                response = await self._send(request, body, data)
        if response.status_code != 401:
            return response
        await response.aread()
        await response.aclose()
        return self._needs_login(body)

    async def _send(self, request: httpx.Request, body: bytes, data: dict):
        headers = dict(request.headers)
        token = data.get("access_token")
        if token:
            headers["authorization"] = f"Bearer {token}"
        else:
            headers.pop("authorization", None)
        return await self._inner.send(
            httpx.Request(
                request.method,
                request.url,
                headers=headers,
                content=body,
                extensions=request.extensions,
            ),
            stream=True,
        )

    def _needs_login(self, body: bytes) -> httpx.Response:
        """Turn a dead end into an answer the model and the user can act on."""
        message = (
            f"MCP server '{self._server}' needs authorization; "
            f"run /mcp_login {self._server}"
        )
        try:
            request_id = json.loads(body)["id"]
        except (ValueError, TypeError, KeyError):
            # A notification has no id to answer, so nothing can be said in
            # band; the plain 401 is the most honest thing to return.
            return httpx.Response(401, json={"error": message})
        return httpx.Response(200, json={
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32001, "message": message},
        })

    async def _refresh(self, data: dict) -> dict:
        client_id, client_secret = self._store.client()
        form = {
            "grant_type": "refresh_token",
            "refresh_token": data.get("refresh_token", ""),
            "client_id": client_id,
        }
        if client_secret:
            form["client_secret"] = client_secret
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                response = await client.post(
                    data.get("token_endpoint", ""),
                    data=form,
                    headers={"Accept": "application/json"},
                )
            payload = response.json() if response.status_code < 400 else {}
        except (httpx.HTTPError, ValueError):
            payload = {}
        if not isinstance(payload, dict) or not payload.get("access_token"):
            return {}  # the refresh token is dead; only a new login helps
        return self._store.save_tokens(
            payload, data.get("token_endpoint", ""), data
        )


def client_factory(server: str, settings: OAuthSettings):
    """Build the ``httpx_client_factory`` the MCP SDK should use for a server."""

    def make(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=headers,
            timeout=timeout if timeout is not None else httpx.Timeout(30.0),
            auth=auth,
            follow_redirects=True,
            transport=AuthTransport(server, settings),
        )

    return make

