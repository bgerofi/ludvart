"""OAuth authorization for remote MCP servers (src/ludvart/mcpauth.py).

The flow is exercised against a *real* HTTP authorization server run in a
thread, driven through the SDK's own ``OAuthClientProvider`` and httpx -- so
metadata discovery, PKCE, the ``state`` check, the token exchange and our token
storage are all the real implementations. Only the browser is simulated, by
handing back the redirect URL it would have landed on.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_mcp_oauth.py
"""

import asyncio
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from ludvart import mcpauth
from ludvart.mcp import McpConfigError, McpManager, _expand

ACCESS_TOKEN = "access-token-abc"
REFRESH_TOKEN = "refresh-token-xyz"
RESOURCE_BODY = "the protected resource"


class _AuthServer(BaseHTTPRequestHandler):
    """A minimal OAuth 2.1 authorization server, resource server and MCP proxy."""

    seen_token_requests: list = []
    registrations: list = []
    #: When set, ``/mcp`` forwards to this real MCP server once authorized.
    mcp_upstream: str = ""

    def log_message(self, *args):  # keep the test output clean
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @property
    def _base(self):
        return f"http://{self.headers.get('Host')}"

    def _unauthorized(self):
        resource = urlparse(self.path).path
        self.send_response(401)
        self.send_header(
            "WWW-Authenticate",
            f'Bearer resource_metadata="{self._base}'
            f'/.well-known/oauth-protected-resource{resource}"',
        )
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _proxy_mcp(self, method: str):
        """Forward an authorized MCP request to the real server behind us."""
        if self.headers.get("Authorization") != f"Bearer {ACCESS_TOKEN}":
            self._unauthorized()
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        # The Authorization header is passed along, not consumed, so the MCP
        # server behind us can report what it was reached with.
        skip = {"host", "content-length", "connection"}
        headers = {k: v for k, v in self.headers.items() if k.lower() not in skip}
        upstream = httpx.request(
            method,
            _AuthServer.mcp_upstream,
            headers=headers,
            content=body,
            timeout=20,
        )
        payload = upstream.content
        self.send_response(upstream.status_code)
        for name in ("content-type", "mcp-session-id", "mcp-protocol-version"):
            if name in upstream.headers:
                self.send_header(name, upstream.headers[name])
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_DELETE(self):
        self._proxy_mcp("DELETE")

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/mcp"):
            self._proxy_mcp("GET")
            return
        if path.startswith("/.well-known/oauth-authorization-server") or path.startswith(
            "/.well-known/openid-configuration"
        ):
            self._json(200, {
                "issuer": self._base,
                "authorization_endpoint": f"{self._base}/authorize",
                "token_endpoint": f"{self._base}/token",
                "registration_endpoint": f"{self._base}/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "code_challenge_methods_supported": ["S256"],
            })
            return
        if path.startswith("/.well-known/oauth-protected-resource"):
            suffix = path[len("/.well-known/oauth-protected-resource"):]
            self._json(200, {
                "resource": f"{self._base}{suffix or '/resource'}",
                "authorization_servers": [self._base],
            })
            return
        if path == "/resource":
            auth = self.headers.get("Authorization", "")
            if auth == f"Bearer {ACCESS_TOKEN}":
                body = RESOURCE_BODY.encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._unauthorized()
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path.startswith("/mcp"):
            self._proxy_mcp("POST")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode()
        if path == "/register":
            _AuthServer.registrations.append(json.loads(raw or "{}"))
            self._json(201, {
                "client_id": "dynamically-registered",
                "client_secret": "dyn-secret",
                "redirect_uris": json.loads(raw or "{}").get("redirect_uris", []),
            })
            return
        if path == "/token":
            form = {k: v[0] for k, v in parse_qs(raw).items()}
            _AuthServer.seen_token_requests.append(form)
            if form.get("grant_type") == "refresh_token":
                if form.get("refresh_token") != REFRESH_TOKEN:
                    self._json(400, {"error": "invalid_grant"})
                    return
            elif not form.get("code_verifier"):
                self._json(400, {"error": "invalid_request"})
                return
            self._json(200, {
                "access_token": ACCESS_TOKEN,
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": REFRESH_TOKEN,
            })
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


def _serve():
    _AuthServer.seen_token_requests = []
    _AuthServer.registrations = []
    _AuthServer.mcp_upstream = ""
    httpd = HTTPServer(("127.0.0.1", 0), _AuthServer)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, f"http://127.0.0.1:{httpd.server_port}"


# A real MCP server that reports the Authorization header it was reached with,
# so a test can prove the token travelled all the way through the transport.
_MCP_SERVER_SRC = textwrap.dedent(
    """
    import sys
    from mcp.server.fastmcp import Context, FastMCP

    mcp = FastMCP(
        "ludvart-oauth-test",
        host="127.0.0.1",
        port=int(sys.argv[1]),
        json_response=True,
        stateless_http=True,
    )

    @mcp.tool()
    def whoami(ctx: Context) -> str:
        "Report the Authorization header this request carried."
        return ctx.request_context.request.headers.get("authorization", "<none>")

    mcp.run(transport="streamable-http")
    """
)


def _start_mcp_server(root: Path):
    """Spawn the real MCP server and wait for it to accept connections."""
    import socket

    root.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    script = root / "oauth_mcp_server.py"
    script.write_text(_MCP_SERVER_SRC)
    proc = subprocess.Popen(
        [sys.executable, str(script), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return proc, f"http://127.0.0.1:{port}/mcp"
        except OSError:
            time.sleep(0.1)
    proc.terminate()
    raise AssertionError("the test MCP server never came up")


@contextmanager
def _home(path: Path):
    """Point ~ at a scratch directory so real tokens are never touched."""
    path.mkdir(parents=True, exist_ok=True)
    prior = os.environ.get("HOME")
    os.environ["HOME"] = str(path)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = prior


def _authorize_like_a_browser(pending: mcpauth.PendingLogin) -> None:
    """Play the part of the user: visit the URL, come back with the redirect."""
    assert pending.url_ready.wait(10), "no authorization URL was produced"
    params = parse_qs(urlparse(pending.url).query)
    assert params["code_challenge_method"] == ["S256"], params
    assert params["response_type"] == ["code"], params
    state = params["state"][0]
    redirect = params["redirect_uri"][0]
    code, returned_state = mcpauth.parse_redirect(
        f"{redirect}?code=granted-code&state={state}"
    )
    pending.code, pending.state = code, returned_state
    pending.code_ready.set()


def _run_flow(tmp_path: Path, settings, base: str) -> str:
    """Drive a real authorized request end to end; return the response body."""
    pending = mcpauth.PendingLogin(
        server="fixture",
        url_ready=threading.Event(),
        code_ready=threading.Event(),
    )
    auth = mcpauth.build_auth("fixture", f"{base}/resource", settings, pending)
    browser = threading.Thread(
        target=_authorize_like_a_browser, args=(pending,), daemon=True
    )
    browser.start()

    async def go():
        async with httpx.AsyncClient(auth=auth, timeout=20) as client:
            resp = await client.get(f"{base}/resource")
            return resp.status_code, resp.text

    status, text = asyncio.run(go())
    browser.join(timeout=5)
    assert status == 200, (status, text)
    return text


def test_a_configured_client_authorizes_and_stores_its_tokens(tmp_path: Path):
    """The whole point: reach a server that answers 401 with an OAuth challenge."""
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            settings = mcpauth.OAuthSettings(
                client_id="OAUTH_CLIENT_ID",
                client_secret="OAUTH_CLIENT_SECRET",
                scopes="gmail.readonly",
            )
            assert _run_flow(tmp_path, settings, base) == RESOURCE_BODY
        finally:
            httpd.shutdown()

        # The configured credentials were used as-is: no dynamic registration.
        assert _AuthServer.registrations == [], _AuthServer.registrations
        form = _AuthServer.seen_token_requests[-1]
        assert form["client_id"] == "OAUTH_CLIENT_ID", form
        assert form["client_secret"] == "OAUTH_CLIENT_SECRET", form
        assert form["code"] == "granted-code", form

        # The refresh token is on disk for next time, readable only by its owner.
        path = Path(mcpauth.auth_dir()) / "fixture.json"
        saved = json.loads(path.read_text())
        assert saved["tokens"]["refresh_token"] == REFRESH_TOKEN, saved
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, oct(path.stat().st_mode)
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    print("a configured client authorizes and stores its tokens: OK")


def test_a_client_without_credentials_registers_itself(tmp_path: Path):
    """No clientId means the server issues one (RFC 7591), not that it fails."""
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            assert _run_flow(tmp_path, mcpauth.OAuthSettings(), base) == RESOURCE_BODY
        finally:
            httpd.shutdown()

        assert len(_AuthServer.registrations) == 1, _AuthServer.registrations
        assert (
            _AuthServer.seen_token_requests[-1]["client_id"]
            == "dynamically-registered"
        )
        saved = json.loads((Path(mcpauth.auth_dir()) / "fixture.json").read_text())
        # A registration is worth keeping: re-registering on every start would
        # leave a trail of dead clients on the provider.
        assert saved["client_info"]["client_id"] == "dynamically-registered", saved
    print("a client without credentials registers itself: OK")


def test_a_stored_token_is_reused_without_a_second_login(tmp_path: Path):
    """Authorizing once must be enough; a browser is not always available."""
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            settings = mcpauth.OAuthSettings(client_id="cid", client_secret="sec")
            _run_flow(tmp_path, settings, base)

            # Second time round nothing is pending, so any attempt to authorize
            # interactively would raise instead of silently blocking.
            auth = mcpauth.build_auth("fixture", f"{base}/resource", settings, None)

            async def go():
                async with httpx.AsyncClient(auth=auth, timeout=20) as client:
                    return await client.get(f"{base}/resource")

            resp = asyncio.run(go())
            assert resp.status_code == 200, resp.text
            assert resp.text == RESOURCE_BODY
        finally:
            httpd.shutdown()
    # Exactly one token exchange: the second request rode on the stored token.
    assert len(_AuthServer.seen_token_requests) == 1, _AuthServer.seen_token_requests
    print("a stored token is reused without a second login: OK")


def test_an_unauthorized_server_says_how_to_authorize(tmp_path: Path):
    """Discovery runs at startup, where nobody can answer a prompt.

    Blocking there would hang the backend before it has a client attached, so
    the server has to fail with an instruction instead.
    """
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            auth = mcpauth.build_auth(
                "gmail",
                f"{base}/resource",
                mcpauth.OAuthSettings(client_id="cid"),
                None,
            )

            async def go():
                async with httpx.AsyncClient(auth=auth, timeout=20) as client:
                    await client.get(f"{base}/resource")

            with pytest.raises(mcpauth.NeedsAuthorization) as excinfo:
                asyncio.run(go())
            assert "/mcp_login gmail" in str(excinfo.value), excinfo.value
        finally:
            httpd.shutdown()
    print("an unauthorized server says how to authorize: OK")


def test_an_expired_access_token_is_refreshed_not_re_authorized(tmp_path: Path):
    """Access tokens last an hour; logging in again every hour is not a design."""
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            settings = mcpauth.OAuthSettings(client_id="cid", client_secret="sec")
            _run_flow(tmp_path, settings, base)

            # Age the stored token past its expiry, as a restart tomorrow would.
            path = Path(mcpauth.auth_dir()) / "fixture.json"
            saved = json.loads(path.read_text())
            saved["token_expires_at"] = 1.0
            path.write_text(json.dumps(saved))

            # No pending login: an attempt to authorize interactively would raise.
            auth = mcpauth.build_auth("fixture", f"{base}/resource", settings, None)

            async def go():
                async with httpx.AsyncClient(auth=auth, timeout=20) as client:
                    return await client.get(f"{base}/resource")

            resp = asyncio.run(go())
            assert resp.status_code == 200, resp.text
            assert resp.text == RESOURCE_BODY
        finally:
            httpd.shutdown()

    grants = [f.get("grant_type") for f in _AuthServer.seen_token_requests]
    assert grants == ["authorization_code", "refresh_token"], grants
    print("an expired access token is refreshed not re-authorized: OK")


def test_the_transport_is_given_the_provider_to_authorize_with(tmp_path: Path):
    """Building a provider is pointless unless the connection actually uses it."""
    import ludvart.mcp as mcp_mod

    seen = {}

    def fake_transport(url, headers=None, auth=None, **kw):
        seen[url] = auth
        raise RuntimeError("stop here: only the wiring is under test")

    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"servers": {
        "plain": {"url": "http://plain.test/mcp"},
        "gmail": {
            "serverUrl": "http://gmail.test/mcp",
            "oauth": {"clientId": "cid"},
        },
        "streamed": {"url": "http://sse.test/mcp", "type": "sse", "oauth": True},
    }}))
    mgr = McpManager(config_file=str(cfg))
    prior = (mcp_mod.streamablehttp_client, mcp_mod.sse_client)
    mcp_mod.streamablehttp_client = fake_transport
    mcp_mod.sse_client = fake_transport
    with _home(tmp_path):
        try:
            status = mgr.refresh()
        finally:
            mcp_mod.streamablehttp_client, mcp_mod.sse_client = prior
            mgr.close()

    assert seen["http://plain.test/mcp"] is None, seen
    # serverUrl is honoured as an alias for url, and both HTTP transports
    # authorize.
    assert seen["http://gmail.test/mcp"] is not None, seen
    assert seen["http://sse.test/mcp"] is not None, seen
    assert sorted(status.servers) == ["gmail", "plain", "streamed"], status.servers
    print("the transport is given the provider to authorize with: OK")


def test_login_then_auth_makes_a_protected_servers_tools_usable(tmp_path: Path):
    """The feature, end to end, driven through the panel commands themselves.

    An OAuth-protected proxy stands in front of a real MCP server, so every
    piece is exercised: startup declining to prompt, /mcp_login producing a URL,
    /mcp_auth completing the exchange, and a tool call finally arriving at the
    server with the bearer token attached.
    """
    from ludvart import server as backend

    proc, upstream = _start_mcp_server(tmp_path / "server")
    httpd, base = _serve()
    _AuthServer.mcp_upstream = upstream
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"servers": {"gmail": {
        "serverUrl": f"{base}/mcp",
        "oauth": {"clientId": "cid", "clientSecret": "sec"},
    }}}))
    mgr = McpManager(config_file=str(cfg), connect_timeout=30.0)

    class _Core:
        mcp = mgr
        tools: list = []
        system_prompt = ""

    class _Channel:
        def __init__(self, sink):
            self._sink = sink

        def send(self, msg):
            if msg.get("kind") == "system":
                self._sink.append(msg.get("text", ""))

    def run(command: str) -> list[str]:
        lines: list[str] = []
        backend._handle_command({"command": command}, None, _Core(), _Channel(lines))
        return lines

    with _home(tmp_path):
        try:
            # Startup: no tokens yet, and nobody to ask. It must report how to
            # fix that rather than block the backend.
            assert "/mcp_login gmail" in "\n".join(run("mcp_refresh"))

            lines = run("mcp_login gmail")
            url = next(line for line in lines if line.startswith("http"))
            assert "/mcp_auth gmail" in "\n".join(lines), lines
            params = parse_qs(urlparse(url).query)
            assert params["client_id"] == ["cid"], params
            redirect = params["redirect_uri"][0]

            done = "\n".join(run(
                f"mcp_auth gmail {redirect}?code=granted-code"
                f"&state={params['state'][0]}"
            ))
            assert "1/1 server(s) connected, 1 tool(s)" in done, done
            assert mgr.call_tool("mcp_gmail_whoami", {}) == f"Bearer {ACCESS_TOKEN}"

            # Asking to log in again must actually re-authorize: the stored
            # token would otherwise satisfy the SDK and no URL would appear.
            again = run("mcp_login gmail")
            assert any(line.startswith("http") for line in again), again
            mgr.cancel_login("gmail")
        finally:
            mgr.close()
            httpd.shutdown()
            proc.terminate()
            proc.wait(timeout=10)
    print("login then auth makes a protected server's tools usable: OK")


def test_a_failed_login_says_what_actually_went_wrong(tmp_path: Path):
    """"Did it ask for authorization?" is a guess; the real error is knowable."""
    import socket

    with socket.socket() as probe:  # a port with nothing behind it
        probe.bind(("127.0.0.1", 0))
        dead = probe.getsockname()[1]
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"servers": {"gmail": {
        "url": f"http://127.0.0.1:{dead}/mcp",
        "oauth": {"clientId": "cid"},
    }}}))
    mgr = McpManager(config_file=str(cfg), connect_timeout=20.0)
    with _home(tmp_path):
        try:
            started = time.time()
            with pytest.raises(McpConfigError) as excinfo:
                mgr.begin_login("gmail")
        finally:
            mgr.close()
    # Reporting must not wait out the whole connect budget for an attempt that
    # already failed.
    assert time.time() - started < 10, time.time() - started
    assert "authorization" not in str(excinfo.value).lower(), excinfo.value
    print("a failed login says what actually went wrong: OK")


def test_a_server_that_needs_no_login_is_not_reported_as_a_timeout(tmp_path: Path):
    """Connecting straight through means the endpoint is not OAuth-protected."""
    proc, upstream = _start_mcp_server(tmp_path / "server")
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"servers": {"gmail": {
        "url": upstream,
        "oauth": {"clientId": "cid"},
    }}}))
    mgr = McpManager(config_file=str(cfg), connect_timeout=20.0)
    with _home(tmp_path):
        try:
            started = time.time()
            with pytest.raises(McpConfigError) as excinfo:
                mgr.begin_login("gmail")
        finally:
            mgr.close()
            proc.terminate()
            proc.wait(timeout=10)
    assert time.time() - started < 15, time.time() - started
    assert "does not use OAuth" in str(excinfo.value), excinfo.value
    print("a server that needs no login is not reported as a timeout: OK")


def test_parse_settings_reads_the_documented_shape():
    cfg = {
        "serverUrl": "https://example.test/mcp/v1",
        "oauth": {
            "clientId": "ID",
            "clientSecret": "${env:LUDVART_TEST_SECRET}",
            "scopes": ["a.read", "b.write"],
        },
    }
    os.environ["LUDVART_TEST_SECRET"] = "shhh"
    settings = mcpauth.parse_settings(cfg, _expand)
    assert settings.client_id == "ID"
    # The secret can stay out of mcp.json entirely.
    assert settings.client_secret == "shhh"
    assert settings.scopes == "a.read b.write"
    assert settings.redirect_uri == mcpauth.DEFAULT_REDIRECT_URI
    assert settings.dynamic is False

    # "oauth": true means "use OAuth, work the details out by discovery".
    bare = mcpauth.parse_settings({"url": "u", "oauth": True}, _expand)
    assert bare is not None and bare.dynamic is True
    assert mcpauth.parse_settings({"url": "u"}, _expand) is None
    assert mcpauth.parse_settings({"url": "u", "oauth": False}, _expand) is None
    print("parse_settings reads the documented shape: OK")


def test_parse_redirect_accepts_what_lands_in_a_paste_buffer():
    assert mcpauth.parse_redirect(
        "http://127.0.0.1:33418/ludvart/callback?code=abc&state=xyz"
    ) == ("abc", "xyz")
    assert mcpauth.parse_redirect("code=abc&state=xyz") == ("abc", "xyz")
    # Some browsers copy the URL wrapped in quotes.
    assert mcpauth.parse_redirect('"http://h/cb?code=abc&state=s"') == ("abc", "s")
    # A code pasted on its own still works; the SDK rejects the missing state.
    assert mcpauth.parse_redirect("just-the-code") == ("just-the-code", None)

    for bad, expected in [
        ("http://h/cb?error=access_denied", "access_denied"),
        ("http://h/cb?state=s", "no 'code'"),
        ("   ", "nothing pasted"),
    ]:
        try:
            mcpauth.parse_redirect(bad)
        except ValueError as exc:
            assert expected in str(exc), (bad, exc)
        else:
            raise AssertionError(f"expected {bad!r} to be rejected")
    print("parse_redirect accepts what lands in a paste buffer: OK")


def test_configured_credentials_are_not_copied_into_the_token_file(tmp_path: Path):
    """mcp.json is the source of truth; a second copy of a secret is a liability."""
    from mcp.shared.auth import OAuthClientInformationFull

    with _home(tmp_path):
        settings = mcpauth.OAuthSettings(client_id="cid", client_secret="sec")
        storage = mcpauth.FileTokenStorage("srv", settings)
        asyncio.run(storage.set_client_info(OAuthClientInformationFull(
            client_id="issued-elsewhere",
            redirect_uris=[settings.redirect_uri],
        )))
        assert not os.path.exists(storage.path), storage.path
        info = asyncio.run(storage.get_client_info())
        assert info.client_id == "cid", info
    print("configured credentials are not copied into the token file: OK")


def test_credentials_for_a_different_client_are_not_reused(tmp_path: Path):
    """Repointing a server at another client must not resurrect stale tokens."""
    from mcp.shared.auth import OAuthToken

    with _home(tmp_path):
        first = mcpauth.FileTokenStorage("srv", mcpauth.OAuthSettings(client_id="one"))
        asyncio.run(first.set_tokens(OAuthToken(access_token="for-client-one")))
        assert first.has_tokens()

        second = mcpauth.FileTokenStorage("srv", mcpauth.OAuthSettings(client_id="two"))
        assert not second.has_tokens()
        assert asyncio.run(second.get_tokens()) is None
    print("credentials for a different client are not reused: OK")


def test_login_rejects_servers_it_cannot_authorize(tmp_path: Path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"servers": {
        "plain": {"url": "http://example.test/mcp"},
        "gmail": {"serverUrl": "http://example.test/mcp", "oauth": {"clientId": "x"}},
    }}))
    mgr = McpManager(config_file=str(cfg))
    assert mgr.oauth_servers() == ["gmail"]
    for name, expected in [("nope", "no MCP server"), ("plain", "not configured")]:
        try:
            mgr.begin_login(name)
        except McpConfigError as exc:
            assert expected in str(exc), exc
        else:
            raise AssertionError(f"expected {name!r} to be rejected")
    try:
        mgr.complete_login("gmail", "code=x")
    except McpConfigError as exc:
        assert "no login in progress" in str(exc), exc
    else:
        raise AssertionError("expected an unstarted login to be rejected")
    print("login rejects servers it cannot authorize: OK")


def main():
    root = Path(tempfile.mkdtemp())
    home = os.environ.get("HOME")
    try:
        test_a_configured_client_authorizes_and_stores_its_tokens(root / "a")
        test_a_client_without_credentials_registers_itself(root / "b")
        test_a_stored_token_is_reused_without_a_second_login(root / "c")
        test_an_expired_access_token_is_refreshed_not_re_authorized(root / "g")
        test_an_unauthorized_server_says_how_to_authorize(root / "d")
        test_the_transport_is_given_the_provider_to_authorize_with(root / "h")
        test_login_then_auth_makes_a_protected_servers_tools_usable(root / "i")
        test_a_failed_login_says_what_actually_went_wrong(root / "k")
        test_a_server_that_needs_no_login_is_not_reported_as_a_timeout(root / "l")
        test_parse_settings_reads_the_documented_shape()
        test_parse_redirect_accepts_what_lands_in_a_paste_buffer()
        test_credentials_for_a_different_client_are_not_reused(root / "e")
        test_configured_credentials_are_not_copied_into_the_token_file(root / "j")
        test_login_rejects_servers_it_cannot_authorize(root / "f")
    finally:
        if home is not None:
            os.environ["HOME"] = home
    print("\nALL MCP OAuth tests passed.")


if __name__ == "__main__":
    main()
