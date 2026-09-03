"""OAuth authorization for remote MCP servers (src/ludvart/mcpauth.py).

The flow is exercised against a *real* HTTP authorization server run in a
thread, standing in front of a *real* MCP server: metadata discovery, PKCE, the
state check, the token exchange, token storage and refresh are all the real
implementations. Only the browser is simulated, by handing back the redirect URL
it would have landed on.

The fixture copies how Google's Gmail MCP endpoint actually behaves -- the tool
list is served in the clear and only ``tools/call`` demands a bearer token --
because that is the case a "wait to be challenged at connect time" design gets
wrong.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_mcp_oauth.py
"""

import asyncio
import json
import os
import socket
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


class _AuthServer(BaseHTTPRequestHandler):
    """An OAuth 2.1 authorization server and a token-checking MCP proxy."""

    seen_token_requests: list = []
    registrations: list = []
    #: When set, ``/mcp`` forwards to this real MCP server.
    mcp_upstream: str = ""
    #: When false, ``/.well-known`` returns 404 so discovery has nothing to find.
    metadata: bool = True
    #: Which resource-metadata paths exist. Google publishes only the one under
    #: the resource path, so that is the default.
    prm_suffixes: tuple = ("/mcp",)
    #: Where the authorization server metadata lives. An OIDC-only provider
    #: publishes just the second one.
    as_paths: tuple = (
        "/.well-known/oauth-authorization-server",
        "/.well-known/openid-configuration",
    )

    def log_message(self, *args):  # keep the test output clean
        pass

    @property
    def _base(self):
        return f"http://{self.headers.get('Host')}"

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _empty(self, code, headers=()):
        self.send_response(code)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _proxy_mcp(self, method: str):
        """Forward to the real MCP server; only tool calls need a token."""
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        if b'"tools/call"' in body and (
            self.headers.get("Authorization") != f"Bearer {ACCESS_TOKEN}"
        ):
            self._empty(401, [(
                "WWW-Authenticate",
                f'Bearer resource_metadata="{self._base}'
                '/.well-known/oauth-protected-resource/mcp"',
            )])
            return
        if not _AuthServer.mcp_upstream:
            # No real server behind this one: the test only cares that the
            # request was let through.
            self._json(200, {"jsonrpc": "2.0", "id": 7, "result": {}})
            return
        skip = {"host", "content-length", "connection"}
        headers = {k: v for k, v in self.headers.items() if k.lower() not in skip}
        upstream = httpx.request(
            method, _AuthServer.mcp_upstream, headers=headers, content=body, timeout=20
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
        if not _AuthServer.metadata or not path.startswith("/.well-known/"):
            self._empty(404)
            return
        if path.startswith("/.well-known/oauth-protected-resource"):
            suffix = path[len("/.well-known/oauth-protected-resource"):]
            if suffix not in _AuthServer.prm_suffixes:
                self._empty(404)
                return
            self._json(200, {
                "resource": f"{self._base}{suffix or '/mcp'}",
                "authorization_servers": [self._base],
                "scopes_supported": ["mail.read", "mail.send"],
            })
            return
        if path not in _AuthServer.as_paths:
            self._empty(404)
            return
        self._json(200, {
            "issuer": self._base,
            "authorization_endpoint": f"{self._base}/authorize",
            "token_endpoint": f"{self._base}/token",
            "registration_endpoint": f"{self._base}/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
        })

    def do_POST(self):
        path = urlparse(self.path).path
        if path.startswith("/mcp"):
            self._proxy_mcp("POST")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode()
        if path == "/register":
            body = json.loads(raw or "{}")
            _AuthServer.registrations.append(body)
            self._json(201, {
                "client_id": "dynamically-registered",
                "client_secret": "dyn-secret",
                "redirect_uris": body.get("redirect_uris", []),
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
                self._json(400, {
                    "error": "invalid_request",
                    "error_description": "PKCE is required",
                })
                return
            self._json(200, {
                "access_token": ACCESS_TOKEN,
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": REFRESH_TOKEN,
            })
            return
        self._empty(404)


def _serve(metadata=True, prm_suffixes=("/mcp",), as_paths=None):
    _AuthServer.seen_token_requests = []
    _AuthServer.registrations = []
    _AuthServer.mcp_upstream = ""
    _AuthServer.metadata = metadata
    _AuthServer.prm_suffixes = prm_suffixes
    _AuthServer.as_paths = as_paths or (
        "/.well-known/oauth-authorization-server",
        "/.well-known/openid-configuration",
    )
    httpd = HTTPServer(("127.0.0.1", 0), _AuthServer)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_port}"


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


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _start_mcp_server(root: Path):
    """Spawn the real MCP server and wait for it to accept connections."""
    root.mkdir(parents=True, exist_ok=True)
    port = _free_port()
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


def _paste_back(url: str) -> str:
    """Play the part of the browser: approve, and report where it landed."""
    params = parse_qs(urlparse(url).query)
    assert params["code_challenge_method"] == ["S256"], params
    assert params["response_type"] == ["code"], params
    return (
        f"{params['redirect_uri'][0]}?code=granted-code&state={params['state'][0]}"
    )


def test_login_then_auth_makes_a_protected_servers_tools_usable(tmp_path: Path):
    """The feature end to end, driven through the panel commands themselves.

    The fixture only demands a token for ``tools/call``, exactly as the Gmail
    endpoint does, so the tools list has to be discoverable before any login and
    the tool call has to start working after one.
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
    mgr = McpManager(config_file=str(cfg), connect_timeout=30.0, call_timeout=15.0)

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
            # Discovery is unauthenticated here, so the tools must show up
            # before any login -- but calling one must not silently fail.
            assert "1 tool(s)" in "\n".join(run("mcp_refresh"))
            denied = mgr.call_tool("mcp_gmail_whoami", {})
            assert "/mcp_login gmail" in denied, denied

            lines = run("mcp_login gmail")
            url = next(line for line in lines if line.startswith("http"))
            joined = "\n".join(lines)
            assert "/mcp_auth gmail" in joined, joined
            # The resource metadata says which scopes it accepts, so the user
            # does not have to look them up to write a config.
            assert "mail.read mail.send" in joined, joined
            params = parse_qs(urlparse(url).query)
            assert params["client_id"] == ["cid"], params
            assert params["scope"] == ["mail.read mail.send"], params

            done = "\n".join(run(f"mcp_auth gmail {_paste_back(url)}"))
            assert "1/1 server(s) connected" in done, done
            assert mgr.call_tool("mcp_gmail_whoami", {}) == f"Bearer {ACCESS_TOKEN}"
        finally:
            mgr.close()
            httpd.shutdown()
            proc.terminate()
            proc.wait(timeout=10)
    print("login then auth makes a protected server's tools usable: OK")


def _post(server: str, url: str, settings: mcpauth.OAuthSettings) -> httpx.Response:
    """Send one authorized MCP request, the way the SDK's client would."""

    async def go():
        transport = mcpauth.AuthTransport(server, settings)
        async with httpx.AsyncClient(transport=transport, timeout=20) as client:
            return await client.post(
                url, json={"jsonrpc": "2.0", "id": 7, "method": "tools/call"}
            )

    return asyncio.run(go())


class _ProxyServer(BaseHTTPRequestHandler):
    """A forward proxy, as sits between this network and the internet."""

    seen: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        _ProxyServer.seen.append(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        if not self.path.startswith("http://"):
            self.send_response(400)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        skip = {"host", "content-length", "connection", "proxy-connection"}
        upstream = httpx.request(
            "POST", self.path, content=body, timeout=20, trust_env=False,
            headers={k: v for k, v in self.headers.items() if k.lower() not in skip},
        )
        payload = upstream.content
        self.send_response(upstream.status_code)
        self.send_header(
            "Content-Type", upstream.headers.get("content-type", "application/json")
        )
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextmanager
def _proxied_through(base: str):
    """Route outbound HTTP through ``base``, as this network's setup does."""
    names = ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY",
             "no_proxy", "NO_PROXY")
    prior = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    os.environ["http_proxy"] = base
    try:
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_requests_still_go_through_the_environments_proxy(tmp_path: Path):
    """Many networks only reach the internet through a proxy.

    httpx applies the environment's proxy settings when it builds a client's
    transport itself; handing it a ready-made one silently skips that, and every
    request then hangs until it times out.
    """
    _ProxyServer.seen = []
    proxy = HTTPServer(("127.0.0.1", 0), _ProxyServer)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            settings = mcpauth.OAuthSettings(client_id="cid")
            mcpauth.TokenStore("srv", settings).write(access_token=ACCESS_TOKEN)
            with _proxied_through(f"http://127.0.0.1:{proxy.server_port}"):
                response = _post("srv", f"{base}/mcp", settings)
        finally:
            httpd.shutdown()
            proxy.shutdown()
    assert response.status_code == 200, response.text
    assert _ProxyServer.seen == [f"{base}/mcp"], _ProxyServer.seen
    print("requests still go through the environment's proxy: OK")


def test_an_expired_token_is_refreshed_without_a_second_login(tmp_path: Path):
    """Access tokens last an hour; logging in again every hour is not a design."""
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            settings = mcpauth.OAuthSettings(client_id="cid", client_secret="sec")
            pending = mcpauth.start_login("srv", f"{base}/mcp", settings)
            mcpauth.finish_login(pending, _paste_back(pending.url))

            # Age the stored token past its expiry, as a restart tomorrow would.
            store = mcpauth.TokenStore("srv", settings)
            store.write(expires_at=1.0)

            response = _post("srv", f"{base}/mcp", settings)
            assert response.status_code == 200, response.text
            assert "error" not in response.json(), response.text
            assert store.read()["access_token"] == ACCESS_TOKEN
        finally:
            httpd.shutdown()

    grants = [f.get("grant_type") for f in _AuthServer.seen_token_requests]
    assert grants == ["authorization_code", "refresh_token"], grants
    # A confidential client has to prove who it is on every grant.
    assert all(f.get("client_secret") == "sec" for f in _AuthServer.seen_token_requests)
    print("an expired token is refreshed without a second login: OK")


def test_a_token_the_server_stopped_accepting_is_renewed(tmp_path: Path):
    """Access can end before its stated expiry, and a 401 is the only warning."""
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            settings = mcpauth.OAuthSettings(client_id="cid")
            store = mcpauth.TokenStore("srv", settings)
            store.write(
                access_token="no-longer-accepted",
                refresh_token=REFRESH_TOKEN,
                token_endpoint=f"{base}/token",
                expires_at=time.time() + 3600,
            )
            response = _post("srv", f"{base}/mcp", settings)
            assert response.status_code == 200, response.text
            assert "error" not in response.json(), response.text
            assert store.read()["access_token"] == ACCESS_TOKEN
        finally:
            httpd.shutdown()
    print("a token the server stopped accepting is renewed: OK")


def test_a_rejected_refresh_asks_for_a_new_login(tmp_path: Path):
    """Revoked access must not look like an ordinary transport failure.

    The answer has to come back *in band* as a JSON-RPC error: the SDK sends
    each request in its own task, so an exception raised here would leave the
    caller waiting for a reply that never comes.
    """
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            settings = mcpauth.OAuthSettings(client_id="cid")
            store = mcpauth.TokenStore("srv", settings)
            store.write(
                access_token="stale",
                refresh_token="revoked",
                token_endpoint=f"{base}/token",
                expires_at=1.0,
            )
            response = _post("srv", f"{base}/mcp", settings)
        finally:
            httpd.shutdown()
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == 7, body
    assert "/mcp_login srv" in body["error"]["message"], body
    print("a rejected refresh asks for a new login: OK")


def test_a_client_without_credentials_registers_itself(tmp_path: Path):
    """No clientId means the server issues one (RFC 7591), not that it fails."""
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            settings = mcpauth.OAuthSettings()
            pending = mcpauth.start_login("srv", f"{base}/mcp", settings)
            assert pending.client_id == "dynamically-registered", pending.client_id
            mcpauth.finish_login(pending, _paste_back(pending.url))
        finally:
            httpd.shutdown()

    assert len(_AuthServer.registrations) == 1, _AuthServer.registrations
    assert _AuthServer.seen_token_requests[-1]["client_id"] == "dynamically-registered"
    print("a client without credentials registers itself: OK")


def test_a_login_that_cannot_start_says_why(tmp_path: Path):
    """"It did not ask for authorization" is a guess; discovery gives a reason."""
    httpd, base = _serve(metadata=False)
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"servers": {"gmail": {
        "url": f"{base}/mcp",
        "oauth": {"clientId": "cid"},
    }}}))
    mgr = McpManager(config_file=str(cfg))
    with _home(tmp_path):
        try:
            started = time.time()
            with pytest.raises(McpConfigError) as excinfo:
                mgr.begin_login("gmail")
        finally:
            mgr.close()
            httpd.shutdown()
    assert "no OAuth metadata" in str(excinfo.value), excinfo.value
    assert time.time() - started < 15, time.time() - started
    print("a login that cannot start says why: OK")


def test_resource_metadata_is_found_wherever_the_server_publishes_it(tmp_path: Path):
    """Providers disagree about where each metadata document lives.

    RFC 9728 puts the resource document under the resource path, but not
    everyone does; and an OIDC provider may only publish the authorization
    server's details as an openid-configuration.
    """
    cases = [
        (("/mcp",), None),
        (("",), None),
        (("/mcp",), ("/.well-known/openid-configuration",)),
    ]
    for prm_suffixes, as_paths in cases:
        httpd, base = _serve(prm_suffixes=prm_suffixes, as_paths=as_paths)
        try:
            endpoints = mcpauth.discover(f"{base}/mcp")
        finally:
            httpd.shutdown()
        where = (prm_suffixes, as_paths)
        assert endpoints.authorization == f"{base}/authorize", where
        assert endpoints.token == f"{base}/token", where
        # The scopes the resource accepts come from that document, so losing it
        # would silently ask for none.
        assert endpoints.scopes == "mail.read mail.send", where
    print("resource metadata is found wherever the server publishes it: OK")


def test_logging_in_again_does_not_keep_the_old_token(tmp_path: Path):
    """A login abandoned half way must not leave the previous access in place."""
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            settings = mcpauth.OAuthSettings(client_id="cid")
            store = mcpauth.TokenStore("srv", settings)
            store.write(access_token="from-an-earlier-login")
            mcpauth.start_login("srv", f"{base}/mcp", settings)
            assert not store.has_tokens(), store.read()
        finally:
            httpd.shutdown()
    print("logging in again does not keep the old token: OK")


def test_the_config_decides_the_scopes_and_the_extra_parameters(tmp_path: Path):
    """Everything the provider needs has to be reachable from mcp.json."""
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            settings = mcpauth.parse_settings({"oauth": {
                "clientId": "cid",
                "scopes": "only.this",
                "authorizationParams": {"prompt": "select_account"},
            }}, _expand)
            pending = mcpauth.start_login("srv", f"{base}/mcp", settings)
        finally:
            httpd.shutdown()
    params = parse_qs(urlparse(pending.url).query)
    assert params["scope"] == ["only.this"], params
    assert params["prompt"] == ["select_account"], params
    print("the config decides the scopes and the extra parameters: OK")


def test_the_redirect_target_is_one_a_provider_will_accept(tmp_path: Path):
    """Providers match the redirect against the client's registered list exactly.

    Google's desktop clients accept any loopback port but no invented path, and
    the out-of-band alternative is withdrawn, so the default has to stay a bare
    loopback address -- and has to be overridable for clients registered with
    something else.
    """
    parsed = urlparse(mcpauth.DEFAULT_REDIRECT_URI)
    assert parsed.scheme == "http", mcpauth.DEFAULT_REDIRECT_URI
    assert parsed.hostname in ("127.0.0.1", "localhost"), mcpauth.DEFAULT_REDIRECT_URI
    assert parsed.path in ("", "/"), mcpauth.DEFAULT_REDIRECT_URI

    mine = "http://localhost:9004/oauth2callback"
    settings = mcpauth.parse_settings(
        {"oauth": {"clientId": "cid", "redirectUri": mine}}, _expand
    )
    assert settings.redirect_uri == mine
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            pending = mcpauth.start_login("srv", f"{base}/mcp", settings)
            assert parse_qs(urlparse(pending.url).query)["redirect_uri"] == [mine]
            mcpauth.finish_login(pending, f"{mine}?code=c&state={pending.state}")
        finally:
            httpd.shutdown()
    # The provider checks it again at the token endpoint, so both must agree.
    assert _AuthServer.seen_token_requests[-1]["redirect_uri"] == mine
    print("the redirect target is one a provider will accept: OK")


def test_a_reachable_redirect_finishes_the_login_without_a_paste(tmp_path: Path):
    """A tunnel from the browser to the backend removes the copying step.

    The backend usually runs on another host, but an SSH forward can carry
    localhost to it, and then the browser delivers the code itself.
    """
    port = _free_port()
    mine = f"http://127.0.0.1:{port}"
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            settings = mcpauth.OAuthSettings(client_id="cid", redirect_uri=mine)
            pending = mcpauth.start_login("srv", f"{base}/mcp", settings)
            assert pending.catcher is not None
            # Nothing has arrived yet, so there is nothing to redeem.
            try:
                mcpauth.finish_login(pending)
            except RuntimeError as exc:
                assert "not reached the redirect address yet" in str(exc), exc
            else:
                raise AssertionError("an empty login was accepted")

            reply = httpx.get(f"{mine}/?code=granted&state={pending.state}", timeout=10)
            assert reply.status_code == 200, reply.status_code
            mcpauth.finish_login(pending)
            assert mcpauth.TokenStore("srv", settings).has_tokens()
        finally:
            if pending.catcher is not None:
                pending.catcher.close()
            httpd.shutdown()
    assert _AuthServer.seen_token_requests[-1]["code"] == "granted"
    print("a reachable redirect finishes the login without a paste: OK")


def test_the_browser_alone_finishes_the_login(tmp_path: Path):
    """Approving in the browser is the whole flow when the redirect arrives.

    The code is redeemed as it lands and the server's tools are picked up, so
    there is no second command and nothing to copy.
    """
    port = _free_port()
    mine = f"http://127.0.0.1:{port}"
    httpd, base = _serve()
    picked_up = []
    with _home(tmp_path):
        try:
            settings = mcpauth.OAuthSettings(client_id="cid", redirect_uri=mine)
            pending = mcpauth.start_login(
                "srv", f"{base}/mcp", settings,
                on_authorized=lambda: picked_up.append(True),
            )
            assert pending.catcher is not None
            reply = httpx.get(f"{mine}/?code=granted&state={pending.state}", timeout=10)
            assert reply.status_code == 200, reply.status_code
            assert "authorized" in reply.text.lower(), reply.text
            assert mcpauth.TokenStore("srv", settings).has_tokens()
            assert picked_up == [True], picked_up
            # Redeeming twice would spend a code the provider has retired.
            again = httpx.get(f"{mine}/?code=granted&state={pending.state}", timeout=10)
            assert "already authorized" in again.text.lower(), again.text
            # /mcp_auth is still allowed to be run, and has nothing left to do.
            mcpauth.finish_login(pending)
        finally:
            pending.catcher.close()
            httpd.shutdown()
    assert len(_AuthServer.seen_token_requests) == 1, _AuthServer.seen_token_requests
    print("the browser alone finishes the login: OK")


def test_a_redirect_that_cannot_be_redeemed_says_so_in_the_browser(tmp_path: Path):
    """The browser is the only place this can be reported: nothing is waiting."""
    port = _free_port()
    mine = f"http://127.0.0.1:{port}"
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            settings = mcpauth.OAuthSettings(client_id="cid", redirect_uri=mine)
            pending = mcpauth.start_login("srv", f"{base}/mcp", settings)
            reply = httpx.get(f"{mine}/?code=granted&state=somebody-elses", timeout=10)
            assert "could not complete" in reply.text.lower(), reply.text
            assert not mcpauth.TokenStore("srv", settings).has_tokens()
            # The panel command surfaces the same reason rather than a timeout.
            try:
                mcpauth.finish_login(pending)
            except RuntimeError as exc:
                assert "state mismatch" in str(exc), exc
            else:
                raise AssertionError("a mismatched state was accepted")
        finally:
            pending.catcher.close()
            httpd.shutdown()
    print("a redirect that cannot be redeemed says so in the browser: OK")


def test_an_idle_browser_connection_does_not_wedge_the_listener(tmp_path: Path):
    """Browsers open speculative connections and send nothing on them.

    A listener that handles one connection at a time sits blocked on such a
    socket, and shutting it down then waits for a request that never comes --
    which strands the login after its tokens were already stored.
    """
    port = _free_port()
    catcher = mcpauth.RedirectCatcher(f"http://127.0.0.1:{port}")
    lurker = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        started = time.time()
        reply = httpx.get(f"http://127.0.0.1:{port}/?code=granted", timeout=30)
        served = time.time() - started
        assert reply.status_code == 200, reply.status_code
        assert "code=granted" in catcher.caught, catcher.caught
        started = time.time()
        catcher.close()
        closed = time.time() - started
    finally:
        lurker.close()
    assert served < 5, f"the redirect waited {served:.1f}s behind an idle socket"
    assert closed < 5, f"closing took {closed:.1f}s"
    print("an idle browser connection does not wedge the listener: OK")


def test_a_redirect_we_cannot_listen_on_still_leaves_the_paste_flow(tmp_path: Path):
    """Losing the listener must not lose the login: most setups have no tunnel."""
    blocker = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = blocker.server_port
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            settings = mcpauth.OAuthSettings(
                client_id="cid", redirect_uri=f"http://127.0.0.1:{port}"
            )
            pending = mcpauth.start_login("srv", f"{base}/mcp", settings)
            assert pending.catcher is None
            mcpauth.finish_login(pending, f"http://h/cb?code=c&state={pending.state}")
            assert mcpauth.TokenStore("srv", settings).has_tokens()
        finally:
            blocker.server_close()
            httpd.shutdown()
    print("a redirect we cannot listen on still leaves the paste flow: OK")


def test_a_url_from_another_attempt_is_refused(tmp_path: Path):
    """The state parameter is only worth sending if it is checked coming back."""
    httpd, base = _serve()
    with _home(tmp_path):
        try:
            settings = mcpauth.OAuthSettings(client_id="cid")
            first = mcpauth.start_login("srv", f"{base}/mcp", settings)
            second = mcpauth.start_login("srv", f"{base}/mcp", settings)
            with pytest.raises(RuntimeError) as excinfo:
                mcpauth.finish_login(second, _paste_back(first.url))
            assert "state mismatch" in str(excinfo.value), excinfo.value
        finally:
            httpd.shutdown()
    print("a url from another attempt is refused: OK")


def test_a_google_login_asks_for_offline_access():
    """Google issues no refresh token unless asked, and re-consents on request."""
    assert mcpauth._default_auth_params("https://accounts.google.com/o/oauth2/v2/auth") == {
        "access_type": "offline",
        "prompt": "consent",
    }
    assert mcpauth._default_auth_params("https://example.test/authorize") == {}
    print("a google login asks for offline access: OK")


def test_the_transport_is_given_the_provider_to_authorize_with(tmp_path: Path):
    """Building an authorizing client is pointless unless the connection uses it."""
    import ludvart.mcp as mcp_mod

    seen = {}

    def fake_transport(url, headers=None, httpx_client_factory=None, **kw):
        seen[url] = httpx_client_factory
        raise RuntimeError("stop here: only the wiring is under test")

    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"servers": {
        "plain": {"url": "http://plain.test/mcp"},
        "gmail": {"serverUrl": "http://gmail.test/mcp", "oauth": {"clientId": "cid"}},
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
    for url in ("http://gmail.test/mcp", "http://sse.test/mcp"):
        client = seen[url]()
        assert isinstance(client._transport, mcpauth.AuthTransport), url
    assert sorted(status.servers) == ["gmail", "plain", "streamed"], status.servers
    print("the transport is given the provider to authorize with: OK")


def test_parse_settings_reads_the_documented_shape():
    cfg = {
        "serverUrl": "https://example.test/mcp/v1",
        "oauth": {
            "clientId": "ID",
            "clientSecret": "${env:LUDVART_TEST_SECRET}",
            "scopes": ["a.read", "b.write"],
        },
    }
    prior = os.environ.get("LUDVART_TEST_SECRET")
    os.environ["LUDVART_TEST_SECRET"] = "shhh"
    try:
        settings = mcpauth.parse_settings(cfg, _expand)
    finally:
        if prior is None:
            os.environ.pop("LUDVART_TEST_SECRET", None)
        else:
            os.environ["LUDVART_TEST_SECRET"] = prior
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
    # A code pasted on its own still works.
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


def test_tokens_are_stored_where_only_their_owner_can_read_them(tmp_path: Path):
    with _home(tmp_path):
        store = mcpauth.TokenStore("srv", mcpauth.OAuthSettings(client_id="one"))
        store.save_tokens(
            {"access_token": "a", "refresh_token": "r", "expires_in": 3600},
            "https://example.test/token",
        )
        path = Path(store.path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, oct(path.stat().st_mode)
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert store.read()["refresh_token"] == "r"

        # Repointing the server at another client must not resurrect these.
        other = mcpauth.TokenStore("srv", mcpauth.OAuthSettings(client_id="two"))
        assert not other.has_tokens()
    print("tokens are stored where only their owner can read them: OK")


def test_a_renewal_keeps_the_refresh_token_it_was_not_sent(tmp_path: Path):
    """Most providers return a refresh token once, on the first exchange only."""
    with _home(tmp_path):
        store = mcpauth.TokenStore("srv", mcpauth.OAuthSettings(client_id="cid"))
        first = store.save_tokens(
            {"access_token": "a1", "refresh_token": "r1", "expires_in": 3600},
            "https://example.test/token",
        )
        store.save_tokens({"access_token": "a2", "expires_in": 3600}, "t", first)
        assert store.read()["refresh_token"] == "r1", store.read()
    print("a renewal keeps the refresh token it was not sent: OK")


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
    test_login_then_auth_makes_a_protected_servers_tools_usable(root / "a")
    test_requests_still_go_through_the_environments_proxy(root / "a2")
    test_an_expired_token_is_refreshed_without_a_second_login(root / "b")
    test_a_token_the_server_stopped_accepting_is_renewed(root / "b2")
    test_a_rejected_refresh_asks_for_a_new_login(root / "c")
    test_a_client_without_credentials_registers_itself(root / "d")
    test_a_login_that_cannot_start_says_why(root / "e")
    test_resource_metadata_is_found_wherever_the_server_publishes_it(root / "e2")
    test_logging_in_again_does_not_keep_the_old_token(root / "e3")
    test_the_config_decides_the_scopes_and_the_extra_parameters(root / "e4")
    test_the_redirect_target_is_one_a_provider_will_accept(root / "e5")
    test_a_reachable_redirect_finishes_the_login_without_a_paste(root / "e6")
    test_the_browser_alone_finishes_the_login(root / "e7")
    test_a_redirect_that_cannot_be_redeemed_says_so_in_the_browser(root / "e8")
    test_an_idle_browser_connection_does_not_wedge_the_listener(root / "e9")
    test_a_redirect_we_cannot_listen_on_still_leaves_the_paste_flow(root / "e10")
    test_a_url_from_another_attempt_is_refused(root / "f")
    test_a_google_login_asks_for_offline_access()
    test_the_transport_is_given_the_provider_to_authorize_with(root / "g")
    test_parse_settings_reads_the_documented_shape()
    test_parse_redirect_accepts_what_lands_in_a_paste_buffer()
    test_tokens_are_stored_where_only_their_owner_can_read_them(root / "h")
    test_a_renewal_keeps_the_refresh_token_it_was_not_sent(root / "i")
    test_login_rejects_servers_it_cannot_authorize(root / "j")
    print("\nALL MCP OAuth tests passed.")


if __name__ == "__main__":
    main()
