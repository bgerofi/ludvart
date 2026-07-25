"""Tests for the GitHub Copilot LiteLLM gateway integration.

Covers the config helpers (``write_copilot_conf`` / ``copilot_model`` /
``copilot_provider_config``) and the :class:`CopilotGateway` lifecycle. The
gateway lifecycle is exercised against a tiny fake ``litellm`` CLI stub, so no
real GitHub Copilot subscription or network access is required.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_copilot_gateway.py
"""

import json
import os
import stat
import tempfile
import textwrap
import time
import urllib.error
import urllib.request

from ludvart import gateway
from ludvart.gateway import GATEWAY_API_KEY, CopilotGateway, GatewayError
from ludvart.llm import _load_conf, copilot_provider_config, write_copilot_conf


# A stand-in for the real `litellm` proxy: serves 200 on /health/liveliness so
# the gateway's readiness poll succeeds, without any auth or model backend.
_FAKE_SERVER = textwrap.dedent(
    """\
    #!{python}
    import argparse
    import http.server

    p = argparse.ArgumentParser()
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--model")
    args, _ = p.parse_known_args()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health/liveliness":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"I'm alive!")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):
            pass

    http.server.HTTPServer((args.host, args.port), H).serve_forever()
    """
)

# A stub that exits immediately, to exercise the "gateway died" error path.
_FAKE_DEAD = textwrap.dedent(
    """\
    #!{python}
    import sys
    sys.exit(3)
    """
)


def _make_cli(tmp, body):
    import sys as _sys

    path = os.path.join(tmp, "fake_litellm")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body.format(python=_sys.executable))
    os.chmod(path, 0o755)
    return path


def test_config_helpers(tmp):
    path = os.path.join(tmp, "llm.conf")
    write_copilot_conf("gpt-4o", path=path)
    conf = _load_conf(path)
    assert conf["COPILOT_MODEL"] == "gpt-4o"
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, oct(mode)

    cfg = copilot_provider_config(
        "http://127.0.0.1:4000/", "github_copilot/gpt-4o", GATEWAY_API_KEY
    )
    assert cfg.name == "custom"
    assert cfg.api_url == "http://127.0.0.1:4000"  # trailing slash stripped
    assert cfg.api_key == GATEWAY_API_KEY
    assert cfg.model == "github_copilot/gpt-4o"
    assert cfg.context_window == 128_000  # gpt-4o known window
    print("config helpers: OK")


def test_gateway_model_and_url(tmp):
    gw = CopilotGateway("gpt-4o", port=12345)
    assert gw.base_url == "http://127.0.0.1:12345"
    assert gw.litellm_model == "github_copilot/gpt-4o"
    print("gateway model/url: OK")


def test_responses_gateway_writes_litellm_mode_config(tmp, monkeypatch_cli):
    monkeypatch_cli(_make_cli(tmp, _FAKE_SERVER))
    gw = CopilotGateway(
        "gpt-5.6-terra", api_mode="responses", log_path=os.path.join(tmp, "gw.log")
    )
    gw.start(timeout=15)
    try:
        assert gw._config_path is not None
        with open(gw._config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
        deployment = config["model_list"][0]
        assert deployment["model_info"]["mode"] == "responses"
        assert deployment["litellm_params"]["model"] == gw.litellm_model
    finally:
        config_path = gw._config_path
        gw.stop()
    assert config_path is not None and not os.path.exists(config_path)
    print("Responses gateway config: OK")


def test_gateway_start_and_stop(tmp, monkeypatch_cli):
    monkeypatch_cli(_make_cli(tmp, _FAKE_SERVER))
    gw = CopilotGateway("gpt-4o", log_path=os.path.join(tmp, "gw.log"))
    gw.start(timeout=15)
    try:
        with urllib.request.urlopen(
            gw.base_url + "/health/liveliness", timeout=3
        ) as resp:
            assert resp.status == 200
    finally:
        gw.stop()
    # After stop the process is gone and the port stops answering.
    assert gw._proc is None
    time.sleep(0.3)
    try:
        urllib.request.urlopen(gw.base_url + "/health/liveliness", timeout=1)
        raise AssertionError("gateway still serving after stop()")
    except (urllib.error.URLError, OSError):
        pass
    gw.stop()  # idempotent
    print("gateway start/stop: OK")


def test_gateway_survives_worker_thread(tmp, monkeypatch_cli):
    """Regression: a gateway started on a short-lived thread must keep running.

    ``/model use`` runs on a transient daemon worker thread. Because the gateway
    child uses ``PR_SET_PDEATHSIG`` (which, on Linux, fires when the *forking
    thread* dies), forking from that worker thread used to kill the gateway the
    moment the switch finished -- the next request then hit a dead proxy. The
    fork is now funnelled through a persistent spawner thread, so the gateway
    must still be serving after the starting thread has exited.
    """
    import threading

    monkeypatch_cli(_make_cli(tmp, _FAKE_SERVER))
    gw = CopilotGateway("gpt-4o", log_path=os.path.join(tmp, "gw.log"))
    error: list = []

    def worker():
        try:
            gw.start(timeout=15)
        except Exception as exc:  # pragma: no cover - surfaced via assert below
            error.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert not error, error
    # The starting thread is now dead; give any stray PR_SET_PDEATHSIG a chance.
    time.sleep(0.5)
    try:
        with urllib.request.urlopen(
            gw.base_url + "/health/liveliness", timeout=3
        ) as resp:
            assert resp.status == 200
    finally:
        gw.stop()
    print("gateway survives worker thread: OK")


def test_gateway_start_failure(tmp, monkeypatch_cli):
    monkeypatch_cli(_make_cli(tmp, _FAKE_DEAD))
    gw = CopilotGateway("gpt-4o", log_path=os.path.join(tmp, "gw.log"))
    try:
        gw.start(timeout=10)
    except GatewayError:
        print("gateway start failure: OK")
        return
    finally:
        gw.stop()
    raise AssertionError("expected GatewayError when the gateway exits early")


def test_missing_cli_raises(tmp, monkeypatch_cli):
    monkeypatch_cli(None)
    gw = CopilotGateway("gpt-4o", log_path=os.path.join(tmp, "gw.log"))
    try:
        gw.start(timeout=5)
    except GatewayError:
        print("missing CLI raises: OK")
        return
    raise AssertionError("expected GatewayError when litellm CLI is missing")


def test_shared_gateway_is_reused_instead_of_spawned(monkeypatch):
    """A pre-started gateway for the same model is used as-is, and not stopped.

    Booting a gateway costs tens of seconds, so a harness that starts many
    ludvart processes (the e2e suite) starts one up front and exports it.
    """
    from ludvart import backend as backend_mod
    from ludvart.gateway import (
        SHARED_GATEWAY_MODEL_ENV,
        SHARED_GATEWAY_URL_ENV,
        shared_gateway,
    )

    monkeypatch.setenv(SHARED_GATEWAY_URL_ENV, "http://127.0.0.1:9999")
    monkeypatch.setenv(SHARED_GATEWAY_MODEL_ENV, "gpt-5.6-terra")

    assert shared_gateway("gpt-5.6-terra") == "http://127.0.0.1:9999"
    # A gateway proxies exactly one model; a different one must not reuse it.
    assert shared_gateway("gpt-4o") is None

    def boom(*a, **k):
        raise AssertionError("a shared gateway was available; none should spawn")

    monkeypatch.setattr(gateway, "CopilotGateway", boom)
    reg = {
        "provider": "copilot",
        "model": "gpt-5.6-terra",
        "api_mode": "responses",
        "api_url": "",
        "api_key": "",
        "active": True,
    }
    built = backend_mod.build_backend(reg)

    assert built.client.config.api_url.startswith("http://127.0.0.1:9999")
    assert built.client.config.api_mode == "responses"
    # Not ours to stop: no gateway is attached, so teardown leaves it running.
    assert built.gateway is None
    built.stop()
    print("a shared gateway is reused instead of spawned: OK")


def _run():
    orig = gateway._litellm_cli
    tests = [
        test_config_helpers,
        test_gateway_model_and_url,
        test_responses_gateway_writes_litellm_mode_config,
        test_gateway_start_and_stop,
        test_gateway_survives_worker_thread,
        test_gateway_start_failure,
        test_missing_cli_raises,
    ]
    try:
        for fn in tests:
            with tempfile.TemporaryDirectory() as tmp:

                def monkeypatch_cli(path):
                    gateway._litellm_cli = lambda: path

                argc = fn.__code__.co_argcount
                if argc == 2:
                    fn(tmp, monkeypatch_cli)
                else:
                    fn(tmp)
                gateway._litellm_cli = orig
    finally:
        gateway._litellm_cli = orig
    print("\nall test_copilot_gateway tests passed")


if __name__ == "__main__":
    _run()
