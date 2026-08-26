"""ModelManager: switch / add / remove registered models.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_backend_model_manager.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart import backend, models as reg  # noqa: E402

_BUILD_BACKEND = backend.build_backend
_VERIFY_BACKEND = backend.verify_backend


class _FakeClient:
    def __init__(self, name="openai", ok=True):
        self.name = name
        self._ok = ok

    def verify(self):
        if not self._ok:
            raise RuntimeError("verify failed")


def _install_fakes(monkey_ok=True):
    """Stub build_backend/verify_backend so no real network/gateway is used."""

    def fake_build(reg_, *, status=None):
        # A registration may carry a private "_ok" flag to force a failure.
        return backend.Backend(_FakeClient(reg_["provider"], reg_.get("_ok", True)))

    backend.build_backend = fake_build
    backend.verify_backend = lambda b: b.client.verify()


def teardown_module():
    """Restore production functions after this module's global test doubles."""
    backend.build_backend = _BUILD_BACKEND
    backend.verify_backend = _VERIFY_BACKEND


def _reg(provider="openai", model="m", active=False, ok=True):
    r = {
        "provider": provider,
        "api_url": "http://x",
        "api_key": "k",
        "model": model,
        "context_window": 0,
        "active": active,
    }
    if not ok:
        r["_ok"] = False
    return r


def _mgr(regs, available=None):
    os.environ["LUDVART_MODELS_FILE"] = os.path.join(tempfile.mkdtemp(), "m.json")
    if available is None:
        available = [True] * len(regs)
    return backend.ModelManager(list(regs), list(available), _FakeClient(), None)


def test_describe_marks_active_and_availability():
    _install_fakes()
    m = _mgr([_reg(model="a", active=True), _reg(model="b")], [True, False])
    lines = m.describe()
    assert "in use" in lines[0] and "available" in lines[0]
    assert "unavailable" in lines[1]
    print("describe marks active/availability: OK")


def test_use_switches_active_and_persists():
    _install_fakes()
    m = _mgr([_reg(model="a", active=True), _reg(model="b")])
    ok, msg = m.use(1)
    assert ok, msg
    assert m.models[1]["active"] and not m.models[0]["active"]
    assert m.client.name == "openai"
    # Persisted to disk.
    saved = reg.load_models()
    assert saved[1]["active"]
    print("use switches active + persists: OK")


def test_use_unavailable_verifies_and_may_fail():
    _install_fakes()
    m = _mgr([_reg(model="a", active=True), _reg(model="b", ok=False)])
    ok, msg = m.use(1)
    assert not ok and "Could not switch" in msg
    assert m.available[1] is False
    assert m.models[0]["active"], "active must stay put on failure"
    print("use failure keeps active + marks unavailable: OK")


def test_add_appends_verified_without_switching():
    _install_fakes()
    m = _mgr([_reg(model="a", active=True)])
    ok, msg = m.add(_reg(model="b"))
    assert ok, msg
    assert [r["model"] for r in m.models] == ["a", "b"]
    assert m.models[0]["active"] and not m.models[1]["active"]
    assert m.available == [True, True]
    print("add appends verified without switching: OK")


def test_add_rejects_unverifiable():
    _install_fakes()
    m = _mgr([_reg(model="a", active=True)])
    ok, msg = m.add(_reg(model="bad", ok=False))
    assert not ok and "Verification failed" in msg
    assert len(m.models) == 1
    print("add rejects unverifiable: OK")


def test_use_builds_a_client_when_the_active_model_has_none():
    """A fresh registry marks its first entry active before anything is built.

    ``use`` used to read that flag alone and answer "Already using ...", so the
    backend kept its empty client and refused every turn as unregistered.
    """
    _install_fakes()
    os.environ["LUDVART_MODELS_FILE"] = os.path.join(tempfile.mkdtemp(), "m.json")
    m = backend.ModelManager([_reg(model="a", active=True)], [True], None, None)

    ok, msg = m.use(0)

    assert ok, msg
    assert m.client is not None, "the active model was never built"
    print("use builds a client for an active-but-unbuilt model: OK")


def test_a_failed_switch_stops_the_gateway_it_started():
    """The gateway is already up by the time verification fails.

    Nothing else references it once ``use`` returns, so a rejected switch used
    to strand a litellm proxy per attempt for the life of the backend.
    """
    stopped = []

    class _Gateway:
        def stop(self):
            stopped.append(1)

    def fake_build(reg_, *, status=None):
        client = _FakeClient(reg_["provider"], reg_.get("_ok", True))
        return backend.Backend(client, _Gateway())

    backend.build_backend = fake_build
    backend.verify_backend = lambda b: b.client.verify()
    m = _mgr([_reg(model="a", active=True), _reg(model="b", ok=False)])

    ok, _ = m.use(1)

    assert not ok
    assert stopped, "the gateway started for the failed switch was left running"
    print("a failed switch stops the gateway it started: OK")


def test_remove_forbids_active():
    _install_fakes()
    m = _mgr([_reg(model="a", active=True), _reg(model="b")])
    ok, msg = m.remove(0)
    assert not ok and "in use" in msg
    ok, msg = m.remove(1)
    assert ok, msg
    assert [r["model"] for r in m.models] == ["a"]
    print("remove forbids active, allows others: OK")


def main():
    test_describe_marks_active_and_availability()
    test_use_switches_active_and_persists()
    test_use_unavailable_verifies_and_may_fail()
    test_add_appends_verified_without_switching()
    test_add_rejects_unverifiable()
    test_use_builds_a_client_when_the_active_model_has_none()
    test_a_failed_switch_stops_the_gateway_it_started()
    test_remove_forbids_active()
    print("\nALL ModelManager tests passed.")


if __name__ == "__main__":
    main()
