"""Model registry (~/.ludvart/models.json): load/save, migration, CRUD.

Run:
    cd /local_home/bgerofi1/src/ludvart && source .venv/bin/activate \
        && python tests/test_models_registry.py
"""

import json
import os
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ludvart import models as reg  # noqa: E402


def _tmp_registry() -> str:
    path = os.path.join(tempfile.mkdtemp(), "models.json")
    os.environ["LUDVART_MODELS_FILE"] = path
    return path


def _sample(provider="openai", model="gpt-4o", active=False):
    return {
        "provider": provider,
        "api_url": "https://api.openai.com/v1",
        "api_key": "sk-test",
        "model": model,
        "context_window": 0,
        "active": active,
    }


def test_load_missing_is_empty():
    _tmp_registry()
    assert reg.load_models() == []
    print("load missing -> empty: OK")


def test_save_and_reload_roundtrip_and_perms():
    path = _tmp_registry()
    models = [_sample(model="gpt-4o", active=True), _sample("anthropic", "claude-x")]
    reg.save_models(models)
    # File is owner-only (holds API keys).
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, oct(mode)
    loaded = reg.load_models()
    assert [m["model"] for m in loaded] == ["gpt-4o", "claude-x"]
    assert loaded[0]["active"] and not loaded[1]["active"]
    print("save/reload roundtrip + 0600 perms: OK")


def test_api_mode_round_trip_and_legacy_default():
    path = _tmp_registry()
    responses = _sample("copilot", "gpt-5.6-terra", active=True)
    responses["api_mode"] = "responses"
    reg.save_models([responses])
    assert reg.load_models()[0]["api_mode"] == "responses"
    # Older registry files have no api_mode and retain Chat-first behavior.
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([_sample("copilot", "gpt-4o", active=True)], fh)
    assert reg.load_models()[0]["api_mode"] == "chat"
    print("api_mode round trip + legacy default: OK")


def test_exactly_one_active_normalized():
    _tmp_registry()
    # Two marked active -> only the first stays active.
    models = [_sample(model="a", active=True), _sample(model="b", active=True)]
    reg.save_models(models)
    loaded = reg.load_models()
    assert [m["active"] for m in loaded] == [True, False]
    # None marked active -> first becomes active.
    models2 = [_sample(model="a"), _sample(model="b")]
    out = reg._normalize_active(models2)
    assert [m["active"] for m in out] == [True, False]
    print("exactly one active normalized: OK")


def test_invalid_entries_dropped():
    path = _tmp_registry()
    with open(path, "w") as fh:
        json.dump(
            [
                {"provider": "openai", "model": "ok"},
                {"provider": "bogus", "model": "x"},  # bad provider
                {"provider": "openai"},  # no model
                "garbage",
            ],
            fh,
        )
    loaded = reg.load_models()
    assert [m["model"] for m in loaded] == ["ok"]
    print("invalid entries dropped: OK")


def test_add_remove_set_active():
    _tmp_registry()
    models: list = []
    models = reg.add_registration(models, _sample(model="a"))
    assert len(models) == 1 and models[0]["active"]  # first is active
    models = reg.add_registration(models, _sample(model="b"), make_active=False)
    assert models[0]["active"] and not models[1]["active"]
    models = reg.add_registration(models, _sample(model="c"))  # active by default
    assert [m["active"] for m in models] == [False, False, True]
    # set_active
    models = reg.set_active(models, 1)
    assert [m["active"] for m in models] == [False, True, False]
    assert reg.active_registration(models)["model"] == "b"
    # remove the active one -> active repoints to the first
    models = reg.remove_registration(models, 1)
    assert [m["model"] for m in models] == ["a", "c"]
    assert models[0]["active"]
    print("add/remove/set_active: OK")


def test_find_registration():
    _tmp_registry()
    models = [_sample(model="gpt-4o"), _sample("anthropic", "claude-sonnet")]
    assert reg.find_registration(models, "1") == 0
    assert reg.find_registration(models, "2") == 1
    assert reg.find_registration(models, "9") is None
    assert reg.find_registration(models, "claude") == 1
    assert reg.find_registration(models, "gpt") == 0
    assert reg.find_registration(models, "nope") is None
    print("find_registration: OK")


def test_registration_to_config():
    _tmp_registry()
    cfg = reg.registration_to_config(_sample(model="gpt-4o"))
    assert cfg.name == "openai" and cfg.model == "gpt-4o"
    assert cfg.api_url == "https://api.openai.com/v1"
    # Copilot cannot be converted directly (needs the gateway).
    try:
        reg.registration_to_config(_sample("copilot", "claude-x"))
    except ValueError:
        pass
    else:
        raise AssertionError("copilot should not convert to a direct config")
    print("registration_to_config: OK")


def test_load_registry_migrates_once_from_conf():
    path = _tmp_registry()
    # No models.json yet; a provider is "configured" -> migrate once.
    from ludvart.llm import ProviderConfig

    orig_resolve = reg.resolve_config
    orig_copilot = reg.copilot_model
    reg.resolve_config = lambda: ProviderConfig(
        name="openai",
        api_url="https://api.openai.com/v1",
        api_key="sk-env",
        model="gpt-4o",
        context_window=0,
    )
    reg.copilot_model = lambda: None
    try:
        models = reg.load_registry()
        assert len(models) == 1
        assert models[0]["provider"] == "openai"
        assert models[0]["model"] == "gpt-4o"
        assert models[0]["active"]
        # File was written; a subsequent load reads the file, not the conf.
        assert os.path.exists(path)
        reg.resolve_config = lambda: ProviderConfig(
            name="openai",
            api_url="https://api.openai.com/v1",
            api_key="sk-env",
            model="changed-in-conf",
            context_window=0,
        )
        again = reg.load_registry()
        assert again[0]["model"] == "gpt-4o", "must read file, not re-migrate"
    finally:
        reg.resolve_config = orig_resolve
        reg.copilot_model = orig_copilot
    print("load_registry migrates once from conf: OK")


def test_migrate_prefers_direct_provider_over_copilot():
    _tmp_registry()
    from ludvart.llm import ProviderConfig

    orig_resolve = reg.resolve_config
    orig_copilot = reg.copilot_model
    reg.resolve_config = lambda: ProviderConfig(
        name="anthropic",
        api_url="https://api.anthropic.com",
        api_key="k",
        model="claude-x",
        context_window=0,
    )
    reg.copilot_model = lambda: "gpt-4o"
    try:
        models = reg.migrate_from_conf()
        assert [m["provider"] for m in models] == ["anthropic", "copilot"]
        assert models[0]["active"] and not models[1]["active"]
    finally:
        reg.resolve_config = orig_resolve
        reg.copilot_model = orig_copilot
    print("migrate prefers direct provider over copilot: OK")


def test_load_registry_empty_when_nothing_configured():
    _tmp_registry()
    orig_resolve = reg.resolve_config
    orig_copilot = reg.copilot_model
    reg.resolve_config = lambda: None
    reg.copilot_model = lambda: None
    try:
        models = reg.load_registry()
        assert models == []
    finally:
        reg.resolve_config = orig_resolve
        reg.copilot_model = orig_copilot
    print("load_registry empty when nothing configured: OK")


def test_private_instance_keeps_its_model_choice_to_itself():
    _tmp_registry()
    reg.save_models([_sample(model="gpt-4o", active=True), _sample(model="claude-x")])
    reg.set_private(True)
    try:
        models = reg.set_active(reg.load_models(), 1)
        reg.save_models(models)
        # The shared registry still points everyone else at the first model.
        assert reg.active_index(reg.load_models()) == 0
        # Registering a model is not a choice, so it is shared as usual.
        reg.save_models([*models, _sample(model="gpt-5")])
        shared = reg.load_models()
        assert [m["model"] for m in shared] == ["gpt-4o", "claude-x", "gpt-5"]
        assert reg.active_index(shared) == 0
    finally:
        reg.set_private(False)
    print("private model choice stays local: OK")


def main():
    test_load_missing_is_empty()
    test_save_and_reload_roundtrip_and_perms()
    test_exactly_one_active_normalized()
    test_invalid_entries_dropped()
    test_add_remove_set_active()
    test_find_registration()
    test_registration_to_config()
    test_load_registry_migrates_once_from_conf()
    test_migrate_prefers_direct_provider_over_copilot()
    test_load_registry_empty_when_nothing_configured()
    test_private_instance_keeps_its_model_choice_to_itself()
    print("\nALL model-registry tests passed.")


if __name__ == "__main__":
    main()
