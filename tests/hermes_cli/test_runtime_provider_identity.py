from types import SimpleNamespace


def test_runtime_provider_includes_pool_credential_identity(monkeypatch):
    import hermes_cli.runtime_provider as rp

    entry = SimpleNamespace(
        provider="openai-codex",
        id="cred-a",
        label="user@example.com",
        source="manual:device_code",
        access_token="token",
        runtime_api_key="token",
        runtime_base_url="https://chatgpt.com/backend-api/codex",
    )

    class Pool:
        def has_credentials(self):
            return True

        def select(self):
            return entry

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openai-codex")
    monkeypatch.setattr(rp, "load_pool", lambda provider: Pool())
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "openai-codex"})

    resolved = rp.resolve_runtime_provider(requested="openai-codex")

    assert resolved["provider"] == "openai-codex"
    assert resolved["credential_id"] == "cred-a"
    assert resolved["credential_label"] == "user@example.com"
    assert resolved["credential_source"] == "manual:device_code"
