"""Focused regressions for the Copilot ACP shim safety layer."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.copilot_acp_client import CopilotACPClient


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()


class CopilotACPClientSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = CopilotACPClient(acp_cwd="/tmp")

    def _dispatch(self, message: dict, *, cwd: str) -> dict:
        process = _FakeProcess()
        handled = self.client._handle_server_message(
            message,
            process=process,
            cwd=cwd,
            text_parts=[],
            reasoning_parts=[],
        )
        self.assertTrue(handled)
        payload = process.stdin.getvalue().strip()
        self.assertTrue(payload)
        return json.loads(payload)

    def test_request_permission_is_not_auto_allowed(self) -> None:
        response = self._dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/request_permission",
                "params": {},
            },
            cwd="/tmp",
        )

        outcome = (((response.get("result") or {}).get("outcome") or {}).get("outcome"))
        self.assertEqual(outcome, "cancelled")

    def test_read_text_file_blocks_internal_hermes_hub_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            blocked = home / ".hermes" / "skills" / ".hub" / "index-cache" / "entry.json"
            blocked.parent.mkdir(parents=True, exist_ok=True)
            blocked.write_text('{"token":"sk-test-secret-1234567890"}')

            with patch.dict(
                os.environ,
                {"HOME": str(home), "HERMES_HOME": str(home / ".hermes")},
                clear=False,
            ):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "fs/read_text_file",
                        "params": {"path": str(blocked)},
                    },
                    cwd=str(home),
                )

        self.assertIn("error", response)

    def test_read_text_file_redacts_sensitive_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            secret_file = root / "config.env"
            secret_file.write_text("OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012")

            # agent.redact snapshots HERMES_REDACT_SECRETS at import time into
            # _REDACT_ENABLED, so patching os.environ is a no-op. Flip the
            # module-level constant directly for the duration of the call.
            with patch("agent.redact._REDACT_ENABLED", True):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "fs/read_text_file",
                        "params": {"path": str(secret_file)},
                    },
                    cwd=str(root),
                )

        content = ((response.get("result") or {}).get("content") or "")
        self.assertNotIn("abc123def456", content)
        self.assertIn("OPENAI_API_KEY=", content)

    def test_write_text_file_reuses_write_denylist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            target = home / ".ssh" / "id_rsa"
            target.parent.mkdir(parents=True, exist_ok=True)

            with patch("agent.copilot_acp_client.is_write_denied", return_value=True, create=True):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 4,
                        "method": "fs/write_text_file",
                        "params": {
                            "path": str(target),
                            "content": "fake-private-key",
                        },
                    },
                    cwd=str(home),
                )

        self.assertIn("error", response)
        self.assertFalse(target.exists())

    def test_write_text_file_respects_safe_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            safe_root = root / "workspace"
            safe_root.mkdir()
            outside = root / "outside.txt"

            with patch.dict(os.environ, {"HERMES_WRITE_SAFE_ROOT": str(safe_root)}, clear=False):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 5,
                        "method": "fs/write_text_file",
                        "params": {
                            "path": str(outside),
                            "content": "should-not-write",
                        },
                    },
                    cwd=str(root),
                )

        self.assertIn("error", response)
        self.assertFalse(outside.exists())


if __name__ == "__main__":
    unittest.main()


# ── HOME env propagation tests (from PR #11285) ─────────────────────

from unittest.mock import patch as _patch
import pytest


def _make_home_client(tmp_path):
    return CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command="copilot",
        acp_args=["--acp", "--stdio"],
        acp_cwd=str(tmp_path),
    )


def _fake_popen_capture(captured):
    def _fake(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        raise FileNotFoundError("copilot not found")
    return _fake


def test_run_prompt_preserves_real_home_when_profile_home_available(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    (hermes_home / "home").mkdir(parents=True)
    real_home = tmp_path / "real-home"
    real_home.mkdir()

    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    captured = {}
    client = _make_home_client(tmp_path)

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
            client._run_prompt("hello", timeout_seconds=1)

    assert captured["kwargs"]["env"]["HOME"] == str(real_home)
    assert captured["kwargs"]["env"]["HERMES_REAL_HOME"] == str(real_home)


def test_run_prompt_passes_home_when_parent_env_is_clean(monkeypatch, tmp_path):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)

    captured = {}
    client = _make_home_client(tmp_path)

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
            client._run_prompt("hello", timeout_seconds=1)

    assert "env" in captured["kwargs"]
    assert captured["kwargs"]["env"]["HOME"]


def test_run_prompt_does_not_pass_devin_model_env_for_copilot_acp(monkeypatch, tmp_path):
    monkeypatch.delenv("DEVIN_MODEL", raising=False)

    captured = {}
    client = _make_home_client(tmp_path)

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
            client._run_prompt("hello", model="opus", timeout_seconds=1)

    assert "DEVIN_MODEL" not in captured["kwargs"]["env"]


def test_run_prompt_passes_devin_model_env_for_devin_acp(monkeypatch, tmp_path):
    monkeypatch.delenv("DEVIN_MODEL", raising=False)
    monkeypatch.delenv("DEVIN_REASONING_EFFORT", raising=False)

    captured = {}
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )

    with _patch.object(client._provider_adapter, "_settings", return_value={"reasoning_effort": "low"}):
        with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
            with pytest.raises(RuntimeError, match="Could not start Devin ACP command"):
                client._run_prompt("hello", model="opus", timeout_seconds=1)

    assert captured["kwargs"]["env"]["DEVIN_MODEL"] == "opus"
    assert captured["kwargs"]["env"]["DEVIN_REASONING_EFFORT"] == "low"
    assert captured["cmd"][:2] == ["devin", "--agent-config"]
    assert captured["cmd"][3:] == ["acp"]
    assert not Path(captured["cmd"][2]).exists()


def test_devin_default_tools_only_attaches_hermes_mcp_and_restricts_native_tools(tmp_path):
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )

    with _patch.object(client._provider_adapter, "_settings", return_value={}):
        args, cleanup = client._provider_adapter.subprocess_args(["acp"], model="opus")
        assert args[:2] == ["--agent-config", args[1]]
        assert args[2:] == ["acp"]
        assert cleanup and cleanup[0] == Path(args[1])
        assert "mcp__hermes__*" in cleanup[0].read_text(encoding="utf-8")
        assert client._provider_adapter.client_capabilities() == {}
        assert client._provider_adapter.supports_client_method("fs/read_text_file") is False
        for path in cleanup:
            path.unlink(missing_ok=True)

        params = client._provider_adapter.session_new_params(
            {"cwd": str(tmp_path), "mcpServers": []},
            model="opus",
        )
        servers = params["mcpServers"]
        assert len(servers) == 1
        assert servers[0]["name"] == "hermes"
        assert servers[0]["command"]
        assert "mcp_hermes_tools.py" in servers[0]["args"][0]
        assert "--platform" in servers[0]["args"]
        env = {item["name"]: item["value"] for item in servers[0]["env"]}
        assert env["HERMES_MCP_TOOL_PLATFORM"] == "discord"


def test_devin_mcp_bridge_can_be_disabled_to_keep_native_tools(tmp_path):
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )

    with _patch.object(client._provider_adapter, "_settings", return_value={"hermes_mcp_bridge": False}):
        assert client._provider_adapter.subprocess_args(["acp"], model="opus") == (["acp"], [])
        assert client._provider_adapter.client_capabilities()["fs"]["readTextFile"] is True
        assert client._provider_adapter.supports_client_method("fs/read_text_file") is True


def test_devin_explicit_deny_tools_generates_agent_config(tmp_path):
    captured = {}
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )

    with _patch.object(client._provider_adapter, "_settings", return_value={"deny_tools": ["*"]}):
        with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
            with pytest.raises(RuntimeError, match="Could not start Devin ACP command"):
                client._run_prompt("hello", model="opus", timeout_seconds=1)

    assert captured["cmd"][:3] == [
        "devin",
        "--agent-config",
        captured["cmd"][2],
    ]
    assert captured["cmd"][3:] == ["acp"]
    assert not Path(captured["cmd"][2]).exists()


def test_run_prompt_does_not_pass_sentinel_model_as_devin_model(monkeypatch, tmp_path):
    monkeypatch.delenv("DEVIN_MODEL", raising=False)

    captured = {}
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start Devin ACP command"):
            client._run_prompt("hello", model="devin-acp", timeout_seconds=1)

    assert "DEVIN_MODEL" not in captured["kwargs"]["env"]


def test_run_prompt_uses_configured_devin_agent_config(monkeypatch, tmp_path):
    configured = tmp_path / "devin-agent.yaml"
    configured.write_text("allowed_tools: []\n")

    captured = {}
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )

    with _patch.object(
        client._provider_adapter,
        "_settings",
        return_value={"agent_config": str(configured)},
    ):
        with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
            with pytest.raises(RuntimeError, match="Could not start Devin ACP command"):
                client._run_prompt("hello", model="opus", timeout_seconds=1)

    assert captured["cmd"] == ["devin", "--agent-config", str(configured), "acp"]


def test_devin_explicit_tools_only_disables_fs_client_capabilities(tmp_path):
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )

    with _patch.object(client._provider_adapter, "_settings", return_value={"deny_tools": ["*"]}):
        assert client._provider_adapter.client_capabilities() == {}
        assert client._provider_adapter.supports_client_method("fs/read_text_file") is False
        assert client._provider_adapter.supports_client_method("session/request_permission") is True


def test_devin_explicit_tools_only_rejects_fs_request(tmp_path):
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )
    process = _FakeProcess()

    with _patch.object(client._provider_adapter, "_settings", return_value={"deny_tools": ["*"]}):
        handled = client._handle_server_message(
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "fs/read_text_file",
                "params": {"path": str(tmp_path / "x.txt")},
            },
            process=process,
            cwd=str(tmp_path),
            text_parts=[],
            reasoning_parts=[],
        )

    assert handled is True
    payload = json.loads(process.stdin.getvalue())
    assert payload["error"]["code"] == -32601
    assert "disabled by Hermes configuration" in payload["error"]["message"]


def test_run_prompt_passes_claude_model_env_for_claude_acp(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    captured = {}
    client = CopilotACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        acp_command="npx",
        acp_args=["-y", "@agentclientprotocol/claude-agent-acp"],
        acp_cwd=str(tmp_path),
    )

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start Claude ACP command"):
            client._run_prompt("hello", model="claude-sonnet-4.5", timeout_seconds=1)

    assert captured["kwargs"]["env"]["ANTHROPIC_MODEL"] == "claude-sonnet-4.5"
    assert captured["cmd"] == ["npx", "-y", "@agentclientprotocol/claude-agent-acp"]


def test_claude_tools_only_disables_fs_client_capabilities(tmp_path):
    client = CopilotACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        acp_command="npx",
        acp_args=["-y", "@agentclientprotocol/claude-agent-acp"],
        acp_cwd=str(tmp_path),
    )

    with _patch.object(client._provider_adapter, "_settings", return_value={}):
        assert client._provider_adapter.client_capabilities() == {}
        assert client._provider_adapter.supports_client_method("fs/read_text_file") is False
        assert client._provider_adapter.supports_client_method("session/request_permission") is True


def test_claude_session_params_inject_hermes_mcp_bridge(tmp_path):
    client = CopilotACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        acp_command="npx",
        acp_args=["-y", "@agentclientprotocol/claude-agent-acp"],
        acp_cwd=str(tmp_path),
    )

    params = client._provider_adapter.session_new_params(
        {"cwd": str(tmp_path), "mcpServers": []},
        model="claude-sonnet-4.5",
    )

    options = params["_meta"]["claudeCode"]["options"]
    assert options["tools"] == []
    assert "hermes" in options["mcpServers"]
    assert "Bash" in options["disallowedTools"]
    assert options["env"]["ANTHROPIC_MODEL"] == "claude-sonnet-4.5"
    assert options["settings"]["availableModels"] == ["claude-sonnet-4.5"]

    hermes_server = options["mcpServers"]["hermes"]
    assert hermes_server["command"]
    assert "mcp_hermes_tools.py" in hermes_server["args"][0]
    env = {item["name"]: item["value"] for item in hermes_server["env"]}
    assert env["HERMES_MCP_TOOL_PLATFORM"] == "discord"


def test_claude_mcp_bridge_can_be_disabled_to_keep_native_tools(tmp_path):
    client = CopilotACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        acp_command="npx",
        acp_args=["-y", "@agentclientprotocol/claude-agent-acp"],
        acp_cwd=str(tmp_path),
    )

    with _patch.object(client._provider_adapter, "_settings", return_value={"hermes_mcp_bridge": False}):
        params = client._provider_adapter.session_new_params(
            {"cwd": str(tmp_path), "mcpServers": {}},
            model="claude-sonnet-4.5",
        )

        assert params == {"cwd": str(tmp_path), "mcpServers": {}}
        assert client._provider_adapter.client_capabilities()["fs"]["readTextFile"] is True
        assert client._provider_adapter.supports_client_method("fs/read_text_file") is True


def test_acp_tools_only_uses_discord_platform_tools(tmp_path):
    client = CopilotACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        acp_command="npx",
        acp_args=["-y", "@agentclientprotocol/claude-agent-acp"],
        acp_cwd=str(tmp_path),
    )
    original = [
        {"type": "function", "function": {"name": "terminal", "parameters": {}}},
    ]
    discord_tools = [
        {"type": "function", "function": {"name": "discord", "parameters": {}}},
    ]

    with _patch.object(client._provider_adapter, "_settings", return_value={}):
        with _patch("agent.copilot_acp_client._platform_tool_definitions", return_value=discord_tools):
            assert client._provider_adapter.prompt_tools(original) == discord_tools
