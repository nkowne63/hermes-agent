"""Focused regressions for the Copilot ACP shim safety layer."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.copilot_acp_client import CopilotACPClient
from agent.copilot_acp_client import _extract_tool_calls_from_text
from agent.copilot_acp_client import _resolve_command


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

    def test_acp_tool_preview_prefers_raw_input_over_generic_hermes_title(self) -> None:
        preview = self.client._acp_tool_trace_preview(
            {
                "sessionUpdate": "tool_call",
                "title": "Calling search_files from hermes",
                "rawInput": {
                    "path": "/home/nkowne63rt/.hermes/hermes-agent",
                    "pattern": "nuxt3",
                    "target": "files",
                },
            },
            kind="tool_call",
        )

        self.assertEqual(preview, "/home/nkowne63rt/.hermes/hermes-agent")

    def test_acp_tool_preview_uses_skill_name_from_raw_input(self) -> None:
        preview = self.client._acp_tool_trace_preview(
            {
                "sessionUpdate": "tool_call",
                "title": "Calling skill_view from hermes",
                "rawInput": {"name": "agent-token-usage-analytics"},
            },
            kind="tool_call",
        )

        self.assertEqual(preview, "agent-token-usage-analytics")


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


def test_resolve_command_defaults_are_provider_specific(monkeypatch):
    monkeypatch.delenv("HERMES_COPILOT_ACP_COMMAND", raising=False)
    monkeypatch.delenv("COPILOT_CLI_PATH", raising=False)
    monkeypatch.delenv("HERMES_DEVIN_ACP_COMMAND", raising=False)
    monkeypatch.delenv("DEVIN_CLI_PATH", raising=False)
    monkeypatch.delenv("HERMES_CLAUDE_ACP_COMMAND", raising=False)
    monkeypatch.delenv("CLAUDE_AGENT_ACP_PATH", raising=False)

    assert _resolve_command("acp://copilot") == "copilot"
    assert _resolve_command("acp://devin") == "devin"
    assert _resolve_command("acp://claude") == "npx"


def test_run_prompt_uses_provider_specific_default_command_for_devin_acp(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_DEVIN_ACP_COMMAND", raising=False)
    monkeypatch.delenv("DEVIN_CLI_PATH", raising=False)

    captured = {}
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start Devin ACP command"):
            client._run_prompt("hello", model="opus", timeout_seconds=1)

    assert captured["cmd"][0] == "devin"
    assert captured["cmd"][1] == "--config"
    assert captured["cmd"][3] == "--agent-config"
    assert captured["cmd"][5] == "acp"
    assert not Path(captured["cmd"][2]).exists()
    assert not Path(captured["cmd"][4]).exists()


def test_run_prompt_uses_devin_acp_subcommand_when_args_are_omitted(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_DEVIN_ACP_ARGS", raising=False)

    captured = {}
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_cwd=str(tmp_path),
    )

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start Devin ACP command"):
            client._run_prompt("hello", model="swe-1.6", timeout_seconds=1)

    assert captured["cmd"][0] == "devin"
    assert captured["cmd"][1] == "--config"
    assert captured["cmd"][3] == "--agent-config"
    assert captured["cmd"][5] == "acp"
    assert not Path(captured["cmd"][2]).exists()
    assert not Path(captured["cmd"][4]).exists()


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
    assert captured["cmd"][:2] == ["devin", "--config"]
    assert captured["cmd"][3] == "--agent-config"
    assert captured["cmd"][5] == "acp"
    assert not Path(captured["cmd"][2]).exists()
    assert not Path(captured["cmd"][4]).exists()


def test_run_prompt_defaults_devin_model_to_swe_1_6(monkeypatch, tmp_path):
    captured = {}
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )

    with _patch.object(client._provider_adapter, "_settings", return_value={}):
        with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
            with pytest.raises(RuntimeError, match="Could not start Devin ACP command"):
                client._run_prompt("hello", model=None, timeout_seconds=1)

    assert captured["kwargs"]["env"]["DEVIN_MODEL"] == "swe-1.6"


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
        assert args[:2] == ["--config", args[1]]
        assert args[2:4] == ["--agent-config", args[3]]
        assert args[4:] == ["acp"]
        assert cleanup and Path(args[1]) in cleanup and Path(args[3]) in cleanup
        devin_config = json.loads(Path(args[1]).read_text(encoding="utf-8"))
        assert "hermes" in devin_config["mcpServers"]
        assert devin_config["mcpServers"]["hermes"]["command"]
        assert devin_config["mcpServers"]["hermes"]["transport"] == "stdio"
        assert "mcp__hermes__*" in devin_config["permissions"]["allow"]
        assert "exec" in devin_config["permissions"]["deny"]
        assert "mcp__hermes__*" in Path(args[3]).read_text(encoding="utf-8")
        assert client._provider_adapter.client_capabilities() == {}
        assert client._provider_adapter.supports_client_method("fs/read_text_file") is False
        for path in cleanup:
            path.unlink(missing_ok=True)


def test_devin_default_tools_only_uses_temp_project_mcp_config(tmp_path):
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )

    with _patch.object(client._provider_adapter, "_settings", return_value={}):
        process_cwd, cleanup = client._provider_adapter.subprocess_cwd(str(tmp_path))

    process_path = Path(process_cwd)
    try:
        assert process_path != tmp_path
        config_path = process_path / ".devin" / "config.local.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        hermes = config["mcpServers"]["hermes"]
        assert hermes["transport"] == "stdio"
        assert hermes["command"]
        assert isinstance(hermes["env"], dict)
        assert hermes["env"]["HERMES_MCP_CWD"] == str(tmp_path)
        assert "--cwd" in hermes["args"]
        assert str(tmp_path) in hermes["args"]
    finally:
        for path in cleanup:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)


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

    assert captured["cmd"][:5] == [
        "devin",
        "--config",
        captured["cmd"][2],
        "--agent-config",
        captured["cmd"][4],
    ]
    assert captured["cmd"][5] == "acp"
    assert not Path(captured["cmd"][2]).exists()
    assert not Path(captured["cmd"][4]).exists()


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

    assert captured["cmd"] == [
        "devin",
        "--config",
        captured["cmd"][2],
        "--agent-config",
        str(configured),
        "acp",
    ]


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

    assert captured["kwargs"]["env"]["ANTHROPIC_MODEL"] == "claude-sonnet-4-5"
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
    assert options["allowedTools"] == ["mcp__hermes__*"]
    assert "Bash" in options["disallowedTools"]
    assert options["env"]["ANTHROPIC_MODEL"] == "claude-sonnet-4-5"
    assert options["settings"]["availableModels"] == ["claude-sonnet-4-5"]

    hermes_server = next(server for server in params["mcpServers"] if server["name"] == "hermes")
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


def test_claude_tools_only_uses_native_mcp_surface_not_prompt_schemas(tmp_path):
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

    with _patch.object(client._provider_adapter, "_settings", return_value={}):
        assert client._provider_adapter.prompt_tools(original) is None
        prompt = client._provider_adapter.format_prompt(
            [{"role": "user", "content": "Inspect the repo"}],
            model="claude-sonnet-4.6",
            tools=original,
        )

    assert "Available tools (OpenAI function schema)." not in prompt
    assert "mcp__hermes__<tool_name>" in prompt
    assert "do not ask for clarification first" in prompt
    assert "mcp__hermes__session_search" in prompt
    assert "mcp__hermes__search_files" in prompt
    assert "ファイル検索" in prompt
    assert "pattern" in prompt
    assert "not `query`" in prompt
    assert "actually called a file tool" in prompt
    assert "only the tool calls" in prompt
    assert "answer from those results" in prompt
    assert "Do not print XML" in prompt


def test_devin_acp_tools_only_uses_hermes_mcp_tools(tmp_path):
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )
    original = [
        {"type": "function", "function": {"name": "terminal", "parameters": {}}},
    ]
    hermes_tools = [
        {"type": "function", "function": {"name": "mcp__hermes__skill_view", "parameters": {}}},
        {"type": "function", "function": {"name": "mcp__hermes__skills_list", "parameters": {}}},
    ]

    with _patch.object(client._provider_adapter, "_settings", return_value={}):
        with _patch("agent.copilot_acp_client._hermes_mcp_tool_definitions", return_value=hermes_tools):
            assert client._provider_adapter.prompt_tools(original) == hermes_tools


def test_create_chat_completion_includes_tools_and_extracts_tool_calls(tmp_path):
    client = CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command="copilot",
        acp_args=["--acp", "--stdio"],
        acp_cwd=str(tmp_path),
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    captured = {}

    def _fake_run_prompt(prompt_text, **kwargs):
        captured["prompt"] = prompt_text
        return (
            '<tool_call>{"id":"1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"/tmp/x.txt\\"}"}}</tool_call>\n'
            "Done.",
            "reasoning text",
        )

    with _patch.object(client._provider_adapter, "prompt_tools", return_value=tools):
        with _patch.object(client, "_run_prompt", side_effect=_fake_run_prompt):
            response = client._create_chat_completion(
                model="claude-sonnet-4.6",
                messages=[{"role": "user", "content": "Inspect /tmp/x.txt"}],
                tools=tools,
            )

    prompt_text = captured["prompt"]
    assert "Available tools (OpenAI function schema)." in prompt_text
    assert '"name": "read_file"' in prompt_text
    assert response.choices[0].message.content == "Done."
    assert response.choices[0].message.tool_calls
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].function.name == "read_file"


def test_devin_prompt_uses_structured_json_payload(tmp_path):
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    captured = {}

    def _fake_run_prompt(prompt_text, **kwargs):
        captured["prompt"] = prompt_text
        return ("Done. <ref_file file=\"/tmp/x.txt\" />", "reasoning text")

    with _patch.object(client._provider_adapter, "prompt_tools", return_value=tools):
        with _patch.object(client, "_run_prompt", side_effect=_fake_run_prompt):
            response = client._create_chat_completion(
                model="swe-1.6",
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Read /tmp/x.txt"},
                ],
                tools=tools,
            )

    prompt_text = captured["prompt"]
    assert "\"type\": \"hermes-conversation\"" in prompt_text
    assert "\"role\": \"system\"" in prompt_text
    assert "\"role\": \"user\"" in prompt_text
    assert "\"name\": \"read_file\"" in prompt_text
    assert "mcp__hermes__<tool_name>" in prompt_text
    assert "do not use Devin's native skill tool" in prompt_text
    assert response.choices[0].message.content == "Done."
    assert response.choices[0].message.reasoning is None
    provider_data = response.choices[0].message.provider_data
    assert provider_data["acp_session_updates"] is None
    assert provider_data["acp_tool_trace"] is None


def test_devin_prompt_uses_hermes_mcp_tools_only(tmp_path):
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )
    original = [
        {"type": "function", "function": {"name": "terminal", "parameters": {}}},
    ]
    hermes_tools = [
        {"type": "function", "function": {"name": "mcp__hermes__skill_view", "parameters": {}}},
        {"type": "function", "function": {"name": "mcp__hermes__skills_list", "parameters": {}}},
    ]

    with _patch.object(client._provider_adapter, "_settings", return_value={}):
        with _patch("agent.copilot_acp_client._hermes_mcp_tool_definitions", return_value=hermes_tools):
            assert client._provider_adapter.prompt_tools(original) == hermes_tools


def test_devin_prompt_builds_mcp_tools_from_platform_tool_surface(tmp_path):
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )
    platform_tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {}}},
        {"type": "function", "function": {"name": "skills_list", "parameters": {}}},
    ]

    with _patch("agent.copilot_acp_client._platform_tool_definitions", return_value=platform_tools):
        tools = client._provider_adapter.prompt_tools(None)

    assert [tool["function"]["name"] for tool in tools] == [
        "mcp__hermes__read_file",
        "mcp__hermes__skills_list",
    ]


def test_devin_prompt_does_not_double_prefix_preprefixed_hermes_mcp_tool_names(tmp_path):
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )
    platform_tools = [
        {"type": "function", "function": {"name": "mcp__hermes__session_search", "parameters": {}}},
    ]

    with _patch("agent.copilot_acp_client._platform_tool_definitions", return_value=platform_tools):
        tools = client._provider_adapter.prompt_tools(None)

    assert [tool["function"]["name"] for tool in tools] == ["mcp__hermes__session_search"]


def test_devin_initial_session_mode_prefers_bypass_when_hermes_bridge_is_on(tmp_path):
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )

    with _patch.object(client._provider_adapter, "_settings", return_value={}):
        assert client._provider_adapter.initial_session_mode() == "bypass"


def test_claude_initial_session_mode_prefers_bypass_when_hermes_bridge_is_on(tmp_path):
    client = CopilotACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        acp_command="npx",
        acp_args=["-y", "@agentclientprotocol/claude-agent-acp"],
        acp_cwd=str(tmp_path),
    )

    with _patch.object(client._provider_adapter, "_settings", return_value={}):
        assert client._provider_adapter.initial_session_mode() == "bypass"


def test_acp_session_update_records_provider_activity(tmp_path):
    events = []
    client = CopilotACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        acp_command="npx",
        acp_args=["-y", "@agentclientprotocol/claude-agent-acp"],
        acp_cwd=str(tmp_path),
        activity_callback=lambda desc: events.append(desc),
    )

    client._record_provider_activity("Claude ACP event received")

    assert events == ["Claude ACP event received"]


def test_devin_session_update_tool_events_are_captured_structurally(tmp_path):
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )
    process = _FakeProcess()
    session_updates = []
    tool_trace = []

    handled = client._handle_server_message(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "sess-1",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tool-1",
                    "title": "Ran command",
                    "kind": "execute",
                    "rawInput": {"command": "echo hello"},
                    "content": [{"type": "content", "content": {"type": "text", "text": "hello"}}],
                    "_meta": {"cognition.ai/inferenceToolName": "exec"},
                },
            },
        },
        process=process,
        cwd=str(tmp_path),
        text_parts=[],
        reasoning_parts=[],
        session_updates=session_updates,
        tool_trace=tool_trace,
    )

    assert handled is True
    assert session_updates[0]["kind"] == "tool_call"
    assert tool_trace[0]["event"] == "tool_call"
    assert tool_trace[0]["tool_call_id"] == "tool-1"
    assert tool_trace[0]["raw_input"] == {"command": "echo hello"}


def test_devin_session_update_tool_events_emit_progress(tmp_path):
    events = []
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
        tool_progress_callback=lambda *args, **kwargs: events.append((args, kwargs)),
    )
    process = _FakeProcess()

    handled = client._handle_server_message(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "sess-1",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tool-1",
                    "title": "Read file",
                    "rawInput": {"path": "/tmp/x.txt"},
                    "_meta": {"cognition.ai/inferenceToolName": "read_file"},
                },
            },
        },
        process=process,
        cwd=str(tmp_path),
        text_parts=[],
        reasoning_parts=[],
        session_updates=[],
        tool_trace=[],
        tool_progress_callback=client._tool_progress_callback,
    )

    assert handled is True
    assert events
    (event_name, tool_name, preview, args), kwargs = events[0]
    assert event_name == "tool.started"
    assert tool_name == "read_file"
    assert preview == "/tmp/x.txt"
    assert args == {"path": "/tmp/x.txt"}
    assert kwargs["tool_call_id"] == "tool-1"
    assert kwargs["session_id"] == "sess-1"


def test_devin_session_update_mcp_tool_names_are_normalized_for_progress(tmp_path):
    events = []
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
        tool_progress_callback=lambda *args, **kwargs: events.append((args, kwargs)),
    )
    process = _FakeProcess()

    handled = client._handle_server_message(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "sess-2",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tool-2",
                    "title": "Skills list",
                    "rawInput": {"category": "docs"},
                    "_meta": {"cognition.ai/inferenceToolName": "mcp__hermes__skills_list"},
                },
            },
        },
        process=process,
        cwd=str(tmp_path),
        text_parts=[],
        reasoning_parts=[],
        session_updates=[],
        tool_trace=[],
        tool_progress_callback=client._tool_progress_callback,
    )

    assert handled is True
    assert events
    (event_name, tool_name, preview, args), kwargs = events[0]
    assert event_name == "tool.started"
    assert tool_name == "skills_list"
    assert preview == '{"category": "docs"}'
    assert args == {"category": "docs"}
    assert kwargs["tool_call_id"] == "tool-2"


def test_devin_tool_call_updates_do_not_emit_started_progress(tmp_path):
    events = []
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
        tool_progress_callback=lambda *args, **kwargs: events.append((args, kwargs)),
    )
    process = _FakeProcess()

    handled = client._handle_server_message(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "sess-3",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tool-3",
                    "status": "running",
                    "_meta": {"cognition.ai/inferenceToolName": "exec"},
                    "content": [{"type": "content", "content": {"type": "text", "text": "chunk"}}],
                },
            },
        },
        process=process,
        cwd=str(tmp_path),
        text_parts=[],
        reasoning_parts=[],
        session_updates=[],
        tool_trace=[],
        tool_progress_callback=client._tool_progress_callback,
    )

    assert handled is True
    assert events == []


def test_devin_terminal_tool_call_update_emits_single_completed_progress(tmp_path):
    events = []
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
        tool_progress_callback=lambda *args, **kwargs: events.append((args, kwargs)),
    )
    process = _FakeProcess()
    message = {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": "sess-4",
            "update": {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tool-4",
                "status": "completed",
                "_meta": {"cognition.ai/inferenceToolName": "mcp__hermes__session_search"},
                "content": [{"type": "content", "content": {"type": "text", "text": "done"}}],
            },
        },
    }

    for _ in range(2):
        handled = client._handle_server_message(
            message,
            process=process,
            cwd=str(tmp_path),
            text_parts=[],
            reasoning_parts=[],
            session_updates=[],
            tool_trace=[],
            tool_progress_callback=client._tool_progress_callback,
        )
        assert handled is True

    assert len(events) == 1
    (event_name, tool_name, preview, args), kwargs = events[0]
    assert event_name == "tool.completed"
    assert tool_name == "session_search"
    assert preview is None
    assert args is None
    assert kwargs["tool_call_id"] == "tool-4"
    assert kwargs["is_error"] is False
    assert kwargs["result"] == "done"


def test_extract_tool_calls_understands_function_calls_blocks():
    tool_calls, cleaned = _extract_tool_calls_from_text(
        "<function_calls><invoke name=\"mcp__hermes__skills_list\"></invoke></function_calls>\nDone."
    )

    assert cleaned == "Done."
    assert tool_calls
    assert tool_calls[0].function.name == "skills_list"
    assert tool_calls[0].function.arguments == "{}"


def test_extract_tool_calls_understands_function_calls_json_arrays():
    tool_calls, cleaned = _extract_tool_calls_from_text(
        "Checking context.\n"
        "<function_calls>\n"
        "[\n"
        '  {"tool_name": "mcp__hermes__skill_view", "parameters": {"name": "nuxt3-regression-context"}},\n'
        '  {"tool_name": "mcp__hermes__session_search", "parameters": {"query": "nuxt3 token", "limit": 5}}\n'
        "]\n"
        "</function_calls>\n"
        "Done."
    )

    assert cleaned == "Checking context.\nDone."
    assert [tc.function.name for tc in tool_calls] == ["skill_view", "session_search"]
    assert json.loads(tool_calls[0].function.arguments) == {
        "name": "nuxt3-regression-context"
    }
    assert json.loads(tool_calls[1].function.arguments) == {
        "query": "nuxt3 token",
        "limit": 5,
    }


def test_extract_tool_calls_understands_function_calls_bracket_syntax():
    tool_calls, cleaned = _extract_tool_calls_from_text(
        "Thinking.\n"
        "<function_calls>\n"
        '[mcp__hermes__search_files(query="nuxt3 migration"), '
        'mcp__hermes__read_file(path="/tmp/notes.md")]\n'
        "</function_calls>\n"
        "Done."
    )

    assert cleaned == "Thinking.\nDone."
    assert [tc.function.name for tc in tool_calls] == ["search_files", "read_file"]
    assert json.loads(tool_calls[0].function.arguments) == {
        "query": "nuxt3 migration",
        "pattern": "nuxt3 migration",
    }
    assert json.loads(tool_calls[1].function.arguments) == {
        "path": "/tmp/notes.md"
    }


def test_extract_tool_calls_maps_search_files_query_to_pattern():
    tool_calls, _cleaned = _extract_tool_calls_from_text(
        '<function_calls>[mcp__hermes__search_files(query="nuxt3")]</function_calls>'
    )

    assert [tc.function.name for tc in tool_calls] == ["search_files"]
    assert json.loads(tool_calls[0].function.arguments) == {
        "query": "nuxt3",
        "pattern": "nuxt3",
    }


def test_extract_tool_calls_parses_claude_invoke_parameters_as_json_arguments():
    tool_calls, cleaned = _extract_tool_calls_from_text(
        "<function_calls>"
        "<invoke name=\"mcp__hermes__execute_code\">"
        "<parameter name=\"code\">from hermes_tools import terminal\n"
        "result = terminal('pwd')\n"
        "print(result)</parameter>"
        "</invoke>"
        "</function_calls>\nDone."
    )

    assert cleaned == "Done."
    assert tool_calls
    assert tool_calls[0].function.name == "execute_code"
    assert json.loads(tool_calls[0].function.arguments) == {
        "code": "from hermes_tools import terminal\nresult = terminal('pwd')\nprint(result)"
    }


def test_extract_tool_calls_parses_direct_claude_mcp_tags():
    tool_calls, cleaned = _extract_tool_calls_from_text(
        'I will load it.\n\n'
        '<mcp__hermes__skill_view name="nuxt3-regression-context">\n'
        '</mcp__hermes__skill_view>\n\n'
        'Done.'
    )

    assert cleaned == "I will load it.\nDone."
    assert tool_calls
    assert tool_calls[0].function.name == "skill_view"
    assert json.loads(tool_calls[0].function.arguments) == {
        "name": "nuxt3-regression-context"
    }


def test_create_chat_completion_extracts_tool_calls_from_reasoning(tmp_path):
    client = CopilotACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        acp_command="npx",
        acp_args=["-y", "@agentclientprotocol/claude-agent-acp"],
        acp_cwd=str(tmp_path),
    )

    def _fake_run_prompt(_prompt_text, **_kwargs):
        return (
            "I will summarize after the search.",
            '<function_calls>[mcp__hermes__search_files(query="nuxt3")]</function_calls>',
        )

    with patch.object(client, "_run_prompt", side_effect=_fake_run_prompt):
        response = client._create_chat_completion(
            model="claude-haiku-4.5",
            messages=[{"role": "user", "content": "ファイル検索して"}],
            tools=None,
        )

    message = response.choices[0].message
    assert response.choices[0].finish_reason == "tool_calls"
    assert [tc.function.name for tc in message.tool_calls] == ["search_files"]
    assert json.loads(message.tool_calls[0].function.arguments) == {
        "query": "nuxt3",
        "pattern": "nuxt3",
    }
    assert message.reasoning is None


def test_devin_agent_config_is_json_permissions_and_mcpservers(tmp_path):
    client = CopilotACPClient(
        api_key="devin-acp",
        base_url="acp://devin",
        acp_command="devin",
        acp_args=["acp"],
        acp_cwd=str(tmp_path),
    )

    args, cleanup = client._provider_adapter.subprocess_args(["acp"], model="swe-1.6")
    assert args[:2] == ["--config", args[1]]
    config_path = Path(args[1])
    assert config_path.exists()
    try:
        config = json.loads(config_path.read_text())
        agent_config_path = Path(args[3])
        agent_config = json.loads(agent_config_path.read_text())
    finally:
        for path in cleanup:
            path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)
        agent_config_path.unlink(missing_ok=True)

    assert "hermes" in config["mcpServers"]
    assert config["mcpServers"]["hermes"]["command"]
    assert isinstance(config["mcpServers"]["hermes"]["env"], dict)
    assert config["mcpServers"]["hermes"]["env"]["HOME"]
    assert config["permissions"]["allow"] == ["mcp__hermes__*"]
    assert config["permissions"]["deny"] == [
        "read",
        "edit",
        "grep",
        "glob",
        "exec",
        "fetch",
        "skill",
        "mcp_list_tools",
        "mcp_list_servers",
        "mcp_list_resources",
        "mcp_list_prompts",
    ]
    assert agent_config["permissions"]["allow"] == ["mcp__hermes__*"]
    assert agent_config["permissions"]["deny"] == [
        "read",
        "edit",
        "grep",
        "glob",
        "exec",
        "fetch",
        "skill",
        "mcp_list_tools",
        "mcp_list_servers",
        "mcp_list_resources",
        "mcp_list_prompts",
    ]
