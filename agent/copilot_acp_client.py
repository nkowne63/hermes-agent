"""OpenAI-compatible shim that forwards Hermes requests to subprocess ACP CLIs.

This adapter lets Hermes treat subprocess ACP servers as chat-style
backends. Each request starts a short-lived ACP session, sends the formatted
conversation as a single prompt, collects text chunks, and converts the result
back into the minimal shape Hermes expects from an OpenAI client.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shlex
import subprocess
import tempfile
import threading
import time
import sys
import uuid
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.file_safety import get_read_block_error, is_write_denied
from agent.anthropic_adapter import normalize_model_name
from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

ACP_MARKER_BASE_URL = "acp://copilot"
_DEFAULT_TIMEOUT_SECONDS = 900.0

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}", re.DOTALL)

# Stderr fingerprint of the deprecated `gh copilot` CLI extension
# (https://github.blog/changelog/2025-09-25-upcoming-deprecation-of-gh-copilot-cli-extension).
# We require BOTH the literal product name ("gh-copilot") AND a deprecation
# marker, so generic stderr from the NEW `@github/copilot` CLI — whose repo
# is github.com/github/copilot-cli and which legitimately mentions "copilot-cli"
# in its own banners and error messages — doesn't get misclassified as the
# deprecated extension.
_DEPRECATION_REQUIRED = ("gh-copilot",)
_DEPRECATION_MARKERS = (
    "has been deprecated",
    "no commands will be executed",
)


def _is_gh_copilot_deprecation_message(stderr_text: str) -> bool:
    """True iff stderr looks like the deprecated gh-copilot extension's banner."""

    lower = stderr_text.lower()
    if not any(req in lower for req in _DEPRECATION_REQUIRED):
        return False
    return any(marker in lower for marker in _DEPRECATION_MARKERS)


def _resolve_command(base_url: str | None = None) -> str:
    marker = (base_url or "").strip().lower()
    if marker.startswith("acp://devin"):
        return (
            os.getenv("HERMES_DEVIN_ACP_COMMAND", "").strip()
            or os.getenv("DEVIN_CLI_PATH", "").strip()
            or "devin"
        )
    if marker.startswith("acp://claude"):
        return (
            os.getenv("HERMES_CLAUDE_ACP_COMMAND", "").strip()
            or os.getenv("CLAUDE_AGENT_ACP_PATH", "").strip()
            or "npx"
        )
    return (
        os.getenv("HERMES_COPILOT_ACP_COMMAND", "").strip()
        or os.getenv("COPILOT_CLI_PATH", "").strip()
        or "copilot"
    )

def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_COPILOT_ACP_ARGS", "").strip()
    if not raw:
        return ["--acp", "--stdio"]
    return shlex.split(raw)


def _resolve_home_dir() -> str:
    """Return a stable HOME for child ACP processes."""
    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok — POSIX fallback inside try/except (pwd import fails on Windows)
        if resolved:
            return resolved
    except Exception:
        pass

    # Last resort: /tmp (writable on any POSIX system). Avoids crashing the
    # subprocess with no HOME; callers can set HERMES_HOME explicitly if they
    # need a different writable dir.
    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    home = _resolve_home_dir()
    env["HOME"] = home
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)
    return env


def _load_acp_settings(provider_key: str, legacy_key: str) -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception:
        cfg = {}
    acp_cfg = cfg.get("acp") if isinstance(cfg, dict) else {}
    provider_cfg = acp_cfg.get(provider_key) if isinstance(acp_cfg, dict) else {}
    legacy_cfg = cfg.get(legacy_key) if isinstance(cfg, dict) else {}
    merged: dict[str, Any] = {}
    if isinstance(legacy_cfg, dict):
        merged.update(legacy_cfg)
    if isinstance(provider_cfg, dict):
        merged.update(provider_cfg)
    return merged


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _platform_tool_definitions(platform: str) -> list[dict[str, Any]]:
    platform = (platform or "").strip()
    if not platform:
        return []
    try:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools
        from model_tools import get_tool_definitions

        cfg = load_config()
        enabled_toolsets = sorted(_get_platform_tools(cfg, platform))
        agent_cfg = cfg.get("agent") or {}
        disabled_toolsets = agent_cfg.get("disabled_toolsets") or None
        return get_tool_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=True,
        )
    except Exception:
        return []


def _hermes_mcp_tool_definitions() -> list[dict[str, Any]]:
    """Return the tool schema surface exposed through the injected Hermes MCP server.

    ACP providers should see the same hermes-side tool surface that their
    ``session/new`` payload injects. Returning the unprefixed platform tool
    list here leaves native provider tools in play, which is exactly the
    class of mix-up we want to avoid for Devin/Claude ACP.
    """
    try:
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        from model_tools import get_tool_definitions
    except Exception:
        return []

    tool_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict)
        and td.get("type") == "function"
        and isinstance(td.get("function"), dict)
    }

    resolved: list[dict[str, Any]] = []
    for name in EXPOSED_TOOLS:
        spec = tool_defs.get(name)
        if not spec:
            continue
        tool_name = str(spec.get("name") or name).strip()
        if not tool_name:
            tool_name = name
        if tool_name.startswith("mcp__hermes__"):
            pass
        elif tool_name.startswith("mcp__"):
            tool_name = f"mcp__hermes__{tool_name.removeprefix('mcp__')}"
        else:
            tool_name = f"mcp__hermes__{tool_name}"
        resolved.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": spec.get("description") or f"Hermes MCP tool {name}",
                    "parameters": spec.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return resolved


def _build_hermes_mcp_server(*, platform: str, session_prefix: str, cwd: str) -> dict[str, Any]:
    """Build the injected Hermes MCP server entry for ACP subprocesses."""
    session_id = f"{session_prefix}-{uuid.uuid4().hex}"
    repo_root = Path(__file__).resolve().parent.parent
    server_path = repo_root / "mcp_hermes_tools.py"

    env_names = [
        "HOME",
        "HERMES_HOME",
        "HERMES_REAL_HOME",
        "PATH",
        "PYTHONPATH",
    ]
    env = [
        {"name": name, "value": value}
        for name in env_names
        if (value := os.environ.get(name))
    ]
    env.extend(
        [
            {"name": "HERMES_MCP_TOOL_PLATFORM", "value": platform},
            {"name": "HERMES_MCP_SESSION_ID", "value": session_id},
            {"name": "HERMES_MCP_TASK_ID", "value": session_id},
            {"name": "HERMES_MCP_CWD", "value": cwd},
        ]
    )

    return {
        "name": "hermes",
        "command": sys.executable,
        "args": [
            str(server_path),
            "--platform",
            platform,
            "--session-id",
            session_id,
            "--task-id",
            session_id,
            "--cwd",
            cwd,
        ],
        "env": env,
    }


class ACPProviderAdapter:
    """Provider-specific behavior for subprocess-backed ACP CLIs."""

    display_name = "ACP"
    default_model = "acp"
    marker_prefixes: tuple[str, ...] = ()
    command_names: tuple[str, ...] = ()

    def matches(self, *, base_url: str, command: str) -> bool:
        marker = (base_url or "").strip().lower()
        if any(marker.startswith(prefix) for prefix in self.marker_prefixes):
            return True
        command_name = Path(command or "").name.lower()
        return command_name in self.command_names

    def subprocess_env(
        self,
        env: dict[str, str],
        *,
        model: str | None,
    ) -> dict[str, str]:
        return env

    def subprocess_args(
        self,
        args: list[str],
        *,
        model: str | None,
    ) -> tuple[list[str], list[Path]]:
        return list(args), []

    def client_capabilities(self) -> dict[str, Any]:
        return {
            "fs": {
                "readTextFile": True,
                "writeTextFile": True,
            }
        }

    def supports_client_method(self, method: str) -> bool:
        return True

    def exposes_reasoning(self) -> bool:
        return True

    def prompt_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        return tools

    def format_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> str:
        return _format_messages_as_prompt(
            messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )

    def session_new_params(self, params: dict[str, Any], *, model: str | None) -> dict[str, Any]:
        del model
        return params

    def missing_command_error(self, command: str) -> str:
        return (
            f"Could not start {self.display_name} command '{command}'. "
            "Install the provider CLI or set the provider-specific command env vars."
        )

    def early_exit_error(self, stderr_text: str) -> RuntimeError:
        return RuntimeError(f"{self.display_name} process exited early: {stderr_text}")

    def timeout_error(self, method: str) -> TimeoutError:
        return TimeoutError(f"Timed out waiting for {self.display_name} response to {method}.")

    def method_error(self, method: str, error: Any) -> RuntimeError:
        message = error.get("message") if isinstance(error, dict) else None
        return RuntimeError(f"{self.display_name} {method} failed: {message or error}")


class CopilotACPProviderAdapter(ACPProviderAdapter):
    display_name = "Copilot ACP"
    default_model = "copilot-acp"
    marker_prefixes = ("acp://copilot",)
    command_names = ("copilot", "copilot.exe")

    def missing_command_error(self, command: str) -> str:
        return (
            f"Could not start Copilot ACP command '{command}'. "
            "Install GitHub Copilot CLI or set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH."
        )

    def early_exit_error(self, stderr_text: str) -> RuntimeError:
        if _is_gh_copilot_deprecation_message(stderr_text):
            return RuntimeError(
                "Hermes ACP mode requires the NEW GitHub Copilot CLI "
                "(github.com/github/copilot-cli), but the binary it just "
                "spawned is the deprecated `gh copilot` extension.\n\n"
                "Install the new CLI:\n"
                "  npm install -g @github/copilot\n"
                "  # then verify with: copilot --help\n\n"
                "If `copilot` already resolves to the new CLI but you still see this,\n"
                "point Hermes at it explicitly:\n"
                "  export HERMES_COPILOT_ACP_COMMAND=/path/to/new/copilot\n\n"
                "Alternative: use the `copilot` provider (no ACP, hits the Copilot API\n"
                "directly with a Copilot subscription token) via `hermes setup`.\n\n"
                f"Original error:\n{stderr_text}"
            )
        return super().early_exit_error(stderr_text)


class DevinACPProviderAdapter(ACPProviderAdapter):
    display_name = "Devin ACP"
    default_model = "devin-acp"
    marker_prefixes = ("acp://devin",)
    command_names = ("devin", "devin.exe")
    _DEFAULT_DENIED_TOOLS = [
        "Read(**)",
        "Write(**)",
        "Grep(**)",
        "Glob(**)",
        "Exec(**)",
        "Fetch(**)",
        "skill",
        "mcp_list_tools",
        "mcp_list_servers",
        "mcp_list_resources",
        "mcp_list_prompts",
    ]

    def _settings(self) -> dict[str, Any]:
        return _load_acp_settings("devin", "devin_acp")

    def _hermes_tools_only(self) -> bool:
        settings = self._settings()
        raw = settings.get("hermes_tools_only", settings.get("tools_only", True))
        if isinstance(raw, bool):
            return raw
        if raw is None:
            return True
        return str(raw).strip().lower() not in {"0", "false", "no", "off"}

    def _agent_config_path(self) -> str:
        settings = self._settings()
        return str(settings.get("agent_config") or settings.get("agent_config_path") or "").strip()

    def _allowed_tools(self) -> list[str]:
        settings = self._settings()
        raw = settings.get("allowed_tools")
        if isinstance(raw, list):
            tools = [str(item).strip() for item in raw if str(item).strip()]
            if tools:
                return tools
        return ["mcp__hermes__*"]

    def _deny_tools(self) -> list[str]:
        settings = self._settings()
        raw = settings.get("deny_tools")
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        if self._use_hermes_mcp_bridge():
            return list(self._DEFAULT_DENIED_TOOLS)
        # Without the Hermes MCP bridge, default-denying every native Devin tool
        # would leave the ACP session with no usable tools.
        return []

    def _tool_platform(self) -> str:
        settings = self._settings()
        return str(settings.get("tool_platform") or "discord").strip()

    def _use_hermes_mcp_bridge(self) -> bool:
        settings = self._settings()
        raw = settings.get("hermes_mcp_bridge", settings.get("mcp_bridge", True))
        if isinstance(raw, bool):
            return raw
        if raw is None:
            return True
        return str(raw).strip().lower() not in {"0", "false", "no", "off"}

    def _restricts_native_tools(self) -> bool:
        if not self._hermes_tools_only():
            return False
        if self._use_hermes_mcp_bridge():
            return True
        if self._agent_config_path():
            return True
        allowed_tools = self._allowed_tools()
        deny_tools = self._deny_tools()
        return bool(deny_tools) or allowed_tools != ["mcp__hermes__*"]

    def _reasoning_effort(self) -> str:
        settings = self._settings()
        return str(settings.get("reasoning_effort") or settings.get("thinking_level") or "").strip()

    def subprocess_env(
        self,
        env: dict[str, str],
        *,
        model: str | None,
    ) -> dict[str, str]:
        model_value = (model or "").strip()
        if model_value and model_value.lower() not in {"devin", "devin-acp"}:
            env["DEVIN_MODEL"] = model_value
        reasoning_effort = self._reasoning_effort()
        if reasoning_effort:
            env["DEVIN_REASONING_EFFORT"] = reasoning_effort
        return env

    def subprocess_args(
        self,
        args: list[str],
        *,
        model: str | None,
    ) -> tuple[list[str], list[Path]]:
        del model
        resolved = list(args)
        if not self._hermes_tools_only() or not self._use_hermes_mcp_bridge():
            return resolved, []
        if "--config" in resolved:
            return resolved, []
        if "--permission-mode" in resolved:
            permission_mode_args: list[str] = []
        else:
            permission_mode_args = ["--permission-mode", "dangerous"]

        cleanup_paths: list[Path] = []
        devin_config_path, config_cleanup = _build_devin_config_path(
            cwd=getattr(self, "_acp_cwd", os.getcwd()),
            session_prefix="devin-acp",
        )
        cleanup_paths.extend(config_cleanup)

        if "--agent-config" in resolved:
            return ["--config", str(devin_config_path), *resolved], cleanup_paths

        configured = self._agent_config_path()
        if configured:
            return ["--config", str(devin_config_path), "--agent-config", configured, *permission_mode_args, *resolved], cleanup_paths
        if not self._restricts_native_tools():
            return ["--config", str(devin_config_path), *permission_mode_args, *resolved], cleanup_paths
        deny_tools = self._deny_tools()
        cfg = {
            "permissions": {
                "allow": ["mcp__hermes__*"],
            },
        }
        if deny_tools:
            cfg["permissions"]["deny"] = deny_tools

        tmp = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            prefix="hermes-devin-acp-",
            suffix=".json",
            delete=False,
        )
        with tmp:
            tmp.write(json.dumps(cfg, ensure_ascii=False, indent=2))
            tmp.write("\n")
        path = Path(tmp.name)
        cleanup_paths.append(path)
        return ["--config", str(devin_config_path), "--agent-config", str(path), *permission_mode_args, *resolved], cleanup_paths

    def client_capabilities(self) -> dict[str, Any]:
        if self._restricts_native_tools():
            return {}
        return super().client_capabilities()

    def supports_client_method(self, method: str) -> bool:
        if self._restricts_native_tools() and method.startswith("fs/"):
            return False
        return True

    def prompt_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not self._hermes_tools_only():
            return tools
        hermes_tools = _hermes_mcp_tool_definitions()
        resolved_tools = hermes_tools or tools
        logger.debug(
            "Devin ACP prompt tools: hermes_mcp=%d fallback=%d",
            len(hermes_tools),
            len(tools or []),
        )
        if not resolved_tools:
            logger.warning(
                "Devin ACP hermes MCP tool surface is empty; falling back to the provided tool list"
            )
        return resolved_tools

    def exposes_reasoning(self) -> bool:
        return False

    def format_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> str:
        structured_messages: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "unknown").strip().lower()
            entry: dict[str, Any] = {
                "role": role,
                "content": _render_message_content(message.get("content")),
            }
            if role == "tool":
                tool_call_id = message.get("tool_call_id")
                if isinstance(tool_call_id, str) and tool_call_id.strip():
                    entry["tool_call_id"] = tool_call_id.strip()
                tool_name = message.get("name")
                if isinstance(tool_name, str) and tool_name.strip():
                    entry["name"] = tool_name.strip()
            structured_messages.append(entry)

        payload: dict[str, Any] = {
            "type": "hermes-conversation",
            "model_hint": model,
            "tool_choice": tool_choice,
            "messages": structured_messages,
        }
        if isinstance(tools, list) and tools:
            payload["tools"] = tools

        return "\n".join(
            [
                "You are being used as the active ACP agent backend for Hermes.",
                "Use the structured JSON conversation payload below as the source of truth.",
                json.dumps(payload, ensure_ascii=False, indent=2),
                "Continue from the latest user request.",
                "Return only the final answer. Start directly with the answer.",
                "Do not narrate tool use, planning, reasoning, or intermediate checks.",
                "Do not echo tool transcripts verbatim.",
            ]
        )

    def session_new_params(self, params: dict[str, Any], *, model: str | None) -> dict[str, Any]:
        del model
        if not self._hermes_tools_only() or not self._use_hermes_mcp_bridge():
            return params

        merged = dict(params)
        cwd = str(merged.get("cwd") or os.getcwd())
        session_id = f"devin-acp-{uuid.uuid4().hex}"
        repo_root = Path(__file__).resolve().parent.parent
        server_path = repo_root / "mcp_hermes_tools.py"

        env_names = [
            "HOME",
            "HERMES_HOME",
            "HERMES_REAL_HOME",
            "PATH",
            "PYTHONPATH",
        ]
        env = [
            {"name": name, "value": value}
            for name in env_names
            if (value := os.environ.get(name))
        ]
        env.extend(
            [
                {"name": "HERMES_MCP_TOOL_PLATFORM", "value": self._tool_platform()},
                {"name": "HERMES_MCP_SESSION_ID", "value": session_id},
                {"name": "HERMES_MCP_TASK_ID", "value": session_id},
                {"name": "HERMES_MCP_CWD", "value": cwd},
            ]
        )

        hermes_server = {
            "name": "hermes",
            "command": sys.executable,
            "args": [
                str(server_path),
                "--platform",
                self._tool_platform(),
                "--session-id",
                session_id,
                "--task-id",
                session_id,
                "--cwd",
                cwd,
            ],
            "env": env,
        }
        existing_raw = merged.get("mcpServers") or []
        if isinstance(existing_raw, dict):
            existing = [
                dict(server)
                for server in existing_raw.values()
                if isinstance(server, dict)
            ]
        elif isinstance(existing_raw, list):
            existing = [
                dict(server)
                for server in existing_raw
                if isinstance(server, dict)
            ]
        else:
            existing = []
        if not any(isinstance(server, dict) and server.get("name") == "hermes" for server in existing):
            existing.append(hermes_server)
        merged["mcpServers"] = existing
        logger.debug(
            "Devin ACP session new params: platform=%s mcp_servers=%d cwd=%s",
            self._tool_platform(),
            len(existing),
            cwd,
        )
        return merged

    def missing_command_error(self, command: str) -> str:
        return (
            f"Could not start Devin ACP command '{command}'. "
            "Install Devin CLI or set HERMES_DEVIN_ACP_COMMAND/DEVIN_CLI_PATH."
        )


class ClaudeACPProviderAdapter(ACPProviderAdapter):
    display_name = "Claude ACP"
    default_model = "claude-acp"
    marker_prefixes = ("acp://claude",)
    command_names = ("claude-agent-acp", "claude-agent-acp.exe", "npx", "npx.cmd")

    _DISALLOWED_BUILTIN_TOOLS = [
        "Agent",
        "Bash",
        "BashOutput",
        "Edit",
        "Glob",
        "Grep",
        "KillBash",
        "LS",
        "MultiEdit",
        "NotebookEdit",
        "Read",
        "Task",
        "TodoWrite",
        "WebFetch",
        "WebSearch",
        "Write",
    ]

    def _settings(self) -> dict[str, Any]:
        return _load_acp_settings("claude", "claude_acp")

    def _agent_config_path(self) -> str:
        settings = self._settings()
        return str(settings.get("agent_config") or settings.get("agent_config_path") or "").strip()

    def _allowed_tools(self) -> list[str]:
        settings = self._settings()
        raw = settings.get("allowed_tools")
        if isinstance(raw, list):
            tools = [str(item).strip() for item in raw if str(item).strip()]
            if tools:
                return tools
        return ["mcp__hermes__*"]

    def _deny_tools(self) -> list[str]:
        settings = self._settings()
        raw = settings.get("deny_tools")
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        if self._use_hermes_mcp_bridge():
            return ["*"]
        return []

    def _use_hermes_mcp_bridge(self) -> bool:
        settings = self._settings()
        raw = settings.get("hermes_mcp_bridge", settings.get("mcp_bridge", True))
        if isinstance(raw, bool):
            return raw
        if raw is None:
            return True
        return str(raw).strip().lower() not in {"0", "false", "no", "off"}

    def _restricts_native_tools(self) -> bool:
        if not self._hermes_tools_only():
            return False
        if self._use_hermes_mcp_bridge():
            return True
        if self._agent_config_path():
            return True
        allowed_tools = self._allowed_tools()
        deny_tools = self._deny_tools()
        return bool(deny_tools) or allowed_tools != ["mcp__hermes__*"]

    def _hermes_tools_only(self) -> bool:
        settings = self._settings()
        raw = settings.get("hermes_tools_only", settings.get("tools_only", True))
        if isinstance(raw, bool):
            return raw
        if raw is None:
            return True
        return str(raw).strip().lower() not in {"0", "false", "no", "off"}

    def _tool_platform(self) -> str:
        settings = self._settings()
        return str(settings.get("tool_platform") or "discord").strip()

    def _normalize_model_for_cli(self, model: str | None) -> str:
        model_value = (model or "").strip()
        if not model_value or model_value.lower() in {"claude", "claude-acp"}:
            return model_value
        return normalize_model_name(model_value)

    def subprocess_env(
        self,
        env: dict[str, str],
        *,
        model: str | None,
    ) -> dict[str, str]:
        model_value = self._normalize_model_for_cli(model)
        if model_value and model_value.lower() not in {"claude", "claude-acp"}:
            env["ANTHROPIC_MODEL"] = model_value
        return env

    def client_capabilities(self) -> dict[str, Any]:
        if self._restricts_native_tools():
            return {}
        return super().client_capabilities()

    def supports_client_method(self, method: str) -> bool:
        if self._restricts_native_tools() and method.startswith("fs/"):
            return False
        return True

    def prompt_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not self._hermes_tools_only():
            return tools
        hermes_tools = _hermes_mcp_tool_definitions()
        resolved_tools = hermes_tools or tools
        logger.debug(
            "Claude ACP prompt tools: hermes_mcp=%d fallback=%d",
            len(hermes_tools),
            len(tools or []),
        )
        if not resolved_tools:
            logger.warning(
                "Claude ACP hermes MCP tool surface is empty; falling back to the provided tool list"
            )
        return resolved_tools

    def session_new_params(self, params: dict[str, Any], *, model: str | None) -> dict[str, Any]:
        if not self._hermes_tools_only() or not self._use_hermes_mcp_bridge():
            return params

        model_value = self._normalize_model_for_cli(model)
        options: dict[str, Any] = {
            "tools": [],
            "disallowedTools": list(self._DISALLOWED_BUILTIN_TOOLS),
            "mcpServers": {},
        }
        if model_value and model_value.lower() not in {"claude", "claude-acp"}:
            options["env"] = {"ANTHROPIC_MODEL": model_value}
            options["settings"] = {"availableModels": [model_value]}

        merged = dict(params)
        meta = dict(merged.get("_meta") or {})
        claude_code = dict(meta.get("claudeCode") or {})
        existing_options = dict(claude_code.get("options") or {})
        existing_servers = existing_options.get("mcpServers") or {}
        if isinstance(existing_servers, list):
            existing_servers = {
                str(server.get("name")): dict(server)
                for server in existing_servers
                if isinstance(server, dict) and str(server.get("name") or "").strip()
            }
        elif not isinstance(existing_servers, dict):
            existing_servers = {}

        hermes_server = _build_hermes_mcp_server(
            platform=self._tool_platform(),
            session_prefix="claude-acp",
            cwd=str(merged.get("cwd") or os.getcwd()),
        )
        existing_servers = dict(existing_servers)
        existing_servers.setdefault("hermes", hermes_server)
        options["mcpServers"] = existing_servers

        merged_options = dict(existing_options)
        merged_options.update(options)
        merged_options["mcpServers"] = existing_servers
        claude_code["options"] = merged_options
        meta["claudeCode"] = claude_code
        merged["_meta"] = meta
        logger.debug(
            "Claude ACP session new params: platform=%s mcp_servers=%d cwd=%s",
            self._tool_platform(),
            len(existing_servers),
            str(merged.get("cwd") or os.getcwd()),
        )
        return merged

    def missing_command_error(self, command: str) -> str:
        return (
            f"Could not start Claude ACP command '{command}'. "
            "Install with `npm install -g @agentclientprotocol/claude-agent-acp`, "
            "or set HERMES_CLAUDE_ACP_COMMAND/HERMES_CLAUDE_ACP_ARGS."
        )


_ACP_PROVIDER_ADAPTERS: tuple[ACPProviderAdapter, ...] = (
    ClaudeACPProviderAdapter(),
    DevinACPProviderAdapter(),
    CopilotACPProviderAdapter(),
)


def _resolve_acp_provider_adapter(*, base_url: str, command: str) -> ACPProviderAdapter:
    for adapter in _ACP_PROVIDER_ADAPTERS:
        if adapter.matches(base_url=base_url, command=command):
            return adapter
    return CopilotACPProviderAdapter()


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _build_devin_config_path(*, cwd: str, session_prefix: str) -> tuple[Path, list[Path]]:
    """Create a temporary Devin config that injects the Hermes MCP server."""
    base_config = _load_json_file(Path.home() / ".config" / "devin" / "config.json")
    merged = dict(base_config)
    existing = merged.get("mcpServers")
    if isinstance(existing, dict):
        servers = {name: dict(cfg) for name, cfg in existing.items() if isinstance(cfg, dict)}
    elif isinstance(existing, list):
        servers = {
            str(cfg.get("name")): dict(cfg)
            for cfg in existing
            if isinstance(cfg, dict) and str(cfg.get("name") or "").strip()
        }
    else:
        servers = {}

    hermes_server = _build_hermes_mcp_server(
        platform="discord",
        session_prefix=session_prefix,
        cwd=cwd,
    )
    servers.setdefault("hermes", hermes_server)
    merged["mcpServers"] = servers

    tmp = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="hermes-devin-config-",
        suffix=".json",
        delete=False,
    )
    with tmp:
        json.dump(merged, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
    return Path(tmp.name), [Path(tmp.name)]


def _permission_denied(message_id: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {
            "outcome": {
                "outcome": "cancelled",
            }
        },
    }


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    sections: list[str] = [
        "You are being used as the active ACP agent backend for Hermes.",
        "Use ACP capabilities to complete tasks.",
        "IMPORTANT: If you take an action with a tool, you MUST output tool calls using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.",
        "If no tool is needed, answer normally.",
    ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_specs.append(
                {
                    "name": name.strip(),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        if tool_specs:
            sections.append(
                "Available tools (OpenAI function schema). "
                "When using a tool, emit ONLY <tool_call>{...}</tool_call> with one JSON object "
                "containing id/type/function{name,arguments}. arguments must be a JSON string.\n"
                + json.dumps(tool_specs, ensure_ascii=False)
            )

    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "tool":
            role = "tool"
        elif role not in {"system", "user", "assistant"}:
            role = "context"

        content = message.get("content")
        rendered = _render_message_content(content)
        if not rendered:
            continue

        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _strip_devin_artifacts(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    cleaned = re.sub(r"\s*<ref_[^>]+/>\s*", " ", text)
    cleaned = re.sub(r"\s*<ref_[^>]+>.*?</ref_[^>]+>\s*", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _extract_tool_calls_from_text(text: str) -> tuple[list[SimpleNamespace], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[SimpleNamespace] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add_tool_call(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        fn = obj.get("function")
        if not isinstance(fn, dict):
            return
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"acp_call_{len(extracted)+1}"

        extracted.append(
            SimpleNamespace(
                id=call_id,
                call_id=call_id,
                response_item_id=None,
                type="function",
                function=SimpleNamespace(name=fn_name.strip(), arguments=fn_args),
            )
        )

    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        raw = m.group(1)
        _try_add_tool_call(raw)
        consumed_spans.append((m.start(), m.end()))

    # Only try bare-JSON fallback when no XML blocks were found.
    if not extracted:
        for m in _TOOL_CALL_JSON_RE.finditer(text):
            raw = m.group(0)
            _try_add_tool_call(raw)
            consumed_spans.append((m.start(), m.end()))

    if not consumed_spans:
        return extracted, text.strip()

    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return extracted, cleaned



def _ensure_path_within_cwd(path_text: str, cwd: str) -> Path:
    candidate = Path(path_text)
    if not candidate.is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved = candidate.resolve()
    root = Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path '{resolved}' is outside the session cwd '{root}'.") from exc
    return resolved


class _ACPChatCompletions:
    def __init__(self, client: "CopilotACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "CopilotACPClient"):
        self.completions = _ACPChatCompletions(client)


class CopilotACPClient:
    """Minimal OpenAI-client-compatible facade for subprocess ACP providers."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        tool_progress_callback: Any = None,
        **_: Any,
    ):
        self.api_key = api_key or "copilot-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._acp_command = acp_command or command or _resolve_command(base_url)
        self._acp_args = list(acp_args or args or _resolve_args())
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self._provider_adapter = _resolve_acp_provider_adapter(
            base_url=self.base_url,
            command=self._acp_command,
        )
        setattr(self._provider_adapter, "_acp_cwd", self._acp_cwd)
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()
        self._tool_progress_callback = tool_progress_callback
        self._last_acp_session_updates: list[dict[str, Any]] = []
        self._last_acp_tool_trace: list[dict[str, Any]] = []
        self._last_acp_reasoning_trace: list[str] = []

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> Any:
        effective_tools = self._provider_adapter.prompt_tools(tools)
        prompt_text = self._provider_adapter.format_prompt(
            messages or [],
            model=model,
            tools=effective_tools,
            tool_choice=tool_choice,
        )
        # Normalise timeout: run_agent.py may pass an httpx.Timeout object
        # (used natively by the OpenAI SDK) rather than a plain float.
        if timeout is None:
            _effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            _effective_timeout = float(timeout)
        else:
            # httpx.Timeout or similar — pick the largest component so the
            # subprocess has enough wall-clock time for the full response.
            _candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            _numeric = [float(v) for v in _candidates if isinstance(v, (int, float))]
            _effective_timeout = max(_numeric) if _numeric else _DEFAULT_TIMEOUT_SECONDS

        response_text, reasoning_text = self._run_prompt(
            prompt_text,
            model=model,
            timeout_seconds=_effective_timeout,
        )

        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        if self._provider_adapter.exposes_reasoning():
            exposed_reasoning = reasoning_text or None
        else:
            exposed_reasoning = None
        assistant_message = SimpleNamespace(
            content=_strip_devin_artifacts(cleaned_text),
            tool_calls=tool_calls,
            reasoning=exposed_reasoning,
            reasoning_content=exposed_reasoning,
            reasoning_details=None,
            provider_data={
                "acp_session_updates": self._last_acp_session_updates or None,
                "acp_tool_trace": self._last_acp_tool_trace or None,
                "acp_reasoning_trace": self._last_acp_reasoning_trace or None,
            },
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or self._provider_adapter.default_model,
        )

    def _run_prompt(
        self,
        prompt_text: str,
        *,
        model: str | None = None,
        timeout_seconds: float,
    ) -> tuple[str, str]:
        env = self._provider_adapter.subprocess_env(
            _build_subprocess_env(),
            model=model,
        )
        acp_args, cleanup_paths = self._provider_adapter.subprocess_args(
            self._acp_args,
            model=model,
        )

        def _cleanup_generated_files() -> None:
            for path in cleanup_paths:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass

        try:
            proc = subprocess.Popen(
                [self._acp_command] + acp_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self._acp_cwd,
                env=env,
            )
        except FileNotFoundError as exc:
            _cleanup_generated_files()
            raise RuntimeError(
                self._provider_adapter.missing_command_error(self._acp_command)
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            _cleanup_generated_files()
            raise RuntimeError(
                f"{self._provider_adapter.display_name} process did not expose stdin/stdout pipes."
            )

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)
        session_updates: list[dict[str, Any]] = []
        tool_trace: list[dict[str, Any]] = []

        def _stdout_reader() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                try:
                    inbox.put(json.loads(line))
                except Exception:
                    inbox.put({"raw": line.rstrip("\n")})

        def _stderr_reader() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))

        out_thread = threading.Thread(target=_stdout_reader, daemon=True)
        err_thread = threading.Thread(target=_stderr_reader, daemon=True)
        out_thread.start()
        err_thread.start()

        next_id = 0

        def _request(method: str, params: dict[str, Any], *, text_parts: list[str] | None = None, reasoning_parts: list[str] | None = None) -> Any:
            nonlocal next_id
            next_id += 1
            request_id = next_id
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()

            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                try:
                    msg = inbox.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self._handle_server_message(
                    msg,
                    process=proc,
                    cwd=self._acp_cwd,
                    text_parts=text_parts,
                    reasoning_parts=reasoning_parts,
                    session_updates=session_updates,
                    tool_trace=tool_trace,
                    tool_progress_callback=self._tool_progress_callback,
                ):
                    continue

                if msg.get("id") != request_id:
                    continue
                if "error" in msg:
                    err = msg.get("error") or {}
                    raise self._provider_adapter.method_error(method, err)
                return msg.get("result")

            stderr_text = "\n".join(stderr_tail).strip()
            if proc.poll() is not None and stderr_text:
                raise self._provider_adapter.early_exit_error(stderr_text)
            raise self._provider_adapter.timeout_error(method)

        try:
            _request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": self._provider_adapter.client_capabilities(),
                    "clientInfo": {
                        "name": "hermes-agent",
                        "title": "Hermes Agent",
                        "version": "0.0.0",
                    },
                },
            )
            session = _request(
                "session/new",
                self._provider_adapter.session_new_params(
                    {
                        "cwd": self._acp_cwd,
                        "mcpServers": [],
                    },
                    model=model,
                ),
            ) or {}
            session_id = str(session.get("sessionId") or "").strip()
            if not session_id:
                raise RuntimeError(
                    f"{self._provider_adapter.display_name} did not return a sessionId."
                )

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            _request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        }
                    ],
                },
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
            )
            self._last_acp_session_updates = session_updates
            self._last_acp_tool_trace = tool_trace
            self._last_acp_reasoning_trace = reasoning_parts
            return "".join(text_parts), "".join(reasoning_parts)
        finally:
            self.close()
            _cleanup_generated_files()

    def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        process: subprocess.Popen[str],
        cwd: str,
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
        session_updates: list[dict[str, Any]] | None = None,
        tool_trace: list[dict[str, Any]] | None = None,
        tool_progress_callback: Any = None,
    ) -> bool:
        method = msg.get("method")
        if not isinstance(method, str):
            return False

        if method == "session/update":
            params = msg.get("params") or {}
            update = params.get("update") or {}
            kind = str(update.get("sessionUpdate") or "").strip()
            content = update.get("content") or {}
            chunk_text = ""
            if isinstance(content, dict):
                chunk_text = str(content.get("text") or "")
            if session_updates is not None:
                session_updates.append(
                    {
                        "session_id": params.get("sessionId"),
                        "kind": kind,
                        "update": update,
                    }
                )
            if tool_trace is not None:
                if kind == "tool_call":
                    tool_trace.append(
                        {
                            "event": "tool_call",
                            "session_id": params.get("sessionId"),
                            "tool_call_id": update.get("toolCallId"),
                            "title": update.get("title"),
                            "tool_kind": update.get("kind"),
                            "raw_input": update.get("rawInput"),
                            "content": update.get("content"),
                            "_meta": update.get("_meta"),
                        }
                    )
                elif kind == "tool_call_update":
                    tool_trace.append(
                        {
                            "event": "tool_call_update",
                            "session_id": params.get("sessionId"),
                            "tool_call_id": update.get("toolCallId"),
                            "status": update.get("status"),
                            "content": update.get("content"),
                            "_meta": update.get("_meta"),
                        }
                    )
            if tool_progress_callback is not None and kind in {"tool_call", "tool_call_update"}:
                tool_name = self._acp_tool_trace_name(update)
                preview = self._acp_tool_trace_preview(update, kind=kind, chunk_text=chunk_text)
                args = self._acp_tool_trace_args(update)
                try:
                    tool_progress_callback(
                        "tool.started",
                        tool_name,
                        preview,
                        args,
                        session_id=params.get("sessionId"),
                        tool_call_id=update.get("toolCallId"),
                        status=update.get("status"),
                        kind=kind,
                    )
                except Exception:
                    logger.debug("ACP tool_progress_callback failed", exc_info=True)
            if kind == "agent_message_chunk" and chunk_text and text_parts is not None:
                text_parts.append(chunk_text)
            elif kind == "agent_thought_chunk" and chunk_text and reasoning_parts is not None:
                reasoning_parts.append(chunk_text)
            return True

        if process.stdin is None:
            return True

        message_id = msg.get("id")
        params = msg.get("params") or {}

        if not self._provider_adapter.supports_client_method(method):
            response = _jsonrpc_error(
                message_id,
                -32601,
                f"{self._provider_adapter.display_name} client method '{method}' is disabled by Hermes configuration.",
            )
        elif method == "session/request_permission":
            response = _permission_denied(message_id)
        elif method == "fs/read_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                block_error = get_read_block_error(str(path))
                if block_error:
                    raise PermissionError(block_error)
                try:
                    content = path.read_text()
                except FileNotFoundError:
                    content = ""
                line = params.get("line")
                limit = params.get("limit")
                if isinstance(line, int) and line > 1:
                    lines = content.splitlines(keepends=True)
                    start = line - 1
                    end = start + limit if isinstance(limit, int) and limit > 0 else None
                    content = "".join(lines[start:end])
                if content:
                    content = redact_sensitive_text(content, force=True)
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "content": content,
                    },
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        elif method == "fs/write_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                if is_write_denied(str(path)):
                    raise PermissionError(
                        f"Write denied: '{path}' is a protected system/credential file."
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(params.get("content") or ""))
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": None,
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        else:
            response = _jsonrpc_error(
                message_id,
                -32601,
                f"ACP client method '{method}' is not supported by Hermes yet.",
            )

        process.stdin.write(json.dumps(response) + "\n")
        process.stdin.flush()
        return True

    @staticmethod
    def _acp_tool_trace_name(update: dict[str, Any]) -> str:
        meta = update.get("_meta")
        if isinstance(meta, dict):
            for key in ("cognition.ai/inferenceToolName", "inferenceToolName"):
                value = meta.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        for key in ("toolName", "tool_name", "name", "title"):
            value = update.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        kind = update.get("kind")
        if isinstance(kind, str) and kind.strip():
            return kind.strip()
        return "acp_tool"

    @staticmethod
    def _acp_tool_trace_args(update: dict[str, Any]) -> dict[str, Any] | None:
        raw = update.get("rawInput")
        if isinstance(raw, dict):
            return raw
        if raw is None:
            return None
        return {"rawInput": raw}

    @staticmethod
    def _acp_tool_trace_preview(
        update: dict[str, Any],
        *,
        kind: str,
        chunk_text: str = "",
    ) -> str | None:
        candidates: list[str] = []
        title = update.get("title")
        if isinstance(title, str) and title.strip():
            candidates.append(title.strip())

        raw_input = update.get("rawInput")
        if isinstance(raw_input, dict):
            for key in ("path", "query", "command", "url", "text", "name", "goal"):
                value = raw_input.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())
                    break
            else:
                try:
                    candidates.append(json.dumps(raw_input, ensure_ascii=False))
                except Exception:
                    pass
        elif raw_input is not None:
            candidates.append(str(raw_input))

        rendered = _render_message_content(update.get("content"))
        if rendered:
            candidates.append(rendered)
        if chunk_text.strip():
            candidates.append(chunk_text.strip())
        if kind == "tool_call_update":
            status = update.get("status")
            if isinstance(status, str) and status.strip():
                candidates.append(status.strip())

        for candidate in candidates:
            candidate = " ".join(str(candidate).split()).strip()
            if candidate:
                return candidate[:177] + "..." if len(candidate) > 180 else candidate
        return None
