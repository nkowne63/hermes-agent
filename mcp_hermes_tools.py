"""Hermes tool-surface MCP server.

This stdio MCP server exposes the same Hermes model tools that a platform
session would normally send in OpenAI ``tools=[...]``.  It is intentionally
separate from ``mcp_serve.py``, which exposes Hermes conversations as a
messaging bridge rather than exposing the active agent tool surface.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from typing import Any

try:
    from mcp.server.lowlevel.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.server.lowlevel.server import NotificationOptions
    from mcp import types
except ModuleNotFoundError:
    Server = None
    stdio_server = None
    NotificationOptions = None
    types = None


def _toolsets_for_platform(platform: str) -> tuple[list[str], list[str] | None]:
    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _get_platform_tools

    cfg = load_config() or {}
    enabled = sorted(_get_platform_tools(cfg, platform or "discord"))
    agent_cfg = cfg.get("agent") or {}
    disabled = agent_cfg.get("disabled_toolsets") or None
    return enabled, disabled


def _tool_definitions(platform: str) -> tuple[list[dict[str, Any]], list[str], list[str] | None]:
    from model_tools import get_tool_definitions

    enabled, disabled = _toolsets_for_platform(platform)
    tools = get_tool_definitions(
        enabled_toolsets=enabled,
        disabled_toolsets=disabled,
        quiet_mode=True,
    )
    return tools or [], enabled, disabled


def _openai_tool_to_mcp_schema(tool: dict[str, Any]) -> dict[str, Any] | None:
    fn = tool.get("function") if isinstance(tool, dict) else None
    if not isinstance(fn, dict):
        return None
    name = str(fn.get("name") or "").strip()
    if not name:
        return None
    parameters = fn.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    return {
        "name": name,
        "description": str(fn.get("description") or ""),
        "inputSchema": parameters,
    }


def _mcp_tool_from_openai_schema(tool: dict[str, Any]) -> types.Tool | None:
    schema = _openai_tool_to_mcp_schema(tool)
    if not schema:
        return None
    return types.Tool(**schema)


def _prefixed_tool_alias(tool: types.Tool) -> types.Tool:
    if tool.name.startswith("mcp__hermes__"):
        return tool
    return types.Tool(
        name=f"mcp__hermes__{tool.name}",
        description=tool.description,
        inputSchema=tool.inputSchema,
    )


def _tool_aliases(enabled_tools: list[str]) -> dict[str, str]:
    aliases = {name: name for name in enabled_tools}
    aliases.update({f"mcp__hermes__{name}": name for name in enabled_tools})
    return aliases


def _limit_tool_result(text: str) -> str:
    raw_limit = os.environ.get("HERMES_MCP_TOOL_RESULT_LIMIT", "24000")
    try:
        limit = int(raw_limit)
    except Exception:
        limit = 24000
    if limit <= 0 or len(text) <= limit:
        return text
    tail_budget = max(1, limit - 96)
    return (
        f"[Hermes MCP: tool output truncated from {len(text)} to {limit} chars; "
        f"showing tail]\n{text[-tail_budget:]}"
    )


def _load_tool_surface(platform: str, task_id: str, cwd: str) -> tuple[list[dict[str, Any]], list[str], list[str] | None, list[str], dict[str, str]]:
    if task_id and cwd:
        try:
            from tools.terminal_tool import register_task_env_overrides

            register_task_env_overrides(task_id, {"cwd": cwd})
        except Exception:
            pass

    tools, enabled_toolsets, disabled_toolsets = _tool_definitions(platform)
    enabled_tools = [
        str((tool.get("function") or {}).get("name") or "")
        for tool in tools
        if isinstance(tool, dict)
    ]
    tool_aliases = _tool_aliases(enabled_tools)
    return tools, enabled_toolsets, disabled_toolsets, enabled_tools, tool_aliases


def _call_hermes_tool_text(
    name: str,
    arguments: dict[str, Any] | None,
    *,
    platform: str,
    task_id: str,
    session_id: str,
    enabled_toolsets: list[str],
    disabled_toolsets: list[str] | None,
    enabled_tools: list[str],
    tool_aliases: dict[str, str],
) -> tuple[bool, str]:
    resolved_name = tool_aliases.get(name)
    if not resolved_name:
        return True, f"Hermes tool '{name}' is not enabled for platform '{platform}'."

    try:
        from model_tools import handle_function_call

        result = handle_function_call(
            resolved_name,
            arguments or {},
            task_id=task_id,
            session_id=session_id,
            enabled_tools=enabled_tools,
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
        )
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return False, _limit_tool_result(text)
    except Exception as exc:
        return True, str(exc)


def create_server(platform: str, task_id: str, session_id: str, cwd: str) -> Server:
    if Server is None or types is None:
        raise RuntimeError("The 'mcp' package is required to run the Hermes tool-surface MCP server.")
    server = Server("hermes-tools", version="0.0.0")
    tools, enabled_toolsets, disabled_toolsets, enabled_tools, tool_aliases = _load_tool_surface(
        platform,
        task_id,
        cwd,
    )

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        converted = [tool for tool in (_mcp_tool_from_openai_schema(tool) for tool in tools) if tool is not None]
        aliases = [_prefixed_tool_alias(tool) for tool in converted]
        return converted + aliases

    @server.call_tool(validate_input=False)
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        resolved_name = tool_aliases.get(name)
        if not resolved_name:
            return types.CallToolResult(
                isError=True,
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Hermes tool '{name}' is not enabled for platform '{platform}'.",
                    )
                ],
            )

        is_error, text = _call_hermes_tool_text(
            name,
            arguments,
            platform=platform,
            task_id=task_id,
            session_id=session_id,
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            enabled_tools=enabled_tools,
            tool_aliases=tool_aliases,
        )
        return types.CallToolResult(
            isError=is_error,
            content=[types.TextContent(type="text", text=text)],
        )

    return server


async def _run_minimal_stdio_mcp(platform: str, task_id: str, session_id: str, cwd: str) -> None:
    tools, enabled_toolsets, disabled_toolsets, enabled_tools, tool_aliases = _load_tool_surface(
        platform,
        task_id,
        cwd,
    )
    converted = [
        schema
        for schema in (_openai_tool_to_mcp_schema(tool) for tool in tools)
        if schema is not None
    ]
    aliases = [
        {**tool, "name": f"mcp__hermes__{tool['name']}"}
        for tool in converted
        if not str(tool.get("name") or "").startswith("mcp__hermes__")
    ]

    def respond(message_id: Any, result: Any = None, error: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": message_id}
        if error is not None:
            payload["error"] = error
        else:
            payload["result"] = result
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            return
        try:
            message = json.loads(line)
        except Exception:
            continue
        if not isinstance(message, dict) or "id" not in message:
            continue
        method = str(message.get("method") or "")
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        message_id = message.get("id")

        if method == "initialize":
            respond(
                message_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "hermes-tools", "version": "0.0.0"},
                },
            )
        elif method == "tools/list":
            respond(message_id, {"tools": converted + aliases})
        elif method == "tools/call":
            name = str(params.get("name") or "")
            arguments = params.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            is_error, text = _call_hermes_tool_text(
                name,
                arguments,
                platform=platform,
                task_id=task_id,
                session_id=session_id,
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                enabled_tools=enabled_tools,
                tool_aliases=tool_aliases,
            )
            result: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
            if is_error:
                result["isError"] = True
            respond(message_id, result)
        else:
            respond(
                message_id,
                error={"code": -32601, "message": f"Unsupported MCP method: {method}"},
            )


async def _run(platform: str, task_id: str, session_id: str, cwd: str) -> None:
    if Server is None or stdio_server is None or NotificationOptions is None:
        await _run_minimal_stdio_mcp(platform, task_id, session_id, cwd)
        return
    server = create_server(platform, task_id, session_id, cwd)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(
                NotificationOptions(),
                {},
            ),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default=os.environ.get("HERMES_MCP_TOOL_PLATFORM", "discord"))
    parser.add_argument("--task-id", default=os.environ.get("HERMES_MCP_TASK_ID", ""))
    parser.add_argument("--session-id", default=os.environ.get("HERMES_MCP_SESSION_ID", ""))
    parser.add_argument("--cwd", default=os.environ.get("HERMES_MCP_CWD", os.getcwd()))
    args = parser.parse_args(argv)

    task_id = args.task_id or f"mcp-hermes-{uuid.uuid4().hex}"
    try:
        asyncio.run(_run(args.platform, task_id, args.session_id, args.cwd))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
