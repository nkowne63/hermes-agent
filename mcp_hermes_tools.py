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

from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.lowlevel.server import NotificationOptions
from mcp import types


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


def _mcp_tool_from_openai_schema(tool: dict[str, Any]) -> types.Tool | None:
    fn = tool.get("function") if isinstance(tool, dict) else None
    if not isinstance(fn, dict):
        return None
    name = str(fn.get("name") or "").strip()
    if not name:
        return None
    parameters = fn.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    return types.Tool(
        name=name,
        description=str(fn.get("description") or ""),
        inputSchema=parameters,
    )


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


def create_server(platform: str, task_id: str, session_id: str, cwd: str) -> Server:
    server = Server("hermes-tools", version="0.0.0")

    if task_id and cwd:
        try:
            from tools.terminal_tool import register_task_env_overrides

            register_task_env_overrides(task_id, {"cwd": cwd})
        except Exception:
            pass

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        tools, _, _ = _tool_definitions(platform)
        converted = [_mcp_tool_from_openai_schema(tool) for tool in tools]
        return [tool for tool in converted if tool is not None]

    @server.call_tool(validate_input=False)
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        tools, enabled_toolsets, disabled_toolsets = _tool_definitions(platform)
        enabled_tools = [
            str((tool.get("function") or {}).get("name") or "")
            for tool in tools
            if isinstance(tool, dict)
        ]
        if name not in enabled_tools:
            return types.CallToolResult(
                isError=True,
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Hermes tool '{name}' is not enabled for platform '{platform}'.",
                    )
                ],
            )

        try:
            from model_tools import handle_function_call

            result = handle_function_call(
                name,
                arguments or {},
                task_id=task_id,
                session_id=session_id,
                enabled_tools=enabled_tools,
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
            )
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=_limit_tool_result(text))]
            )
        except Exception as exc:
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(type="text", text=str(exc))],
            )

    return server


async def _run(platform: str, task_id: str, session_id: str, cwd: str) -> None:
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
