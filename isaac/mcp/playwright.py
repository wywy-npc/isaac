"""Playwright MCP — auto-spawn and connect to @anthropic-ai/playwright-mcp-server.

Provides structured browser automation tools via MCP protocol.
Requires: npx @anthropic-ai/playwright-mcp-server
"""
from __future__ import annotations

from typing import Any

from isaac.core.types import ToolDef
from isaac.mcp.client import MCPManager


async def connect_playwright(mcp: MCPManager) -> dict[str, tuple[ToolDef, Any]]:
    """Connect to Playwright MCP server and return discovered tools with handlers."""
    registry: dict[str, tuple[ToolDef, Any]] = {}

    try:
        conn = await mcp.connect_stdio("playwright", "npx", ["@anthropic-ai/playwright-mcp-server"])
        tool_defs = await conn.discover_tools()

        for tdef in tool_defs:
            # Capture conn and tool name in closure
            async def make_handler(c, tn):
                async def handler(**kwargs):
                    return await c.call_tool(tn, kwargs)
                return handler

            registry[tdef.name] = (tdef, await make_handler(conn, tdef.name))

    except Exception:
        pass  # Playwright MCP not available — degrade gracefully

    return registry
