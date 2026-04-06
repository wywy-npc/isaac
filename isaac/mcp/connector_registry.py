"""ConnectorRegistry — tracks per-connector state, tools, and health.

Single source of truth for "what connectors are alive and what tools they gave us."
Each connector connects in isolation — one dying doesn't affect others.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from isaac.core.types import PermissionLevel, ToolDef
from isaac.mcp.client import MCPManager

log = logging.getLogger(__name__)


@dataclass
class ConnectorState:
    """State of a single connector."""
    name: str
    status: str = "disconnected"  # "connected" | "failed" | "needs_auth" | "disconnected"
    tools: list[str] = field(default_factory=list)
    error: str = ""
    transport: str = "stdio"
    connected_at: float = 0.0
    missing_credentials: list[str] = field(default_factory=list)


class ConnectorRegistry:
    """Manages MCP connector lifecycle and tracks health."""

    def __init__(self) -> None:
        self._manager = MCPManager()
        self._states: dict[str, ConnectorState] = {}

    async def connect(
        self,
        name: str,
        command: str = "",
        args: list[str] | None = None,
        url: str = "",
        transport: str = "stdio",
        env: dict[str, str] | None = None,
        registry: dict | None = None,
    ) -> ConnectorState:
        """Connect to a single MCP server. Returns state (never raises)."""
        state = ConnectorState(name=name, transport=transport)
        self._states[name] = state

        try:
            if transport == "stdio" and command:
                conn = await self._manager.connect_stdio(name, command, args or [], env)
            elif transport in ("http", "sse") and url:
                conn = await self._manager.connect_http(name, url)
            else:
                state.status = "failed"
                state.error = f"Invalid config: transport={transport}, command={command}, url={url}"
                return state

            # Discover tools
            tool_defs = await conn.discover_tools()
            tool_names: list[str] = []

            if registry is not None:
                for tdef in tool_defs:
                    tdef.source = f"connector:{name}"
                    if tdef.name not in registry:
                        async def _make_handler(c, tn):
                            async def handler(**kwargs):
                                try:
                                    return await c.call_tool(tn, kwargs)
                                except Exception as e:
                                    return {"error": f"Connector {name} tool failed: {e}"}
                            return handler
                        registry[tdef.name] = (tdef, await _make_handler(conn, tdef.name))
                    tool_names.append(tdef.name)

            state.status = "connected"
            state.tools = tool_names
            state.connected_at = time.time()
            log.info(f"Connector {name}: {len(tool_names)} tools")

        except Exception as e:
            state.status = "failed"
            state.error = str(e)
            log.warning(f"Connector {name} failed: {e}")

        return state

    async def reconnect(
        self, name: str, command: str = "", args: list[str] | None = None,
        url: str = "", transport: str = "stdio", env: dict[str, str] | None = None,
        registry: dict | None = None,
    ) -> ConnectorState:
        """Tear down and reconnect a connector."""
        # Remove old tools from registry
        old_state = self._states.get(name)
        if old_state and registry:
            for tool_name in old_state.tools:
                registry.pop(tool_name, None)

        return await self.connect(
            name, command=command, args=args, url=url,
            transport=transport, env=env, registry=registry,
        )

    def get_status(self) -> list[ConnectorState]:
        """All connector states."""
        return list(self._states.values())

    def get_tool_map(self) -> dict[str, list[str]]:
        """Map of connector name → tool names (connected only)."""
        return {
            s.name: s.tools
            for s in self._states.values()
            if s.status == "connected"
        }

    def get_state(self, name: str) -> ConnectorState | None:
        return self._states.get(name)

    async def close_all(self) -> None:
        """Graceful shutdown."""
        await self._manager.close_all()
        for state in self._states.values():
            state.status = "disconnected"
