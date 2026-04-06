"""MCP client — connects to external MCP servers, discovers tools, dispatches calls."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from isaac.core.types import PermissionLevel, ToolDef

log = logging.getLogger(__name__)


class MCPConnection:
    """A live connection to one MCP server."""

    def __init__(self, name: str, session: ClientSession) -> None:
        self.name = name
        self.session = session
        self.tools: dict[str, ToolDef] = {}

    async def discover_tools(self) -> list[ToolDef]:
        """Call tools/list and register discovered tools."""
        result = await self.session.list_tools()
        self.tools.clear()
        defs: list[ToolDef] = []
        for tool in result.tools:
            prefixed_name = f"mcp__{self.name}__{tool.name}"
            tdef = ToolDef(
                name=prefixed_name,
                description=tool.description or "",
                input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
                permission=PermissionLevel.AUTO,
            )
            self.tools[prefixed_name] = tdef
            defs.append(tdef)
        return defs

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Dispatch a tool call to this MCP server."""
        # Strip the mcp__servername__ prefix to get the real tool name
        real_name = tool_name.split("__", 2)[-1] if "__" in tool_name else tool_name
        result = await self.session.call_tool(real_name, arguments)
        # Extract content from MCP result
        if hasattr(result, "content") and result.content:
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts) if parts else str(result)
        return str(result)


class MCPManager:
    """Manages multiple MCP server connections."""

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._contexts: list[Any] = []

    async def connect_stdio(
        self, name: str, command: str, args: list[str] | None = None, env: dict[str, str] | None = None,
    ) -> MCPConnection:
        """Connect to an MCP server via stdio transport."""
        params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )
        # Create the stdio client context
        ctx = stdio_client(params)
        read_stream, write_stream = await ctx.__aenter__()
        self._contexts.append(ctx)

        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        self._contexts.append(session)
        await session.initialize()

        conn = MCPConnection(name, session)
        self._connections[name] = conn
        log.info(f"Connected to MCP server: {name}")
        return conn

    async def connect_http(
        self, name: str, url: str, headers: dict[str, str] | None = None,
    ) -> MCPConnection:
        """Connect to a remote MCP server via HTTP/SSE transport.

        Used for OAuth-based servers (Granola, Notion, etc.) that expose
        a remote MCP endpoint. Auth tokens are passed via headers.
        """
        try:
            from mcp.client.sse import sse_client
        except ImportError:
            raise RuntimeError("MCP SSE client not available. Update mcp package.")

        transport_headers = headers or {}
        ctx = sse_client(url, headers=transport_headers)
        read_stream, write_stream = await ctx.__aenter__()
        self._contexts.append(ctx)

        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        self._contexts.append(session)
        await session.initialize()

        conn = MCPConnection(name, session)
        self._connections[name] = conn
        log.info(f"Connected to remote MCP server: {name} ({url})")
        return conn

    async def discover_all_tools(self) -> dict[str, tuple[ToolDef, Any]]:
        """Discover tools from all connected servers. Returns registry entries."""
        registry: dict[str, tuple[ToolDef, Any]] = {}
        for name, conn in self._connections.items():
            tools = await conn.discover_tools()
            for tdef in tools:
                async def make_handler(c: MCPConnection, tn: str):
                    async def handler(**kwargs: Any) -> Any:
                        return await c.call_tool(tn, kwargs)
                    return handler

                registry[tdef.name] = (tdef, await make_handler(conn, tdef.name))
        return registry

    def get_connection(self, name: str) -> MCPConnection | None:
        return self._connections.get(name)

    async def close_all(self) -> None:
        """Graceful shutdown of all connections.

        Note: MCP SDK's stdio_client uses anyio task groups that don't
        clean up properly when exited from a different async task.
        We suppress the RuntimeError and just let the subprocess die.
        """
        for ctx in reversed(self._contexts):
            try:
                await ctx.__aexit__(None, None, None)
            except (Exception, BaseExceptionGroup):
                pass  # Suppress anyio task group cleanup errors
        self._connections.clear()
        self._contexts.clear()
