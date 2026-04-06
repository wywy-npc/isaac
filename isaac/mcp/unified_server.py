"""Unified MCP server — one server, all tools, hot-reloadable.

Every agent in the constellation connects to this single server.
Toolsmith drops a .py file in ~/.isaac/tools/ → all agents get the tool.

Taxonomy:
  - Tools   = callable functions from ~/.isaac/tools/*.py, served here
  - Skills  = Claude skills framework (higher-level workflows that compose tools)
  - Meta tools = server management (reload_tools, list_tools)

Usage:
    isaac serve                          # stdio (for agent connections)
    isaac serve --transport http         # HTTP (for remote/browser access)
    isaac serve --transport http --port 9100

Tool file format:
    # ~/.isaac/tools/my_tool.py
    TOOLS = [
        {
            "name": "my_tool",
            "description": "Does a thing",
            "params": {"query": str},
            "handler": my_handler,
        }
    ]

    async def my_handler(query: str) -> dict:
        return {"result": "done"}
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from isaac.core.config import TOOLS_DIR, ensure_dirs
from isaac.mcp.tool_loader import ToolLoader, LoadedTool

log = logging.getLogger(__name__)

# The single server instance
_server: FastMCP | None = None
_loader: ToolLoader | None = None
_upstream_manager = None  # MCPManager for upstream connections
_last_scan: float = 0.0
_registered_tools: set[str] = set()


def create_server(tools_dir: Path | None = None) -> FastMCP:
    """Create the unified MCP server and register all tools.

    Loads tools from three sources:
    1. Local tool files (~/.isaac/tools/*.py)
    2. Upstream MCP servers (from ~/.isaac/connections.yaml)
    3. Meta tools (reload_tools, list_tools, connect_service, etc.)
    """
    global _server, _loader, _last_scan

    _server = FastMCP(
        "isaac-tools",
        instructions=(
            "ISAAC unified tool server. Tools from ~/.isaac/tools/ and connected "
            "services (connections.yaml) are all available here. "
            "Tools are hot-reloadable — new tools appear automatically."
        ),
    )

    _loader = ToolLoader(tools_dir or TOOLS_DIR)
    _registered_tools.clear()

    # Register meta tools (server management + service connections)
    _register_meta_tools(_server)
    _registered_tools.add("reload_tools")
    _registered_tools.add("list_tools")
    _registered_tools.add("connect_service")
    _registered_tools.add("disconnect_service")
    _registered_tools.add("list_services")

    # Load and register local tool files
    _register_tools(_server, _loader)

    # Note: upstream MCP connections are async and happen at runtime
    # via connect_service or when the server starts with connections.yaml

    _last_scan = time.time()
    return _server


def _register_tools(server: FastMCP, loader: ToolLoader) -> int:
    """Scan tools directory and register each tool on the MCP server."""
    tools = loader.scan()
    count = 0

    for name, tool in tools.items():
        if name in _registered_tools:
            continue
        _register_one_tool(server, name, tool)
        _registered_tools.add(name)
        count += 1

    return count


def _register_one_tool(server: FastMCP, name: str, tool: LoadedTool) -> None:
    """Register a single tool on the FastMCP server."""
    handler = tool.handler

    async def tool_handler(arguments: str = "{}") -> str:
        """Dynamic tool handler that parses JSON arguments."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            args = {}

        if asyncio.iscoroutinefunction(handler):
            result = await handler(**args)
        else:
            result = handler(**args)

        return json.dumps(result) if isinstance(result, (dict, list)) else str(result)

    server.add_tool(
        fn=tool_handler,
        name=name,
        description=tool.description,
    )


def _register_meta_tools(server: FastMCP) -> None:
    """Register server management (meta) tools + service connection tools."""

    @server.tool()
    def reload_tools() -> str:
        """Hot-reload all tools. Call after toolsmith creates a new tool file."""
        if _loader:
            old_count = len(_registered_tools)
            _loader.reload()
            new_count = _register_tools(server, _loader)
            return json.dumps({
                "reloaded": True,
                "new_tools": new_count,
                "total_tools": old_count + new_count,
            })
        return json.dumps({"error": "No tool loader available"})

    @server.tool()
    def list_tools() -> str:
        """List all loaded tools (local + connected services)."""
        local = []
        if _loader:
            for n, t in _loader.all_tools.items():
                local.append({"name": n, "description": t.description, "source": "local"})

        # Add upstream MCP tools
        upstream = [
            {"name": n, "source": "service"} for n in _registered_tools
            if n not in (t["name"] for t in local)
            and n not in ("reload_tools", "list_tools", "connect_service", "disconnect_service", "list_services")
        ]

        all_tools = local + upstream
        return json.dumps({"tools": all_tools, "count": len(all_tools)})

    @server.tool()
    def connect_service(
        name: str,
        command: str = "",
        args: str = "",
        url: str = "",
        transport: str = "stdio",
        description: str = "",
    ) -> str:
        """Connect a new external service (MCP server). Persists to connections.yaml.

        For stdio servers: provide command and args (space-separated).
          Example: connect_service(name="google", command="npx", args="@anthropic-ai/google-mcp")
        For HTTP servers: provide url.
          Example: connect_service(name="financial", url="http://localhost:8200/mcp", transport="http")
        """
        from isaac.mcp.connections import add_connection
        args_list = args.split() if args else []
        conn = add_connection(
            name=name,
            type="mcp",
            command=command,
            args=args_list,
            url=url,
            transport=transport,
            description=description,
        )
        return json.dumps({
            "connected": name,
            "type": conn.type,
            "transport": conn.transport,
            "note": "Service added to connections.yaml. Restart `isaac serve` to activate, or call reload_tools.",
        })

    @server.tool()
    def disconnect_service(name: str) -> str:
        """Remove an external service connection."""
        from isaac.mcp.connections import remove_connection
        removed = remove_connection(name)
        if removed:
            return json.dumps({"disconnected": name})
        return json.dumps({"error": f"Service '{name}' not found"})

    @server.tool()
    def list_services() -> str:
        """List all configured external service connections."""
        from isaac.mcp.connections import list_connections
        services = list_connections()
        return json.dumps({"services": services, "count": len(services)})


async def _connect_upstream_services(server: FastMCP) -> int:
    """Connect to all enabled MCP servers from connections.yaml and register their tools."""
    global _upstream_manager
    from isaac.mcp.connections import load_connections
    from isaac.mcp.client import MCPManager

    connections = load_connections()
    if not connections:
        return 0

    _upstream_manager = MCPManager()
    connected = 0

    for name, conn in connections.items():
        if not conn.enabled or conn.type != "mcp":
            continue

        try:
            if conn.transport == "stdio" and conn.command:
                # Skip if env vars are missing (empty strings)
                if conn.env:
                    missing = [k for k, v in conn.env.items() if not v]
                    if missing:
                        log.debug(f"Skipping {name}: missing env vars {missing}")
                        continue
                mcp_conn = await _upstream_manager.connect_stdio(
                    name, conn.command, conn.args, conn.env or None
                )
            elif conn.transport == "http" and conn.url:
                mcp_conn = await _upstream_manager.connect_http(name, conn.url)
            else:
                continue

            # Discover tools from this upstream server
            tool_defs = await mcp_conn.discover_tools()
            for tdef in tool_defs:
                if tdef.name not in _registered_tools:
                    # Create a proxy handler
                    async def _make_proxy(c, tn):
                        async def proxy_handler(arguments: str = "{}") -> str:
                            import json as _json
                            args = _json.loads(arguments) if isinstance(arguments, str) else arguments
                            result = await c.call_tool(tn, args)
                            return _json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                        return proxy_handler

                    handler = await _make_proxy(mcp_conn, tdef.name)
                    server.add_tool(fn=handler, name=tdef.name, description=tdef.description)
                    _registered_tools.add(tdef.name)

            connected += 1
            log.info(f"Connected upstream: {name} ({len(tool_defs)} tools)")

        except Exception as e:
            log.warning(f"Failed to connect {name}: {e}")

    return connected


def run_server(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 9100,
    tools_dir: Path | None = None,
) -> None:
    """Start the unified MCP server."""
    import asyncio

    ensure_dirs()
    server = create_server(tools_dir)

    # Connect to upstream MCP services from connections.yaml
    try:
        loop = asyncio.new_event_loop()
        upstream_count = loop.run_until_complete(_connect_upstream_services(server))
        loop.close()
        if upstream_count > 0:
            log.info(f"Connected to {upstream_count} upstream services")
    except Exception as e:
        log.warning(f"Upstream connection phase failed: {e}")

    meta_count = 5  # reload_tools, list_tools, connect_service, disconnect_service, list_services
    tool_count = len(_registered_tools) - meta_count
    log.info(f"ISAAC unified MCP server starting ({tool_count} tools)")

    if transport == "stdio":
        server.run(transport="stdio")
    elif transport in ("http", "streamable-http"):
        server.run(transport="streamable-http", host=host, port=port)
    else:
        raise ValueError(f"Unknown transport: {transport}. Use 'stdio' or 'http'.")
