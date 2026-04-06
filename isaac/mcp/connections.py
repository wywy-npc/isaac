"""Service connections — manage upstream MCP servers and API integrations.

connections.yaml is the source of truth for all external service connections.
Agents can add/remove connections via tools, or users can edit the file directly.

~/.isaac/connections.yaml:
    services:
      google:
        type: mcp
        command: npx
        args: ["@anthropic-ai/google-mcp-server"]
        env:
          GOOGLE_API_KEY: "${GOOGLE_API_KEY}"
        enabled: true

      granola:
        type: mcp
        command: npx
        args: ["granola-mcp-server"]
        enabled: true

      financial_data:
        type: mcp
        url: "http://localhost:8200/mcp"
        transport: http
        enabled: true

      custom_api:
        type: rest
        base_url: "https://api.example.com/v1"
        auth: bearer
        env:
          API_KEY: "${EXAMPLE_API_KEY}"
        enabled: false
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from isaac.core.config import ISAAC_HOME

CONNECTIONS_FILE = ISAAC_HOME / "connections.yaml"


@dataclass
class ServiceConnection:
    """A single external service connection."""
    name: str
    type: str = "mcp"  # mcp | rest
    enabled: bool = True

    # For MCP stdio servers
    command: str = ""
    args: list[str] = field(default_factory=list)

    # For MCP HTTP servers
    url: str = ""
    transport: str = "stdio"  # stdio | http

    # Environment variables (supports ${VAR} expansion)
    env: dict[str, str] = field(default_factory=dict)

    # For REST APIs
    base_url: str = ""
    auth: str = ""  # bearer | basic | none

    # Metadata
    description: str = ""


def load_connections(credential_store: Any = None) -> dict[str, ServiceConnection]:
    """Load connections from ~/.isaac/connections.yaml.

    Credential resolution order:
    1. ${CRED:service.key} → credential store (encrypted on disk)
    2. ${ENV_VAR} → environment variable
    3. Literal value
    """
    if not CONNECTIONS_FILE.exists():
        return {}

    with open(CONNECTIONS_FILE) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    connections: dict[str, ServiceConnection] = {}
    for name, cfg in raw.get("services", {}).items():
        # Expand credentials and environment variables in env dict
        env = {}
        for k, v in cfg.get("env", {}).items():
            if isinstance(v, str) and v.startswith("${CRED:") and v.endswith("}"):
                # Credential store reference: ${CRED:service.key}
                cred_ref = v[7:-1]  # "service.key"
                if "." in cred_ref:
                    svc, cred_key = cred_ref.split(".", 1)
                else:
                    svc, cred_key = name, cred_ref
                if credential_store:
                    env[k] = credential_store.get(svc, cred_key) or ""
                else:
                    env[k] = ""
            elif isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                # Environment variable reference: ${VAR}
                env_key = v[2:-1]
                # Try credential store first (service name matches connection name)
                resolved = ""
                if credential_store:
                    resolved = credential_store.get(name, env_key) or ""
                if not resolved:
                    resolved = os.environ.get(env_key, "")
                env[k] = resolved
            else:
                env[k] = str(v)

        connections[name] = ServiceConnection(
            name=name,
            type=cfg.get("type", "mcp"),
            enabled=cfg.get("enabled", True),
            command=cfg.get("command", ""),
            args=cfg.get("args", []),
            url=cfg.get("url", ""),
            transport=cfg.get("transport", "stdio"),
            env=env,
            base_url=cfg.get("base_url", ""),
            auth=cfg.get("auth", ""),
            description=cfg.get("description", ""),
        )
    return connections


def save_connections(connections: dict[str, ServiceConnection]) -> Path:
    """Write connections back to ~/.isaac/connections.yaml."""
    CONNECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    services: dict[str, Any] = {}
    for name, conn in connections.items():
        entry: dict[str, Any] = {"type": conn.type, "enabled": conn.enabled}
        if conn.command:
            entry["command"] = conn.command
        if conn.args:
            entry["args"] = conn.args
        if conn.url:
            entry["url"] = conn.url
        if conn.transport != "stdio":
            entry["transport"] = conn.transport
        if conn.env:
            entry["env"] = conn.env
        if conn.base_url:
            entry["base_url"] = conn.base_url
        if conn.auth:
            entry["auth"] = conn.auth
        if conn.description:
            entry["description"] = conn.description

        services[name] = entry

    CONNECTIONS_FILE.write_text(yaml.dump({"services": services}, default_flow_style=False, sort_keys=False))
    return CONNECTIONS_FILE


def add_connection(
    name: str,
    type: str = "mcp",
    command: str = "",
    args: list[str] | None = None,
    url: str = "",
    transport: str = "stdio",
    env: dict[str, str] | None = None,
    description: str = "",
) -> ServiceConnection:
    """Add a new service connection (persists to connections.yaml)."""
    connections = load_connections()
    conn = ServiceConnection(
        name=name,
        type=type,
        enabled=True,
        command=command,
        args=args or [],
        url=url,
        transport=transport,
        env=env or {},
        description=description,
    )
    connections[name] = conn
    save_connections(connections)
    return conn


def remove_connection(name: str) -> bool:
    """Remove a service connection."""
    connections = load_connections()
    if name not in connections:
        return False
    del connections[name]
    save_connections(connections)
    return True


def list_connections() -> list[dict[str, Any]]:
    """List all connections with status."""
    connections = load_connections()
    return [
        {
            "name": c.name,
            "type": c.type,
            "enabled": c.enabled,
            "transport": c.transport,
            "description": c.description,
            "command": c.command or c.url or c.base_url,
        }
        for c in connections.values()
    ]
