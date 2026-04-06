"""REST adapter — creates tools from API endpoints for services without MCP servers.

When a service doesn't have an MCP server, define its endpoints in the catalog
and the adapter wraps each one as a callable tool with auth injection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isaac.core.types import PermissionLevel, ToolDef


@dataclass
class RESTEndpoint:
    """A single API endpoint that becomes a tool."""
    name: str                           # Tool name (e.g. "search_emails")
    method: str = "GET"                 # HTTP method
    path: str = ""                      # URL path (e.g. "/api/v1/search")
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)  # JSON schema for input params
    body_template: dict[str, Any] | None = None  # Request body template


@dataclass
class RESTServiceConfig:
    """Configuration for a REST API service."""
    base_url: str
    auth_type: str = "bearer"           # "bearer" | "basic" | "header" | "query" | "none"
    auth_header: str = "Authorization"  # Header name for auth
    auth_prefix: str = "Bearer"         # Prefix for auth value
    auth_key: str = ""                  # Credential key name in store
    endpoints: list[RESTEndpoint] = field(default_factory=list)
    service_name: str = ""              # For credential store lookup


def build_rest_tools(
    config: RESTServiceConfig,
    get_credential: Any = None,  # callable(service, key) -> str
) -> dict[str, tuple[ToolDef, Any]]:
    """Create tools from REST API endpoint definitions.

    Each endpoint becomes a tool that makes an HTTP request with auth injected.
    """
    registry: dict[str, tuple[ToolDef, Any]] = {}

    for endpoint in config.endpoints:
        tool_name = endpoint.name

        # Build input schema from endpoint params
        input_schema: dict[str, Any] = {
            "type": "object",
            "properties": dict(endpoint.params),
        }

        async def _make_handler(ep, cfg):
            async def handler(**kwargs) -> dict[str, Any]:
                import httpx

                # Build URL
                url = cfg.base_url.rstrip("/") + ep.path

                # Substitute path params
                for key, value in kwargs.items():
                    url = url.replace(f"{{{key}}}", str(value))

                # Build headers with auth
                headers: dict[str, str] = {"Content-Type": "application/json"}
                if cfg.auth_type != "none" and get_credential:
                    token = get_credential(cfg.service_name, cfg.auth_key)
                    if token:
                        if cfg.auth_type == "bearer":
                            headers[cfg.auth_header] = f"{cfg.auth_prefix} {token}"
                        elif cfg.auth_type == "basic":
                            import base64
                            headers[cfg.auth_header] = f"Basic {base64.b64encode(token.encode()).decode()}"
                        elif cfg.auth_type == "header":
                            headers[cfg.auth_header] = token
                        elif cfg.auth_type == "query":
                            url += f"{'&' if '?' in url else '?'}{cfg.auth_key}={token}"

                try:
                    async with httpx.AsyncClient(timeout=30) as client:
                        if ep.method.upper() == "GET":
                            # Non-path params become query params
                            query_params = {k: v for k, v in kwargs.items() if f"{{{k}}}" not in ep.path}
                            resp = await client.get(url, headers=headers, params=query_params)
                        elif ep.method.upper() == "POST":
                            body = ep.body_template.copy() if ep.body_template else {}
                            body.update(kwargs)
                            resp = await client.post(url, headers=headers, json=body)
                        elif ep.method.upper() == "PUT":
                            resp = await client.put(url, headers=headers, json=kwargs)
                        elif ep.method.upper() == "DELETE":
                            resp = await client.delete(url, headers=headers)
                        else:
                            return {"error": f"Unsupported method: {ep.method}"}

                        if resp.status_code >= 400:
                            return {"error": f"HTTP {resp.status_code}", "body": resp.text[:2000]}

                        try:
                            return resp.json()
                        except Exception:
                            return {"text": resp.text[:5000]}

                except Exception as e:
                    return {"error": f"Request failed: {e}"}

            return handler

        import asyncio
        handler = asyncio.get_event_loop().run_until_complete(_make_handler(endpoint, config))

        registry[tool_name] = (
            ToolDef(
                name=tool_name,
                description=endpoint.description,
                input_schema=input_schema,
                permission=PermissionLevel.AUTO,
                source=f"connector:{config.service_name}",
            ),
            handler,
        )

    return registry
