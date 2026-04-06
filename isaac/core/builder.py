"""HarnessBuilder — factory that assembles HarnessConfig.

Extracts the assembly logic from terminal.py into a reusable builder.
CLI, SDK, gateway, delegation all use this to create HarnessCore instances.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from isaac.core.config import load_agents_config
from isaac.core.harness import HarnessConfig, HarnessCore
from isaac.core.llm import LLMClient
from isaac.core.permissions import PermissionGate
from isaac.core.types import AgentConfig, PermissionLevel, ToolDef


class HarnessBuilder:
    """Fluent builder for HarnessCore. Replaces manual assembly in terminal.py."""

    def __init__(self, agent_name: str = "default", cwd: str | None = None) -> None:
        agents = load_agents_config()
        self._config = agents.get(agent_name, AgentConfig(name=agent_name))
        if cwd:
            self._config.cwd = cwd
        self._memory_store = None
        self._embedding_store = None
        self._llm_client: LLMClient | None = None
        self._approval_fn: Callable | None = None
        self._event_handler: Callable | None = None
        self._soul_mode: str = "full"
        self._soul_override: str = ""
        self._extra_tools: dict[str, tuple[ToolDef, Any]] = {}
        self._enable_plugins: bool = True
        self._enable_delegation: bool = True
        self._enable_gateway: bool = True
        self._enable_sandbox: bool = True

    def with_config(self, config: AgentConfig) -> HarnessBuilder:
        """Override the agent config directly."""
        self._config = config
        return self

    def with_memory(self, store: Any = None, embedding_store: Any = None) -> HarnessBuilder:
        """Provide memory store and optional embeddings."""
        self._memory_store = store
        self._embedding_store = embedding_store
        return self

    def with_llm(self, client: LLMClient) -> HarnessBuilder:
        """Override the LLM client."""
        self._llm_client = client
        return self

    def with_approval(self, fn: Callable) -> HarnessBuilder:
        """Provide an approval callback for ASK-level tools."""
        self._approval_fn = fn
        return self

    def with_events(self, handler: Callable) -> HarnessBuilder:
        """Provide an event callback (for non-streaming consumers)."""
        self._event_handler = handler
        return self

    def with_soul(self, mode: str = "full", override: str = "") -> HarnessBuilder:
        """Configure soul mode and override."""
        self._soul_mode = mode
        self._soul_override = override
        return self

    def with_tools(self, tools: dict[str, tuple[ToolDef, Any]]) -> HarnessBuilder:
        """Add extra tools to the registry."""
        self._extra_tools.update(tools)
        return self

    def without_plugins(self) -> HarnessBuilder:
        self._enable_plugins = False
        return self

    def without_delegation(self) -> HarnessBuilder:
        self._enable_delegation = False
        return self

    def without_gateway(self) -> HarnessBuilder:
        self._enable_gateway = False
        return self

    def without_sandbox(self) -> HarnessBuilder:
        self._enable_sandbox = False
        return self

    def _resolve_llm_client(self) -> LLMClient:
        """Auto-select LLM client based on model name."""
        if self._llm_client:
            return self._llm_client

        model = self._config.model
        # Anthropic models get native client (prompt caching, extended thinking)
        if "claude" in model or "anthropic" in model:
            from isaac.core.llm_anthropic import AnthropicClient
            return AnthropicClient()

        # Everything else goes through litellm
        try:
            from isaac.core.llm_litellm import LiteLLMClient
            return LiteLLMClient()
        except ImportError:
            # Fallback to Anthropic if litellm not installed
            from isaac.core.llm_anthropic import AnthropicClient
            return AnthropicClient()

    def _build_memory(self) -> tuple[Any, Any, Callable | None]:
        """Initialize memory store, embeddings, and scout function.

        Builds both agent/company memory and personal memory scouts.
        The combined memory_fn returns both in a single context block.
        """
        if self._memory_store is None:
            from isaac.memory.store import MemoryStore
            self._memory_store = MemoryStore()

        # Embeddings — graceful fallback
        if self._embedding_store is None:
            try:
                from isaac.memory.embeddings import EmbeddingStore
                self._embedding_store = EmbeddingStore(self._memory_store.dir)
            except ImportError:
                pass

        # Agent/company scout
        from isaac.memory.scout import MemoryScout
        scout = MemoryScout(self._memory_store, self._embedding_store)

        # Personal memory scout (same MemoryScout, different store + budget)
        personal_scout = None
        try:
            from isaac.personal.store import get_personal_store
            from isaac.core.context import PERSONAL_SCOUT_BUDGET
            personal_store = get_personal_store()

            # Personal embeddings (separate from agent)
            personal_embedding = None
            try:
                from isaac.memory.embeddings import EmbeddingStore
                personal_embedding = EmbeddingStore(personal_store.dir)
            except ImportError:
                pass

            personal_scout = MemoryScout(personal_store, personal_embedding)
        except Exception:
            pass

        async def memory_fn(query: str) -> str:
            parts = []
            # Agent/company memory (2000 tokens)
            try:
                agent_ctx = await scout.search(query)
                if agent_ctx:
                    parts.append(agent_ctx)
            except Exception:
                pass
            # Personal memory (1000 tokens)
            if personal_scout:
                try:
                    from isaac.core.context import PERSONAL_SCOUT_BUDGET
                    personal_ctx = await personal_scout.search(query, budget=PERSONAL_SCOUT_BUDGET)
                    if personal_ctx:
                        # Re-label the section header
                        personal_ctx = personal_ctx.replace("## Memory Context", "## Personal Context")
                        parts.append(personal_ctx)
                except Exception:
                    pass
            return "\n\n".join(parts)

        return self._memory_store, self._embedding_store, memory_fn

    def _build_tools(self, memory: Any, embedding_store: Any) -> dict[str, tuple[ToolDef, Any]]:
        """Assemble tool registry: built-in + plugins + delegation + skills + extra.

        Each tool is tagged with its source for the 3-layer taxonomy:
          Skills (HOW) → Tools (WHAT) → Connectors (WHERE + bundled tools)
        """
        from isaac.agents.tools import build_builtin_tools
        registry = build_builtin_tools(memory, self._config.cwd, embedding_store)
        # Tag built-in tools
        for name, (tdef, handler) in registry.items():
            if tdef.source == "built_in":
                pass  # Already tagged

        # Plugin tools from ~/.isaac/tools/
        if self._enable_plugins:
            try:
                from isaac.mcp.tool_loader import ToolLoader
                import asyncio
                loader = ToolLoader()
                plugin_tools = loader.scan()
                for name, ptool in plugin_tools.items():
                    async def _make_handler(pt):
                        async def handler(**kwargs):
                            if asyncio.iscoroutinefunction(pt.handler):
                                return await pt.handler(**kwargs)
                            return pt.handler(**kwargs)
                        return handler

                    registry[name] = (
                        ToolDef(
                            name=name,
                            description=ptool.description,
                            input_schema=ptool.input_schema,
                            permission=PermissionLevel.AUTO,
                            source="plugin",
                        ),
                        asyncio.get_event_loop().run_until_complete(_make_handler(ptool)),
                    )
            except Exception:
                pass

        # Delegation
        if self._enable_delegation:
            try:
                from isaac.agents.delegation import AgentDelegator
                agents = load_agents_config()
                delegator = AgentDelegator(agents, memory, embedding_store)
                if "delegate_agent" in registry:
                    tdef, _ = registry["delegate_agent"]
                    async def real_delegate(agent_name: str, task: str) -> dict:
                        return await delegator.delegate(agent_name, task)
                    registry["delegate_agent"] = (tdef, real_delegate)
                for tool_name, (tdef, handler) in delegator.get_exposable_tools().items():
                    registry[tool_name] = (tdef, handler)
            except Exception:
                pass

        # Workspace tools (always available)
        try:
            from isaac.plugins.workspace import build_workspace_tools
            registry.update(build_workspace_tools(self._config.cwd))
        except Exception:
            pass

        # Computer-scope tools (only if enabled in agent config)
        if self._config.computer_scope:
            try:
                from isaac.plugins.computer_scope import build_computer_scope_tools
                registry.update(build_computer_scope_tools())
            except Exception:
                pass

        # App tools (loaded as plugin, not baked into core)
        try:
            from isaac.plugins.apps import build_app_tools
            memory_ref = memory if memory else self._memory_store
            app_tools = build_app_tools(memory=memory_ref, parent_tools=registry)
            for name, (tdef, handler) in app_tools.items():
                tdef.source = "app"
                registry[name] = (tdef, handler)
        except Exception:
            pass

        # Skill tools (use_skill, list_skills)
        try:
            from isaac.plugins.skills import build_skill_tools
            registry.update(build_skill_tools())
        except Exception:
            pass

        # Wiki tools (personal knowledge bases)
        try:
            from isaac.wiki.tools import build_wiki_tools
            registry.update(build_wiki_tools())
        except Exception:
            pass

        # Personal memory tools (remember, search, read)
        try:
            from isaac.personal.tools import build_personal_tools
            registry.update(build_personal_tools())
        except Exception:
            pass

        # Extra tools (from caller)
        registry.update(self._extra_tools)

        # --- Apply tool filter from agent config ---
        # tools: ["*"] = all tools (default), otherwise filter by glob patterns
        if self._config.tools and self._config.tools != ["*"]:
            import fnmatch
            allowed = set()
            # Always keep internal entries (like _overflow_store)
            for key in list(registry.keys()):
                if key.startswith("_"):
                    allowed.add(key)
                    continue
                for pattern in self._config.tools:
                    if fnmatch.fnmatch(key, pattern):
                        allowed.add(key)
                        break
            registry = {k: v for k, v in registry.items() if k in allowed}

        return registry

    async def _connect_mcp_services(self, registry: dict) -> Any:
        """Connect to MCP services directly (no gateway subprocess).

        Each connector runs in an isolated task with timeout.
        Failed connectors are recorded but don't block startup.
        Missing credentials → "needs_auth" state (not silently skipped).
        Returns a ConnectorRegistry with health state.
        """
        from isaac.mcp.connector_registry import ConnectorRegistry, ConnectorState
        from isaac.mcp.connections import load_connections
        from isaac.mcp.credentials import CredentialStore

        cr = ConnectorRegistry()
        cred_store = CredentialStore()
        connections = load_connections(credential_store=cred_store)
        if not connections:
            return cr

        async def _connect_one(name, conn):
            """Connect a single service in isolation."""
            # Check for missing credentials BEFORE trying to connect
            if conn.env:
                missing = [k for k, v in conn.env.items() if not v]
                if missing:
                    state = ConnectorState(
                        name=name,
                        status="needs_auth",
                        error=f"Missing credentials: {', '.join(missing)}",
                        missing_credentials=missing,
                    )
                    cr._states[name] = state
                    return state

            try:
                return await asyncio.wait_for(
                    cr.connect(
                        name=name,
                        command=conn.command,
                        args=conn.args,
                        url=conn.url,
                        transport=conn.transport,
                        env=conn.env if conn.env else None,
                        registry=registry,
                    ),
                    timeout=15,
                )
            except asyncio.TimeoutError:
                state = ConnectorState(name=name, status="failed", error="Connection timed out")
                cr._states[name] = state
                return state
            except Exception as e:
                state = ConnectorState(name=name, status="failed", error=str(e))
                cr._states[name] = state
                return state

        # Connect all enabled MCP services in parallel
        tasks = []
        service_conns = {}
        for name, conn in connections.items():
            if not conn.enabled or conn.type != "mcp":
                continue
            # Skip if required env vars are missing
            if conn.env:
                missing = [k for k, v in conn.env.items() if not v]
                if missing:
                    continue
            tasks.append(_connect_one(name, conn))
            service_conns[name] = conn

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Late-bind connector management tools
        self._register_connector_tools(cr, registry, service_conns)

        connected = sum(1 for s in cr.get_status() if s.status == "connected")
        failed = sum(1 for s in cr.get_status() if s.status == "failed")
        needs_auth = sum(1 for s in cr.get_status() if s.status == "needs_auth")
        if connected or failed or needs_auth:
            import logging
            parts = []
            if connected: parts.append(f"{connected} connected")
            if needs_auth: parts.append(f"{needs_auth} needs_auth")
            if failed: parts.append(f"{failed} failed")
            logging.getLogger(__name__).info(f"Connectors: {', '.join(parts)}")

        return cr

    def _register_connector_tools(
        self, cr: Any, registry: dict, service_conns: dict
    ) -> None:
        """Late-bind connector management tools into the registry."""
        from isaac.mcp.credentials import CredentialStore

        cred_store = CredentialStore()

        async def connector_status() -> dict:
            states = cr.get_status()
            return {
                "connectors": [
                    {
                        "name": s.name,
                        "status": s.status,
                        "tools": s.tools[:10],
                        "tool_count": len(s.tools),
                        "error": s.error,
                        "missing_credentials": s.missing_credentials,
                    }
                    for s in states
                ],
                "connected": sum(1 for s in states if s.status == "connected"),
                "failed": sum(1 for s in states if s.status == "failed"),
                "needs_auth": sum(1 for s in states if s.status == "needs_auth"),
            }

        registry["connector_status"] = (
            ToolDef(
                name="connector_status",
                description="Show status of all connectors: connected, failed, or needs_auth. Shows tools per connector and missing credentials.",
                input_schema={"type": "object", "properties": {}},
                permission=PermissionLevel.AUTO,
                is_read_only=True,
                source="built_in",
            ),
            connector_status,
        )

        async def connector_reconnect(name: str, credentials: dict[str, str] | None = None) -> dict:
            """Reconnect a connector. Optionally provide credentials to fix needs_auth."""
            # If credentials provided, save them first
            if credentials:
                cred_store.save_many(name, credentials)

            conn = service_conns.get(name)
            if not conn:
                # Try loading from connections.yaml (might have been added via connect_service)
                from isaac.mcp.connections import load_connections
                all_conns = load_connections(credential_store=cred_store)
                if name in all_conns:
                    conn = all_conns[name]
                    service_conns[name] = conn
                else:
                    return {"error": f"Unknown connector: {name}"}

            # Re-resolve env with credential store
            if conn.env:
                from isaac.mcp.credentials import resolve_credential
                resolved_env = {}
                for k, v in conn.env.items():
                    if not v:
                        resolved_env[k] = resolve_credential(name, k, cred_store)
                    else:
                        resolved_env[k] = v
                conn.env = resolved_env

            state = await cr.reconnect(
                name, command=conn.command, args=conn.args,
                url=conn.url, transport=conn.transport,
                env=conn.env if conn.env else None,
                registry=registry,
            )
            return {
                "name": name,
                "status": state.status,
                "tools": state.tools,
                "tool_count": len(state.tools),
                "error": state.error,
            }

        registry["connector_reconnect"] = (
            ToolDef(
                name="connector_reconnect",
                description=(
                    "Reconnect a failed or needs_auth connector. Optionally provide credentials "
                    "to fix authentication. Credentials are stored securely and persist across sessions."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Connector name"},
                        "credentials": {
                            "type": "object",
                            "description": "Credentials to store (e.g. {\"GITHUB_PERSONAL_ACCESS_TOKEN\": \"ghp_...\"})",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": ["name"],
                },
                permission=PermissionLevel.AUTO,
                source="built_in",
            ),
            connector_reconnect,
        )

        # Upgrade connect_service — credentials go to store, hot-connect immediately
        async def connect_service_live(
            name: str, command: str = "", args: str = "", url: str = "",
            transport: str = "stdio", description: str = "",
            credentials: dict[str, str] | None = None,
        ) -> dict:
            """Connect a new service with optional credentials. Activates immediately."""
            # Save credentials if provided
            if credentials:
                cred_store.save_many(name, credentials)

            # Write to connections.yaml
            from isaac.mcp.connections import add_connection
            args_list = args.split() if args else []
            env_refs = {}
            if credentials:
                for k in credentials:
                    env_refs[k] = f"${{CRED:{name}.{k}}}"
            add_connection(
                name=name, type="mcp", command=command, args=args_list,
                url=url, transport=transport, env=env_refs, description=description,
            )

            # Hot-connect immediately
            from isaac.mcp.connections import load_connections
            all_conns = load_connections(credential_store=cred_store)
            conn = all_conns.get(name)
            if conn:
                service_conns[name] = conn
                try:
                    state = await asyncio.wait_for(
                        cr.connect(
                            name=name, command=conn.command, args=conn.args,
                            url=conn.url, transport=conn.transport,
                            env=conn.env if conn.env else None,
                            registry=registry,
                        ),
                        timeout=15,
                    )
                    return {
                        "connected": name,
                        "status": state.status,
                        "tools": state.tools,
                        "tool_count": len(state.tools),
                        "error": state.error,
                    }
                except Exception as e:
                    return {"connected": name, "status": "failed", "error": str(e),
                            "note": "Saved to connections.yaml. Will retry on next restart."}

            return {"connected": name, "note": "Saved to connections.yaml."}

        # Override the built-in connect_service with the live version
        if "connect_service" in registry:
            old_tdef, _ = registry["connect_service"]
            new_tdef = ToolDef(
                name="connect_service",
                description=(
                    "Connect a new service. Provide credentials to store them securely and activate immediately. "
                    "Use catalog_setup to find the command, args, and required credentials for a service."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Service name (e.g. 'github', 'slack')"},
                        "command": {"type": "string", "description": "Command for stdio MCP server (e.g. 'npx')"},
                        "args": {"type": "string", "description": "Space-separated args"},
                        "url": {"type": "string", "description": "URL for HTTP MCP server"},
                        "transport": {"type": "string", "description": "stdio or http", "default": "stdio"},
                        "description": {"type": "string", "description": "What this service does"},
                        "credentials": {
                            "type": "object",
                            "description": "API keys/tokens to store securely (e.g. {\"GITHUB_PERSONAL_ACCESS_TOKEN\": \"ghp_...\"})",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": ["name"],
                },
                permission=PermissionLevel.AUTO,
                source="built_in",
            )
            registry["connect_service"] = (new_tdef, connect_service_live)

    async def build(self) -> HarnessCore:
        """Assemble and return a HarnessCore. Async because gateway may need connection."""
        memory, embedding_store, memory_fn = self._build_memory()
        registry = self._build_tools(memory, embedding_store)
        gate = PermissionGate()
        llm_client = self._resolve_llm_client()

        # Sandbox bridge (wrap tools to route bash/files to VM)
        sandbox_session = None
        sandbox_backend = None
        if self._enable_sandbox and self._config.sandbox:
            try:
                from isaac.sandbox.bridge import SessionBridge
                from isaac.sandbox.lifecycle import ensure_sandbox
                from isaac.sandbox.registry import get_sandbox_backend

                sandbox_backend = get_sandbox_backend(self._config.sandbox)
                sandbox_session = await ensure_sandbox(
                    sandbox_backend, self._config.name, self._config.sandbox_template,
                    size=self._config.sandbox_size, disk_gb=self._config.sandbox_disk_gb,
                )
                bridge = SessionBridge(sandbox_backend, sandbox_session.info.sandbox_id)
                registry = bridge.wrap_tools(registry)
            except Exception as e:
                # Surface HTTP response body if available (Fly API errors)
                detail = str(e)
                if hasattr(e, "response"):
                    try:
                        detail = f"{e} — {e.response.text}"
                    except Exception:
                        pass
                if not detail:
                    detail = repr(e)
                raise RuntimeError(
                    f"Sandbox required but failed to start for {self._config.name}: {detail}"
                ) from e

        # --- Direct MCP connections (no gateway subprocess) ---
        # Each connector connects independently. One dying doesn't affect others.
        connector_registry = None
        if self._enable_gateway:
            connector_registry = await self._connect_mcp_services(registry)

        config = HarnessConfig(
            agent_config=self._config,
            tool_registry=registry,
            permission_gate=gate,
            llm_client=llm_client,
            memory_fn=memory_fn,
            approval_fn=self._approval_fn,
            soul_mode=self._soul_mode,
            soul_override=self._soul_override,
            event_handler=self._event_handler,
            sandbox_session=sandbox_session,
            sandbox_backend=sandbox_backend,
            connector_registry=connector_registry,
        )

        return HarnessCore(config)

    def build_sync(self) -> HarnessCore:
        """Synchronous build for simple use cases."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.build())
