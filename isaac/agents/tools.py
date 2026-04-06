"""Built-in tools — the core toolset every agent gets."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from isaac.core.types import PermissionLevel, ToolDef
from isaac.memory.store import MemoryStore


def build_builtin_tools(
    memory: MemoryStore, cwd: str | None = None, embedding_store: Any = None,
) -> dict[str, tuple[ToolDef, Any]]:
    """Build the registry of built-in tools."""
    working_dir = cwd or os.getcwd()
    registry: dict[str, tuple[ToolDef, Any]] = {}

    # --- Memory tools (auto-approved, read/write) ---

    async def memory_search(query: str) -> dict[str, Any]:
        results = memory.search(query, max_results=5)
        return {
            "results": [
                {"path": n.path, "content": n.content[:500], "tags": n.tags}
                for n in results
            ]
        }

    registry["memory_search"] = (
        ToolDef(
            name="memory_search",
            description="Search memory for relevant knowledge. Returns matching memory nodes.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        memory_search,
    )

    async def memory_read(path: str) -> dict[str, Any]:
        node = memory.read(path)
        if not node:
            return {"error": f"Not found: {path}"}
        return {"path": node.path, "content": node.content, "meta": node.meta}

    registry["memory_read"] = (
        ToolDef(
            name="memory_read",
            description="Read a specific memory node by path.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Memory node path"}},
                "required": ["path"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        memory_read,
    )

    async def memory_write(path: str, content: str, tags: list[str] | None = None, importance: float = 0.5) -> dict[str, Any]:
        from isaac.memory.linker import auto_link
        meta = {"tags": tags or [], "importance": importance}
        # Auto-link to related nodes and create stubs for new entities
        content, created_stubs = auto_link(content, path, memory, embedding_store)
        node = memory.write(path, content, meta)
        # Generate embedding if store available
        if embedding_store:
            try:
                embedding_store.embed_and_store(path, content)
            except Exception:
                pass  # never block memory writes on embedding failures
        return {"written": node.path, "links": node.outgoing_links, "created_stubs": created_stubs}

    registry["memory_write"] = (
        ToolDef(
            name="memory_write",
            description=(
                "Write or update a memory node. Auto-links to related existing nodes via [[wikilinks]] "
                "and creates stub nodes for new entities mentioned in the content. "
                "Format: use markdown with a # heading, body content, and [[path/to/node.md]] links. "
                "Organize paths: people/name.md, entities/name.md, projects/name.md, topics/name.md, "
                "deals/name.md, logs/date.md. Scout follows links automatically for context retrieval."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Memory node path (e.g. 'people/jane-doe.md', 'entities/acme-corp.md', 'projects/isaac.md')"},
                    "content": {"type": "string", "description": "Markdown content with # heading, body, and optional [[wikilinks]]"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization"},
                    "importance": {"type": "number", "description": "Importance score 0-1"},
                },
                "required": ["path", "content"],
            },
            permission=PermissionLevel.AUTO,
        ),
        memory_write,
    )

    # --- File tools ---

    async def file_read(path: str) -> dict[str, Any]:
        target = Path(path) if os.path.isabs(path) else Path(working_dir) / path
        if not target.exists():
            return {"error": f"File not found: {target}"}
        try:
            content = target.read_text()
            if len(content) > 50_000:
                content = content[:50_000] + "\n\n[... truncated ...]"
            return {"path": str(target), "content": content}
        except Exception as e:
            return {"error": str(e)}

    registry["file_read"] = (
        ToolDef(
            name="file_read",
            description="Read a file from the filesystem.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path (absolute or relative to cwd)"}},
                "required": ["path"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        file_read,
    )

    async def file_write(path: str, content: str) -> dict[str, Any]:
        target = Path(path) if os.path.isabs(path) else Path(working_dir) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return {"written": str(target), "bytes": len(content)}

    registry["file_write"] = (
        ToolDef(
            name="file_write",
            description="Write content to a file. Creates parent directories if needed.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
            permission=PermissionLevel.AUTO,
        ),
        file_write,
    )

    async def file_list(path: str = ".", pattern: str = "*") -> dict[str, Any]:
        target = Path(path) if os.path.isabs(path) else Path(working_dir) / path
        if not target.exists():
            return {"error": f"Directory not found: {target}"}
        files = sorted(str(p.relative_to(target)) for p in target.glob(pattern) if p.is_file())
        dirs = sorted(str(p.relative_to(target)) for p in target.iterdir() if p.is_dir())
        return {"files": files[:100], "dirs": dirs[:50]}

    registry["file_list"] = (
        ToolDef(
            name="file_list",
            description="List files and directories.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path", "default": "."},
                    "pattern": {"type": "string", "description": "Glob pattern", "default": "*"},
                },
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        file_list,
    )

    async def file_search(pattern: str, path: str = ".") -> dict[str, Any]:
        target = Path(path) if os.path.isabs(path) else Path(working_dir) / path
        try:
            proc = await asyncio.create_subprocess_exec(
                "grep", "-rl", "--include=*.py", "--include=*.md", "--include=*.yaml",
                "--include=*.json", "--include=*.txt", "--include=*.ts", "--include=*.js",
                pattern, str(target),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            files = stdout.decode().strip().split("\n")[:20]
            return {"matches": [f for f in files if f]}
        except Exception as e:
            return {"error": str(e)}

    registry["file_search"] = (
        ToolDef(
            name="file_search",
            description="Search file contents using grep. Returns matching file paths.",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern (regex)"},
                    "path": {"type": "string", "description": "Directory to search in", "default": "."},
                },
                "required": ["pattern"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        file_search,
    )

    # --- Shell tool ---

    async def bash(command: str, timeout: int = 30) -> dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode()
            if len(output) > 30_000:
                output = output[:15_000] + "\n\n[... truncated ...]\n\n" + output[-15_000:]
            return {
                "exit_code": proc.returncode,
                "stdout": output,
                "stderr": stderr.decode()[:5000],
            }
        except asyncio.TimeoutError:
            return {"error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}

    registry["bash"] = (
        ToolDef(
            name="bash",
            description="Execute a shell command. Use for system operations, git, package management, etc.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                },
                "required": ["command"],
            },
            permission=PermissionLevel.AUTO,
        ),
        bash,
    )

    # --- Web search (DuckDuckGo free, Brave if API key set) ---

    async def web_search(query: str) -> dict[str, Any]:
        # Prefer Brave if key is set
        brave_key = os.environ.get("BRAVE_API_KEY")
        if brave_key:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        params={"q": query, "count": 5},
                        headers={"X-Subscription-Token": brave_key},
                    )
                    data = resp.json()
                    results = []
                    for r in data.get("web", {}).get("results", [])[:5]:
                        results.append({
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "snippet": r.get("description", ""),
                        })
                    return {"results": results}
            except Exception as e:
                return {"results": [], "error": f"Brave search failed: {e}"}

        # Fallback: DuckDuckGo (free, no API key)
        try:
            from duckduckgo_search import DDGS
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, lambda: DDGS().text(query, max_results=5)
            )
            results = [
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                for r in (raw or [])
            ]
            return {"results": results}
        except Exception as e:
            return {"results": [], "error": f"Web search failed: {e}"}

    registry["web_search"] = (
        ToolDef(
            name="web_search",
            description="Search the web. Returns titles, URLs, and snippets.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        web_search,
    )

    # --- Agent delegation tool (stub — wired to real delegation in terminal.py) ---

    async def delegate_agent(agent_name: str, task: str) -> dict[str, Any]:
        return {
            "status": "delegated",
            "agent": agent_name,
            "task": task,
            "note": "Task delegated. Check agent output for results.",
        }

    registry["delegate_agent"] = (
        ToolDef(
            name="delegate_agent",
            description="Delegate a task to another agent in the constellation.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string", "description": "Name of the agent to delegate to"},
                    "task": {"type": "string", "description": "Task description for the agent"},
                },
                "required": ["agent_name", "task"],
            },
            permission=PermissionLevel.AUTO,
        ),
        delegate_agent,
    )

    # --- Continuation tool (clean-room handoff for heartbeats / long tasks) ---

    async def write_continuation(
        what_was_done: str,
        what_remains: str = "Nothing — task complete.",
        blocking_on: str = "Nothing.",
        artifacts: list[str] | None = None,
        next_priority: str = "",
    ) -> dict[str, Any]:
        from isaac.core.heartbeat import write_continuation as _write
        # Use the agent name from the working directory context
        agent_name = os.environ.get("ISAAC_AGENT", "default")
        path = _write(agent_name, what_was_done, what_remains, blocking_on, artifacts, next_priority)
        return {"written": path, "agent": agent_name}

    registry["write_continuation"] = (
        ToolDef(
            name="write_continuation",
            description="Write a structured handoff for the next run. Use when finishing a heartbeat or long task.",
            input_schema={
                "type": "object",
                "properties": {
                    "what_was_done": {"type": "string", "description": "Factual summary of actions taken"},
                    "what_remains": {"type": "string", "description": "Specific unfinished items"},
                    "blocking_on": {"type": "string", "description": "External dependencies or missing info"},
                    "artifacts": {"type": "array", "items": {"type": "string"}, "description": "Paths to files/memory created"},
                    "next_priority": {"type": "string", "description": "What the next run should focus on first"},
                },
                "required": ["what_was_done"],
            },
            permission=PermissionLevel.AUTO,
        ),
        write_continuation,
    )

    # --- E2B sandbox tools (only if E2B_API_KEY is set) ---

    if os.environ.get("E2B_API_KEY"):
        try:
            from isaac.sandbox.e2b import E2BSandbox
            _sandbox = E2BSandbox()

            async def sandbox_execute(code: str, language: str = "python") -> dict[str, Any]:
                return await _sandbox.execute(code, language)

            registry["sandbox_execute"] = (
                ToolDef(
                    name="sandbox_execute",
                    description="Execute code in a cloud sandbox. Safe for untrusted code.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Code to execute"},
                            "language": {"type": "string", "description": "Language (python, javascript)", "default": "python"},
                        },
                        "required": ["code"],
                    },
                    permission=PermissionLevel.AUTO,
                ),
                sandbox_execute,
            )

            async def sandbox_upload(path: str) -> dict[str, Any]:
                return await _sandbox.upload(path)

            registry["sandbox_upload"] = (
                ToolDef(
                    name="sandbox_upload",
                    description="Upload a local file to the cloud sandbox.",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Local file path to upload"}},
                        "required": ["path"],
                    },
                    permission=PermissionLevel.AUTO,
                ),
                sandbox_upload,
            )

            async def sandbox_download(path: str) -> dict[str, Any]:
                return await _sandbox.download(path)

            registry["sandbox_download"] = (
                ToolDef(
                    name="sandbox_download",
                    description="Download a file from the cloud sandbox.",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Remote file path in sandbox"}},
                        "required": ["path"],
                    },
                    permission=PermissionLevel.AUTO,
                ),
                sandbox_download,
            )
        except ImportError:
            pass  # e2b-code-interpreter not installed

    # --- App tools (bolt any app as a tool) ---

    async def app_list() -> dict[str, Any]:
        from isaac.apps.manifest import list_manifests
        apps = list_manifests()
        return {"apps": apps, "count": len(apps)}

    registry["app_list"] = (
        ToolDef(
            name="app_list",
            description="List available external apps that can be run as tools.",
            input_schema={"type": "object", "properties": {}},
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        app_list,
    )

    async def app_run(app: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        from isaac.apps.runner import AppRunner
        # Pass the full parent tool registry so the VM agent gets access to
        # web search, memory, MCP tools, connected apps — the full ISAAC surface
        runner = AppRunner(memory=memory, parent_tools=registry)
        result = await runner.run(app, inputs or {})
        return {
            "status": result.status,
            "summary": result.summary[:3000],
            "artifacts": result.artifacts,
            "duration": result.duration,
            "cost": result.cost,
            "error": result.error,
        }

    registry["app_run"] = (
        ToolDef(
            name="app_run",
            description="Run an external app on cloud GPU. Provisions compute, clones repo, runs the app, collects artifacts.",
            input_schema={
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "App name (from app_list)"},
                    "inputs": {
                        "type": "object",
                        "description": "App-specific inputs (see manifest for schema)",
                        "additionalProperties": True,
                    },
                },
                "required": ["app"],
            },
            permission=PermissionLevel.AUTO,
        ),
        app_run,
    )

    # --- Service connection tools (manage upstream MCP servers) ---

    async def list_services() -> dict[str, Any]:
        from isaac.mcp.connections import list_connections
        return {"services": list_connections()}

    registry["list_services"] = (
        ToolDef(
            name="list_services",
            description="List all connected external services (MCP servers, APIs).",
            input_schema={"type": "object", "properties": {}},
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        list_services,
    )

    async def connect_service(
        name: str, command: str = "", args: str = "", url: str = "",
        transport: str = "stdio", description: str = "",
    ) -> dict[str, Any]:
        from isaac.mcp.connections import add_connection
        args_list = args.split() if args else []
        conn = add_connection(
            name=name, type="mcp", command=command, args=args_list,
            url=url, transport=transport, description=description,
        )
        return {
            "connected": name,
            "note": "Added to connections.yaml. Restart `isaac serve` to activate.",
        }

    registry["connect_service"] = (
        ToolDef(
            name="connect_service",
            description="Connect a new external service (MCP server). Saves to connections.yaml.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Service name (e.g. 'google', 'slack')"},
                    "command": {"type": "string", "description": "Command for stdio MCP server (e.g. 'npx')"},
                    "args": {"type": "string", "description": "Space-separated args (e.g. '@anthropic-ai/google-mcp')"},
                    "url": {"type": "string", "description": "URL for HTTP MCP server"},
                    "transport": {"type": "string", "description": "stdio or http", "default": "stdio"},
                    "description": {"type": "string", "description": "What this service does"},
                },
                "required": ["name"],
            },
            permission=PermissionLevel.AUTO,
        ),
        connect_service,
    )

    async def disconnect_service(name: str) -> dict[str, Any]:
        from isaac.mcp.connections import remove_connection
        if remove_connection(name):
            return {"disconnected": name}
        return {"error": f"Service '{name}' not found"}

    registry["disconnect_service"] = (
        ToolDef(
            name="disconnect_service",
            description="Remove an external service connection.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Service name to disconnect"},
                },
                "required": ["name"],
            },
            permission=PermissionLevel.AUTO,
        ),
        disconnect_service,
    )

    # --- Service catalog tools (discover + auto-connect) ---

    async def catalog_search(query: str) -> dict[str, Any]:
        """Search the catalog of known connectable services."""
        from isaac.mcp.catalog import search_catalog
        results = search_catalog(query)
        return {
            "results": [
                {
                    "name": e.name,
                    "description": e.description,
                    "category": e.category,
                    "auth_type": e.auth_type,
                    "provides": e.provides,
                    "setup_url": e.setup_url,
                }
                for e in results
            ]
        }

    registry["catalog_search"] = (
        ToolDef(
            name="catalog_search",
            description="Search available services that can be connected (Gmail, Slack, GitHub, etc.).",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search by name, description, or category (communication, development, search, database, productivity, automation)"},
                },
                "required": ["query"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        catalog_search,
    )

    async def catalog_list() -> dict[str, Any]:
        """List all services in the catalog."""
        from isaac.mcp.catalog import get_full_catalog
        catalog = get_full_catalog()
        return {
            "services": [
                {
                    "name": e.name,
                    "description": e.description,
                    "category": e.category,
                    "auth_type": e.auth_type,
                }
                for e in catalog.values()
            ],
            "count": len(catalog),
        }

    registry["catalog_list"] = (
        ToolDef(
            name="catalog_list",
            description="List all known services in the catalog that can be connected.",
            input_schema={"type": "object", "properties": {}},
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        catalog_list,
    )

    async def catalog_setup(name: str) -> dict[str, Any]:
        """Get full setup instructions for connecting a service.

        Returns everything needed: command, env vars, auth steps, URLs.
        For services requiring API keys, the agent should tell the user
        exactly where to get the key and ask them to set it.
        """
        from isaac.mcp.catalog import get_full_catalog
        catalog = get_full_catalog()
        entry = catalog.get(name)
        if not entry:
            return {"error": f"Service '{name}' not in catalog. Use catalog_search to find it."}

        return {
            "name": entry.name,
            "description": entry.description,
            "command": entry.command,
            "args": entry.args,
            "auth_type": entry.auth_type,
            "env_vars": entry.env_vars,
            "setup_url": entry.setup_url,
            "setup_instructions": entry.setup_instructions,
            "provides": entry.provides,
            "npm_package": entry.npm_package,
            "next_steps": (
                "To connect this service:\n"
                "1. Ask the user to provide the required credentials listed in env_vars\n"
                "2. Use connect_service to add it to connections.yaml with the command and args shown\n"
                "3. The credentials go in env vars on the user's machine (never in the yaml file)\n"
                "4. Remind the user to restart the session or run /reload to activate"
            ),
        }

    registry["catalog_setup"] = (
        ToolDef(
            name="catalog_setup",
            description="Get full setup instructions for connecting a service from the catalog.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Service name from catalog (e.g. 'google', 'slack', 'github')"},
                },
                "required": ["name"],
            },
            permission=PermissionLevel.AUTO,
            is_read_only=True,
        ),
        catalog_setup,
    )

    async def catalog_add(
        name: str, description: str, command: str, args: str = "",
        auth_type: str = "none", env_vars: str = "",
        provides: str = "", npm_package: str = "",
        setup_instructions: str = "",
    ) -> dict[str, Any]:
        """Add a new service to the catalog (Toolsmith can do this autonomously).

        This doesn't connect the service — it adds it to the catalog so
        any agent can later connect it via catalog_setup + connect_service.
        """
        from isaac.mcp.catalog import CatalogEntry, save_custom_catalog_entry

        env_dict = {}
        if env_vars:
            for pair in env_vars.split(","):
                if "=" in pair:
                    k, v = pair.strip().split("=", 1)
                    env_dict[k.strip()] = v.strip()

        entry = CatalogEntry(
            name=name,
            description=description,
            command=command,
            args=args.split() if args else [],
            auth_type=auth_type,
            env_vars=env_dict,
            provides=provides.split(",") if provides else [],
            npm_package=npm_package,
            setup_instructions=setup_instructions,
            category="custom",
        )
        save_custom_catalog_entry(entry)
        return {"added": name, "catalog": "custom"}

    registry["catalog_add"] = (
        ToolDef(
            name="catalog_add",
            description="Add a new service to the catalog. Use when Toolsmith discovers a new MCP server or builds a connector.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Service name"},
                    "description": {"type": "string", "description": "What this service does"},
                    "command": {"type": "string", "description": "Command to run (e.g. 'npx')"},
                    "args": {"type": "string", "description": "Space-separated args"},
                    "auth_type": {"type": "string", "description": "api_key, oauth, or none"},
                    "env_vars": {"type": "string", "description": "Comma-separated KEY=description pairs"},
                    "provides": {"type": "string", "description": "Comma-separated list of tool names this service provides"},
                    "npm_package": {"type": "string", "description": "NPM package name if applicable"},
                    "setup_instructions": {"type": "string", "description": "Human-readable setup steps"},
                },
                "required": ["name", "description", "command"],
            },
            permission=PermissionLevel.AUTO,
        ),
        catalog_add,
    )

    return registry
