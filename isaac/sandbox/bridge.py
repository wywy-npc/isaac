"""
Session Bridge — routes agent tool calls between local and sandbox.

The bridge intercepts bash/file tools and routes them to the VM when a sandbox
is active. Memory tools, MCP tools, and delegation always stay local.

This is the "brain on Mac, hands in VM" implementation.

Routing rules:
  bash         → VM (agent's compute happens there)
  file_read    → VM (agent reads VM filesystem)
  file_write   → VM (agent writes to VM filesystem)
  file_list    → VM (agent browses VM filesystem)
  file_search  → VM (agent searches VM files)
  local_read   → Mac (whitelisted paths only)
  local_write  → Mac (whitelisted paths only)
  memory_*     → Mac (always local — brain stays home)
  mcp__*       → Mac (credentials never leave)
  web_search   → Mac (API key stays local)
  delegate_*   → Mac (orchestration is local)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from isaac.core.types import PermissionLevel, ToolDef
from isaac.sandbox.base import ExecResult, Sandbox, SandboxInfo, SandboxState

log = logging.getLogger(__name__)

# Default whitelisted paths for local_read/local_write
DEFAULT_READ_PATHS = [
    "~/Documents/",
    "~/Downloads/",
    "~/Desktop/",
]
DEFAULT_WRITE_PATHS = [
    "~/Documents/isaac-output/",
    "~/.isaac/tools/",
]


class SessionBridge:
    """Routes tool calls between local execution and sandbox VM.

    The bridge wraps the built-in tools. For tools that should run in the VM,
    it replaces the handler with one that executes via the sandbox API.
    For tools that stay local, it passes through unchanged.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        sandbox_id: str,
        read_paths: list[str] | None = None,
        write_paths: list[str] | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.sandbox_id = sandbox_id
        self.read_paths = [os.path.expanduser(p) for p in (read_paths or DEFAULT_READ_PATHS)]
        self.write_paths = [os.path.expanduser(p) for p in (write_paths or DEFAULT_WRITE_PATHS)]

    def wrap_tools(
        self, registry: dict[str, tuple[ToolDef, Any]]
    ) -> dict[str, tuple[ToolDef, Any]]:
        """Wrap tool registry — redirect VM-bound tools, add local_read/local_write."""
        wrapped: dict[str, tuple[ToolDef, Any]] = {}

        for name, (tdef, handler) in registry.items():
            if name == "bash":
                wrapped[name] = (tdef, self._make_sandbox_bash())
            elif name == "file_read":
                wrapped[name] = (tdef, self._make_sandbox_file_read())
            elif name == "file_write":
                wrapped[name] = (tdef, self._make_sandbox_file_write())
            elif name == "file_list":
                wrapped[name] = (tdef, self._make_sandbox_file_list())
            elif name == "file_search":
                wrapped[name] = (tdef, self._make_sandbox_file_search())
            else:
                # Memory, MCP, web_search, delegation — stay local
                wrapped[name] = (tdef, handler)

        # Add local_read and local_write
        wrapped["local_read"] = (
            ToolDef(
                name="local_read",
                description="Read a file from the user's Mac. Restricted to whitelisted paths.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": f"File path on the user's Mac. Allowed: {', '.join(self.read_paths)}",
                        },
                    },
                    "required": ["path"],
                },
                permission=PermissionLevel.AUTO,
                is_read_only=True,
            ),
            self._make_local_read(),
        )

        wrapped["local_write"] = (
            ToolDef(
                name="local_write",
                description="Write a file to the user's Mac. Restricted to whitelisted paths.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": f"File path on the user's Mac. Allowed: {', '.join(self.write_paths)}",
                        },
                        "content": {"type": "string", "description": "File content"},
                    },
                    "required": ["path", "content"],
                },
                permission=PermissionLevel.AUTO,
            ),
            self._make_local_write(),
        )

        wrapped["local_bash"] = (
            ToolDef(
                name="local_bash",
                description=(
                    "Run a shell command on the user's Mac. Use for VM management "
                    "(fly CLI), network checks, installing tools, or anything that "
                    "must run on the host machine rather than the sandbox VM."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute on the Mac",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds",
                            "default": 30,
                        },
                    },
                    "required": ["command"],
                },
                permission=PermissionLevel.ASK,  # Always ask — this runs on the Mac
            ),
            self._make_local_bash(),
        )

        wrapped["bootstrap_update"] = (
            ToolDef(
                name="bootstrap_update",
                description=(
                    "Install newly registered apps on your VM. Reads ~/.isaac/apps/*.yaml "
                    "manifests, checks what's already installed, and only sets up new apps. "
                    "Use after a new app has been added via `isaac app add`."
                ),
                input_schema={"type": "object", "properties": {}},
                permission=PermissionLevel.AUTO,
            ),
            self._make_bootstrap_update(),
        )

        # --- Sandbox management tools (agent controls its own VM) ---

        wrapped["sandbox_scale"] = (
            ToolDef(
                name="sandbox_scale",
                description=(
                    "Scale your VM up or down. Use when you need more compute (GPU, CPU, RAM) "
                    "or want to downsize after heavy work. Machine restarts with new size, "
                    "disk preserved. GPU sizes: a10 ($1.50/hr), l40s ($2.50/hr), "
                    "a100-40gb ($3.50/hr), a100-80gb ($5/hr). "
                    "CPU sizes: shared-cpu-1x to performance-16x."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "size": {
                            "type": "string",
                            "description": (
                                "Target size. CPU: shared-cpu-1x, shared-cpu-4x, performance-4x, performance-16x. "
                                "GPU: a10, l40s, a100-40gb, a100-80gb"
                            ),
                        },
                    },
                    "required": ["size"],
                },
                permission=PermissionLevel.AUTO,
            ),
            self._make_sandbox_scale(),
        )

        wrapped["sandbox_info"] = (
            ToolDef(
                name="sandbox_info",
                description="Get current sandbox VM state: size, region, status, IP.",
                input_schema={"type": "object", "properties": {}},
                permission=PermissionLevel.AUTO,
                is_read_only=True,
            ),
            self._make_sandbox_info(),
        )

        return wrapped

    # --- Sandbox-routed tools ---

    def _make_sandbox_bash(self):
        sandbox = self.sandbox
        sid = self.sandbox_id

        async def bash(command: str, timeout: int = 30) -> dict[str, Any]:
            result = await sandbox.exec(sid, command, timeout=timeout)
            return {
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_ms": result.duration_ms,
                "sandbox": True,
            }
        return bash

    def _make_sandbox_file_read(self):
        sandbox = self.sandbox
        sid = self.sandbox_id

        async def file_read(path: str) -> dict[str, Any]:
            try:
                content = await sandbox.read_file(sid, path)
                if len(content) > 50_000:
                    content = content[:50_000] + "\n\n[... truncated ...]"
                return {"path": path, "content": content, "sandbox": True}
            except Exception as e:
                return {"error": str(e)}
        return file_read

    def _make_sandbox_file_write(self):
        sandbox = self.sandbox
        sid = self.sandbox_id

        async def file_write(path: str, content: str) -> dict[str, Any]:
            try:
                await sandbox.write_file(sid, path, content)
                return {"written": path, "bytes": len(content), "sandbox": True}
            except Exception as e:
                return {"error": str(e)}
        return file_write

    def _make_sandbox_file_list(self):
        sandbox = self.sandbox
        sid = self.sandbox_id

        async def file_list(path: str = "/workspace", pattern: str = "*") -> dict[str, Any]:
            result = await sandbox.exec(sid, f"find {path} -maxdepth 1 -name '{pattern}' | head -100", timeout=10)
            if result.exit_code != 0:
                return {"error": result.stderr}
            entries = [e for e in result.stdout.strip().split("\n") if e]
            return {"entries": entries, "sandbox": True}
        return file_list

    def _make_sandbox_file_search(self):
        sandbox = self.sandbox
        sid = self.sandbox_id

        async def file_search(pattern: str, path: str = "/workspace") -> dict[str, Any]:
            result = await sandbox.exec(
                sid,
                f"grep -rl '{pattern}' {path} --include='*.py' --include='*.md' --include='*.yaml' --include='*.json' --include='*.txt' --include='*.js' --include='*.ts' 2>/dev/null | head -20",
                timeout=15,
            )
            files = [f for f in result.stdout.strip().split("\n") if f]
            return {"matches": files, "sandbox": True}
        return file_search

    # --- Sandbox management tools ---

    def _make_sandbox_scale(self):
        sandbox = self.sandbox
        sid = self.sandbox_id

        async def sandbox_scale(size: str) -> dict[str, Any]:
            """Scale the VM to a new size. Restarts the machine, disk preserved."""
            from isaac.sandbox.fly import MACHINE_SIZES

            if size not in MACHINE_SIZES:
                available = ", ".join(sorted(MACHINE_SIZES.keys()))
                return {"error": f"Unknown size '{size}'. Available: {available}"}

            size_info = MACHINE_SIZES[size]
            cost = size_info["cost_hr"]
            gpu = size_info.get("gpu", "none")

            try:
                # Fly Machines API: update the machine config, then restart
                import httpx
                fly_sandbox = sandbox
                headers = fly_sandbox._headers()
                url = fly_sandbox._url(f"/machines/{sid}")

                # Get current config
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    current = resp.json()

                # Update size
                config = current.get("config", {})
                config["size"] = size

                # GPU machines need guest config
                if size_info.get("gpu"):
                    config["guest"] = {
                        "cpus": size_info["cpu"],
                        "memory_mb": size_info["ram"],
                        "gpu_kind": size_info["gpu"],
                    }
                elif "guest" in config:
                    del config["guest"]

                # Update the machine
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        url, headers=headers, json={"config": config}
                    )
                    resp.raise_for_status()

                    # Restart to apply
                    await client.post(
                        fly_sandbox._url(f"/machines/{sid}/restart"),
                        headers=headers,
                    )

                return {
                    "scaled": True,
                    "new_size": size,
                    "cpu": size_info["cpu"],
                    "ram_mb": size_info["ram"],
                    "gpu": gpu,
                    "cost_per_hour": cost,
                    "note": "Machine restarting with new size. Disk preserved. Give it 5-10 seconds.",
                }
            except Exception as e:
                return {"error": f"Scale failed: {e}"}

        return sandbox_scale

    def _make_sandbox_info(self):
        sandbox = self.sandbox
        sid = self.sandbox_id

        async def sandbox_info() -> dict[str, Any]:
            try:
                info = await sandbox.info(sid)
                return {
                    "sandbox_id": info.sandbox_id,
                    "agent": info.agent_name,
                    "state": info.state.value,
                    "ip": info.ip,
                    "region": info.region,
                    "meta": info.meta,
                }
            except Exception as e:
                return {"error": str(e)}

        return sandbox_info

    def _make_bootstrap_update(self):
        sandbox = self.sandbox
        sid = self.sandbox_id

        async def bootstrap_update() -> dict[str, Any]:
            """Install newly added apps on the VM (incremental)."""
            from isaac.sandbox.bootstrap import run_incremental_bootstrap
            return await run_incremental_bootstrap(sandbox, sid)

        return bootstrap_update

    # --- Local tools (whitelisted Mac access) ---

    def _make_local_read(self):
        read_paths = self.read_paths

        async def local_read(path: str) -> dict[str, Any]:
            resolved = os.path.expanduser(path)
            if not any(resolved.startswith(p) for p in read_paths):
                return {"error": f"Path not in whitelist. Allowed: {read_paths}"}
            target = Path(resolved)
            if not target.exists():
                return {"error": f"File not found: {resolved}"}
            try:
                content = target.read_text()
                if len(content) > 50_000:
                    content = content[:50_000] + "\n\n[... truncated ...]"
                return {"path": resolved, "content": content, "local": True}
            except Exception as e:
                return {"error": str(e)}
        return local_read

    def _make_local_write(self):
        write_paths = self.write_paths

        async def local_write(path: str, content: str) -> dict[str, Any]:
            resolved = os.path.expanduser(path)
            if not any(resolved.startswith(p) for p in write_paths):
                return {"error": f"Path not in whitelist. Allowed: {write_paths}"}
            target = Path(resolved)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            return {"written": resolved, "bytes": len(content), "local": True}
        return local_write

    def _make_local_bash(self):
        async def local_bash(command: str, timeout: int = 30) -> dict[str, Any]:
            """Run a command on the user's Mac (not the sandbox)."""
            import subprocess
            try:
                proc = await asyncio.wait_for(
                    asyncio.create_subprocess_shell(
                        command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    ),
                    timeout=timeout,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout,
                )
                output = stdout.decode()
                if len(output) > 30_000:
                    output = output[:15_000] + "\n\n[... truncated ...]\n\n" + output[-15_000:]
                return {
                    "exit_code": proc.returncode,
                    "stdout": output,
                    "stderr": stderr.decode()[:5_000],
                    "local": True,
                }
            except asyncio.TimeoutError:
                return {"error": f"Command timed out after {timeout}s", "local": True}
            except Exception as e:
                return {"error": str(e), "local": True}
        return local_bash
