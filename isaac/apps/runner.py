"""AppRunner — orchestrates external apps as tools.

Full lifecycle: provision → clone → setup → execute → collect → teardown.
For agent-mode apps, spawns a child ISAAC agent inside the VM.
For command-mode apps, runs a single command and collects output.
"""
from __future__ import annotations

import asyncio
import glob as globmod
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from isaac.apps.compute import ComputeBackend, ComputeInstance, ExecResult, get_backend
from isaac.apps.manifest import AppManifest, CHECKPOINTS_DIR, load_manifest, list_manifests

from isaac.core.config import get_env
from isaac.core.types import AgentConfig, PermissionLevel
from isaac.memory.store import MemoryStore


# --- Events streamed during app execution ---

class AppEvent:
    pass


@dataclass
class AppProvisionEvent(AppEvent):
    backend: str
    gpu_type: str
    status: str


@dataclass
class AppSetupEvent(AppEvent):
    output: str


@dataclass
class AppExecEvent(AppEvent):
    command: str
    stdout: str
    stderr: str
    exit_code: int


@dataclass
class AppAgentEvent(AppEvent):
    text: str


@dataclass
class AppArtifactEvent(AppEvent):
    path: str
    size: int


@dataclass
class AppDoneEvent(AppEvent):
    summary: str
    duration: float
    artifacts_collected: int
    state_action: str  # "destroyed" | "checkpointed" | "snapshotted"


@dataclass
class AppErrorEvent(AppEvent):
    error: str


@dataclass
class AppResult:
    """Final result of an app run."""
    status: str  # "success" | "error" | "timeout"
    summary: str = ""
    artifacts: list[dict[str, str]] = field(default_factory=list)
    duration: float = 0.0
    cost: float = 0.0
    error: str | None = None


class AppRunner:
    """Orchestrates external app execution with managed compute."""

    def __init__(
        self,
        memory: MemoryStore | None = None,
        backend_name: str | None = None,
        parent_tools: dict[str, tuple[Any, Any]] | None = None,
    ) -> None:
        self.memory = memory or MemoryStore()
        self._backend_name = backend_name
        self.parent_tools = parent_tools or {}

    def _get_backend(self) -> ComputeBackend:
        return get_backend(self._backend_name)

    async def run(self, app_name: str, inputs: dict[str, Any] | None = None) -> AppResult:
        """Run an app by name. Full lifecycle management."""
        manifest = load_manifest(app_name)
        if not manifest:
            return AppResult(status="error", error=f"Unknown app: {app_name}. Run `isaac app list`.")

        inputs = inputs or {}
        start_time = time.time()
        instance = None

        try:
            backend = self._get_backend()

            # --- PROVISION ---
            instance = await backend.provision(manifest)
            if instance.metadata.get("setup_error"):
                return AppResult(
                    status="error",
                    error=f"Setup failed: {instance.metadata['setup_error']}",
                    duration=time.time() - start_time,
                )

            # --- RESTORE CHECKPOINT if applicable ---
            if manifest.state == "checkpoint":
                await self._restore_checkpoint(backend, instance, manifest)

            # --- EXECUTE ---
            if manifest.mode == "agent":
                result = await self._run_agent_mode(backend, instance, manifest, inputs)
            else:
                result = await self._run_command_mode(backend, instance, manifest, inputs)

            result.duration = time.time() - start_time

            # --- COLLECT ARTIFACTS ---
            artifacts = await self._collect_artifacts(backend, instance, manifest)
            result.artifacts = artifacts

            # --- STATE MANAGEMENT ---
            state_action = await self._handle_state(backend, instance, manifest)

            # --- SAVE TO MEMORY ---
            if artifacts:
                summary_lines = [f"# App Run: {manifest.name}", f"Duration: {result.duration:.0f}s", ""]
                for a in artifacts:
                    summary_lines.append(f"- {a['path']} ({a['size']} bytes)")
                if result.summary:
                    summary_lines.extend(["", result.summary])
                self.memory.write(
                    f"apps/{manifest.name}/run_{int(start_time)}.md",
                    "\n".join(summary_lines),
                    {"tags": ["app-run", manifest.name], "importance": 0.7},
                )

            return result

        except Exception as e:
            return AppResult(
                status="error",
                error=str(e),
                duration=time.time() - start_time,
            )
        finally:
            # Ensure teardown on error (if state policy doesn't keep it alive)
            if instance and instance.status != "stopped" and manifest and manifest.state == "ephemeral":
                try:
                    backend = self._get_backend()
                    await backend.teardown(instance)
                except Exception:
                    pass

    async def _run_command_mode(
        self,
        backend: ComputeBackend,
        instance: ComputeInstance,
        manifest: AppManifest,
        inputs: dict[str, Any],
    ) -> AppResult:
        """Execute a single command and collect output."""
        # Substitute inputs into the run command
        command = manifest.run
        for key, val in inputs.items():
            command = command.replace(f"{{{key}}}", str(val))

        workdir = instance.metadata.get("workdir", "/home/user/app")
        full_cmd = f"cd {workdir} && {command}"

        result = await backend.exec(instance, full_cmd, timeout=manifest.timeout)

        if result.error or result.exit_code != 0:
            return AppResult(
                status="error",
                summary=result.stdout[:2000],
                error=result.stderr[:2000] or result.error,
            )

        return AppResult(
            status="success",
            summary=result.stdout[:5000],
        )

    async def _run_agent_mode(
        self,
        backend: ComputeBackend,
        instance: ComputeInstance,
        manifest: AppManifest,
        inputs: dict[str, Any],
    ) -> AppResult:
        """Spawn a child ISAAC agent that operates inside the VM.

        The agent gets bash/file tools that route through the compute backend's
        exec API, so every command runs inside the VM, not locally.
        """
        import anthropic
        from isaac.core.orchestrator import Orchestrator, TextEvent, CostEvent, ErrorEvent

        # Build tools that execute on the remote VM
        remote_tools = self._build_remote_tools(backend, instance, manifest)

        # Configure the child agent
        config = AgentConfig(
            name=f"app-{manifest.name}",
            soul="default",
            model="claude-sonnet-4-6",
            max_iterations=min(manifest.timeout // 30, 50),  # rough: 30s per iteration
            context_budget=180_000,
        )

        from isaac.core.permissions import PermissionGate
        gate = PermissionGate()
        for tool_name in remote_tools:
            gate.set_override(tool_name, PermissionLevel.AUTO)

        from isaac.agents.session import new_session
        state = new_session(config.name)

        orch = Orchestrator(
            agent_config=config,
            tool_registry=remote_tools,
            permission_gate=gate,
        )

        # Build the task prompt
        task = self._build_agent_prompt(manifest, inputs)

        # Run the agent
        response_parts: list[str] = []
        total_cost = 0.0

        async for event in orch.run(task, state):
            if isinstance(event, TextEvent):
                response_parts.append(event.text)
            elif isinstance(event, CostEvent):
                total_cost += event.cost
            elif isinstance(event, ErrorEvent):
                return AppResult(status="error", error=event.error, cost=total_cost)

        return AppResult(
            status="success",
            summary="\n".join(response_parts)[-5000:],
            cost=total_cost,
        )

    def _build_remote_tools(
        self,
        backend: ComputeBackend,
        instance: ComputeInstance,
        manifest: AppManifest,
    ) -> dict[str, tuple[Any, Any]]:
        """Build a merged tool registry for the VM agent.

        Three layers, in priority order:
        1. Remote-override tools (bash, file_read, file_write) — these execute
           on the VM via the compute backend, NOT locally.
        2. Parent tools passed through — web_search, memory_*, MCP tools,
           delegate_agent, etc. These execute locally on the ISAAC host.
           The VM agent can search the web, read memory, call connected apps.
        3. Excluded: app_run (prevent recursive app launches), file_list,
           file_search (these would search local filesystem, not VM).
        """
        from isaac.core.types import ToolDef
        registry: dict[str, tuple[ToolDef, Any]] = {}
        workdir = instance.metadata.get("workdir", "/home/user/app")

        # --- Layer 2: Pass through parent tools (local execution on ISAAC host) ---
        # These give the VM agent access to web search, memory, MCP servers,
        # connected apps — the full ISAAC tool surface.
        EXCLUDE_FROM_PASSTHROUGH = {
            "bash", "file_read", "file_write",  # overridden with remote versions
            "file_list", "file_search",  # would search local fs, not VM
            "app_run",  # prevent recursive app launches
        }
        for name, entry in self.parent_tools.items():
            if name not in EXCLUDE_FROM_PASSTHROUGH:
                registry[name] = entry

        # --- Layer 1: Remote-override tools (execute on VM) ---

        async def remote_bash(command: str, timeout: int = 120) -> dict[str, Any]:
            result = await backend.exec(instance, f"cd {workdir} && {command}", timeout=timeout)
            return {
                "stdout": result.stdout[:30_000],
                "stderr": result.stderr[:5000],
                "exit_code": result.exit_code,
                "error": result.error,
            }

        registry["bash"] = (
            ToolDef(
                name="bash",
                description="Execute a command on the remote GPU VM.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command"},
                        "timeout": {"type": "integer", "default": 120},
                    },
                    "required": ["command"],
                },
                permission=PermissionLevel.AUTO,
            ),
            remote_bash,
        )

        async def remote_file_read(path: str) -> dict[str, Any]:
            result = await backend.exec(instance, f"cat {workdir}/{path}")
            if result.exit_code != 0:
                return {"error": result.stderr or "File not found"}
            content = result.stdout
            if len(content) > 50_000:
                content = content[:50_000] + "\n[... truncated ...]"
            return {"path": path, "content": content}

        registry["file_read"] = (
            ToolDef(
                name="file_read",
                description="Read a file from the remote VM.",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "File path (relative to app dir)"}},
                    "required": ["path"],
                },
                permission=PermissionLevel.AUTO,
                is_read_only=True,
            ),
            remote_file_read,
        )

        async def remote_file_write(path: str, content: str) -> dict[str, Any]:
            escaped = content.replace("'", "'\\''")
            result = await backend.exec(
                instance,
                f"cat > {workdir}/{path} << 'ISAAC_EOF'\n{escaped}\nISAAC_EOF"
            )
            if result.exit_code != 0:
                return {"error": result.stderr}
            return {"written": path, "bytes": len(content)}

        registry["file_write"] = (
            ToolDef(
                name="file_write",
                description="Write a file on the remote VM.",
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
            remote_file_write,
        )

        # Remote file_list (list files on the VM, not locally)
        async def remote_file_list(path: str = ".", pattern: str = "*") -> dict[str, Any]:
            target = f"{workdir}/{path}" if path != "." else workdir
            result = await backend.exec(instance, f"ls -1 {target}/{pattern} 2>/dev/null")
            files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            return {"files": files[:100]}

        registry["file_list"] = (
            ToolDef(
                name="file_list",
                description="List files on the remote VM.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                        "pattern": {"type": "string", "default": "*"},
                    },
                },
                permission=PermissionLevel.AUTO,
                is_read_only=True,
            ),
            remote_file_list,
        )

        # Remote file_search (grep on the VM)
        async def remote_file_search(pattern: str, path: str = ".") -> dict[str, Any]:
            target = f"{workdir}/{path}" if path != "." else workdir
            result = await backend.exec(instance, f"grep -rl '{pattern}' {target} 2>/dev/null | head -20")
            matches = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            return {"matches": matches}

        registry["file_search"] = (
            ToolDef(
                name="file_search",
                description="Search file contents on the remote VM.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Search pattern"},
                        "path": {"type": "string", "default": "."},
                    },
                    "required": ["pattern"],
                },
                permission=PermissionLevel.AUTO,
                is_read_only=True,
            ),
            remote_file_search,
        )

        return registry

    def _build_agent_prompt(self, manifest: AppManifest, inputs: dict[str, Any]) -> str:
        """Build the initial prompt for the VM agent."""
        parts = []

        if manifest.agent_soul:
            parts.append(manifest.agent_soul.strip())

        parts.append(f"\n## App: {manifest.name}")
        parts.append(f"Repo: {manifest.repo}")

        if inputs:
            parts.append("\n## User Inputs:")
            for k, v in inputs.items():
                parts.append(f"- {k}: {v}")

        parts.append(f"\n## Constraints:")
        parts.append(f"- Timeout: {manifest.timeout}s")
        parts.append("- bash, file_read, file_write execute on the remote GPU VM")
        parts.append("- memory_*, web_search, and any MCP tools execute locally on the ISAAC host")
        parts.append("- You can search the web, read/write memory, and call connected apps from inside the VM")
        parts.append("- Save important findings to memory (persists after VM shutdown)")
        parts.append("- When done, provide a clear summary of results")

        return "\n".join(parts)

    async def _collect_artifacts(
        self,
        backend: ComputeBackend,
        instance: ComputeInstance,
        manifest: AppManifest,
    ) -> list[dict[str, str]]:
        """Download artifacts from the VM and save to memory."""
        collected: list[dict[str, str]] = []
        workdir = instance.metadata.get("workdir", "/home/user/app")

        for spec in manifest.artifacts:
            try:
                # Handle glob patterns
                if "*" in spec.path:
                    result = await backend.exec(instance, f"ls -1 {workdir}/{spec.path} 2>/dev/null")
                    paths = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
                else:
                    paths = [f"{workdir}/{spec.path}"]

                for remote_path in paths[:10]:  # cap at 10 files per pattern
                    try:
                        data = await backend.download(instance, remote_path)
                        filename = remote_path.split("/")[-1]
                        # Save to memory
                        content = data.decode(errors="replace")
                        if len(content) > 50_000:
                            content = content[:50_000] + "\n[... truncated ...]"
                        mem_path = f"apps/{manifest.name}/artifacts/{filename}"
                        self.memory.write(mem_path, content, {
                            "tags": ["artifact", manifest.name],
                            "importance": 0.8,
                        })
                        collected.append({
                            "path": filename,
                            "memory_path": mem_path,
                            "size": str(len(data)),
                            "description": spec.description,
                        })
                    except Exception:
                        continue
            except Exception:
                continue

        return collected

    async def _handle_state(
        self,
        backend: ComputeBackend,
        instance: ComputeInstance,
        manifest: AppManifest,
    ) -> str:
        """Handle post-run state management per manifest policy."""
        if manifest.state == "ephemeral":
            await backend.teardown(instance)
            return "destroyed"

        elif manifest.state == "checkpoint":
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            workdir = instance.metadata.get("workdir", "/home/user/app")
            # Tar the working directory
            await backend.exec(instance, f"cd {workdir} && tar czf /tmp/checkpoint.tar.gz .")
            try:
                data = await backend.download(instance, "/tmp/checkpoint.tar.gz")
                checkpoint_path = CHECKPOINTS_DIR / f"{manifest.name}_latest.tar.gz"
                checkpoint_path.write_bytes(data)
            except Exception:
                pass
            await backend.teardown(instance)
            return "checkpointed"

        elif manifest.state == "persistent":
            try:
                snapshot_id = await backend.snapshot(instance)
                # Store snapshot reference
                state_file = CHECKPOINTS_DIR / f"{manifest.name}_snapshot.json"
                state_file.parent.mkdir(parents=True, exist_ok=True)
                state_file.write_text(json.dumps({"snapshot_id": snapshot_id, "backend": backend.name}))
            except NotImplementedError:
                pass
            await backend.teardown(instance)
            return "snapshotted"

        await backend.teardown(instance)
        return "destroyed"

    async def _restore_checkpoint(
        self,
        backend: ComputeBackend,
        instance: ComputeInstance,
        manifest: AppManifest,
    ) -> None:
        """Restore a checkpoint if one exists."""
        checkpoint_path = CHECKPOINTS_DIR / f"{manifest.name}_latest.tar.gz"
        if not checkpoint_path.exists():
            return
        workdir = instance.metadata.get("workdir", "/home/user/app")
        try:
            await backend.upload(instance, str(checkpoint_path), "/tmp/checkpoint.tar.gz")
            await backend.exec(instance, f"cd {workdir} && tar xzf /tmp/checkpoint.tar.gz")
        except Exception:
            pass
