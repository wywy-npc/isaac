"""
Sandbox Lifecycle Manager — handles VM boot, checkpoint, suspend, resume.

The user never touches this. The lifecycle manager:
1. On session start: checks if agent has an existing VM → resume, or create new
2. During session: keeps VM alive, routes tools through bridge
3. On session end: agent writes checkpoint, VM suspends (not kills)
4. On next session: VM resumes, agent reads checkpoint, picks up where it left off

The checkpoint is a memory node with machine state — NOT a VM snapshot.
The VM disk survives suspend natively. The checkpoint just gives the agent
cognitive context about what's on the machine.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import isaac.core.config as _cfg
from isaac.core.heartbeat import read_continuation, write_continuation
from isaac.sandbox.base import Sandbox, SandboxInfo, SandboxState

log = logging.getLogger(__name__)

# Sandbox registry — maps agent names to sandbox IDs across sessions
_REGISTRY_FILE = _cfg.ISAAC_HOME / "sandbox_registry.yaml"


@dataclass
class SandboxSession:
    """A live sandbox session for an agent."""
    info: SandboxInfo
    agent_name: str
    started_at: float
    is_new: bool = False  # True if freshly created (not resumed)


def _load_registry() -> dict[str, str]:
    """Load agent_name → sandbox_id mapping."""
    import yaml
    if not _REGISTRY_FILE.exists():
        return {}
    with open(_REGISTRY_FILE) as f:
        data = yaml.safe_load(f) or {}
    return data.get("sandboxes", {})


def _save_registry(registry: dict[str, str]) -> None:
    """Save agent_name → sandbox_id mapping."""
    import yaml
    _REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_REGISTRY_FILE, "w") as f:
        yaml.dump({"sandboxes": registry}, f)


async def ensure_sandbox(
    sandbox: Sandbox,
    agent_name: str,
    template: str = "ubuntu-24",
    size: str = "",
    disk_gb: int = 0,
) -> SandboxSession:
    """Ensure a sandbox VM is running for the agent.

    - If agent has an existing VM (in registry) → resume it
    - If no existing VM → create a new one
    - Supports GPU tiers: size="a10", "l40s", "a100-40gb", "a100-80gb"
    - Returns a SandboxSession ready for use
    """
    registry = _load_registry()
    existing_id = registry.get(agent_name)

    if existing_id:
        try:
            info = await sandbox.info(existing_id)
            if info.state == SandboxState.SLEEPING:
                log.info(f"Resuming sandbox for {agent_name}: {existing_id}")
                info = await sandbox.start(existing_id)
                return SandboxSession(
                    info=info,
                    agent_name=agent_name,
                    started_at=time.time(),
                    is_new=False,
                )
            elif info.state == SandboxState.ACTIVE:
                log.info(f"Sandbox already active for {agent_name}: {existing_id}")
                return SandboxSession(
                    info=info,
                    agent_name=agent_name,
                    started_at=time.time(),
                    is_new=False,
                )
        except Exception as e:
            log.warning(f"Existing sandbox {existing_id} not reachable: {e}")
            # Fall through to create new

    # Create new sandbox
    log.info(f"Creating new sandbox for {agent_name} (template: {template})")
    info = await sandbox.create(agent_name, template, size=size, disk_gb=disk_gb)
    info = await sandbox.start(info.sandbox_id)

    # Bootstrap — install registered app dependencies on new VMs
    try:
        from isaac.sandbox.bootstrap import run_bootstrap
        log.info(f"Bootstrapping new sandbox for {agent_name}")
        bootstrap_result = await run_bootstrap(sandbox, info.sandbox_id)
        if bootstrap_result["exit_code"] != 0:
            log.warning(f"Bootstrap had errors (exit {bootstrap_result['exit_code']})")
    except Exception as e:
        log.warning(f"Bootstrap failed: {e}")
        # Non-fatal — agent can still use the VM, just without pre-installed apps

    # Register it
    registry[agent_name] = info.sandbox_id
    _save_registry(registry)

    return SandboxSession(
        info=info,
        agent_name=agent_name,
        started_at=time.time(),
        is_new=True,
    )


async def suspend_sandbox(
    sandbox: Sandbox,
    session: SandboxSession,
    machine_state_summary: str = "",
) -> None:
    """Suspend a sandbox VM after writing a checkpoint.

    The agent should have already written its continuation via write_continuation.
    This function handles the VM-level suspend and writes a machine state checkpoint.
    """
    agent = session.agent_name

    # Write machine state checkpoint if provided
    if machine_state_summary:
        from isaac.memory.store import MemoryStore
        store = MemoryStore()
        store.write(
            f"machines/{agent}.md",
            machine_state_summary,
            {
                "type": "machine_state",
                "agent": agent,
                "sandbox_id": session.info.sandbox_id,
                "importance": 0.85,
                "tags": ["machine_state", "sandbox", agent],
            },
        )

    # Suspend the VM
    await sandbox.stop(session.info.sandbox_id)
    log.info(
        f"Sandbox {agent} suspended (id: {session.info.sandbox_id}, "
        f"active for {time.time() - session.started_at:.0f}s)"
    )


async def checkpoint_and_suspend(
    sandbox: Sandbox,
    session: SandboxSession,
) -> str:
    """Auto-generate a machine state checkpoint, then suspend.

    Runs commands in the VM to inspect what's installed, what's running,
    and what's on disk. Writes this as a memory node, then suspends.

    Returns the checkpoint content.
    """
    sid = session.info.sandbox_id

    # Gather machine state
    parts: list[str] = [f"## Machine State: {session.agent_name}"]
    parts.append(f"- Sandbox ID: {sid}")
    parts.append(f"- Session duration: {time.time() - session.started_at:.0f}s")

    # What's installed
    try:
        result = await sandbox.exec(sid, "dpkg --get-selections 2>/dev/null | wc -l", timeout=5)
        parts.append(f"- System packages: {result.stdout.strip()}")
    except Exception:
        pass

    try:
        result = await sandbox.exec(sid, "pip list --format=freeze 2>/dev/null | wc -l", timeout=5)
        parts.append(f"- Python packages: {result.stdout.strip()}")
    except Exception:
        pass

    # What's in /workspace
    try:
        result = await sandbox.exec(sid, "find /workspace -maxdepth 2 -type f | head -30", timeout=5)
        if result.stdout.strip():
            parts.append(f"\n### Workspace files")
            for f in result.stdout.strip().split("\n"):
                parts.append(f"- {f}")
    except Exception:
        pass

    # Running processes
    try:
        result = await sandbox.exec(sid, "ps aux --no-headers | grep -v 'ps aux' | head -10", timeout=5)
        if result.stdout.strip():
            parts.append(f"\n### Running processes")
            parts.append(f"```\n{result.stdout.strip()}\n```")
    except Exception:
        pass

    # Git repos
    try:
        result = await sandbox.exec(
            sid, "find /workspace -name .git -maxdepth 3 -type d 2>/dev/null", timeout=5
        )
        if result.stdout.strip():
            parts.append(f"\n### Git repos")
            for repo in result.stdout.strip().split("\n"):
                repo_dir = repo.replace("/.git", "")
                parts.append(f"- {repo_dir}")
    except Exception:
        pass

    checkpoint = "\n".join(parts)

    # Write and suspend
    await suspend_sandbox(sandbox, session, checkpoint)

    return checkpoint


async def destroy_sandbox(sandbox: Sandbox, agent_name: str) -> None:
    """Permanently destroy an agent's sandbox."""
    registry = _load_registry()
    sandbox_id = registry.get(agent_name)
    if not sandbox_id:
        log.warning(f"No sandbox registered for {agent_name}")
        return

    await sandbox.destroy(sandbox_id)
    del registry[agent_name]
    _save_registry(registry)
    log.info(f"Sandbox for {agent_name} destroyed")
