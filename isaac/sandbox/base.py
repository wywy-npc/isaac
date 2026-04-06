"""Sandbox base — abstract interface for VM lifecycle and command execution."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SandboxState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    ACTIVE = "active"
    SLEEPING = "sleeping"
    WAKING = "waking"
    ERROR = "error"


@dataclass
class SandboxInfo:
    """Current state of a sandbox VM."""
    sandbox_id: str
    agent_name: str
    state: SandboxState
    ip: str = ""
    region: str = ""
    created_at: float = 0.0
    disk_gb: int = 10
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecResult:
    """Result of a command executed in the sandbox."""
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int = 0


class Sandbox(ABC):
    """Abstract sandbox — one VM per agent."""

    @abstractmethod
    async def create(
        self, agent_name: str, template: str = "ubuntu-24",
        size: str = "", disk_gb: int = 0, region: str = "",
    ) -> SandboxInfo:
        """Create a new sandbox VM for an agent.

        size: Machine preset. GPU tiers: "a10", "l40s", "a100-40gb", "a100-80gb"
        """

    @abstractmethod
    async def start(self, sandbox_id: str) -> SandboxInfo:
        """Start/resume a sleeping sandbox."""

    @abstractmethod
    async def stop(self, sandbox_id: str) -> None:
        """Suspend a sandbox (preserves disk)."""

    @abstractmethod
    async def destroy(self, sandbox_id: str) -> None:
        """Permanently destroy a sandbox and its disk."""

    @abstractmethod
    async def exec(self, sandbox_id: str, command: str, timeout: int = 30) -> ExecResult:
        """Execute a shell command in the sandbox."""

    @abstractmethod
    async def write_file(self, sandbox_id: str, path: str, content: str) -> None:
        """Write a file to the sandbox filesystem."""

    @abstractmethod
    async def read_file(self, sandbox_id: str, path: str) -> str:
        """Read a file from the sandbox filesystem."""

    @abstractmethod
    async def info(self, sandbox_id: str) -> SandboxInfo:
        """Get current sandbox state."""

    @abstractmethod
    async def list_all(self) -> list[SandboxInfo]:
        """List all sandboxes."""
