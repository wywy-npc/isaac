"""Compute backend abstraction — pluggable GPU provisioning.

Backends implement this interface to provide VM lifecycle management.
The AppRunner doesn't care whether it's E2B, Modal, or RunPod behind it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from isaac.apps.manifest import AppManifest


@dataclass
class ExecResult:
    """Result of executing a command on a compute instance."""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: str | None = None


@dataclass
class ComputeInstance:
    """Handle to a running compute instance."""
    id: str
    backend: str
    status: str = "provisioning"  # provisioning | ready | running | stopped
    metadata: dict[str, Any] = field(default_factory=dict)
    # Backend-specific handle (E2B Sandbox, Modal Sandbox, etc.)
    _handle: Any = None


class ComputeBackend(ABC):
    """Abstract compute backend. Implementations provide GPU VMs."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e2b, modal, runpod, etc.)."""
        ...

    @abstractmethod
    async def provision(self, manifest: AppManifest) -> ComputeInstance:
        """Provision a new compute instance matching the manifest's requirements."""
        ...

    @abstractmethod
    async def exec(self, instance: ComputeInstance, command: str, timeout: int = 300) -> ExecResult:
        """Execute a shell command on the instance."""
        ...

    @abstractmethod
    async def upload(self, instance: ComputeInstance, local_path: str, remote_path: str) -> None:
        """Upload a file to the instance."""
        ...

    @abstractmethod
    async def download(self, instance: ComputeInstance, remote_path: str) -> bytes:
        """Download a file from the instance."""
        ...

    @abstractmethod
    async def teardown(self, instance: ComputeInstance) -> None:
        """Destroy the instance and free resources."""
        ...

    async def snapshot(self, instance: ComputeInstance) -> str:
        """Snapshot the instance for later restoration. Returns snapshot ID."""
        raise NotImplementedError(f"{self.name} does not support snapshots")

    async def restore(self, snapshot_id: str, manifest: AppManifest) -> ComputeInstance:
        """Restore an instance from a snapshot."""
        raise NotImplementedError(f"{self.name} does not support restore")


def get_backend(name: str | None = None) -> ComputeBackend:
    """Get a compute backend by name. Auto-detects if name is None."""
    import os

    if name == "modal" or (name is None and os.environ.get("MODAL_TOKEN_ID")):
        from isaac.apps.backends.modal_backend import ModalBackend
        return ModalBackend()

    if name == "e2b" or (name is None and os.environ.get("E2B_API_KEY")):
        from isaac.apps.backends.e2b_backend import E2BBackend
        return E2BBackend()

    # Fallback: try E2B first, then Modal
    if os.environ.get("E2B_API_KEY"):
        from isaac.apps.backends.e2b_backend import E2BBackend
        return E2BBackend()

    if os.environ.get("MODAL_TOKEN_ID"):
        from isaac.apps.backends.modal_backend import ModalBackend
        return ModalBackend()

    raise RuntimeError(
        "No compute backend available. Set E2B_API_KEY or MODAL_TOKEN_ID."
    )
