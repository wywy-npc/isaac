"""Modal compute backend — serverless GPU containers.

Modal is ideal for ML workloads: automatic GPU scheduling, fast cold starts,
and built-in file persistence. Requires MODAL_TOKEN_ID + MODAL_TOKEN_SECRET.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from isaac.apps.compute import ComputeBackend, ComputeInstance, ExecResult
from isaac.apps.manifest import AppManifest

# GPU type mapping: manifest name -> Modal GPU spec
GPU_MAP = {
    "T4": "T4",
    "A10G": "A10G",
    "A100": "A100",
    "H100": "H100",
}


class ModalBackend(ComputeBackend):
    """Modal serverless GPU containers as compute backend."""

    @property
    def name(self) -> str:
        return "modal"

    async def provision(self, manifest: AppManifest) -> ComputeInstance:
        """Provision a Modal sandbox with GPU."""
        loop = asyncio.get_event_loop()

        def _create():
            import modal
            app = modal.App.lookup("isaac-apps", create_if_missing=True)
            gpu_spec = GPU_MAP.get(manifest.gpu_type, "T4") if manifest.gpu else None

            sandbox = modal.Sandbox.create(
                app=app,
                timeout=manifest.timeout,
                gpu=gpu_spec,
                image=modal.Image.debian_slim(python_version="3.12")
                    .apt_install("git", "curl")
                    .pip_install("uv"),
            )
            return sandbox

        sandbox = await loop.run_in_executor(None, _create)
        instance = ComputeInstance(
            id=f"modal-{uuid.uuid4().hex[:8]}",
            backend="modal",
            status="ready",
            _handle=sandbox,
        )

        # Clone repo
        if manifest.repo:
            await self.exec(instance, f"git clone --depth 1 -b {manifest.version} {manifest.repo} /root/app")
            instance.metadata["workdir"] = "/root/app"

        # Run setup
        if manifest.setup:
            result = await self.exec(instance, f"cd /root/app && {manifest.setup}", timeout=600)
            if result.exit_code != 0:
                instance.metadata["setup_error"] = result.stderr

        instance.status = "running"
        return instance

    async def exec(self, instance: ComputeInstance, command: str, timeout: int = 300) -> ExecResult:
        """Execute a command in the Modal sandbox."""
        loop = asyncio.get_event_loop()
        sandbox = instance._handle

        def _exec():
            process = sandbox.exec("bash", "-c", command)
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            exit_code = process.returncode
            return ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code or 0)

        try:
            return await asyncio.wait_for(loop.run_in_executor(None, _exec), timeout=timeout + 30)
        except asyncio.TimeoutError:
            return ExecResult(error=f"Command timed out after {timeout}s", exit_code=1)

    async def upload(self, instance: ComputeInstance, local_path: str, remote_path: str) -> None:
        """Upload a file via exec + base64 (Modal doesn't have direct file API in sandbox)."""
        import base64
        with open(local_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        await self.exec(instance, f"echo '{data}' | base64 -d > {remote_path}")

    async def download(self, instance: ComputeInstance, remote_path: str) -> bytes:
        """Download a file via exec + base64."""
        import base64
        result = await self.exec(instance, f"base64 {remote_path}")
        if result.error or result.exit_code != 0:
            raise RuntimeError(f"Download failed: {result.stderr or result.error}")
        return base64.b64decode(result.stdout.strip())

    async def teardown(self, instance: ComputeInstance) -> None:
        loop = asyncio.get_event_loop()
        sandbox = instance._handle
        if sandbox:
            await loop.run_in_executor(None, sandbox.terminate)
        instance.status = "stopped"
        instance._handle = None
