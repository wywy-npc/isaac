"""E2B compute backend — cloud sandboxes with optional GPU support."""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from isaac.apps.compute import ComputeBackend, ComputeInstance, ExecResult
from isaac.apps.manifest import AppManifest


class E2BBackend(ComputeBackend):
    """E2B Code Interpreter sandbox as compute backend."""

    @property
    def name(self) -> str:
        return "e2b"

    async def provision(self, manifest: AppManifest) -> ComputeInstance:
        """Provision an E2B sandbox."""
        loop = asyncio.get_event_loop()

        def _create():
            from e2b_code_interpreter import Sandbox
            kwargs: dict[str, Any] = {
                "timeout": manifest.timeout,
            }
            # E2B GPU support — specify template if GPU needed
            if manifest.gpu:
                kwargs["template"] = "gpu"
            sandbox = Sandbox(**kwargs)
            return sandbox

        sandbox = await loop.run_in_executor(None, _create)
        instance = ComputeInstance(
            id=f"e2b-{uuid.uuid4().hex[:8]}",
            backend="e2b",
            status="ready",
            _handle=sandbox,
        )

        # Clone repo if specified
        if manifest.repo:
            await self.exec(instance, f"git clone --depth 1 -b {manifest.version} {manifest.repo} /home/user/app")
            await self.exec(instance, "cd /home/user/app")
            instance.metadata["workdir"] = "/home/user/app"

        # Run setup commands
        if manifest.setup:
            result = await self.exec(instance, f"cd /home/user/app && {manifest.setup}", timeout=600)
            if result.exit_code != 0:
                instance.metadata["setup_error"] = result.stderr

        instance.status = "running"
        return instance

    async def exec(self, instance: ComputeInstance, command: str, timeout: int = 300) -> ExecResult:
        """Execute a command in the E2B sandbox."""
        loop = asyncio.get_event_loop()
        sandbox = instance._handle

        def _exec():
            result = sandbox.run_code(f"import subprocess; r = subprocess.run({command!r}, shell=True, capture_output=True, text=True, timeout={timeout}); print(r.stdout); import sys; print(r.stderr, file=sys.stderr); exit(r.returncode)")
            stdout = "".join(r.text for r in result.logs.stdout)
            stderr = "".join(r.text for r in result.logs.stderr)
            error = str(result.error) if result.error else None
            return ExecResult(stdout=stdout, stderr=stderr, exit_code=0 if not error else 1, error=error)

        try:
            return await asyncio.wait_for(loop.run_in_executor(None, _exec), timeout=timeout + 30)
        except asyncio.TimeoutError:
            return ExecResult(error=f"Command timed out after {timeout}s", exit_code=1)

    async def upload(self, instance: ComputeInstance, local_path: str, remote_path: str) -> None:
        loop = asyncio.get_event_loop()
        sandbox = instance._handle

        def _upload():
            with open(local_path, "rb") as f:
                sandbox.files.write(remote_path, f)

        await loop.run_in_executor(None, _upload)

    async def download(self, instance: ComputeInstance, remote_path: str) -> bytes:
        loop = asyncio.get_event_loop()
        sandbox = instance._handle

        def _download():
            content = sandbox.files.read(remote_path)
            return content if isinstance(content, bytes) else content.encode()

        return await loop.run_in_executor(None, _download)

    async def teardown(self, instance: ComputeInstance) -> None:
        loop = asyncio.get_event_loop()
        sandbox = instance._handle
        if sandbox:
            await loop.run_in_executor(None, sandbox.close)
        instance.status = "stopped"
        instance._handle = None
