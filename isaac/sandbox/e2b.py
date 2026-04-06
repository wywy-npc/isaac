"""E2B sandbox — cloud code execution in isolated environments.

Wraps e2b_code_interpreter for safe, sandboxed execution.
Requires E2B_API_KEY environment variable.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any


class E2BSandbox:
    """Thin wrapper around E2B code interpreter sandbox."""

    def __init__(self) -> None:
        self._sandbox = None

    def _get_sandbox(self) -> Any:
        """Lazy-init the sandbox instance (reused across calls in a session)."""
        if self._sandbox is None:
            from e2b_code_interpreter import Sandbox
            self._sandbox = Sandbox()
        return self._sandbox

    async def execute(self, code: str, language: str = "python") -> dict[str, Any]:
        """Execute code in the sandbox."""
        loop = asyncio.get_event_loop()
        sandbox = self._get_sandbox()

        def _run():
            result = sandbox.run_code(code, language=language)
            return {
                "stdout": "".join(r.text for r in result.logs.stdout),
                "stderr": "".join(r.text for r in result.logs.stderr),
                "results": [str(r) for r in result.results],
                "error": str(result.error) if result.error else None,
            }

        return await loop.run_in_executor(None, _run)

    async def upload(self, local_path: str) -> dict[str, Any]:
        """Upload a file to the sandbox."""
        loop = asyncio.get_event_loop()
        sandbox = self._get_sandbox()
        p = Path(local_path)
        if not p.exists():
            return {"error": f"File not found: {local_path}"}

        def _upload():
            with open(p, "rb") as f:
                remote_path = sandbox.files.write(p.name, f)
            return {"uploaded": p.name, "remote_path": str(remote_path)}

        return await loop.run_in_executor(None, _upload)

    async def download(self, remote_path: str) -> dict[str, Any]:
        """Download a file from the sandbox."""
        loop = asyncio.get_event_loop()
        sandbox = self._get_sandbox()

        def _download():
            content = sandbox.files.read(remote_path)
            return {"path": remote_path, "content": content.decode() if isinstance(content, bytes) else str(content)}

        return await loop.run_in_executor(None, _download)

    def close(self) -> None:
        if self._sandbox:
            try:
                self._sandbox.close()
            except Exception:
                pass
            self._sandbox = None


# Auto-register with sandbox registry on import
from isaac.sandbox.registry import register_sandbox
register_sandbox("e2b", E2BSandbox)
