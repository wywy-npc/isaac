"""Fly.io Machine sandbox — real VMs with suspend/resume.

Uses the Fly Machines API to create, start, stop, and exec commands on
per-agent VMs. Each agent gets a persistent machine with its own disk.

Requires: FLY_API_TOKEN env var and a Fly.io app (created via `fly apps create`).

Machine lifecycle:
  create → start (ACTIVE) → [multi-turn usage] → stop (SLEEPING) → start (ACTIVE) → ...
  At any point: destroy (permanently remove)

Suspend preserves the full disk image. Resume takes 2-3 seconds.
Cost: ~$0.03/hr active, ~$0.01/day sleeping (disk storage only).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx

from isaac.sandbox.base import ExecResult, Sandbox, SandboxInfo, SandboxState

log = logging.getLogger(__name__)

FLY_API_BASE = "https://api.machines.dev/v1"

# Default machine config
DEFAULT_IMAGE = "ubuntu:24.04"
DEFAULT_SIZE = "shared-cpu-1x"  # 1 shared vCPU, 256MB RAM — cheapest
DEFAULT_DISK_GB = 10
DEFAULT_REGION = "ord"  # Chicago — low latency to most US locations

# Machine size presets — agent picks the right size for the task
MACHINE_SIZES = {
    # CPU tiers
    "shared-cpu-1x":      {"cpu": 1, "ram": 256,   "gpu": None, "cost_hr": 0.003},
    "shared-cpu-2x":      {"cpu": 2, "ram": 512,   "gpu": None, "cost_hr": 0.006},
    "shared-cpu-4x":      {"cpu": 4, "ram": 1024,  "gpu": None, "cost_hr": 0.012},
    "shared-cpu-8x":      {"cpu": 8, "ram": 2048,  "gpu": None, "cost_hr": 0.024},
    "performance-1x":     {"cpu": 1, "ram": 2048,  "gpu": None, "cost_hr": 0.031},
    "performance-2x":     {"cpu": 2, "ram": 4096,  "gpu": None, "cost_hr": 0.062},
    "performance-4x":     {"cpu": 4, "ram": 8192,  "gpu": None, "cost_hr": 0.124},
    "performance-8x":     {"cpu": 8, "ram": 16384, "gpu": None, "cost_hr": 0.248},
    "performance-16x":    {"cpu": 16, "ram": 32768, "gpu": None, "cost_hr": 0.496},
    # GPU tiers
    "a10":                {"cpu": 8, "ram": 16384, "gpu": "a10",   "cost_hr": 1.50},
    "l40s":               {"cpu": 8, "ram": 32768, "gpu": "l40s",  "cost_hr": 2.50},
    "a100-40gb":          {"cpu": 12, "ram": 65536, "gpu": "a100-40gb", "cost_hr": 3.50},
    "a100-80gb":          {"cpu": 12, "ram": 131072, "gpu": "a100-80gb", "cost_hr": 5.00},
}


class FlySandbox(Sandbox):
    """Fly.io Machine-backed sandbox."""

    def __init__(self, app_name: str | None = None, token: str | None = None) -> None:
        self.app = app_name or os.environ.get("FLY_APP", "isaac-sandbox")
        self.token = token or os.environ.get("FLY_API_TOKEN", "")
        if not self.token:
            log.warning("FLY_API_TOKEN not set — sandbox operations will fail")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{FLY_API_BASE}/apps/{self.app}{path}"

    async def create(
        self, agent_name: str, template: str = "ubuntu-24",
        size: str = "", disk_gb: int = 0, region: str = "",
    ) -> SandboxInfo:
        """Create a new Fly Machine for an agent.

        Args:
            agent_name: Agent this sandbox belongs to
            template: Base image template (ubuntu-24, ubuntu-24-python, cuda-ubuntu-24)
            size: Machine size preset (see MACHINE_SIZES). Supports GPU tiers:
                  "a10" (24GB VRAM, $1.50/hr), "l40s" (48GB, $2.50/hr),
                  "a100-40gb" ($3.50/hr), "a100-80gb" ($5.00/hr)
            disk_gb: Disk size in GB (default 10, GPU machines may want 50+)
            region: Fly region (default: ord). GPU regions: ord, sjc, iad
        """
        machine_size = size or DEFAULT_SIZE
        machine_disk = disk_gb or DEFAULT_DISK_GB
        machine_region = region or DEFAULT_REGION

        # Image selection
        image = DEFAULT_IMAGE
        if template == "ubuntu-24-python":
            image = "python:3.12-slim"
        elif template == "cuda-ubuntu-24":
            image = "nvidia/cuda:12.4.0-devel-ubuntu24.04"
            if not size:
                machine_size = "a10"  # Default GPU if using CUDA template
            if not disk_gb:
                machine_disk = 50  # GPU workloads need more disk

        # Validate size
        size_info = MACHINE_SIZES.get(machine_size)
        if not size_info:
            log.warning(f"Unknown size '{machine_size}', falling back to {DEFAULT_SIZE}")
            machine_size = DEFAULT_SIZE
            size_info = MACHINE_SIZES[DEFAULT_SIZE]

        if size_info.get("gpu"):
            log.info(f"Creating GPU sandbox: {machine_size} ({size_info['gpu']}, ~${size_info['cost_hr']}/hr)")

        async with httpx.AsyncClient(timeout=30) as client:
            # Create or find volume — we need the volume ID for the mount
            vol_name = f"isaac_{agent_name}_vol"
            vol_id = ""

            # Check for existing volume first
            try:
                vol_resp = await client.get(
                    f"{FLY_API_BASE}/apps/{self.app}/volumes",
                    headers=self._headers(),
                )
                vol_resp.raise_for_status()
                for vol in vol_resp.json():
                    if vol.get("name") == vol_name:
                        vol_id = vol["id"]
                        break
            except Exception:
                pass

            # Create volume if it doesn't exist
            if not vol_id:
                vol_create_resp = await client.post(
                    f"{FLY_API_BASE}/apps/{self.app}/volumes",
                    headers=self._headers(),
                    json={
                        "name": vol_name,
                        "size_gb": machine_disk,
                        "region": machine_region,
                    },
                )
                vol_create_resp.raise_for_status()
                vol_id = vol_create_resp.json()["id"]

            machine_config: dict[str, Any] = {
                "image": image,
                "size": machine_size,
                "init": {
                    "cmd": ["tail", "-f", "/dev/null"],  # Keep container alive for exec
                },
                "env": {
                    "ISAAC_AGENT": agent_name,
                    "DEBIAN_FRONTEND": "noninteractive",
                },
                "mounts": [{
                    "volume": vol_id,
                    "path": "/workspace",
                }],
                "services": [],
                "auto_destroy": False,
                "restart": {"policy": "no"},
            }

            # GPU machines need the gpu_kind field
            if size_info.get("gpu"):
                machine_config["guest"] = {
                    "cpus": size_info["cpu"],
                    "memory_mb": size_info["ram"],
                    "gpu_kind": size_info["gpu"],
                }

            config = {
                "name": f"isaac-{agent_name}",
                "config": machine_config,
                "region": machine_region,
            }

            resp = await client.post(
                self._url("/machines"),
                headers=self._headers(),
                json=config,
            )
            resp.raise_for_status()
            data = resp.json()

        return SandboxInfo(
            sandbox_id=data["id"],
            agent_name=agent_name,
            state=SandboxState.STARTING,
            ip=data.get("private_ip", ""),
            region=data.get("region", DEFAULT_REGION),
            created_at=time.time(),
            disk_gb=DEFAULT_DISK_GB,
            meta={"fly_machine_id": data["id"]},
        )

    async def start(self, sandbox_id: str) -> SandboxInfo:
        """Start/resume a stopped or sleeping machine."""
        # Wait for machine to leave "created" state before starting
        for _ in range(10):
            info = await self.info(sandbox_id)
            if info.state == SandboxState.ACTIVE:
                return info  # Already running
            if info.meta.get("fly_state") != "created":
                break
            await asyncio.sleep(1)

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._url(f"/machines/{sandbox_id}/start"),
                headers=self._headers(),
            )
            resp.raise_for_status()

        # Wait for machine to be ready
        for _ in range(30):
            info = await self.info(sandbox_id)
            if info.state == SandboxState.ACTIVE:
                return info
            await asyncio.sleep(1)

        return await self.info(sandbox_id)

    async def stop(self, sandbox_id: str) -> None:
        """Suspend a machine (preserves disk, stops billing for compute)."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._url(f"/machines/{sandbox_id}/stop"),
                headers=self._headers(),
            )
            resp.raise_for_status()
        log.info(f"Sandbox {sandbox_id} suspended")

    async def destroy(self, sandbox_id: str) -> None:
        """Permanently destroy a machine and its data."""
        async with httpx.AsyncClient() as client:
            # Force stop first
            try:
                await client.post(
                    self._url(f"/machines/{sandbox_id}/stop"),
                    headers=self._headers(),
                )
                await asyncio.sleep(2)
            except Exception:
                pass

            resp = await client.delete(
                self._url(f"/machines/{sandbox_id}"),
                headers=self._headers(),
                params={"force": "true"},
            )
            resp.raise_for_status()
        log.info(f"Sandbox {sandbox_id} destroyed")

    async def exec(self, sandbox_id: str, command: str, timeout: int = 30) -> ExecResult:
        """Execute a command in the machine via the Fly exec API."""
        start = time.monotonic()

        async with httpx.AsyncClient(timeout=timeout + 5) as client:
            resp = await client.post(
                self._url(f"/machines/{sandbox_id}/exec"),
                headers=self._headers(),
                json={
                    "cmd": "bash -c " + json.dumps(command),
                    "timeout": timeout,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        duration = int((time.monotonic() - start) * 1000)

        stdout = data.get("stdout", "")
        stderr = data.get("stderr", "")
        exit_code = data.get("exit_code", -1)

        # Truncate large outputs
        if len(stdout) > 50_000:
            stdout = stdout[:25_000] + "\n\n[... truncated ...]\n\n" + stdout[-25_000:]
        if len(stderr) > 10_000:
            stderr = stderr[:10_000]

        return ExecResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration,
        )

    async def write_file(self, sandbox_id: str, path: str, content: str) -> None:
        """Write a file to the machine via exec (base64 encoded)."""
        import base64
        encoded = base64.b64encode(content.encode()).decode()
        # Ensure parent dir exists, then write
        parent = "/".join(path.split("/")[:-1])
        cmd = f"mkdir -p {parent} && echo '{encoded}' | base64 -d > {path}"
        result = await self.exec(sandbox_id, cmd, timeout=10)
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to write file {path}: {result.stderr}")

    async def read_file(self, sandbox_id: str, path: str) -> str:
        """Read a file from the machine via exec."""
        result = await self.exec(sandbox_id, f"cat {path}", timeout=10)
        if result.exit_code != 0:
            raise FileNotFoundError(f"File not found on sandbox: {path}")
        return result.stdout

    async def info(self, sandbox_id: str) -> SandboxInfo:
        """Get current machine state."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                self._url(f"/machines/{sandbox_id}"),
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        fly_state = data.get("state", "unknown")
        state_map = {
            "started": SandboxState.ACTIVE,
            "stopped": SandboxState.SLEEPING,
            "starting": SandboxState.STARTING,
            "stopping": SandboxState.SLEEPING,
            "created": SandboxState.STOPPED,
            "destroyed": SandboxState.STOPPED,
        }

        return SandboxInfo(
            sandbox_id=sandbox_id,
            agent_name=data.get("name", "").replace("isaac-", ""),
            state=state_map.get(fly_state, SandboxState.ERROR),
            ip=data.get("private_ip", ""),
            region=data.get("region", ""),
            meta={"fly_state": fly_state},
        )

    async def list_all(self) -> list[SandboxInfo]:
        """List all ISAAC machines in the Fly app."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                self._url("/machines"),
                headers=self._headers(),
            )
            resp.raise_for_status()
            machines = resp.json()

        results: list[SandboxInfo] = []
        for m in machines:
            if m.get("name", "").startswith("isaac-"):
                info = SandboxInfo(
                    sandbox_id=m["id"],
                    agent_name=m["name"].replace("isaac-", ""),
                    state=SandboxState.ACTIVE if m.get("state") == "started" else SandboxState.SLEEPING,
                    ip=m.get("private_ip", ""),
                    region=m.get("region", ""),
                )
                results.append(info)

        return results


# Auto-register with sandbox registry on import
from isaac.sandbox.registry import register_sandbox
register_sandbox("fly", FlySandbox)
