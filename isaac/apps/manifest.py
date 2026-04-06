"""App manifest — YAML-driven app definitions that become callable tools.

Each manifest in ~/.isaac/apps/ declares a repo, compute needs, setup,
execution mode, inputs/outputs, and state policy. The AppRunner reads
these to provision and orchestrate external apps as tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from isaac.core.config import ISAAC_HOME

APPS_DIR = ISAAC_HOME / "apps"
CHECKPOINTS_DIR = ISAAC_HOME / "checkpoints"


@dataclass
class ArtifactSpec:
    path: str
    description: str = ""


@dataclass
class InputSpec:
    type: str = "string"
    required: bool = False
    default: Any = None
    description: str = ""


@dataclass
class AppManifest:
    """Everything needed to run an external app as a tool."""
    name: str
    description: str = ""
    repo: str = ""
    version: str = "main"

    # Compute
    gpu: bool = False
    gpu_type: str = "T4"
    memory_gb: int = 16
    timeout: int = 3600  # seconds

    # Setup
    setup: str = ""

    # Execution
    mode: str = "command"  # "command" or "agent"
    run: str = ""  # for mode: command
    agent_soul: str = ""  # for mode: agent
    agent_tools: list[str] = field(default_factory=lambda: ["bash", "file_read", "file_write"])

    # I/O
    inputs: dict[str, InputSpec] = field(default_factory=dict)
    artifacts: list[ArtifactSpec] = field(default_factory=list)

    # State
    state: str = "ephemeral"  # ephemeral | checkpoint | persistent


def load_manifest(name: str) -> AppManifest | None:
    """Load a manifest by name from ~/.isaac/apps/."""
    path = APPS_DIR / f"{name}.yaml"
    if not path.exists():
        return None
    return _parse_manifest(path)


def list_manifests() -> list[dict[str, str]]:
    """List all available app manifests."""
    APPS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for f in APPS_DIR.glob("*.yaml"):
        try:
            m = _parse_manifest(f)
            results.append({
                "name": m.name,
                "description": m.description,
                "mode": m.mode,
                "gpu": str(m.gpu),
                "gpu_type": m.gpu_type if m.gpu else "-",
                "state": m.state,
            })
        except Exception:
            continue
    return results


def _parse_manifest(path: Path) -> AppManifest:
    """Parse a YAML manifest file into an AppManifest."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Parse inputs
    inputs: dict[str, InputSpec] = {}
    for k, v in raw.get("inputs", {}).items():
        if isinstance(v, dict):
            inputs[k] = InputSpec(
                type=v.get("type", "string"),
                required=v.get("required", False),
                default=v.get("default"),
                description=v.get("description", ""),
            )
        else:
            inputs[k] = InputSpec(type="string", description=str(v))

    # Parse artifacts
    artifacts: list[ArtifactSpec] = []
    for a in raw.get("artifacts", []):
        if isinstance(a, dict):
            artifacts.append(ArtifactSpec(path=a.get("path", ""), description=a.get("description", "")))
        elif isinstance(a, str):
            artifacts.append(ArtifactSpec(path=a))

    return AppManifest(
        name=raw.get("name", path.stem),
        description=raw.get("description", ""),
        repo=raw.get("repo", ""),
        version=raw.get("version", "main"),
        gpu=raw.get("gpu", False),
        gpu_type=raw.get("gpu_type", "T4"),
        memory_gb=raw.get("memory_gb", 16),
        timeout=raw.get("timeout", 3600),
        setup=raw.get("setup", ""),
        mode=raw.get("mode", "command"),
        run=raw.get("run", ""),
        agent_soul=raw.get("agent_soul", ""),
        agent_tools=raw.get("agent_tools", ["bash", "file_read", "file_write"]),
        inputs=inputs,
        artifacts=artifacts,
        state=raw.get("state", "ephemeral"),
    )
