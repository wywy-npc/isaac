"""Configuration loading — agents.yaml, souls, env."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from isaac.core.types import AgentConfig

ISAAC_HOME = Path(os.environ.get("ISAAC_HOME", Path.home() / ".isaac"))
SOULS_DIR = ISAAC_HOME / "souls"
TOOLS_DIR = ISAAC_HOME / "tools"
PLUGINS_DIR = TOOLS_DIR  # backward compat alias
SESSIONS_DIR = ISAAC_HOME / "sessions"
APPS_DIR = ISAAC_HOME / "apps"
CHECKPOINTS_DIR = ISAAC_HOME / "checkpoints"
SKILLS_DIR = ISAAC_HOME / "skills"
WIKIS_DIR = ISAAC_HOME / "wikis"
PERSONAL_DIR = ISAAC_HOME / "personal"
CONFIG_FILE = ISAAC_HOME / "agents.yaml"


def ensure_dirs() -> None:
    for d in (ISAAC_HOME, SOULS_DIR, TOOLS_DIR, SESSIONS_DIR, APPS_DIR, CHECKPOINTS_DIR, SKILLS_DIR, WIKIS_DIR, PERSONAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    _ensure_bundled_skills()
    _ensure_bundled_souls()


def _ensure_bundled_skills() -> None:
    """Copy bundled starter skills to SKILLS_DIR if they don't exist."""
    bundled_dir = Path(__file__).parent.parent / "data" / "skills"
    if not bundled_dir.exists():
        return
    for skill_file in bundled_dir.glob("*.md"):
        target = SKILLS_DIR / skill_file.name
        if not target.exists():
            target.write_text(skill_file.read_text())


def _ensure_bundled_souls() -> None:
    """Copy bundled starter souls to SOULS_DIR if they don't exist."""
    bundled_dir = Path(__file__).parent.parent / "data" / "souls"
    if not bundled_dir.exists():
        return
    for soul_file in bundled_dir.glob("*.md"):
        target = SOULS_DIR / soul_file.name
        if not target.exists():
            target.write_text(soul_file.read_text())


def load_agents_config(path: Path | None = None) -> dict[str, AgentConfig]:
    """Load agents.yaml → dict of AgentConfig."""
    path = path or CONFIG_FILE
    if not path.exists():
        return {"default": AgentConfig(name="default")}

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    agents: dict[str, AgentConfig] = {}
    for name, cfg in raw.get("agents", {}).items():
        agents[name] = AgentConfig(
            name=name,
            soul=cfg.get("soul", "default"),
            model=cfg.get("model", "claude-sonnet-4-6"),
            tools=cfg.get("tools", ["*"]),
            mcp_servers=cfg.get("mcp_servers", []),
            max_iterations=cfg.get("max_iterations", 25),
            context_budget=cfg.get("context_budget", 180_000),
            cwd=cfg.get("cwd"),
            auto_start=cfg.get("auto_start", False),
            expose_as_tool=cfg.get("expose_as_tool", False),
            tool_description=cfg.get("tool_description", ""),
            computer_use=cfg.get("computer_use", False),
            sandbox=cfg.get("sandbox", "fly"),
            sandbox_template=cfg.get("sandbox_template", "ubuntu-24"),
            sandbox_size=cfg.get("sandbox_size", ""),
            sandbox_disk_gb=cfg.get("sandbox_disk_gb", 0),
            scope=cfg.get("scope", "cwd"),
            computer_scope=cfg.get("computer_scope", False),
        )
    return agents or {"default": AgentConfig(name="default")}


def save_agent_config(name: str, config: AgentConfig) -> None:
    """Append or update an agent in agents.yaml. Preserves existing agents."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw: dict[str, Any] = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            raw = yaml.safe_load(f) or {}

    agents = raw.setdefault("agents", {})

    # Build the agent entry — only include non-default values to keep YAML clean
    entry: dict[str, Any] = {"soul": config.soul, "model": config.model}
    if config.tools != ["*"]:
        entry["tools"] = config.tools
    if config.expose_as_tool:
        entry["expose_as_tool"] = True
        if config.tool_description:
            entry["tool_description"] = config.tool_description
    if config.cwd:
        entry["cwd"] = config.cwd
    if config.auto_start:
        entry["auto_start"] = True
    if config.sandbox and config.sandbox != "fly":
        entry["sandbox"] = config.sandbox
    if config.sandbox_size:
        entry["sandbox_size"] = config.sandbox_size
    if config.computer_scope:
        entry["computer_scope"] = True

    agents[name] = entry

    with open(CONFIG_FILE, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)


def get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val
