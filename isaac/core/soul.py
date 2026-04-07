"""
Soul/personality system — runtime-aware, multi-layer resolution.

Layers (all in prompt Layer 1, cached after turn 1):
1. Platform soul — base identity and behavior
2. Architecture reference — static system knowledge (ARCHITECTURE.md)
3. Runtime context — live system state (agent identity, OS, active plugins,
   other agents, workspace files, model info)
4. Soul file — role-specific personality (e.g. research.md, toolsmith.md)
5. Agent overlay — per-agent overrides (agent-{name}.md)

Layers 1-2 never change (cached permanently).
Layer 3 changes when the system state changes (plugins added, agents changed).
Layers 4-5 change when soul files are edited.

Why runtime context matters (learned from OpenClaw):
- A Toolsmith on Linux needs to know it's on Linux, not macOS
- An agent needs to know what plugins are ACTUALLY loaded right now
- An agent needs to know what other agents exist and what they do
- An agent working in a project directory needs that project's conventions
- Subagents need minimal context, not the full architecture doc
"""
from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

import isaac.core.config as _cfg

_SOURCE_ROOT = Path(__file__).resolve().parent.parent.parent

PLATFORM_SOUL = """# Soul

## Who You Are
You are Isaac — a thoughtful, curious companion who happens to have powerful tools.
You're not a task executor that occasionally chats. You're a thinking partner who
genuinely engages with the person you're working with. You have perspective. You have
preferences. You remember things. You care about doing good work together.

## Your Voice
- Warm but sharp. You're not saccharine — you're the colleague who's genuinely
  interested in what you're building together and isn't afraid to push back.
- You think out loud when it's useful. You don't just execute — you share your
  reasoning, your hunches, your "wait, actually..." moments.
- Match the energy. If someone's excited about an idea, engage with that excitement.
  If they're frustrated, acknowledge it before jumping to solutions.
- You're direct without being terse. "Keep responses concise" doesn't mean stripped
  of personality — it means don't ramble. Say what matters, with life in it.

## How You Work
- You're proactive. When you see something that needs doing, do it. When you notice
  a pattern, mention it. When you have a better idea, say so.
- You persist. You remember conversations, learn preferences, and build on shared
  context. You're not starting from zero each time.
- You use tools naturally — searching, reading, writing, building — the way a person
  would reach for a reference or open a terminal. Tools serve your thinking, not
  the other way around.
- When a task requires multiple steps, think through the approach, then execute.
  Write findings to memory so they persist.

## Your Boundaries
- Never fabricate data. If unsure, search or say so honestly.
- Respect permission levels. Never bypass approval gates.
- If you lack a capability, say so — another agent (Toolsmith) can build it.
"""


def _load_agent_guide() -> str:
    """Load AGENT.md — what the agent can do and how to act."""
    for candidate in [
        _SOURCE_ROOT / "AGENT.md",
        _cfg.ISAAC_HOME / "AGENT.md",
    ]:
        if candidate.exists():
            return candidate.read_text()
    return ""


def _build_runtime_context(
    agent_name: str,
    agent_config: Any = None,
    active_tools: list[str] | None = None,
    mode: str = "full",
    connector_registry: Any = None,
) -> str:
    """Build live runtime context — what's true RIGHT NOW about this system.

    This is what OpenClaw does with their 6-file resolution system.
    We do it in one function that assembles actual system state.

    mode:
        "full" — complete context (for primary agents)
        "minimal" — just identity + workspace (for subagents/delegated tasks)
    """
    parts: list[str] = []

    # --- Agent identity ---
    parts.append(f"## Runtime")
    parts.append(f"- Agent: {agent_name}")
    parts.append(f"- Platform: {platform.system()} {platform.machine()}")
    parts.append(f"- Python: {platform.python_version()}")

    if agent_config:
        parts.append(f"- Model: {agent_config.model}")
        parts.append(f"- Max iterations: {agent_config.max_iterations}")
        parts.append(f"- Context budget: {agent_config.context_budget:,} tokens")
        if agent_config.cwd:
            parts.append(f"- Working directory: {agent_config.cwd}")

    cwd = agent_config.cwd if agent_config and agent_config.cwd else os.getcwd()
    parts.append(f"- CWD: {cwd}")
    parts.append(f"- ISAAC_HOME: {_cfg.ISAAC_HOME}")

    # --- User context (from hatch + personal memory) ---
    try:
        from isaac.cli.hatch import load_user_context
        user_ctx = load_user_context()
        if user_ctx:
            parts.append(f"\n## Who You're Working With")
            parts.append("This is your person. You know them. Use what you know to be helpful")
            parts.append("in the way *they* need — not generically.")
            for line in user_ctx.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and ":" in line:
                    parts.append(f"- {line}")

            # Pull in recent personal facts for richer relationship context
            try:
                from isaac.personal.store import get_personal_store
                store = get_personal_store()
                recent_facts = store.search("user preferences context", max_results=5)
                if recent_facts:
                    parts.append("")
                    parts.append("Things you've learned about them:")
                    for node in recent_facts:
                        # Extract the heading line as the fact
                        first_line = node.content.strip().split("\n")[0].lstrip("# ").strip()
                        if first_line:
                            parts.append(f"- {first_line}")
            except Exception:
                pass
    except Exception:
        pass

    if mode == "minimal":
        return "\n".join(parts)

    # === 3-LAYER TAXONOMY ===
    # Skills (HOW — workflows) → Tools (WHAT — actions) → Connectors (WHERE — data + bundled tools)

    # --- Layer 1: Skills (prompt workflows) ---
    try:
        from isaac.core.skills import load_skills
        skills = load_skills(_cfg.SKILLS_DIR)
        if skills:
            parts.append(f"\n## Skills ({len(skills)})")
            parts.append("Skills are reusable workflows. Use `use_skill(name, params)` to activate.")
            for name, skill in skills.items():
                invocable = " [user: /{name}]" if skill.user_invocable else ""
                parts.append(f"- **{name}**: {skill.description}{invocable}")
                if skill.params:
                    param_str = ", ".join(f"{k}" for k in skill.params)
                    parts.append(f"  Params: {param_str}")
    except Exception:
        pass

    # --- Layer 2: Tools (by source) ---
    if active_tools:
        # Group tools by source for clarity
        built_in = sorted([t for t in active_tools if not t.startswith("mcp__")])
        connector_tools = sorted([t for t in active_tools if t.startswith("mcp__")])

        parts.append(f"\n## Tools ({len(active_tools)})")
        parts.append("Tools are actions. Built-in + plugins + apps + connector-provided.")
        if built_in:
            parts.append(f"Built-in/Plugin/App: {', '.join(built_in[:30])}")
            if len(built_in) > 30:
                parts.append(f"  ...and {len(built_in) - 30} more")
        if connector_tools:
            parts.append(f"From connectors: {', '.join(connector_tools[:20])}")
            if len(connector_tools) > 20:
                parts.append(f"  ...and {len(connector_tools) - 20} more")

    # --- Layer 3: Connectors (data sources + their tools) ---
    if connector_registry:
        try:
            states = connector_registry.get_status()
            if states:
                connected = sum(1 for s in states if s.status == "connected")
                failed = sum(1 for s in states if s.status == "failed")
                label = f"{connected} connected"
                if failed:
                    label += f", {failed} failed"
                parts.append(f"\n## Connectors ({label})")
                parts.append("Connectors provide data and bring their own tools. Use connector_status for details.")
                for s in states:
                    if s.status == "connected":
                        tool_preview = ", ".join(s.tools[:5])
                        more = f" +{len(s.tools) - 5} more" if len(s.tools) > 5 else ""
                        parts.append(f"- **{s.name}** [connected]: {tool_preview}{more}")
                    elif s.status == "failed":
                        parts.append(f"- **{s.name}** [FAILED]: {s.error[:80]}")
        except Exception:
            pass
    else:
        # Fallback for subagents without registry
        try:
            from isaac.mcp.connections import load_connections
            connections = load_connections()
            enabled = {k: v for k, v in connections.items() if v.enabled}
            if enabled:
                parts.append(f"\n## Connectors ({len(enabled)})")
                for name, conn in enabled.items():
                    desc = conn.description if hasattr(conn, "description") and conn.description else ""
                    parts.append(f"- **{name}**{': ' + desc if desc else ''}")
        except Exception:
            pass

    # --- Other agents in the constellation ---
    try:
        agents = _cfg.load_agents_config()
        if len(agents) > 1:
            parts.append(f"\n## Agent Constellation")
            for name, cfg in agents.items():
                marker = " ← you" if name == agent_name else ""
                expose = " [exposed as tool]" if cfg.expose_as_tool else ""
                parts.append(f"- **{name}**: soul={cfg.soul}, model={cfg.model}{expose}{marker}")
                if cfg.tool_description:
                    parts.append(f"  _{cfg.tool_description}_")
    except Exception:
        pass

    # --- Plugins on disk ---
    try:
        tool_files = sorted(_cfg.TOOLS_DIR.glob("*.py"))
        plugin_names = [f.stem for f in tool_files if not f.name.startswith("_")]
        if plugin_names:
            parts.append(f"\n## Plugins on Disk")
            for t in plugin_names:
                parts.append(f"- {t}")
    except Exception:
        pass

    # --- Workspace context (project README, CLAUDE.md, etc.) ---
    workspace_context = _load_workspace_context(cwd)
    if workspace_context:
        parts.append(f"\n## Workspace Context")
        parts.append(workspace_context)

    return "\n".join(parts)


def _load_workspace_context(cwd: str) -> str:
    """Load project-level context files from the working directory.

    Checks for CLAUDE.md, README.md, .agents/ conventions — same idea
    as OpenClaw's bootstrap context but without the 150K char budget
    machinery. We cap at 2000 chars to keep the static layer lean.
    """
    candidates = ["CLAUDE.md", ".claude/AGENTS.md", "AGENTS.md"]
    cwd_path = Path(cwd)

    for name in candidates:
        path = cwd_path / name
        if path.exists():
            content = path.read_text()
            if len(content) > 2000:
                content = content[:2000] + "\n\n[... truncated]"
            return f"From `{name}`:\n{content}"

    return ""


def load_soul(
    soul_name: str,
    agent_name: str = "",
    agent_config: Any = None,
    active_tools: list[str] | None = None,
    mode: str = "full",
    connector_registry: Any = None,
) -> str:
    """Multi-layer soul resolution with runtime awareness.

    Args:
        soul_name: which soul file to load (e.g. "default", "research")
        agent_name: the agent's name in the constellation
        agent_config: AgentConfig for runtime identity
        active_tools: list of tool names currently available
        mode: "full" for primary agents, "minimal" for subagents
        connector_registry: ConnectorRegistry for showing connector health + tools

    Returns:
        Complete soul text ready for system prompt injection.
    """
    layers: list[str] = [PLATFORM_SOUL]

    # Layer 2: Agent guide (what you can do, how to act)
    if mode == "full":
        guide = _load_agent_guide()
        if guide:
            layers.append(guide)

    # Layer 3: Runtime context (live system state)
    runtime = _build_runtime_context(agent_name, agent_config, active_tools, mode, connector_registry)
    if runtime:
        layers.append(runtime)

    # Layer 4: Role-specific soul
    soul_file = _cfg.SOULS_DIR / f"{soul_name}.md"
    if soul_file.exists():
        layers.append(soul_file.read_text())

    # Layer 5: Per-agent overlay
    agent_soul = _cfg.SOULS_DIR / f"agent-{agent_name}.md"
    if agent_name and agent_soul.exists():
        layers.append(agent_soul.read_text())

    return "\n\n---\n\n".join(layers)


def save_soul(name: str, content: str) -> Path:
    _cfg.SOULS_DIR.mkdir(parents=True, exist_ok=True)
    path = _cfg.SOULS_DIR / f"{name}.md"
    path.write_text(content)
    return path
