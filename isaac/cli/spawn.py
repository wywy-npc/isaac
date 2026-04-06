"""Spawn — interactive agent creation.

Opens a conversation that defines a new agent's personality, tools, and goals,
then registers it in agents.yaml with a persistent VM.

Usage:
    isaac new              # interactive spawn
    isaac new --model M    # override model for the spawn conversation
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel

from isaac.core.config import ISAAC_HOME, SOULS_DIR, ensure_dirs, save_agent_config
from isaac.core.types import AgentConfig

console = Console()

# Available built-in tools the user can choose from
AVAILABLE_TOOLS = [
    "memory_search", "memory_read", "memory_write",
    "file_read", "file_write", "file_list", "file_search",
    "bash", "web_search", "delegate_agent",
]

SPAWN_SOUL = """# Spawn Mode

You are ISAAC, spawning a new agent. Your job: learn what this agent should be
through natural conversation, then create it.

## What You Need to Learn
1. **Purpose** — What does this agent do? What's its specialty?
2. **Name** — Short, kebab-case identifier (e.g. "competitor-intel", "code-review")
3. **Tools** — Which tools should it have? Available: {tools}
   - ["*"] means all tools. Only restrict if the user wants a focused agent.
4. **Personality** — How should it behave? Direct? Thorough? Creative?
5. **Goals** — Any standing instructions or ongoing objectives?

## How to Do It
- Be conversational. Ask naturally, not as a checklist.
- Suggest reasonable defaults. Most agents want ["*"] for tools.
- If the user is vague, make smart suggestions based on the purpose.
- Keep it to 2-4 exchanges, not a long interview.

## When You Have Enough
Do TWO things:

1. Call `file_write` to create the soul file at `{souls_dir}/{{agent_name}}.md` with:
```markdown
# {{Agent Name}}

## Identity
{{Who this agent is and what it does}}

## Behavior
{{How it should act — communication style, approach}}

## Goals
{{Standing instructions, ongoing objectives}}

## Rules
{{Any constraints or guidelines}}
```

2. Call `memory_write` to save a summary to `agents/{{agent_name}}.md` with tags
   ["agent", "spawn", "{{agent_name}}"] and importance 0.9

3. End your final message with a JSON block (the system will parse it):
```json
{{{{
  "name": "agent-name",
  "soul": "agent-name",
  "tools": ["*"],
  "model": "claude-sonnet-4-6",
  "description": "One-line description",
  "expose_as_tool": false
}}}}
```

## Tone
Efficient and collaborative. You're configuring a teammate, not filling out a form.
""".format(
    tools=", ".join(AVAILABLE_TOOLS),
    souls_dir=str(SOULS_DIR),
)


def _extract_agent_json(text: str) -> dict | None:
    """Extract the JSON config block from the agent's final message."""
    # Look for JSON block in markdown code fence or bare
    patterns = [
        r"```json\s*\n(.*?)\n```",
        r"```\s*\n(\{.*?\})\n```",
        r"(\{[^{}]*\"name\"[^{}]*\})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    return None


def run_spawn(model: str = "claude-sonnet-4-6") -> str | None:
    """Run the interactive spawn flow. Returns the new agent name, or None if cancelled."""
    ensure_dirs()

    console.print()
    console.print(Panel(
        "[bold]Spawning a new ISAAC agent[/bold]\n\n"
        "[dim]Describe what you need this agent to do.\n"
        "I'll set up its personality, tools, and VM.[/dim]",
        border_style="cyan",
        title="[cyan]isaac new[/cyan]",
        title_align="left",
        padding=(0, 1),
    ))
    console.print()

    from isaac.cli.terminal import TerminalREPL

    config = AgentConfig(
        name="spawn",
        soul="default",
        model=model,
        sandbox="",  # Spawn conversation runs locally — no VM needed
    )
    config._hatch_soul = SPAWN_SOUL

    repl = TerminalREPL(config)
    repl._hatch_mode = True

    # Run the spawn conversation
    try:
        asyncio.run(repl.run())
    except (KeyboardInterrupt, EOFError):
        console.print("\n  [yellow]Spawn cancelled.[/yellow]")
        return None

    # Parse the agent config from the conversation
    if not repl.state or not repl.state.messages:
        return None

    # Find the last assistant message with JSON config
    agent_data = None
    for msg in reversed(repl.state.messages):
        if msg.role == "assistant" and msg.content:
            agent_data = _extract_agent_json(msg.content)
            if agent_data:
                break

    if not agent_data or "name" not in agent_data:
        console.print("  [yellow]Could not parse agent config from conversation.[/yellow]")
        console.print("  [dim]Try again with `isaac new`[/dim]")
        return None

    # Register the agent
    name = agent_data["name"]
    agent_config = AgentConfig(
        name=name,
        soul=agent_data.get("soul", name),
        model=agent_data.get("model", "claude-sonnet-4-6"),
        tools=agent_data.get("tools", ["*"]),
        expose_as_tool=agent_data.get("expose_as_tool", False),
        tool_description=agent_data.get("description", ""),
    )

    save_agent_config(name, agent_config)

    console.print()
    console.print(Panel(
        f"[bold green]Agent '{name}' created[/bold green]\n\n"
        f"  Soul: ~/.isaac/souls/{agent_data.get('soul', name)}.md\n"
        f"  Tools: {agent_data.get('tools', ['*'])}\n"
        f"  Model: {agent_data.get('model', 'claude-sonnet-4-6')}\n"
        f"  Registered in agents.yaml\n\n"
        f"[dim]VM will be provisioned on first run.\n"
        f"Starting agent session...[/dim]",
        border_style="green",
        title="[green]Agent Ready[/green]",
        title_align="left",
        padding=(0, 1),
    ))

    return name
