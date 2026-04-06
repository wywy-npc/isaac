"""Hatch — first-run onboarding.

Three steps:
  1. API key setup (required to talk to Claude)
  2. Model selection (Haiku/Sonnet/Opus)
  3. Natural conversation to learn about the user

Usage:
    isaac hatch              # full onboarding
    isaac hatch --reset      # re-run from scratch
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from isaac.cli.ui import hatch_banner
from isaac.core.config import ISAAC_HOME, SOULS_DIR, ensure_dirs

console = Console()

USER_FILE = ISAAC_HOME / "user.md"
ENV_FILE = ISAAC_HOME / ".env"

MODELS = [
    ("claude-sonnet-4-6", "Sonnet", "$3/$15 per 1M tokens", "Best balance of speed and quality (recommended)"),
    ("claude-opus-4-6", "Opus", "$15/$75 per 1M tokens", "Most capable, best for complex reasoning"),
    ("claude-haiku-4-5-20251001", "Haiku", "$0.80/$4 per 1M tokens", "Fastest and cheapest"),
]

HATCH_SOUL = """# Hatch Mode

You are ISAAC, hatching for the first time. This is your onboarding conversation.

## Your Goal
Learn about your user through natural conversation. You need to understand:
- Their name
- What they do (role, company, domain)
- How they like to communicate (direct/detailed/casual)
- What they're working on right now
- Any other context (timezone, interests, tools, preferences)

## How to Do It
- Be conversational, not robotic. Don't ask a numbered list of questions.
- Start with a warm intro — you're meeting for the first time.
- Let the conversation flow naturally. Ask follow-ups based on what they say.
- Be genuinely curious. If they mention something interesting, dig in.
- Keep it brief — this should take 1-2 minutes, not 10.

## When You Have Enough
Once you have a good picture of the user, do TWO things:

1. Call `file_write` to write `{user_file}` with this format:
```
# User Profile
name: <their name>
role: <what they do>
communication: <how they like to communicate>
working_on: <current projects/focus>
context: <timezone, interests, anything else>
hatched: <current timestamp>
```

2. Call `memory_write` to save the profile to `user/profile.md` with tags ["user", "profile", "hatch"] and importance 1.0

Then tell them you're all set and they can start using `isaac chat`.

## Tone
Warm but not cheesy. You're a sharp AI meeting your new operator for the first time.
""".format(user_file=str(USER_FILE))


def run_hatch(reset: bool = False) -> None:
    """Run the full hatch: API key → model → conversation."""
    import asyncio
    ensure_dirs()

    if USER_FILE.exists() and not reset:
        console.print()
        console.print("[yellow]Already hatched![/yellow] Your profile is at:")
        console.print(f"  {USER_FILE}")
        console.print()
        console.print("[dim]To re-hatch: [bold]isaac hatch --reset[/bold][/dim]")
        return

    if reset and USER_FILE.exists():
        USER_FILE.unlink()

    console.print()
    console.print(hatch_banner())
    console.print()

    session = PromptSession()

    # --- Step 1: API Key ---
    api_key = _setup_api_key(session)
    if not api_key:
        return

    # --- Step 2: Model Selection ---
    model = _setup_model(session)

    # --- Step 3: Save config ---
    _save_env(api_key)
    console.print()
    console.print("  [green]Config saved.[/green]")
    console.print()

    # --- Step 4: Natural conversation ---
    console.print(Panel(
        "[bold]Now let's get to know each other.[/bold]\n"
        "[dim]Just chat naturally — I'll learn what I need to know.[/dim]",
        border_style="cyan",
        padding=(0, 1),
    ))
    console.print()

    from isaac.cli.terminal import TerminalREPL
    from isaac.core.types import AgentConfig

    config = AgentConfig(name="isaac", soul="default", model=model)
    config._hatch_soul = HATCH_SOUL

    repl = TerminalREPL(config)
    repl._hatch_mode = True
    asyncio.run(repl.run())


def _setup_api_key(session: PromptSession) -> str | None:
    """Prompt for API key if not already set."""
    existing = os.environ.get("ANTHROPIC_API_KEY", "")

    # Check .env file too
    if not existing and ENV_FILE.exists():
        for line in ENV_FILE.read_text().split("\n"):
            if line.startswith("ANTHROPIC_API_KEY=") and not line.endswith("="):
                existing = line.split("=", 1)[1].strip()
                break

    if existing and existing.startswith("sk-ant-"):
        masked = existing[:10] + "..." + existing[-4:]
        console.print(f"  [green]API key found:[/green] [dim]{masked}[/dim]")
        console.print()

        try:
            answer = session.prompt("  Keep this key? [Y/n] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if answer not in ("n", "no"):
            return existing

    console.print("  [bold]Anthropic API Key[/bold]")
    console.print("  [dim]Get one at: https://console.anthropic.com/settings/keys[/dim]")
    console.print()

    try:
        key = session.prompt("  Paste your key > ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n  [yellow]Hatch cancelled.[/yellow]")
        return None

    if not key:
        console.print("  [red]No key provided. Can't proceed without an API key.[/red]")
        return None

    # Set it for the current process so the conversation works
    os.environ["ANTHROPIC_API_KEY"] = key
    return key


def _setup_model(session: PromptSession) -> str:
    """Interactive model selection."""
    console.print()
    console.print("  [bold]Default Model[/bold]")
    console.print("  [dim]The router will auto-switch per query, but this sets the ceiling.[/dim]")
    console.print()

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("num", style="bold cyan", width=4)
    t.add_column("name", style="bold", width=10)
    t.add_column("price", style="dim", width=24)
    t.add_column("desc")

    for i, (model_id, name, price, desc) in enumerate(MODELS, 1):
        rec = " (recommended)" if i == 1 else ""
        t.add_row(f"  {i}.", name, price, f"{desc}{rec}")

    console.print(t)
    console.print()

    try:
        choice = session.prompt("  Pick [1/2/3] > ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    idx = int(choice) - 1 if choice in ("1", "2", "3") else 0
    model_id, name, _, _ = MODELS[idx]

    console.print(f"  [green]Selected:[/green] [bold]{name}[/bold] ({model_id})")
    return model_id


def _save_env(api_key: str) -> None:
    """Save API key and any other config to ~/.isaac/.env."""
    lines: dict[str, str] = {}

    # Load existing .env
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().split("\n"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                lines[k.strip()] = v.strip()

    lines["ANTHROPIC_API_KEY"] = api_key

    # Write back
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    content = "# ISAAC Environment\n"
    for k, v in lines.items():
        content += f"{k}={v}\n"
    ENV_FILE.write_text(content)


def is_hatched() -> bool:
    """Check if the user has completed the hatch."""
    return USER_FILE.exists()


def load_user_context() -> str:
    """Load user.md content for soul injection."""
    if not USER_FILE.exists():
        return ""
    return USER_FILE.read_text()
