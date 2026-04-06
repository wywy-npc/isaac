"""Shared Rich UI components — banner, status, formatting.

Every visual element in the CLI goes through here.
"""
from __future__ import annotations

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# ── Dexter-inspired design tokens ──────────────────────────────
PROMPT_CHAR = "❯"
CIRCLE = "⏺"
TREE = "⎿  "
THINKING_VERBS = [
    "Thinking", "Reasoning", "Analyzing", "Pondering",
    "Working", "Processing", "Reflecting", "Considering",
]

# Agent role → color mapping
ROLE_COLORS = {
    "lead": "cyan",
    "research": "green",
    "ops": "yellow",
    "toolsmith": "magenta",
    "default": "blue",
}

# Model tier → color + label
MODEL_BADGES = {
    "claude-haiku-4-5-20251001": ("dim white", "HAIKU"),
    "claude-sonnet-4-6": ("bright_blue", "SONNET"),
    "claude-opus-4-6": ("bright_magenta", "OPUS"),
}

ISAAC_LOGO = (
    "[bold bright_cyan] ___  ____    _    ____[/bold bright_cyan]\n"
    "[bold cyan]|_ _|/ ___|  / \\  / ___|[/bold cyan]\n"
    "[cyan] | | \\___ \\ / _ \\| |[/cyan]\n"
    "[dim cyan] | |  ___) / ___ \\ |___[/dim cyan]\n"
    "[dim cyan]|___|____/_/   \\_\\____|[/dim cyan]"
)

ISAAC_LOGO_SMALL = "[bold cyan]ISAAC[/bold cyan]"


def banner(
    agent_name: str,
    model: str,
    tool_count: int = 0,
    session_id: str = "",
    services: int = 0,
) -> Panel:
    """Render the startup banner."""
    color = ROLE_COLORS.get(agent_name, "blue")
    model_color, model_label = MODEL_BADGES.get(model, ("white", model.split("-")[-1].upper()))

    # Compact info line beneath logo
    parts = [
        f"[bold {color}]{agent_name}[/bold {color}]",
        f"[{model_color}]{model_label}[/{model_color}]",
        f"[dim]{tool_count} tools[/dim]",
    ]
    if services > 0:
        parts.append(f"[dim]{services} services[/dim]")
    if session_id:
        parts.append(f"[dim]{session_id[:8]}[/dim]")
    info_line = "  ·  ".join(parts)

    content = f"{ISAAC_LOGO}\n\n{info_line}"

    return Panel(
        Text.from_markup(content),
        title="[bold bright_cyan]ISAAC[/bold bright_cyan] [dim]v0.2.0[/dim]",
        subtitle="[dim]/help · Ctrl+C to exit[/dim]",
        border_style="dim cyan",
        padding=(1, 2),
    )


def hatch_banner() -> Panel:
    """Render the hatch intro banner."""
    return Panel(
        Align.center(
            Text.from_markup(
                f"{ISAAC_LOGO}\n\n"
                "[bold]Hatching a new agent...[/bold]\n"
                "[dim]Let's get to know each other.[/dim]"
            )
        ),
        border_style="dim cyan",
        padding=(1, 2),
    )


def hatch_complete_banner(agent_name: str, user_name: str) -> Panel:
    """Render the hatch complete celebration."""
    return Panel(
        Align.center(
            f"[bold green]Hatch complete![/bold green]\n\n"
            f"[bold]{agent_name}[/bold] is ready.\n"
            f"Configured for [bold cyan]{user_name}[/bold cyan].\n\n"
            f"[dim]Run [bold]isaac chat[/bold] to start talking.[/dim]"
        ),
        border_style="green",
        padding=(1, 2),
    )


def status_panel(
    agent_name: str,
    model: str,
    tokens: int,
    cost: float,
    turns: int,
    tool_count: int,
) -> Panel:
    """Render a status dashboard panel."""
    color = ROLE_COLORS.get(agent_name, "blue")
    model_color, model_label = MODEL_BADGES.get(model, ("white", "?"))

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("key", style="dim")
    t.add_column("value")
    t.add_row("Agent", f"[bold {color}]{agent_name}[/bold {color}]")
    t.add_row("Model", f"[{model_color}]{model_label}[/{model_color}]")
    t.add_row("Tokens", f"{tokens:,}")
    t.add_row("Cost", f"${cost:.4f}")
    t.add_row("Turns", str(turns))
    t.add_row("Tools", str(tool_count))

    return Panel(t, title="Status", border_style=color)


def tool_table(tools: dict) -> Panel:
    """Render a styled tool list."""
    t = Table(box=None, padding=(0, 2), show_header=True, header_style="bold")
    t.add_column("Tool", style="bold")
    t.add_column("Permission", width=8)
    t.add_column("Description", style="dim")

    for name in sorted(tools.keys()):
        tdef, _ = tools[name]
        perm_color = {"auto": "green", "ask": "yellow", "deny": "red"}.get(tdef.permission.value, "white")
        desc = tdef.description[:60] if tdef.description else ""
        t.add_row(name, f"[{perm_color}]{tdef.permission.value}[/{perm_color}]", desc)

    return Panel(t, title=f"Tools ({len(tools)})", border_style="dim")


def model_badge(model: str) -> str:
    """Inline model badge for event rendering."""
    color, label = MODEL_BADGES.get(model, ("white", "?"))
    return f"[{color}]{label}[/{color}]"


def first_run_notice() -> Panel:
    """Show when user.md doesn't exist yet."""
    return Panel(
        "[bold yellow]First run detected![/bold yellow]\n\n"
        "Run [bold]isaac hatch[/bold] to set up your agent.\n"
        "This teaches ISAAC who you are and how you work.\n\n"
        "[dim]Or just start chatting — you can hatch later.[/dim]",
        border_style="yellow",
        padding=(0, 1),
    )
