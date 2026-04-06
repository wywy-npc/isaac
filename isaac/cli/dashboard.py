"""Cost dashboard — Rich tables for session and historical cost tracking."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from isaac.core.config import SESSIONS_DIR


def _load_session_headers(sessions_dir: Path | None = None) -> list[dict]:
    """Read the first line (JSON header) of every .jsonl session file."""
    d = sessions_dir or SESSIONS_DIR
    headers = []
    for f in d.glob("*.jsonl"):
        try:
            first_line = f.open().readline().strip()
            if first_line:
                headers.append(json.loads(first_line))
        except Exception:
            continue
    return headers


def render_session(total_tokens: int, total_cost: float, turn_count: int) -> Panel:
    """Render current session cost summary."""
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("metric", style="bold")
    t.add_column("value")
    t.add_row("Tokens", f"{total_tokens:,}")
    t.add_row("Cost", f"${total_cost:.4f}")
    t.add_row("Turns", str(turn_count))
    if turn_count > 0:
        t.add_row("Avg $/turn", f"${total_cost / turn_count:.4f}")
    return Panel(t, title="Session Cost", border_style="cyan")


def render_historical(sessions_dir: Path | None = None, agent_filter: str | None = None) -> Panel:
    """Render historical cost breakdown across all sessions."""
    headers = _load_session_headers(sessions_dir)

    if agent_filter:
        headers = [h for h in headers if h.get("agent_name") == agent_filter]

    if not headers:
        return Panel("[dim]No session data found[/dim]", title="Cost History")

    # Aggregate
    by_agent: dict[str, dict] = defaultdict(lambda: {"cost": 0.0, "tokens": 0, "turns": 0, "sessions": 0})
    total_cost = 0.0
    total_tokens = 0

    for h in headers:
        agent = h.get("agent_name", "unknown")
        cost = h.get("total_cost", 0.0)
        tokens = h.get("total_tokens", 0)
        turns = h.get("turn_count", 0)
        by_agent[agent]["cost"] += cost
        by_agent[agent]["tokens"] += tokens
        by_agent[agent]["turns"] += turns
        by_agent[agent]["sessions"] += 1
        total_cost += cost
        total_tokens += tokens

    t = Table(title="By Agent", box=None, padding=(0, 2))
    t.add_column("Agent", style="bold")
    t.add_column("Sessions", justify="right")
    t.add_column("Turns", justify="right")
    t.add_column("Tokens", justify="right")
    t.add_column("Cost", justify="right", style="green")

    for agent, data in sorted(by_agent.items()):
        t.add_row(
            agent,
            str(data["sessions"]),
            str(data["turns"]),
            f"{data['tokens']:,}",
            f"${data['cost']:.4f}",
        )

    t.add_section()
    t.add_row("TOTAL", str(len(headers)), "", f"{total_tokens:,}", f"${total_cost:.4f}")

    return Panel(t, title="Cost History", border_style="cyan")
