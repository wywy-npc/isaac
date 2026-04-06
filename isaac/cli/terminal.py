"""Terminal REPL — rich-powered interactive agent session.

Now a thin consumer of HarnessCore. Assembly logic lives in builder.py.
"""
from __future__ import annotations

import asyncio
import random
import sys
import time as _time

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from isaac.agents.session import list_sessions, load_session, new_session, save_session
from isaac.core.builder import HarnessBuilder
from isaac.core.config import ISAAC_HOME, ensure_dirs, load_agents_config
from isaac.agents.delegation import DelegationEvent
from isaac.core.orchestrator import (
    ApprovalEvent,
    CompactEvent,
    CostEvent,
    ErrorEvent,
    Event,
    ModelRouteEvent,
    Orchestrator,
    ProgressEvent,
    TextDeltaEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from isaac.core.permissions import PermissionGate
from isaac.core.types import AgentConfig, PermissionLevel, SessionState, ToolCall
from isaac.memory.store import MemoryStore

console = Console()


class TerminalREPL:
    """Interactive terminal session for a single agent.

    Now uses HarnessBuilder for assembly. This class is just:
    1. Input handling (prompt_toolkit)
    2. Event rendering (Rich)
    3. Slash commands
    """

    def __init__(self, agent_config: AgentConfig, session_id: str | None = None) -> None:
        self.config = agent_config
        self.state: SessionState | None = None
        self._session_id = session_id
        self._hatch_mode = False
        self._harness = None
        self._memory = None
        self._pending_skill_prompt: str | None = None
        self._status_hint: str = ""

    def _bottom_toolbar(self) -> HTML:
        """Dexter-style hint bar — contextual info at the bottom of the terminal."""
        left_parts = [
            f'<style bg="ansibrightblack" fg="ansiwhite"> {self.config.name} </style>',
        ]
        if self.state:
            left_parts.append(
                f'<style fg="ansigray"> {self.state.turn_count} turns · ${self.state.total_cost:.4f} </style>'
            )
        if self._status_hint:
            left_parts.append(
                f'<style fg="ansicyan"> {self._status_hint} </style>'
            )

        right = '<style fg="ansigray"> / commands · ctrl+c exit </style>'
        left = " ".join(left_parts)
        return HTML(f'{left}  {right}')

    async def _approval_handler(self, tool_call: ToolCall) -> bool:
        """Ask user for tool approval in terminal."""
        console.print()
        console.print(Panel(
            f"[bold]{tool_call.name}[/bold]\n{_format_input(tool_call.input)}",
            title="[yellow]Tool Approval Required[/yellow]",
            border_style="yellow",
        ))
        try:
            session = PromptSession()
            answer = await asyncio.get_event_loop().run_in_executor(
                None, lambda: session.prompt("  Allow? [Y/n/always] > ")
            )
            answer = answer.strip().lower()
            if answer in ("always", "a"):
                if self._harness:
                    self._harness.config.permission_gate.session_allow(tool_call.name)
                return True
            if answer in ("n", "no"):
                return False
            return True
        except (EOFError, KeyboardInterrupt):
            return False

    async def run(self) -> None:
        """Main REPL loop."""
        from isaac.cli.ui import banner, first_run_notice
        from isaac.cli.hatch import is_hatched

        ensure_dirs()

        # Load or create session
        if self._session_id:
            self.state = load_session(self.config.name, self._session_id)
        if not self.state:
            self.state = new_session(self.config.name)

        # --- Build harness via builder (replaces 90+ lines of manual assembly) ---
        hatch_soul = getattr(self.config, "_hatch_soul", "") if self._hatch_mode else ""
        builder = HarnessBuilder(self.config.name)
        builder.with_config(self.config)
        builder.with_approval(self._approval_handler)
        if hatch_soul:
            builder.with_soul(override=hatch_soul)

        try:
            self._harness = await builder.build()
        except Exception as e:
            console.print(f"[red]Failed to build harness: {e}[/red]")
            return

        # Count tools and services for banner
        tool_count = len(self._harness.config.tool_registry)
        service_count = 0
        try:
            from isaac.mcp.connections import load_connections
            service_count = sum(1 for c in load_connections().values() if c.enabled)
        except Exception:
            pass

        # --- Startup banner ---
        console.print()
        console.print(banner(
            agent_name=self.config.name,
            model=self.config.model,
            tool_count=tool_count,
            session_id=self.state.session_id,
            services=service_count,
        ))

        if self._session_id and self.state.turn_count > 0:
            console.print(f"  [dim]Resumed session ({self.state.turn_count} turns, ${self.state.total_cost:.4f})[/dim]")

        if not is_hatched() and not self._hatch_mode:
            console.print()
            console.print(first_run_notice())

        # Prompt session with history
        history_file = ISAAC_HOME / "history" / f"{self.config.name}.txt"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_session = PromptSession(history=FileHistory(str(history_file)))

        console.print()

        while True:
            try:
                self._status_hint = ""
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: prompt_session.prompt("❯ ", bottom_toolbar=self._bottom_toolbar),
                )
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Saving session...[/dim]")
                save_session(self.state)

                # Checkpoint and suspend sandbox VM
                if self._harness and self._harness.config.sandbox_session:
                    try:
                        from isaac.sandbox.lifecycle import checkpoint_and_suspend
                        console.print("  [dim]Checkpointing and suspending sandbox...[/dim]")
                        await checkpoint_and_suspend(
                            self._harness.config.sandbox_backend,
                            self._harness.config.sandbox_session,
                        )
                        console.print("  [green]Sandbox suspended[/green]")
                    except Exception as e:
                        console.print(f"  [yellow]Sandbox suspend failed: {e}[/yellow]")

                # Close MCP connector connections
                if self._harness and self._harness.config.connector_registry:
                    try:
                        await self._harness.config.connector_registry.close_all()
                    except Exception:
                        pass

                break
            except (asyncio.CancelledError, Exception) as e:
                if "CancelledError" in type(e).__name__ or "cancel" in str(e).lower():
                    continue  # MCP server died — ignore and keep REPL alive
                raise

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle slash commands
            self._pending_skill_prompt = None
            if user_input.startswith("/"):
                if self._handle_command(user_input):
                    continue
                # If a skill was matched, _pending_skill_prompt has the rendered prompt
                if self._pending_skill_prompt:
                    user_input = self._pending_skill_prompt
                    self._pending_skill_prompt = None

            # Overwrite prompt_toolkit's plain echo with styled version
            display_text = user_input if len(user_input) < 500 else user_input[:200] + "...[skill prompt]"
            sys.stdout.write("\033[A\033[2K")
            sys.stdout.flush()
            console.print(Text.from_markup(f"[white on grey23] ❯ {display_text} [/white on grey23]"))

            # --- Mutable state for the live display ---
            turn_state = _TurnState()

            # --- Live renderable ---
            live_view = _LiveTurnView(turn_state)

            with Live(live_view, console=console, refresh_per_second=10, transient=True) as live:
                try:
                    async for event in self._harness.run(user_input, self.state):
                        if isinstance(event, ThinkingEvent):
                            turn_state.is_thinking = event.active
                            if event.active:
                                turn_state.think_start = _time.time()

                        elif isinstance(event, TextDeltaEvent):
                            turn_state.is_thinking = False
                            turn_state.text_chunks.append(event.delta)

                        elif isinstance(event, TextEvent):
                            turn_state.is_thinking = False
                            if event.text and not turn_state.text_chunks:
                                turn_state.text_chunks.append(event.text)

                        elif isinstance(event, ToolCallEvent):
                            turn_state.is_thinking = False
                            turn_state.active_tool = event.tool_call.name
                            turn_state.tool_start = _time.time()

                        elif isinstance(event, ToolResultEvent):
                            elapsed = _time.time() - turn_state.tool_start if turn_state.tool_start else 0
                            if event.result.is_error:
                                turn_state.tool_log.append(("error", event.tool_name, elapsed))
                            else:
                                turn_state.tool_log.append(("ok", event.tool_name, elapsed))
                            turn_state.active_tool = ""

                        elif isinstance(event, CostEvent):
                            elapsed = _time.time() - (turn_state.think_start or _time.time())
                            cache = ""
                            if event.cache_read > 0:
                                pct = event.cache_read * 100 // max(event.input_tokens, 1)
                                cache = f" cache:{pct}%"
                            turn_state.cost_info = (
                                f"{event.input_tokens + event.output_tokens:,}tok "
                                f"${event.cost:.4f} {elapsed:.1f}s{cache}"
                            )

                        elif isinstance(event, ProgressEvent):
                            tools_str = ", ".join(event.tools_used[-5:]) if event.tools_used else "thinking"
                            turn_state.tool_log.append((
                                "info",
                                f"⏱ {event.elapsed_seconds}s | step {event.iteration} | ${event.cost:.4f} | {tools_str}",
                                0,
                            ))

                        elif isinstance(event, DelegationEvent):
                            style = "info"
                            if event.event_type == "error":
                                style = "error"
                            elif event.event_type == "done":
                                style = "ok"
                            turn_state.tool_log.append((style, event.detail, 0))

                        elif isinstance(event, CompactEvent):
                            turn_state.tool_log.append(("warn", "compacted", 0))

                        elif isinstance(event, ErrorEvent):
                            turn_state.is_thinking = False
                            turn_state.active_tool = ""
                            turn_state.errors.append(event.error)

                        elif isinstance(event, ApprovalEvent):
                            live.stop()
                            approved = await self._approval_handler(event.tool_call)
                            live.start()
                            if not approved:
                                turn_state.tool_log.append(("deny", event.tool_call.name, 0))

                except (KeyboardInterrupt, asyncio.CancelledError):
                    pass

            # --- Final response (Dexter-style: blue circle + markdown) ---
            if turn_state.text_chunks:
                full_text = "".join(turn_state.text_chunks)
                if full_text.strip():
                    console.print()
                    console.print(Text.from_markup(f"[bright_blue]⏺[/bright_blue]"))
                    console.print(Markdown(full_text))

            # --- Analytics (Dexter-style: circles + tree connectors) ---
            for status, name, elapsed in turn_state.tool_log:
                if status == "ok":
                    console.print(Text.from_markup(
                        f"  [bright_blue]⏺[/bright_blue] {_title_case_tool(name)}"
                    ))
                    console.print(Text.from_markup(
                        f"  [dim]⎿  {elapsed:.1f}s[/dim]"
                    ))
                elif status == "error":
                    console.print(Text.from_markup(
                        f"  [bold red]⏺[/bold red] {_title_case_tool(name)}"
                    ))
                    console.print(Text.from_markup(
                        f"  [dim]⎿  [/dim][red]Error[/red]"
                    ))
                elif status == "warn":
                    console.print(Text.from_markup(
                        f"  [yellow]⏺[/yellow] {name}"
                    ))
                elif status == "deny":
                    console.print(Text.from_markup(
                        f"  [yellow]⏺[/yellow] {_title_case_tool(name)}"
                    ))
                    console.print(Text.from_markup(
                        f"  [dim]⎿  [/dim][yellow]Denied[/yellow]"
                    ))
            if turn_state.cost_info:
                console.print(Text.from_markup(f"  [dim]⎿  {turn_state.cost_info}[/dim]"))

            for err in turn_state.errors:
                console.print(Text.from_markup(f"  [bold red]⏺[/bold red] {err}"))

            console.print()

            # Auto-save after each turn
            save_session(self.state)

            # Auto-extract personal facts (fire-and-forget, never blocks)
            if turn_state.text_chunks:
                full_response = "".join(turn_state.text_chunks)
                asyncio.create_task(self._extract_personal_facts(user_input, full_response))

    async def _extract_personal_facts(self, user_input: str, assistant_response: str) -> None:
        """Background: extract personal facts from the conversation turn."""
        try:
            from isaac.personal.store import get_personal_store
            from isaac.personal.extractor import extract_personal_facts
            store = get_personal_store()
            await extract_personal_facts(user_input, assistant_response, store)
        except Exception:
            pass  # never let extraction errors surface to user

    def _handle_command(self, cmd: str) -> bool:
        """Handle /slash commands. Returns True if handled."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command == "/help":
            console.print(Panel(
                "[bold]Commands:[/bold]\n"
                "  /help          — Show this help\n"
                "  /tools         — List available tools\n"
                "  /connectors    — Show connector status\n"
                "  /skills        — List available skills\n"
                "  /status        — Show session dashboard\n"
                "  /sessions      — List saved sessions\n"
                "  /memory <q>    — Search memory\n"
                "  /cost [all]    — Show session cost (all = historical)\n"
                "  /apps          — List available external apps\n"
                "  /dream         — Run memory consolidation\n"
                "  /save          — Save session\n"
                "  /clear         — Clear conversation (keep memory)\n"
                "  /soul          — Show current soul\n"
                "  /reload        — Reload tools\n"
                "  /quit          — Save and exit\n"
                "\n"
                "[bold]Skills:[/bold] (use /<skill-name> <args> to run)\n"
                "  Run /skills to see available skill workflows",
                title="ISAAC Help",
                border_style="cyan",
            ))
            return True

        if command == "/tools" and self._harness:
            from isaac.cli.ui import tool_table
            console.print(tool_table(self._harness.config.tool_registry))
            return True

        if command == "/connectors" and self._harness:
            cr = self._harness.config.connector_registry
            if not cr:
                console.print("  [dim]No connector registry (gateway disabled)[/dim]")
            else:
                states = cr.get_status()
                if not states:
                    console.print("  [dim]No connectors configured. Edit ~/.isaac/connections.yaml[/dim]")
                else:
                    table = Table(title="Connectors", border_style="cyan")
                    table.add_column("Name", style="bold")
                    table.add_column("Status")
                    table.add_column("Tools")
                    table.add_column("Error", style="dim")
                    for s in states:
                        status_style = "green" if s.status == "connected" else "red"
                        tools_str = f"{len(s.tools)} tools" if s.tools else "-"
                        table.add_row(
                            s.name,
                            f"[{status_style}]{s.status}[/{status_style}]",
                            tools_str,
                            s.error[:60] if s.error else "",
                        )
                    console.print(table)
            return True

        if command == "/status" and self.state:
            from isaac.cli.ui import status_panel
            tool_count = len(self._harness.config.tool_registry) if self._harness else 0
            console.print(status_panel(
                agent_name=self.config.name,
                model=self.config.model,
                tokens=self.state.total_tokens,
                cost=self.state.total_cost,
                turns=self.state.turn_count,
                tool_count=tool_count,
            ))
            return True

        if command == "/sessions":
            sessions = list_sessions(self.config.name)
            if sessions:
                for s in sessions:
                    console.print(
                        f"  {s['session_id']} | {s['agent_name']} | "
                        f"{s['turn_count']} turns | ${s['total_cost']:.4f} | {s['timestamp']}"
                    )
            else:
                console.print("  [dim]No saved sessions[/dim]")
            return True

        if command == "/memory":
            memory = MemoryStore()
            if not arg:
                paths = memory.list_all()
                console.print(f"  [dim]{len(paths)} memory nodes[/dim]")
                for p in paths[:20]:
                    console.print(f"    {p}")
            else:
                results = memory.search(arg)
                for node in results:
                    console.print(f"  [bold]{node.path}[/bold]: {node.content[:100]}...")
            return True

        if command == "/cost" and self.state:
            from isaac.cli.dashboard import render_session, render_historical
            console.print(render_session(self.state.total_tokens, self.state.total_cost, self.state.turn_count))
            if arg == "all":
                console.print(render_historical())
            return True

        if command == "/save" and self.state:
            path = save_session(self.state)
            console.print(f"  [dim]Saved to {path}[/dim]")
            return True

        if command == "/clear" and self.state:
            self.state.messages.clear()
            self.state.summary = ""
            console.print("  [dim]Conversation cleared (memory preserved)[/dim]")
            return True

        if command == "/soul":
            from isaac.core.soul import load_soul
            soul = load_soul(self.config.soul, self.config.name)
            console.print(Markdown(soul))
            return True

        if command == "/apps":
            from isaac.apps.manifest import list_manifests
            apps = list_manifests()
            if not apps:
                console.print("  [dim]No apps. Run `isaac app add <repo>` or `isaac init`[/dim]")
            else:
                for a in apps:
                    gpu = f"GPU:{a['gpu_type']}" if a['gpu'] == "True" else "CPU"
                    console.print(f"  [bold]{a['name']}[/bold] — {a['description'][:60]} [{gpu}]")
            return True

        if command == "/dream":
            console.print("  [dim]Running memory consolidation...[/dim]")
            from isaac.memory.autodream import AutoDream
            memory = MemoryStore()
            try:
                from isaac.memory.embeddings import EmbeddingStore
                emb = EmbeddingStore(memory.dir)
            except ImportError:
                emb = None
            dreamer = AutoDream(memory, emb)
            import asyncio
            report = asyncio.get_event_loop().run_until_complete(dreamer.run())
            console.print(
                f"  scanned={report['scanned']} clusters={report['clusters']} "
                f"consolidated={report['consolidated']} pruned={report['pruned']}"
            )
            return True

        if command == "/skills":
            try:
                from isaac.core.skills import load_skills
                from isaac.core.config import SKILLS_DIR
                skills = load_skills(SKILLS_DIR)
                if not skills:
                    console.print("  [dim]No skills. Add .md files to ~/.isaac/skills/[/dim]")
                else:
                    for name, skill in skills.items():
                        invocable = f"  [dim]→ /{name} <args>[/dim]" if skill.user_invocable else ""
                        console.print(f"  [bold]{name}[/bold] — {skill.description}{invocable}")
                        if skill.params:
                            for pname, pdesc in skill.params.items():
                                console.print(f"    [dim]{pname}[/dim]: {pdesc}")
            except Exception as e:
                console.print(f"  [red]Error loading skills: {e}[/red]")
            return True

        if command == "/reload":
            console.print("  [dim]Rebuilding harness...[/dim]")
            console.print("  [dim]Use /quit and restart for now[/dim]")
            return True

        if command in ("/quit", "/exit"):
            if self.state:
                save_session(self.state)
                console.print("  [dim]Session saved.[/dim]")
            raise EOFError()

        # --- Check if it's a user-invocable skill ---
        skill_name = command.lstrip("/")
        try:
            from isaac.core.skills import load_skills, render_skill
            from isaac.core.config import SKILLS_DIR
            skills = load_skills(SKILLS_DIR)
            skill = skills.get(skill_name)
            if skill and skill.user_invocable:
                # Parse args as the first param value
                params: dict[str, str] = {}
                if arg and skill.params:
                    first_param = next(iter(skill.params))
                    params[first_param] = arg
                rendered = render_skill(skill, params)
                # Return False so the rendered prompt gets sent as the user message
                self._pending_skill_prompt = rendered
                return False
        except Exception:
            pass

        console.print(f"  [yellow]Unknown command: {command}[/yellow]")
        return True


def _format_input(inp: dict) -> str:
    """Format tool input for display."""
    lines = []
    for k, v in inp.items():
        val = str(v)
        if len(val) > 200:
            val = val[:200] + "..."
        lines.append(f"  {k}: {val}")
    return "\n".join(lines)


def _brief_input(inp: dict) -> str:
    """One-line summary of tool input."""
    parts = []
    for k, v in inp.items():
        val = str(v)[:60]
        parts.append(f"{k}={val}")
    return ", ".join(parts)[:120]


def _title_case_tool(name: str) -> str:
    """Convert 'file_read' → 'File Read' like Dexter."""
    import re
    stripped = re.sub(r"^(get)_", "", name)
    return " ".join(w.capitalize() for w in stripped.split("_"))


class _TurnState:
    """Mutable state for a single turn — shared between event loop and renderer."""

    def __init__(self) -> None:
        self.is_thinking: bool = False
        self.think_start: float = _time.time()
        self.text_chunks: list[str] = []
        self.active_tool: str = ""
        self.tool_start: float = 0.0
        self.tool_log: list[tuple[str, str, float]] = []
        self.cost_info: str = ""
        self.errors: list[str] = []


# Braille spinner frames for smooth animation
_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


_THINKING_VERBS = [
    "Thinking", "Reasoning", "Analyzing", "Pondering",
    "Working", "Processing", "Reflecting", "Considering",
]


class _LiveTurnView:
    """Dexter-inspired live renderable — circles, tree connectors, colored spinner."""

    def __init__(self, state: _TurnState) -> None:
        self.state = state
        self._thinking_verb = random.choice(_THINKING_VERBS)

    def __rich_console__(self, console, options):
        s = self.state
        now = _time.time()
        frame_idx = int(now * 10) % len(_SPINNER_FRAMES)
        frame = _SPINNER_FRAMES[frame_idx]

        # --- Completed tools (always show first) ---
        for status, name, t in s.tool_log:
            display = _title_case_tool(name) if status in ("ok", "error") else name
            if status == "ok":
                yield Text.from_markup(f"  [bright_blue]⏺[/bright_blue] {display}")
                yield Text.from_markup(f"  [dim]⎿  {t:.1f}s[/dim]")
            elif status == "error":
                yield Text.from_markup(f"  [bold red]⏺[/bold red] {display}")
                yield Text.from_markup(f"  [dim]⎿  [/dim][red]Error[/red]")
            elif status == "info":
                yield Text.from_markup(f"  [dim]⎿  {name}[/dim]")

        # --- Active tool (blinking green circle) ---
        if s.active_tool:
            elapsed = now - s.tool_start if s.tool_start else 0
            blink_on = int(now * 2.5) % 2 == 0
            circle = "[green]⏺[/green]" if blink_on else "[dim]⏺[/dim]"
            yield Text.from_markup(f"  {circle} {_title_case_tool(s.active_tool)} [dim]{elapsed:.1f}s[/dim]")
            return

        # --- Thinking (colored spinner + random verb) ---
        if s.is_thinking and not s.text_chunks:
            elapsed = now - s.think_start
            yield Text.from_markup(
                f"  [bright_blue]{frame} {self._thinking_verb}...[/bright_blue] [dim]{elapsed:.0f}s[/dim]"
            )
            return

        # --- Streaming response (blue circle + markdown) ---
        if s.text_chunks:
            partial = "".join(s.text_chunks)
            if partial.strip():
                yield Text("")
                yield Text.from_markup("[bright_blue]⏺[/bright_blue]")
                yield Markdown(partial)
            return

        yield Text("")
