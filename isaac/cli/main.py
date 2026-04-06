"""ISAAC CLI — main entry point.

Usage:
    isaac                     # Start default agent REPL
    isaac chat [agent]        # Start specific agent REPL
    isaac start               # Launch all agents in tmux tabs
    isaac stop                # Graceful shutdown all agents
    isaac status              # Show running agents
    isaac add <name>          # Add agent to running session
    isaac logs <name>         # Tail agent's session
    isaac sessions            # List saved sessions
    isaac gateway <type>      # Start a chat gateway (telegram, webhook)
    isaac plugins             # List/reload plugins
    isaac memory <query>      # Search memory
    isaac init                # Initialize ~/.isaac with defaults
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

from isaac.core.config import (
    CONFIG_FILE,
    ISAAC_HOME,
    TOOLS_DIR,
    SESSIONS_DIR,
    SOULS_DIR,
    ensure_dirs,
    load_agents_config,
)


def _load_env() -> None:
    """Load ~/.isaac/.env into the process environment."""
    env_file = ISAAC_HOME / ".env"
    if env_file.exists():
        for line in env_file.read_text().split("\n"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if val and key not in os.environ:
                    os.environ[key] = val


def main() -> None:
    _load_env()
    parser = argparse.ArgumentParser(
        prog="isaac",
        description="ISAAC — Intelligent Self-Assembling Agent Constellation",
    )
    sub = parser.add_subparsers(dest="command")

    # isaac chat [agent] [--session ID]
    chat_p = sub.add_parser("chat", help="Start an interactive agent session")
    chat_p.add_argument("agent", nargs="?", default="default", help="Agent name")
    chat_p.add_argument("--session", "-s", help="Resume a specific session ID")
    chat_p.add_argument("--model", "-m", help="Override model")
    chat_p.add_argument("--cwd", help="Working directory for the agent")

    # isaac start
    start_p = sub.add_parser("start", help="Launch all agents in tmux tabs")
    start_p.add_argument("--config", "-c", help="Path to agents.yaml")

    # isaac stop
    sub.add_parser("stop", help="Stop all running agents")

    # isaac status
    sub.add_parser("status", help="Show running agents and stats")

    # isaac add <name>
    add_p = sub.add_parser("add", help="Add an agent to the running tmux session")
    add_p.add_argument("name", help="Agent name")

    # isaac logs <name>
    logs_p = sub.add_parser("logs", help="Tail an agent's session")
    logs_p.add_argument("name", help="Agent name")

    # isaac sessions
    sessions_p = sub.add_parser("sessions", help="List saved sessions")
    sessions_p.add_argument("--agent", "-a", help="Filter by agent name")

    # isaac gateway <type>
    gw_p = sub.add_parser("gateway", help="Start a chat gateway")
    gw_p.add_argument("type", choices=["telegram", "webhook"], help="Gateway type")
    gw_p.add_argument("--agent", "-a", default="default", help="Agent to route messages to")
    gw_p.add_argument("--port", "-p", type=int, default=8080, help="Port for webhook gateway")

    # isaac tools
    tools_p = sub.add_parser("tools", help="List and manage tools")
    tools_p.add_argument("--reload", action="store_true", help="Reload all tools")

    # isaac memory <query>
    mem_p = sub.add_parser("memory", help="Search or list memory")
    mem_p.add_argument("query", nargs="?", help="Search query")

    # isaac hatch
    hatch_p = sub.add_parser("hatch", help="First-run onboarding — teach ISAAC about you")
    hatch_p.add_argument("--reset", action="store_true", help="Re-run onboarding")

    # isaac new
    new_p = sub.add_parser("new", help="Spawn a new agent — interactive setup with VM")
    new_p.add_argument("--model", "-m", default="claude-sonnet-4-6", help="Model for spawn conversation")

    # isaac cost
    cost_p = sub.add_parser("cost", help="Show cost dashboard")
    cost_p.add_argument("--agent", "-a", help="Filter by agent name")

    # isaac dream
    sub.add_parser("dream", help="Run memory consolidation (autoDream)")

    # isaac toolsmith
    ts_p = sub.add_parser("toolsmith", help="Generate a plugin from a description")
    ts_p.add_argument("description", help="Natural language description of the tool to build")
    ts_p.add_argument("--model", "-m", default="claude-sonnet-4-6", help="Model to use")
    ts_p.add_argument("--dry-run", action="store_true", help="Show generated code without saving")

    # isaac connect
    conn_p = sub.add_parser("connect", help="Manage service connections")
    conn_sub = conn_p.add_subparsers(dest="connect_command")
    conn_sub.add_parser("list", help="List connected services")
    conn_add_p = conn_sub.add_parser("add", help="Add a service connection")
    conn_add_p.add_argument("name", help="Service name")
    conn_add_p.add_argument("--command", "-c", help="Command for stdio MCP (e.g. npx)")
    conn_add_p.add_argument("--args", "-a", help="Args for the command")
    conn_add_p.add_argument("--url", "-u", help="URL for HTTP MCP server")
    conn_add_p.add_argument("--description", "-d", default="", help="Description")
    conn_rm_p = conn_sub.add_parser("remove", help="Remove a service connection")
    conn_rm_p.add_argument("name", help="Service name to remove")

    # isaac serve
    serve_p = sub.add_parser("serve", help="Start the unified MCP tool server")
    serve_p.add_argument("--transport", "-t", default="stdio", choices=["stdio", "http"], help="Transport protocol")
    serve_p.add_argument("--host", default="127.0.0.1", help="Host for HTTP transport")
    serve_p.add_argument("--port", "-p", type=int, default=9100, help="Port for HTTP transport")

    # isaac app
    app_p = sub.add_parser("app", help="Manage and run external apps")
    app_sub = app_p.add_subparsers(dest="app_command")
    app_sub.add_parser("list", help="List available apps")
    app_run_p = app_sub.add_parser("run", help="Run an app")
    app_run_p.add_argument("name", help="App name")
    app_run_p.add_argument("--input", "-i", action="append", default=[], help="Input key=value pairs")
    app_run_p.add_argument("--backend", "-b", help="Compute backend (e2b, modal)")
    app_add_p = app_sub.add_parser("add", help="Generate manifest from a GitHub repo")
    app_add_p.add_argument("repo_url", help="GitHub repo URL")

    # isaac heartbeat
    hb_p = sub.add_parser("heartbeat", help="Start the heartbeat scheduler")
    hb_p.add_argument("--agent", "-a", help="Run heartbeat for a specific agent only")
    hb_p.add_argument("--once", action="store_true", help="Run once and exit (no loop)")
    hb_p.add_argument("--interval", type=int, help="Override interval in seconds")

    # isaac cron
    cron_p = sub.add_parser("cron", help="Manage scheduled tasks")
    cron_sub = cron_p.add_subparsers(dest="cron_command")
    cron_sub.add_parser("list", help="List cron jobs")
    cron_add_p = cron_sub.add_parser("add", help="Add a cron job")
    cron_add_p.add_argument("name", help="Job name")
    cron_add_p.add_argument("--agent", "-a", default="default", help="Agent to run")
    cron_add_p.add_argument("--schedule", "-s", required=True, help="Cron expr or 'every:30m'")
    cron_add_p.add_argument("--task", "-t", required=True, help="Task description")
    cron_rm_p = cron_sub.add_parser("remove", help="Remove a cron job")
    cron_rm_p.add_argument("name", help="Job name to remove")
    cron_run_p = cron_sub.add_parser("run", help="Run a cron job immediately")
    cron_run_p.add_argument("name", help="Job name to run")

    # isaac personal
    personal_p = sub.add_parser("personal", help="Manage personal memory")
    personal_sub = personal_p.add_subparsers(dest="personal_command")
    personal_sub.add_parser("list", help="List personal memories")
    personal_search_p = personal_sub.add_parser("search", help="Search personal memory")
    personal_search_p.add_argument("query", help="Search query")
    personal_sub.add_parser("dream", help="Run personal memory consolidation")

    # isaac wiki
    wiki_p = sub.add_parser("wiki", help="Manage personal knowledge wikis")
    wiki_sub = wiki_p.add_subparsers(dest="wiki_command")
    wiki_sub.add_parser("list", help="List all wikis")
    wiki_create_p = wiki_sub.add_parser("create", help="Create a new wiki")
    wiki_create_p.add_argument("name", help="Wiki name (lowercase, hyphens)")
    wiki_create_p.add_argument("--description", "-d", default="", help="Description")
    wiki_ingest_p = wiki_sub.add_parser("ingest", help="Ingest a source into a wiki")
    wiki_ingest_p.add_argument("name", help="Wiki name")
    wiki_ingest_p.add_argument("source", help="URL, file path, or text to ingest")
    wiki_compile_p = wiki_sub.add_parser("compile", help="Compile raw sources into wiki pages")
    wiki_compile_p.add_argument("name", help="Wiki name")
    wiki_compile_p.add_argument("--source", "-s", help="Specific raw source filename (default: all new)")
    wiki_compile_p.add_argument("--model", "-m", default="claude-haiku-4-5-20251001", help="Model to use")
    wiki_query_p = wiki_sub.add_parser("query", help="Query a wiki")
    wiki_query_p.add_argument("name", help="Wiki name")
    wiki_query_p.add_argument("question", help="Question to ask")
    wiki_query_p.add_argument("--file-back", action="store_true", help="Save answer as wiki page")
    wiki_query_p.add_argument("--model", "-m", default="claude-sonnet-4-6", help="Model to use")
    wiki_lint_p = wiki_sub.add_parser("lint", help="Health-check a wiki")
    wiki_lint_p.add_argument("name", help="Wiki name")
    wiki_pages_p = wiki_sub.add_parser("pages", help="List pages in a wiki")
    wiki_pages_p.add_argument("name", help="Wiki name")
    wiki_read_p = wiki_sub.add_parser("read", help="Read a wiki page")
    wiki_read_p.add_argument("name", help="Wiki name")
    wiki_read_p.add_argument("page", help="Page path (e.g. overview.md, index.md)")
    wiki_log_p = wiki_sub.add_parser("log", help="Show wiki log")
    wiki_log_p.add_argument("name", help="Wiki name")
    wiki_search_p = wiki_sub.add_parser("search", help="Search wiki pages")
    wiki_search_p.add_argument("name", help="Wiki name")
    wiki_search_p.add_argument("query", help="Search query")

    # isaac init
    sub.add_parser("init", help="Initialize ~/.isaac with default configs")

    args = parser.parse_args()

    if args.command is None or args.command == "chat":
        # Default: launch REPL
        agent_name = getattr(args, "agent", "default")
        session_id = getattr(args, "session", None)
        model = getattr(args, "model", None)
        cwd = getattr(args, "cwd", None)
        _cmd_chat(agent_name, session_id, model, cwd)

    elif args.command == "start":
        config_path = Path(args.config) if args.config else None
        _cmd_start(config_path)

    elif args.command == "stop":
        _cmd_stop()

    elif args.command == "status":
        _cmd_status()

    elif args.command == "add":
        _cmd_add(args.name)

    elif args.command == "logs":
        _cmd_logs(args.name)

    elif args.command == "sessions":
        _cmd_sessions(getattr(args, "agent", None))

    elif args.command == "gateway":
        _cmd_gateway(args.type, args.agent, args.port)

    elif args.command in ("tools", "plugins"):
        _cmd_tools(getattr(args, "reload", False))

    elif args.command == "memory":
        _cmd_memory(getattr(args, "query", None))

    elif args.command == "hatch":
        _cmd_hatch(getattr(args, "reset", False))

    elif args.command == "new":
        _cmd_new(getattr(args, "model", "claude-sonnet-4-6"))

    elif args.command == "cost":
        _cmd_cost(getattr(args, "agent", None))

    elif args.command == "dream":
        _cmd_dream()

    elif args.command == "toolsmith":
        _cmd_toolsmith(args.description, args.model, getattr(args, "dry_run", False))

    elif args.command == "connect":
        conn_cmd = getattr(args, "connect_command", None)
        if conn_cmd == "list":
            _cmd_connect_list()
        elif conn_cmd == "add":
            _cmd_connect_add(args.name, getattr(args, "command", ""), getattr(args, "args", ""), getattr(args, "url", ""), getattr(args, "description", ""))
        elif conn_cmd == "remove":
            _cmd_connect_remove(args.name)
        else:
            print("Usage: isaac connect [list|add|remove]")

    elif args.command == "serve":
        _cmd_serve(args.transport, args.host, args.port)

    elif args.command == "app":
        app_cmd = getattr(args, "app_command", None)
        if app_cmd == "list":
            _cmd_app_list()
        elif app_cmd == "run":
            inputs = {}
            for pair in getattr(args, "input", []):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    inputs[k] = v
            _cmd_app_run(args.name, inputs, getattr(args, "backend", None))
        elif app_cmd == "add":
            _cmd_app_add(args.repo_url)
        else:
            print("Usage: isaac app [list|run|add]")

    elif args.command == "heartbeat":
        _cmd_heartbeat(
            getattr(args, "agent", None),
            getattr(args, "once", False),
            getattr(args, "interval", None),
        )

    elif args.command == "cron":
        cron_cmd = getattr(args, "cron_command", None)
        if cron_cmd == "list":
            _cmd_cron_list()
        elif cron_cmd == "add":
            _cmd_cron_add(args.name, args.agent, args.schedule, args.task)
        elif cron_cmd == "remove":
            _cmd_cron_remove(args.name)
        elif cron_cmd == "run":
            _cmd_cron_run(args.name)
        else:
            print("Usage: isaac cron [list|add|remove|run]")

    elif args.command == "personal":
        personal_cmd = getattr(args, "personal_command", None)
        if personal_cmd == "list":
            _cmd_personal_list()
        elif personal_cmd == "search":
            _cmd_personal_search(args.query)
        elif personal_cmd == "dream":
            _cmd_personal_dream()
        else:
            print("Usage: isaac personal [list|search|dream]")

    elif args.command == "wiki":
        wiki_cmd = getattr(args, "wiki_command", None)
        if wiki_cmd == "list":
            _cmd_wiki_list()
        elif wiki_cmd == "create":
            _cmd_wiki_create(args.name, getattr(args, "description", ""))
        elif wiki_cmd == "ingest":
            _cmd_wiki_ingest(args.name, args.source)
        elif wiki_cmd == "compile":
            _cmd_wiki_compile(args.name, getattr(args, "source", None), getattr(args, "model", "claude-haiku-4-5-20251001"))
        elif wiki_cmd == "query":
            _cmd_wiki_query(args.name, args.question, getattr(args, "file_back", False), getattr(args, "model", "claude-sonnet-4-6"))
        elif wiki_cmd == "lint":
            _cmd_wiki_lint(args.name)
        elif wiki_cmd == "pages":
            _cmd_wiki_pages(args.name)
        elif wiki_cmd == "read":
            _cmd_wiki_read(args.name, args.page)
        elif wiki_cmd == "log":
            _cmd_wiki_log(args.name)
        elif wiki_cmd == "search":
            _cmd_wiki_search(args.name, args.query)
        else:
            print("Usage: isaac wiki [list|create|ingest|compile|query|lint|pages|read|log|search]")

    elif args.command == "init":
        _cmd_init()

    else:
        parser.print_help()


def _cmd_chat(agent_name: str, session_id: str | None, model: str | None, cwd: str | None) -> None:
    """Launch interactive REPL for one agent."""
    from isaac.cli.terminal import TerminalREPL

    ensure_dirs()
    agents = load_agents_config()
    config = agents.get(agent_name)

    if not config:
        from isaac.core.types import AgentConfig
        config = AgentConfig(name=agent_name)

    if model:
        config.model = model
    if cwd:
        config.cwd = cwd

    repl = TerminalREPL(config, session_id)

    # Suppress MCP SDK cleanup noise on exit.
    # The stdio_client async generator throws RuntimeError when exiting
    # from a different task context — harmless but noisy.
    import warnings
    import logging
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*")

    try:
        asyncio.run(repl.run())
    except (KeyboardInterrupt, EOFError):
        pass  # Clean exit
    except BaseException as e:
        # MCP cancel scopes can throw CancelledError that escapes asyncio.run
        if "CancelledError" in type(e).__name__:
            pass
        else:
            raise
    finally:
        logging.getLogger("asyncio").setLevel(logging.WARNING)


def _cmd_start(config_path: Path | None = None) -> None:
    """Launch all agents in a tmux session with tabs."""
    ensure_dirs()
    agents = load_agents_config(config_path)

    if not shutil.which("tmux"):
        print("Error: tmux not found. Install tmux to use multi-agent mode.")
        sys.exit(1)

    session_name = "isaac"

    # Kill existing session if any
    subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True)

    # Create new tmux session with first agent
    first_agent = list(agents.keys())[0]
    isaac_cmd = f"isaac chat {first_agent}"

    subprocess.run([
        "tmux", "new-session", "-d", "-s", session_name, "-n", first_agent, isaac_cmd
    ])

    # Add tabs for remaining agents
    for name in list(agents.keys())[1:]:
        cmd = f"isaac chat {name}"
        subprocess.run([
            "tmux", "new-window", "-t", session_name, "-n", name, cmd
        ])

    # Add a tab for the unified MCP tool server
    subprocess.run([
        "tmux", "new-window", "-t", session_name, "-n", "mcp-server", "isaac serve"
    ])

    # Attach
    print(f"Launching {len(agents)} agents + MCP server in tmux session '{session_name}'...")
    print(f"  Agents: {', '.join(agents.keys())}")
    print(f"  MCP: isaac-tools (unified tool server)")
    os.execvp("tmux", ["tmux", "attach-session", "-t", session_name])


def _cmd_stop() -> None:
    """Stop all running agents gracefully."""
    result = subprocess.run(
        ["tmux", "kill-session", "-t", "isaac"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("ISAAC agents stopped.")
    else:
        print("No running ISAAC session found.")


def _cmd_status() -> None:
    """Show running agents and stats."""
    if not shutil.which("tmux"):
        print("tmux not installed — multi-agent status unavailable.")
        print("Install tmux or use `isaac chat <agent>` for single-agent mode.")
        return

    result = subprocess.run(
        ["tmux", "list-windows", "-t", "isaac", "-F", "#{window_name} #{window_active}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("No running ISAAC session.")
        return

    print("Running agents:")
    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        name = parts[0] if parts else "?"
        active = " ← active" if len(parts) > 1 and parts[1] == "1" else ""
        print(f"  {name}{active}")

    # Show session stats
    from isaac.agents.session import list_sessions
    sessions = list_sessions()
    if sessions:
        total_cost = sum(s["total_cost"] for s in sessions)
        total_turns = sum(s["turn_count"] for s in sessions)
        print(f"\nTotal: {total_turns} turns | ${total_cost:.4f}")


def _cmd_add(agent_name: str) -> None:
    """Add an agent tab to the running tmux session."""
    cmd = f"isaac chat {agent_name}"
    result = subprocess.run(
        ["tmux", "new-window", "-t", "isaac", "-n", agent_name, cmd],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"Added agent '{agent_name}' to ISAAC session.")
    else:
        print(f"Error: {result.stderr.strip()}")


def _cmd_logs(agent_name: str) -> None:
    """Show recent session activity for an agent."""
    from isaac.agents.session import list_sessions
    sessions = list_sessions(agent_name)
    if not sessions:
        print(f"No sessions for agent '{agent_name}'.")
        return

    latest = sessions[0]
    print(f"Agent: {latest['agent_name']}")
    print(f"Session: {latest['session_id']}")
    print(f"Turns: {latest['turn_count']} | Cost: ${latest['total_cost']:.4f}")
    print(f"Last active: {latest['timestamp']}")

    # Show last few messages
    from isaac.agents.session import load_session
    state = load_session(agent_name, latest["session_id"])
    if state:
        print("\nRecent messages:")
        for msg in state.messages[-6:]:
            prefix = "YOU" if msg.role.value == "user" else "BOT"
            text = msg.content[:200] if msg.content else "[tool call]"
            print(f"  [{prefix}] {text}")


def _cmd_sessions(agent_name: str | None) -> None:
    """List saved sessions."""
    from isaac.agents.session import list_sessions
    sessions = list_sessions(agent_name)
    if not sessions:
        print("No saved sessions.")
        return

    for s in sessions:
        print(
            f"  {s['session_id']} | {s['agent_name']:12} | "
            f"{s['turn_count']:3} turns | ${s['total_cost']:.4f} | {s['timestamp']}"
        )


def _cmd_gateway(gw_type: str, agent_name: str, port: int) -> None:
    """Start a chat gateway connected to an agent."""
    asyncio.run(_run_gateway(gw_type, agent_name, port))


async def _run_gateway(gw_type: str, agent_name: str, port: int) -> None:
    from isaac.agents.session import new_session, save_session
    from isaac.agents.tools import build_builtin_tools
    from isaac.core.orchestrator import Orchestrator, TextEvent
    from isaac.core.permissions import PermissionGate
    from isaac.core.types import AgentConfig
    from isaac.memory.scout import MemoryScout
    from isaac.memory.store import MemoryStore

    ensure_dirs()
    agents = load_agents_config()
    config = agents.get(agent_name, AgentConfig(name=agent_name))

    memory = MemoryStore()
    scout = MemoryScout(memory)
    gate = PermissionGate()

    # Auto-approve all tools in gateway mode (no human in the loop)
    gate.set_override("bash", PermissionLevel("deny"))  # except bash
    for tool_name in ["memory_search", "memory_read", "memory_write", "web_search",
                      "file_read", "file_list", "file_search"]:
        gate.set_override(tool_name, PermissionLevel("auto"))

    registry = build_builtin_tools(memory, config.cwd)

    # Per-sender session tracking
    sessions: dict[str, tuple] = {}

    async def handle_message(inbound) -> str:
        sender_key = f"{inbound.channel}:{inbound.sender_id}"

        if sender_key not in sessions:
            state = new_session(config.name)
            orch = Orchestrator(
                agent_config=config,
                tool_registry=registry,
                permission_gate=gate,
                memory_fn=scout.search,
            )
            sessions[sender_key] = (state, orch)

        state, orch = sessions[sender_key]
        response_parts: list[str] = []

        async for event in orch.run(inbound.text, state):
            if isinstance(event, TextEvent):
                response_parts.append(event.text)

        save_session(state)
        return "\n".join(response_parts) or "I processed your message but have no text response."

    # Create gateway
    if gw_type == "telegram":
        from isaac.gateway.telegram import TelegramGateway
        gw = TelegramGateway()
    elif gw_type == "webhook":
        from isaac.gateway.webhook import WebhookGateway
        gw = WebhookGateway(port=port)
    else:
        print(f"Unknown gateway type: {gw_type}")
        return

    gw.on_message(handle_message)
    print(f"Starting {gw_type} gateway with agent '{agent_name}'...")

    await gw.start()
    try:
        await asyncio.Event().wait()  # Run forever
    except KeyboardInterrupt:
        pass
    finally:
        await gw.stop()


def _cmd_tools(reload: bool) -> None:
    """List or reload tools."""
    from isaac.mcp.tool_loader import ToolLoader
    loader = ToolLoader()

    if reload:
        tools = loader.reload()
    else:
        tools = loader.scan()

    if not tools:
        print(f"No tools found in {TOOLS_DIR}")
        print(f"\nTo add a tool, create a .py file in {TOOLS_DIR}")
        print("Or use: isaac toolsmith 'description of the tool'")
        return

    print(f"Tools ({len(tools)}):")
    for name, tool in tools.items():
        print(f"  {name}: {tool.description}")


def _cmd_memory(query: str | None) -> None:
    """Search or list memory."""
    from isaac.memory.store import MemoryStore
    store = MemoryStore()

    if query:
        results = store.search(query)
        if not results:
            print("No results.")
            return
        for node in results:
            print(f"\n  [{node.path}]")
            print(f"  {node.content[:200]}")
    else:
        paths = store.list_all()
        print(f"Memory nodes ({len(paths)}):")
        for p in paths[:30]:
            print(f"  {p}")


def _cmd_connect_list() -> None:
    """List service connections."""
    from isaac.mcp.connections import list_connections
    services = list_connections()
    if not services:
        print("No service connections. Add one with: isaac connect add <name> --command <cmd> --args <args>")
        return
    print(f"Services ({len(services)}):")
    for s in services:
        status = "enabled" if s["enabled"] else "disabled"
        print(f"  {s['name']:20} {s['description'][:40]:40} [{s['transport']}] [{status}]")


def _cmd_connect_add(name: str, command: str, args: str, url: str, description: str) -> None:
    """Add a service connection."""
    from isaac.mcp.connections import add_connection
    args_list = args.split() if args else []
    transport = "http" if url else "stdio"
    add_connection(name=name, command=command, args=args_list, url=url, transport=transport, description=description)
    print(f"Added service '{name}' to connections.yaml")
    print(f"Restart `isaac serve` to activate, or the agent can call reload_tools.")


def _cmd_connect_remove(name: str) -> None:
    """Remove a service connection."""
    from isaac.mcp.connections import remove_connection
    if remove_connection(name):
        print(f"Removed service '{name}'")
    else:
        print(f"Service '{name}' not found")


def _cmd_serve(transport: str, host: str, port: int) -> None:
    """Start the unified MCP tool server."""
    from isaac.mcp.unified_server import run_server
    ensure_dirs()
    print(f"Starting ISAAC unified MCP server ({transport})")
    if transport == "http":
        print(f"  Listening on {host}:{port}")
    print(f"  Tools dir: {TOOLS_DIR}")
    print("  Tools auto-discovered from ~/.isaac/tools/")
    print()
    run_server(transport=transport, host=host, port=port)


def _cmd_app_list() -> None:
    """List available apps."""
    from isaac.apps.manifest import list_manifests
    apps = list_manifests()
    if not apps:
        print("No apps found. Run `isaac app add <repo_url>` or `isaac init` to get started.")
        return
    print(f"Available apps ({len(apps)}):")
    for a in apps:
        gpu = f"GPU:{a['gpu_type']}" if a['gpu'] == "True" else "CPU"
        print(f"  {a['name']:20} {a['description'][:50]:50} [{gpu}] [{a['mode']}] [{a['state']}]")


def _cmd_app_run(name: str, inputs: dict, backend: str | None) -> None:
    """Run an app."""
    from isaac.apps.runner import AppRunner
    from isaac.memory.store import MemoryStore
    runner = AppRunner(memory=MemoryStore(), backend_name=backend)
    print(f"Running app '{name}'...")
    if inputs:
        print(f"  Inputs: {inputs}")
    result = asyncio.run(runner.run(name, inputs))
    print(f"\nStatus: {result.status}")
    if result.error:
        print(f"Error: {result.error}")
    if result.summary:
        print(f"\n{result.summary[:2000]}")
    if result.artifacts:
        print(f"\nArtifacts ({len(result.artifacts)}):")
        for a in result.artifacts:
            print(f"  {a['path']} ({a['size']} bytes) → memory:{a['memory_path']}")
    print(f"\nDuration: {result.duration:.0f}s | Cost: ${result.cost:.4f}")


def _cmd_app_add(repo_url: str) -> None:
    """Auto-generate a manifest from a GitHub repo."""
    from isaac.apps.manifest import APPS_DIR
    APPS_DIR.mkdir(parents=True, exist_ok=True)

    # Extract repo name from URL
    name = repo_url.rstrip("/").split("/")[-1].lower().replace("-", "_")

    # Use Toolsmith pattern: have Claude generate the manifest
    import anthropic
    from isaac.core.config import get_env
    client = anthropic.Anthropic(api_key=get_env("ANTHROPIC_API_KEY"))

    print(f"Analyzing {repo_url}...")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""Generate an ISAAC app manifest (YAML) for this GitHub repo: {repo_url}

Output ONLY valid YAML, no markdown fences, no explanation. Use this format:
name: {name}
description: "one-line description"
repo: {repo_url}
version: main
gpu: true/false
gpu_type: T4/A10G/A100/H100
memory_gb: 16
timeout: 3600
setup: |
  commands to install dependencies
mode: command  # or agent
run: "command to run"  # for mode: command
agent_soul: |  # for mode: agent
  instructions for the agent
agent_tools: [bash, file_read, file_write]
inputs:
  query:
    type: string
    required: true
    description: "what the user provides"
artifacts:
  - path: "output/*"
    description: "what the app produces"
state: ephemeral"""
        }],
    )

    yaml_content = response.content[0].text if response.content else ""
    manifest_path = APPS_DIR / f"{name}.yaml"
    manifest_path.write_text(yaml_content)
    print(f"Manifest saved to {manifest_path}")
    print(f"\nGenerated manifest:\n{yaml_content[:1000]}")
    print(f"\nEdit {manifest_path} to customize.")
    print(f"\n  New agent VMs will auto-install this app on first boot.")
    print(f"  Existing agents: use the bootstrap_update tool to install it now.")


def _cmd_hatch(reset: bool) -> None:
    """Run the hatch onboarding."""
    from isaac.cli.hatch import run_hatch
    run_hatch(reset=reset)


def _cmd_new(model: str) -> None:
    """Spawn a new agent interactively, then launch it."""
    from isaac.cli.spawn import run_spawn

    agent_name = run_spawn(model=model)
    if agent_name:
        # Seamlessly transition into the new agent's REPL
        _cmd_chat(agent_name, session_id=None, model=None, cwd=None)


def _cmd_cost(agent_name: str | None) -> None:
    """Show cost dashboard."""
    from rich.console import Console
    from isaac.cli.dashboard import render_historical
    Console().print(render_historical(agent_filter=agent_name))


def _cmd_dream() -> None:
    """Run memory consolidation."""
    from isaac.memory.autodream import AutoDream
    from isaac.memory.store import MemoryStore
    store = MemoryStore()
    embedding_store = None
    try:
        from isaac.memory.embeddings import EmbeddingStore
        embedding_store = EmbeddingStore(store.dir)
    except ImportError:
        pass
    dreamer = AutoDream(store, embedding_store)
    report = asyncio.run(dreamer.run())
    print(f"Dream complete: scanned={report['scanned']}, clusters={report['clusters']}, "
          f"consolidated={report['consolidated']}, pruned={report['pruned']}")


def _cmd_toolsmith(description: str, model: str, dry_run: bool) -> None:
    """Generate a plugin from natural language description."""
    from isaac.agents.toolsmith import Toolsmith
    smith = Toolsmith(model=model)
    result = asyncio.run(smith.generate(description))
    if result.get("errors"):
        print(f"Validation errors: {result['errors']}")
        print(f"\nGenerated code:\n{result.get('code', '')}")
        return
    print(f"Generated plugin: {result.get('tool_name', 'unknown')}")
    print(f"\n{result['code']}")
    if dry_run:
        print("\n[dry-run] Plugin not saved.")
    else:
        path = smith.save(result)
        print(f"\nSaved to {path}")
        print("Run `isaac tools --reload` to activate.")


def _cmd_init() -> None:
    """Initialize ~/.isaac with default configuration."""
    ensure_dirs()

    # Default agents.yaml
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text("""# ISAAC Agent Configuration
# Each agent gets its own tmux tab when you run `isaac start`

agents:
  lead:
    soul: default
    model: claude-sonnet-4-6
    tools: ["*"]
    auto_start: true

  research:
    soul: research
    model: claude-sonnet-4-6
    tools: ["web_search", "memory_search", "memory_write", "file_read"]
    expose_as_tool: true
    tool_description: "Deep multi-pass research on any topic"

  ops:
    soul: ops
    model: claude-sonnet-4-6
    tools: ["bash", "file_read", "file_write", "file_list", "memory_write"]

  toolsmith:
    soul: toolsmith
    model: claude-sonnet-4-6
    tools: ["*"]
    cwd: "{tools_dir}"
""".format(tools_dir=str(TOOLS_DIR)))
        print(f"  Created {CONFIG_FILE}")

    # Default soul
    default_soul = SOULS_DIR / "default.md"
    if not default_soul.exists():
        default_soul.write_text("""# Default Soul

## Identity
You are ISAAC, a sharp and autonomous AI agent. You work in a constellation of specialized agents.

## Tone
- Concise. Act first, explain briefly.
- Use tools proactively — don't ask permission to think.
- Write important findings to memory.

## Rules
- Always check memory before starting new research.
- When delegating, be specific about the task and expected output.
- If you create a file, ensure it follows the project's existing structure.
""")
        print(f"  Created {default_soul}")

    # Research soul
    research_soul = SOULS_DIR / "research.md"
    if not research_soul.exists():
        research_soul.write_text("""# Research Agent Soul

## Identity
You are a research specialist. Your job is deep, multi-pass investigation.

## Behavior
- Search broadly first, then narrow.
- Always verify claims with multiple sources.
- Write findings to memory as structured notes.
- Cite sources.

## Output Format
Structure your findings as:
1. Summary (2-3 sentences)
2. Key findings (bullet points)
3. Sources
4. Open questions
""")
        print(f"  Created {research_soul}")

    # Ops soul
    ops_soul = SOULS_DIR / "ops.md"
    if not ops_soul.exists():
        ops_soul.write_text("""# Ops Agent Soul

## Identity
You are an operations agent. You execute system tasks, manage files, and run commands.

## Behavior
- Prefer safe, reversible operations.
- Always check before overwriting files.
- Log what you did to memory.
- Use git for version control when modifying code.
""")
        print(f"  Created {ops_soul}")

    # Toolsmith soul
    toolsmith_soul = SOULS_DIR / "toolsmith.md"
    if not toolsmith_soul.exists():
        toolsmith_soul.write_text("""# Toolsmith Agent Soul

## Identity
You are a toolsmith. You build tools that other agents can use.

## Behavior
- When asked to build a tool, first research the target API/service.
- Write tool files following the standard template (TOOLS list + handler functions).
- Test every tool before declaring it done.
- Drop working tools in the tools/ directory.

## Tool Template
```python
# tools/example.py
import httpx

TOOLS = [
    {
        "name": "example_action",
        "description": "What this tool does",
        "params": {"query": str},
        "handler": do_action,
    }
]

async def do_action(query: str) -> dict:
    # Implementation here
    return {"result": "done"}
```
""")
        print(f"  Created {toolsmith_soul}")

    # Example tool
    example_plugin = TOOLS_DIR / "_example.py"
    if not example_plugin.exists():
        example_plugin.write_text("""\"\"\"Example tool — rename without underscore to activate.\"\"\"

TOOLS = [
    {
        "name": "hello_world",
        "description": "A simple example tool that returns a greeting.",
        "params": {"name": str},
        "handler": None,  # set below
    }
]


async def hello(name: str) -> dict:
    return {"greeting": f"Hello, {name}! This is an ISAAC tool."}


TOOLS[0]["handler"] = hello
""")
        print(f"  Created {example_plugin}")

    # AutoResearch app manifest
    from isaac.apps.manifest import APPS_DIR
    autoresearch_manifest = APPS_DIR / "autoresearch.yaml"
    if not autoresearch_manifest.exists():
        autoresearch_manifest.write_text("""# AutoResearch — Karpathy's autonomous ML research agent
# An ISAAC agent iterates on train.py to minimize val_bpb
name: autoresearch
description: "Autonomous ML research — iterates on train.py to minimize val_bpb"
repo: https://github.com/karpathy/autoresearch
version: main

# Compute — needs a real NVIDIA GPU
gpu: true
gpu_type: H100
memory_gb: 32
timeout: 28800  # 8 hours max

# Setup — install uv, sync deps, prepare data
setup: |
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv sync
  uv run prepare.py

# Agent mode — ISAAC agent runs inside the VM, iterating on experiments
mode: agent
agent_soul: |
  You are running AutoResearch. Your job is to autonomously improve a small
  LLM by iterating on train.py. Each experiment runs for exactly 5 minutes.

  Protocol:
  1. Read program.md to understand the research framework
  2. Read the current train.py to understand the baseline
  3. Hypothesize an improvement (architecture, optimizer, data, etc.)
  4. Modify train.py with your change
  5. Run: uv run train.py
  6. Read the output — look for val_bpb (lower is better)
  7. Log results to memory_write with the hypothesis and outcome
  8. Decide: keep the change or revert? Then try the next hypothesis.
  9. Repeat until max_experiments reached or diminishing returns.

  Always save your best train.py and a summary of all experiments to memory.
agent_tools:
  - bash
  - file_read
  - file_write
  - memory_write

# Inputs
inputs:
  query:
    type: string
    required: true
    description: "Research direction or hypothesis to explore"
  max_experiments:
    type: integer
    default: 12
    description: "Max experiments (~5 min each, 12 = 1 hour)"

# Artifacts to collect when done
artifacts:
  - path: "train.py"
    description: "Best performing training script"
  - path: "*.log"
    description: "Experiment logs"

# State — ephemeral (save artifacts to memory, kill VM)
state: ephemeral
""")
        print(f"  Created {autoresearch_manifest}")

    # .env template
    env_file = ISAAC_HOME / ".env.example"
    if not env_file.exists():
        env_file.write_text("""# ISAAC Environment Variables

# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Web search
BRAVE_API_KEY=

# Optional: Telegram gateway
TELEGRAM_BOT_TOKEN=

# Optional: Cloud compute (for app_run)
E2B_API_KEY=
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=

# Optional: Custom model
# ISAAC_MODEL=claude-sonnet-4-6
""")
        print(f"  Created {env_file}")

    # Copy AGENT.md to ISAAC_HOME for self-awareness
    # This tells agents what they can do and how to act
    agent_md_src = Path(__file__).resolve().parent.parent.parent / "AGENT.md"
    agent_md_dst = ISAAC_HOME / "AGENT.md"
    if agent_md_src.exists() and not agent_md_dst.exists():
        import shutil
        shutil.copy2(agent_md_src, agent_md_dst)
        print(f"  Created {agent_md_dst}")

    print(f"\nISAAC initialized at {ISAAC_HOME}")
    print(f"\nNext steps:")
    print(f"  1. Set ANTHROPIC_API_KEY in your environment")
    print(f"  2. Edit {CONFIG_FILE} to configure agents")
    print(f"  3. Run: isaac chat          (single agent)")
    print(f"     Run: isaac start         (all agents in tmux)")
    print(f"     Run: isaac app list      (see available apps)")
    print(f"     Run: isaac app run autoresearch -i query='explore attention'")
    print(f"     Run: isaac gateway telegram  (Telegram bot)")


def _cmd_heartbeat(agent_name: str | None, once: bool, interval: int | None) -> None:
    """Start the heartbeat scheduler."""
    from isaac.core.heartbeat import (
        HeartbeatConfig,
        HeartbeatState,
        run_heartbeat,
        start_heartbeat_loop,
    )

    ensure_dirs()
    agents = load_agents_config()

    if once:
        # Run a single heartbeat for one agent
        name = agent_name or "default"
        config = agents.get(name)
        if not config:
            from isaac.core.types import AgentConfig
            config = AgentConfig(name=name)

        hb_config = HeartbeatConfig(
            agent_name=name,
            interval_seconds=interval or 1800,
        )

        # Load standing orders from heartbeats/ dir
        orders_file = ISAAC_HOME / "heartbeats" / f"{name}.md"
        if orders_file.exists():
            hb_config.standing_orders = orders_file.read_text()

        state = HeartbeatState(agent_name=name)
        print(f"Running heartbeat for '{name}'...")
        result = asyncio.run(run_heartbeat(config, hb_config, state))
        if result.get("ran"):
            print(f"  Cost: ${result['cost']:.4f} | Tokens: {result['tokens']:,}")
            if result.get("response_preview"):
                print(f"  Response: {result['response_preview']}")
        else:
            print(f"  Skipped: {result.get('reason', 'unknown')}")
        return

    # Start the scheduler loop
    heartbeats: dict[str, HeartbeatConfig] = {}
    for name, config in agents.items():
        if agent_name and name != agent_name:
            continue
        hb_config = HeartbeatConfig(
            agent_name=name,
            interval_seconds=interval or 1800,
        )
        orders_file = ISAAC_HOME / "heartbeats" / f"{name}.md"
        if orders_file.exists():
            hb_config.standing_orders = orders_file.read_text()
        heartbeats[name] = hb_config

    if not heartbeats:
        print("No agents configured for heartbeat.")
        return

    print(f"Starting heartbeat scheduler for {len(heartbeats)} agents:")
    for name, hb in heartbeats.items():
        print(f"  {name}: every {hb.interval_seconds}s")
    print("Press Ctrl+C to stop.\n")

    try:
        asyncio.run(start_heartbeat_loop(agents, heartbeats))
    except KeyboardInterrupt:
        print("\nHeartbeat scheduler stopped.")


def _cmd_cron_list() -> None:
    """List cron jobs."""
    from isaac.core.heartbeat import load_cron_jobs
    jobs = load_cron_jobs()
    if not jobs:
        print("No cron jobs configured.")
        print(f"Add one: isaac cron add my-job -a research -s 'every:30m' -t 'Check for new papers'")
        return
    for job in jobs:
        status = "enabled" if job.enabled else "disabled"
        print(f"  {job.name}: agent={job.agent_name}, schedule={job.schedule}, {status}")
        if job.task:
            print(f"    task: {job.task[:100]}")


def _cmd_cron_add(name: str, agent: str, schedule: str, task: str) -> None:
    """Add a cron job."""
    from isaac.core.heartbeat import CronJob, save_cron_job
    job = CronJob(name=name, agent_name=agent, schedule=schedule, task=task)
    save_cron_job(job)
    print(f"Cron job '{name}' added: agent={agent}, schedule={schedule}")


def _cmd_cron_remove(name: str) -> None:
    """Remove a cron job."""
    import yaml
    cron_file = ISAAC_HOME / "cron.yaml"
    if not cron_file.exists():
        print(f"No cron jobs configured.")
        return
    with open(cron_file) as f:
        data = yaml.safe_load(f) or {}
    jobs = data.get("jobs", {})
    if name in jobs:
        del jobs[name]
        with open(cron_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"Removed cron job '{name}'.")
    else:
        print(f"Cron job '{name}' not found.")


def _cmd_cron_run(name: str) -> None:
    """Run a cron job immediately."""
    from isaac.core.heartbeat import (
        HeartbeatConfig,
        HeartbeatState,
        load_cron_jobs,
        run_heartbeat,
    )

    jobs = load_cron_jobs()
    job = next((j for j in jobs if j.name == name), None)
    if not job:
        print(f"Cron job '{name}' not found.")
        return

    agents = load_agents_config()
    config = agents.get(job.agent_name)
    if not config:
        from isaac.core.types import AgentConfig
        config = AgentConfig(name=job.agent_name)

    hb_config = HeartbeatConfig(
        agent_name=job.agent_name,
        standing_orders=job.task,
    )
    state = HeartbeatState(agent_name=job.agent_name)

    print(f"Running cron job '{name}' (agent: {job.agent_name})...")
    result = asyncio.run(run_heartbeat(config, hb_config, state))
    if result.get("ran"):
        print(f"  Cost: ${result['cost']:.4f} | Tokens: {result['tokens']:,}")
        if result.get("response_preview"):
            print(f"  Response: {result['response_preview']}")
    else:
        print(f"  Skipped: {result.get('reason', 'unknown')}")


# --- Personal memory commands ---


def _cmd_personal_list() -> None:
    from isaac.personal.store import get_personal_store
    store = get_personal_store()
    paths = store.list_all()
    if not paths:
        print("No personal memories yet. They're auto-extracted from conversations,")
        print("or use the 'remember' tool in chat to save facts explicitly.")
        return
    print(f"Personal memories ({len(paths)}):")
    for p in paths[:30]:
        node = store.read(p)
        if node:
            tags_str = f" [{', '.join(node.tags)}]" if node.tags else ""
            preview = node.content[:80].replace("\n", " ")
            print(f"  {p:40} {preview}{tags_str}")
    if len(paths) > 30:
        print(f"  ... and {len(paths) - 30} more")


def _cmd_personal_search(query_str: str) -> None:
    from isaac.personal.store import get_personal_store
    store = get_personal_store()
    results = store.search(query_str)
    if not results:
        print("No results.")
        return
    for node in results:
        print(f"\n  [{node.path}]")
        print(f"  {node.content[:200]}")


def _cmd_personal_dream() -> None:
    from isaac.personal.store import get_personal_store
    from isaac.memory.autodream import AutoDream
    store = get_personal_store()

    # Optional embeddings
    embedding_store = None
    try:
        from isaac.memory.embeddings import EmbeddingStore
        embedding_store = EmbeddingStore(store.dir)
    except ImportError:
        pass

    dreamer = AutoDream(store, embedding_store)
    result = asyncio.run(dreamer.run())
    print(f"Personal dream complete:")
    print(f"  Scanned:      {result['scanned']}")
    print(f"  Clusters:     {result['clusters']}")
    print(f"  Consolidated: {result['consolidated']}")
    print(f"  Pruned:       {result['pruned']}")


# --- Wiki commands ---


def _cmd_wiki_list() -> None:
    from isaac.wiki.store import WikiStore
    wikis = WikiStore.list_wikis()
    if not wikis:
        print("No wikis. Create one with: isaac wiki create <name>")
        return
    print(f"Wikis ({len(wikis)}):")
    for w in wikis:
        print(f"  {w['name']:20} {w['raw_count']:3} raw | {w['page_count']:3} pages")


def _cmd_wiki_create(name: str, description: str) -> None:
    from isaac.wiki.store import WikiStore
    try:
        store = WikiStore.create(name, description)
        print(f"Created wiki '{name}' at {store.dir}")
        print(f"  raw/       — drop source documents here")
        print(f"  pages/     — LLM-maintained wiki pages")
        print(f"  index.md   — content catalog")
        print(f"  log.md     — operation log")
        print(f"  schema.md  — wiki conventions")
        print(f"\nNext: isaac wiki ingest {name} <url-or-file>")
    except FileExistsError:
        print(f"Wiki '{name}' already exists.")


def _cmd_wiki_ingest(wiki_name: str, source: str) -> None:
    from isaac.wiki.store import WikiStore
    from isaac.wiki.ingest import ingest_file, ingest_url, ingest_text

    store = WikiStore(wiki_name)
    if not store.exists:
        print(f"Wiki '{wiki_name}' not found. Create it first: isaac wiki create {wiki_name}")
        return

    if source.startswith("http://") or source.startswith("https://"):
        result = asyncio.run(ingest_url(store, source))
    elif "/" in source or source.endswith((".md", ".txt", ".pdf")):
        result = asyncio.run(ingest_file(store, source))
    else:
        result = asyncio.run(ingest_text(store, source[:50], source))

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Ingested → raw/{result['filename']} ({result['chars']:,} chars)")
        print(f"Next: isaac wiki compile {wiki_name}")


def _cmd_wiki_compile(wiki_name: str, source: str | None, model: str) -> None:
    from isaac.wiki.store import WikiStore
    from isaac.wiki.compiler import compile_source, compile_all_new

    store = WikiStore(wiki_name)
    if not store.exists:
        print(f"Wiki '{wiki_name}' not found.")
        return

    async def _run():
        if source:
            gen = compile_source(store, source, model)
        else:
            gen = compile_all_new(store, model)

        async for event in gen:
            etype = event["type"]
            if etype == "status":
                print(f"  {event['content']}")
            elif etype == "page":
                print(f"  ✓ {event['content']}")
            elif etype == "index":
                print(f"  ✓ {event['content']}")
            elif etype == "done":
                print(f"\n{event['content']}")
            elif etype == "error":
                print(f"  ✗ {event['content']}")

    print(f"Compiling wiki '{wiki_name}'...")
    asyncio.run(_run())


def _cmd_wiki_query(wiki_name: str, question: str, file_back: bool, model: str) -> None:
    from isaac.wiki.store import WikiStore
    from isaac.wiki.compiler import query

    store = WikiStore(wiki_name)
    if not store.exists:
        print(f"Wiki '{wiki_name}' not found.")
        return

    result = asyncio.run(query(store, question, file_back=file_back, model=model))
    print(result["answer"])
    if result.get("pages_consulted"):
        print(f"\n--- Consulted: {', '.join(result['pages_consulted'])}")
    if result.get("filed_back"):
        print(f"--- Filed as: {result['filed_back']}")


def _cmd_wiki_lint(wiki_name: str) -> None:
    from isaac.wiki.store import WikiStore
    from isaac.wiki.compiler import lint

    store = WikiStore(wiki_name)
    if not store.exists:
        print(f"Wiki '{wiki_name}' not found.")
        return

    result = asyncio.run(lint(store))
    print(result["summary"])
    for issue in result["issues"]:
        if issue["type"] == "llm_analysis":
            print(f"\n{issue['description']}")
        else:
            print(f"  [{issue['type']}] {issue['description']}")


def _cmd_wiki_pages(wiki_name: str) -> None:
    from isaac.wiki.store import WikiStore
    store = WikiStore(wiki_name)
    if not store.exists:
        print(f"Wiki '{wiki_name}' not found.")
        return
    pages = store.list_pages()
    if not pages:
        print(f"No pages in '{wiki_name}'. Run: isaac wiki compile {wiki_name}")
        return
    print(f"Pages in '{wiki_name}' ({len(pages)}):")
    for p in pages:
        print(f"  {p}")


def _cmd_wiki_read(wiki_name: str, page: str) -> None:
    from isaac.wiki.store import WikiStore
    store = WikiStore(wiki_name)
    if not store.exists:
        print(f"Wiki '{wiki_name}' not found.")
        return
    # Try pages/ first, then top-level
    content = store.read_page(page)
    if content is None:
        content = store.read(page)
    if content is None:
        print(f"Page not found: {page}")
        return
    print(content)


def _cmd_wiki_log(wiki_name: str) -> None:
    from isaac.wiki.store import WikiStore
    store = WikiStore(wiki_name)
    if not store.exists:
        print(f"Wiki '{wiki_name}' not found.")
        return
    log = store.read_log()
    if not log.strip():
        print("Log is empty.")
    else:
        print(log)


def _cmd_wiki_search(wiki_name: str, query_str: str) -> None:
    from isaac.wiki.store import WikiStore
    from isaac.wiki.search import search

    store = WikiStore(wiki_name)
    if not store.exists:
        print(f"Wiki '{wiki_name}' not found.")
        return

    results = search(store, query_str)
    if not results:
        print("No results.")
        return
    for r in results:
        print(f"\n  [{r['path']}] (score: {r['score']})")
        print(f"  {r['snippet'][:200]}")


if __name__ == "__main__":
    main()
