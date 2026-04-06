"""
Heartbeat — scheduled agent execution with clean-room handoff.

Combines three patterns:
1. OpenClaw heartbeat: interval-based agent execution with cost budgeting
2. AutoResearch: agent reads reality (not a handoff doc) to decide what to do
3. Clean-room principle: agent leaves state clean for the next run

The key innovation: instead of writing a "continuation prompt" (fragile, goes stale),
the agent writes ARTIFACTS that ARE the state. The next run reads those artifacts
and decides what to do. Just like AutoResearch reads git log + results.tsv + train.py,
our agents read memory nodes + session transcripts + workspace files.

On each heartbeat:
1. Load the agent's standing orders (HEARTBEAT.md or heartbeat config)
2. Run memory scout with the standing orders as query (what's relevant NOW)
3. Execute the agentic loop (agent sees current state, decides what to do)
4. Agent writes results to memory (artifacts, not handoff notes)
5. Session auto-saves (transcript is the audit trail)
6. If work is incomplete, agent writes a CONTINUATION memory node
   with structured fields the next run can pick up

The continuation node is NOT a prompt — it's structured data:
- what_was_done: factual summary of actions taken
- what_remains: specific unfinished items
- blocking_on: external dependencies or missing info
- artifacts: paths to files/memory nodes created
- next_priority: what the next run should focus on first

The next run's memory scout finds this node (high importance, recent timestamp)
and the agent decides what to do based on reality, not a stale prompt.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import isaac.core.config as _cfg
from isaac.core.types import AgentConfig

log = logging.getLogger(__name__)


@dataclass
class HeartbeatConfig:
    """Configuration for a scheduled heartbeat."""
    agent_name: str
    interval_seconds: int = 1800  # 30 minutes default
    standing_orders: str = ""     # What the agent should focus on each run
    cron: str = ""                # Cron expression (alternative to interval)
    enabled: bool = True
    quiet_hours: tuple[int, int] = (0, 6)  # Don't run between midnight and 6am
    max_cost_per_run: float = 0.10  # Cost cap per heartbeat run ($0.10 default)
    model_override: str = "claude-haiku-4-5-20251001"  # Haiku by default — cheap
    light_context: bool = True    # Minimal context by default (no ARCHITECTURE.md)


@dataclass
class HeartbeatState:
    """Persistent state for a heartbeat schedule."""
    agent_name: str
    last_run_at: float = 0.0
    last_run_cost: float = 0.0
    total_runs: int = 0
    total_cost: float = 0.0
    last_continuation: str = ""  # Path to last continuation memory node
    consecutive_no_ops: int = 0  # How many runs did nothing useful


@dataclass
class CronJob:
    """A scheduled task."""
    name: str
    agent_name: str
    schedule: str  # Cron expression: "0 9 * * 1-5" or interval: "every:30m"
    task: str      # What to do (natural language or path to orders file)
    enabled: bool = True
    last_run: float = 0.0
    next_run: float = 0.0
    run_count: int = 0


# --- Continuation node contract ---

CONTINUATION_TEMPLATE = """\
---
type: "continuation"
agent: "{agent_name}"
importance: 0.9
tags: ["continuation", "heartbeat", "{agent_name}"]
created: "{timestamp}"
---

## What Was Done
{what_was_done}

## What Remains
{what_remains}

## Blocking On
{blocking_on}

## Artifacts
{artifacts}

## Next Priority
{next_priority}
"""


def write_continuation(
    agent_name: str,
    what_was_done: str,
    what_remains: str = "Nothing — task complete.",
    blocking_on: str = "Nothing.",
    artifacts: list[str] | None = None,
    next_priority: str = "",
) -> str:
    """Write a continuation memory node. Returns the path."""
    from isaac.memory.store import MemoryStore
    store = MemoryStore()

    path = f"continuations/{agent_name}.md"
    content = CONTINUATION_TEMPLATE.format(
        agent_name=agent_name,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        what_was_done=what_was_done,
        what_remains=what_remains,
        blocking_on=blocking_on,
        artifacts="\n".join(f"- {a}" for a in (artifacts or [])) or "None.",
        next_priority=next_priority or what_remains,
    )

    store.write(path, content, {
        "type": "continuation",
        "agent": agent_name,
        "importance": 0.9,
        "tags": ["continuation", "heartbeat", agent_name],
    })

    return path


def read_continuation(agent_name: str) -> str | None:
    """Read the latest continuation for an agent, if any."""
    from isaac.memory.store import MemoryStore
    store = MemoryStore()
    node = store.read(f"continuations/{agent_name}.md")
    return node.content if node else None


# --- Heartbeat runner ---

async def run_heartbeat(
    agent_config: AgentConfig,
    heartbeat_config: HeartbeatConfig,
    state: HeartbeatState,
) -> dict[str, Any]:
    """Execute a single heartbeat run.

    Returns dict with: ran (bool), cost, tokens, continuation_path
    """
    from isaac.agents.session import new_session, save_session
    from isaac.agents.tools import build_builtin_tools
    from isaac.core.orchestrator import Orchestrator, TextEvent, ErrorEvent, CostEvent
    from isaac.core.permissions import PermissionGate
    from isaac.memory.scout import MemoryScout
    from isaac.memory.store import MemoryStore

    # Check quiet hours
    current_hour = time.localtime().tm_hour
    quiet_start, quiet_end = heartbeat_config.quiet_hours
    if quiet_start <= current_hour < quiet_end:
        return {"ran": False, "reason": "quiet_hours"}

    # Check cost cap
    if state.last_run_cost > heartbeat_config.max_cost_per_run:
        log.warning(f"Last run cost ${state.last_run_cost:.4f} exceeded cap, skipping")

    # Build standing orders prompt
    # The agent gets: standing orders + any continuation from last run + current state
    prompt_parts: list[str] = []

    # Standing orders (the agent's recurring mission)
    if heartbeat_config.standing_orders:
        prompt_parts.append(heartbeat_config.standing_orders)
    else:
        # Load from HEARTBEAT.md if it exists
        heartbeat_file = _cfg.ISAAC_HOME / "heartbeats" / f"{agent_config.name}.md"
        if heartbeat_file.exists():
            prompt_parts.append(heartbeat_file.read_text())
        else:
            prompt_parts.append(
                f"You are {agent_config.name} running on a scheduled heartbeat. "
                "Check your continuation memory for any pending work. "
                "If nothing is pending, review recent memory for anything that needs attention."
            )

    # Append continuation awareness
    continuation = read_continuation(agent_config.name)
    if continuation:
        prompt_parts.append(
            "\n\n## Previous Run State\n"
            "Your last heartbeat run left this continuation. Read it carefully — "
            "pick up where you left off. Do NOT repeat work that's already done.\n\n"
            + continuation
        )

    # Instruction: clean up after yourself
    prompt_parts.append(
        "\n\n## Heartbeat Protocol\n"
        "When you finish this run:\n"
        "1. Write any findings/results to memory (memory_write tool)\n"
        "2. If work remains, use memory_write to update your continuation "
        f"at path 'continuations/{agent_config.name}.md' with what was done, "
        "what remains, what you're blocked on, and what the next run should prioritize.\n"
        "3. If the task is complete, write a final summary to memory and clear "
        "the continuation by writing 'Task complete. No pending work.' to it.\n"
        "4. Do NOT ask for permission or wait for input. You are autonomous."
    )

    user_message = "\n\n".join(prompt_parts)

    # Override model + cap iterations for cost control
    config = AgentConfig(
        name=agent_config.name,
        soul=agent_config.soul,
        model=heartbeat_config.model_override or agent_config.model,
        tools=agent_config.tools,
        mcp_servers=agent_config.mcp_servers,
        max_iterations=min(agent_config.max_iterations, 10),  # Hard cap for heartbeats
        context_budget=agent_config.context_budget,
        cwd=agent_config.cwd,
    )

    # Setup — reduced scout budget for heartbeats (500 tokens, not 2000)
    memory = MemoryStore()
    scout = MemoryScout(memory)
    gate = PermissionGate()

    soul_mode = "minimal" if heartbeat_config.light_context else "full"

    # Auto-approve read tools, deny bash in heartbeat mode
    for tool_name in ["memory_search", "memory_read", "memory_write", "web_search",
                      "file_read", "file_list", "file_search"]:
        from isaac.core.types import PermissionLevel
        gate.set_override(tool_name, PermissionLevel.AUTO)

    registry = build_builtin_tools(memory, config.cwd)
    session = new_session(config.name)

    orch = Orchestrator(
        agent_config=config,
        tool_registry=registry,
        permission_gate=gate,
        memory_fn=scout.search,
        soul_mode=soul_mode,
    )

    # Run
    run_cost = 0.0
    run_tokens = 0
    response_text: list[str] = []

    async for event in orch.run(user_message, session):
        if isinstance(event, TextEvent):
            response_text.append(event.text)
        elif isinstance(event, CostEvent):
            run_cost += event.cost
            run_tokens += event.input_tokens + event.output_tokens

            # Cost circuit breaker
            if run_cost > heartbeat_config.max_cost_per_run:
                log.warning(f"Heartbeat cost cap reached (${run_cost:.4f}), stopping")
                break
        elif isinstance(event, ErrorEvent):
            log.error(f"Heartbeat error: {event.error}")

    # Save session
    save_session(session)

    # Update state
    state.last_run_at = time.time()
    state.last_run_cost = run_cost
    state.total_runs += 1
    state.total_cost += run_cost

    # Check if anything meaningful happened
    if not response_text or all(not t.strip() for t in response_text):
        state.consecutive_no_ops += 1
    else:
        state.consecutive_no_ops = 0

    return {
        "ran": True,
        "cost": run_cost,
        "tokens": run_tokens,
        "turns": session.turn_count,
        "response_preview": " ".join(response_text)[:200],
    }


# --- Heartbeat scheduler ---

async def start_heartbeat_loop(
    agents: dict[str, AgentConfig],
    heartbeats: dict[str, HeartbeatConfig],
) -> None:
    """Run the heartbeat scheduler loop. Fires agents on their configured intervals."""
    states: dict[str, HeartbeatState] = {}
    for name in heartbeats:
        states[name] = HeartbeatState(agent_name=name)

    log.info(f"Heartbeat scheduler started for {len(heartbeats)} agents")

    while True:
        now = time.time()

        for name, hb_config in heartbeats.items():
            if not hb_config.enabled:
                continue

            state = states[name]
            elapsed = now - state.last_run_at

            if elapsed < hb_config.interval_seconds:
                continue

            agent_config = agents.get(name)
            if not agent_config:
                continue

            # Back off if agent keeps doing nothing
            if state.consecutive_no_ops >= 3:
                backoff = min(state.consecutive_no_ops * hb_config.interval_seconds, 7200)
                if elapsed < backoff:
                    continue

            log.info(f"Heartbeat firing: {name} (run #{state.total_runs + 1})")

            try:
                result = await run_heartbeat(agent_config, hb_config, state)
                if result["ran"]:
                    log.info(
                        f"Heartbeat {name}: cost=${result['cost']:.4f}, "
                        f"tokens={result['tokens']}, turns={result['turns']}"
                    )
            except Exception as e:
                log.error(f"Heartbeat {name} failed: {e}")

        # Sleep until next check (1 minute resolution)
        await asyncio.sleep(60)


# --- Cron job management ---

_CRON_FILE = _cfg.ISAAC_HOME / "cron.yaml"


def load_cron_jobs() -> list[CronJob]:
    """Load cron jobs from ~/.isaac/cron.yaml."""
    import yaml
    if not _CRON_FILE.exists():
        return []

    with open(_CRON_FILE) as f:
        data = yaml.safe_load(f) or {}

    jobs: list[CronJob] = []
    for name, cfg in data.get("jobs", {}).items():
        jobs.append(CronJob(
            name=name,
            agent_name=cfg.get("agent", "default"),
            schedule=cfg.get("schedule", "every:30m"),
            task=cfg.get("task", ""),
            enabled=cfg.get("enabled", True),
        ))
    return jobs


def save_cron_job(job: CronJob) -> None:
    """Save/update a cron job to ~/.isaac/cron.yaml."""
    import yaml
    _CRON_FILE.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {}
    if _CRON_FILE.exists():
        with open(_CRON_FILE) as f:
            data = yaml.safe_load(f) or {}

    jobs = data.setdefault("jobs", {})
    jobs[job.name] = {
        "agent": job.agent_name,
        "schedule": job.schedule,
        "task": job.task,
        "enabled": job.enabled,
    }

    with open(_CRON_FILE, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
