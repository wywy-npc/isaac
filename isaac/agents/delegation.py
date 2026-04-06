"""Agent-as-a-tool — delegate tasks to other agents in the constellation.

Spawns a child orchestrator with the target agent's config, auto-approved
permissions, and STREAMS progress back to the parent. Child agents do NOT
get delegate_agent to prevent infinite recursion.

The key fix: child events are yielded upward as DelegationEvents so the
parent terminal shows what the child is doing. No more silent black hole.
"""
from __future__ import annotations

import time
from typing import Any, AsyncIterator

from isaac.core.config import load_agents_config
from isaac.core.orchestrator import (
    CostEvent,
    ErrorEvent,
    Event,
    Orchestrator,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from isaac.core.permissions import PermissionGate
from isaac.core.types import AgentConfig, PermissionLevel, ToolDef
from isaac.agents.session import new_session
from isaac.memory.store import MemoryStore


class DelegationEvent(Event):
    """Progress from a delegated child agent — surfaces in the parent terminal."""
    def __init__(self, agent_name: str, event_type: str, detail: str) -> None:
        self.agent_name = agent_name
        self.event_type = event_type
        self.detail = detail


class AgentDelegator:
    """Manages delegation to child agents with progress streaming."""

    def __init__(self, agents: dict[str, AgentConfig], memory: MemoryStore, embedding_store: Any = None) -> None:
        self.agents = agents
        self.memory = memory
        self.embedding_store = embedding_store

    async def delegate_streaming(self, agent_name: str, task: str) -> AsyncIterator[Event]:
        """Run a task on a child agent, yielding progress events to the parent.

        This is the fix for the "silent delegation" bug. The parent terminal
        sees DelegationEvents showing what the child is doing in real time.
        """
        config = self.agents.get(agent_name)
        if not config:
            yield ErrorEvent(f"Unknown agent: {agent_name}. Available: {list(self.agents.keys())}")
            return

        yield DelegationEvent(agent_name, "start", f"Delegating to {agent_name}...")

        # Build child via HarnessBuilder — NO delegate_agent to prevent loops
        from isaac.core.builder import HarnessBuilder
        child_config = AgentConfig(
            name=config.name,
            soul=config.soul,
            model=config.model,
            tools=config.tools,
            max_iterations=min(config.max_iterations, 15),
            context_budget=config.context_budget,
            cwd=config.cwd,
            scope=config.scope,
            computer_scope=config.computer_scope,
            sandbox="",  # Sub-agents are ephemeral — no VM
        )

        builder = HarnessBuilder(config.name)
        builder.with_config(child_config)
        builder.with_memory(self.memory, self.embedding_store)
        builder.with_soul(mode="minimal")
        builder.without_delegation()  # Prevent infinite recursion
        builder.without_gateway()  # Child doesn't need gateway overhead

        try:
            harness = await builder.build()
        except Exception as e:
            yield ErrorEvent(f"Failed to build child harness for {agent_name}: {e}")
            return

        # Remove delegate_agent from child registry
        harness.config.tool_registry.pop("delegate_agent", None)

        # Auto-approve everything in child context
        gate = harness.config.permission_gate
        for name in harness.config.tool_registry:
            gate.set_override(name, PermissionLevel.AUTO)

        state = new_session(config.name)
        orch = Orchestrator(
            agent_config=child_config,
            tool_registry=harness.config.tool_registry,
            permission_gate=gate,
            llm_client=harness.config.llm_client,
            memory_fn=harness.config.memory_fn,
            soul_mode="minimal",
        )

        response_parts: list[str] = []
        total_cost = 0.0
        tool_count = 0
        start_time = time.time()

        async for event in orch.run(task, state):
            if isinstance(event, TextEvent) and event.text:
                response_parts.append(event.text)

            elif isinstance(event, ToolCallEvent):
                tool_count += 1
                yield DelegationEvent(
                    agent_name, "tool",
                    f"[{agent_name}] → {event.tool_call.name}",
                )

            elif isinstance(event, ToolResultEvent):
                status = "✗" if event.result.is_error else "✓"
                preview = event.result.content[:80] if event.result.content else ""
                yield DelegationEvent(
                    agent_name, "result",
                    f"[{agent_name}] {status} {event.tool_name}: {preview}",
                )

            elif isinstance(event, CostEvent):
                total_cost += event.cost

            elif isinstance(event, ErrorEvent):
                yield DelegationEvent(agent_name, "error", f"[{agent_name}] Error: {event.error}")

        elapsed = time.time() - start_time
        response = "\n".join(response_parts)

        yield DelegationEvent(
            agent_name, "done",
            f"[{agent_name}] Done in {elapsed:.0f}s | {tool_count} tools | ${total_cost:.4f}",
        )

        # Yield the final text response as a regular TextEvent
        # so the parent LLM sees the result in its tool_result
        if response:
            yield TextEvent(response)

    async def delegate(self, agent_name: str, task: str) -> dict[str, Any]:
        """Non-streaming delegate for backward compat. Collects all events."""
        response_parts: list[str] = []
        total_cost = 0.0

        async for event in self.delegate_streaming(agent_name, task):
            if isinstance(event, TextEvent) and event.text:
                response_parts.append(event.text)
            elif isinstance(event, CostEvent):
                total_cost += event.cost

        return {
            "agent": agent_name,
            "response": "\n".join(response_parts),
            "cost": total_cost,
        }

    def get_exposable_tools(self) -> dict[str, tuple[ToolDef, Any]]:
        """Generate tool entries for agents with expose_as_tool=True."""
        tools: dict[str, tuple[ToolDef, Any]] = {}

        for name, config in self.agents.items():
            if not config.expose_as_tool:
                continue

            tool_name = f"agent_{name}"
            description = config.tool_description or f"Delegate a task to the {name} agent"

            async def handler(task: str, _agent_name: str = name) -> dict[str, Any]:
                return await self.delegate(_agent_name, task)

            tools[tool_name] = (
                ToolDef(
                    name=tool_name,
                    description=description,
                    input_schema={
                        "type": "object",
                        "properties": {"task": {"type": "string", "description": "Task for the agent"}},
                        "required": ["task"],
                    },
                    permission=PermissionLevel.AUTO,
                ),
                handler,
            )

        return tools
