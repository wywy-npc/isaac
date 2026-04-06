"""ISAAC SDK — programmatic API for embedding the harness.

Usage:
    import asyncio
    from isaac.sdk import run, create_harness

    # One-shot
    result = asyncio.run(run("default", "What files are in this directory?"))

    # Session-based
    harness = asyncio.run(create_harness("default"))
    async for event in harness.run("List my memory nodes", state):
        print(event)
"""
from __future__ import annotations

from typing import Any

from isaac.agents.session import new_session
from isaac.core.builder import HarnessBuilder
from isaac.core.harness import HarnessCore
from isaac.core.orchestrator import TextEvent


async def run(agent: str = "default", message: str = "", **kwargs: Any) -> str:
    """One-shot: run a message through an agent, return the text response."""
    core = await HarnessBuilder(agent, **kwargs).build()
    state = new_session(agent)
    result = ""
    async for event in core.run(message, state):
        if isinstance(event, TextEvent):
            result += event.text
    return result


async def create_harness(
    agent: str = "default",
    cwd: str | None = None,
    **builder_kwargs: Any,
) -> HarnessCore:
    """Create a HarnessCore for session-based usage."""
    builder = HarnessBuilder(agent, cwd=cwd)
    return await builder.build()
