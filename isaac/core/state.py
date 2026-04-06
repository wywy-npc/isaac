"""Unified harness state — frozen snapshot for external consumers.

The orchestrator mutates SessionState internally. External consumers
(CLI, SDK, dashboard) see immutable HarnessState snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaac.core.harness import HarnessCore


@dataclass(frozen=True)
class HarnessState:
    """Immutable snapshot of harness state."""
    session_id: str
    agent_name: str
    model: str
    turn_count: int
    total_tokens: int
    total_cost: float
    active_tools: tuple[str, ...]
    active_connections: tuple[str, ...]
    sandbox_status: str  # "none" | "active" | "sleeping"
    memory_node_count: int

    @staticmethod
    def from_core(core: HarnessCore) -> HarnessState:
        session = core.session
        return HarnessState(
            session_id=session.session_id if session else "",
            agent_name=core.config.agent_config.name,
            model=core.config.agent_config.model,
            turn_count=session.turn_count if session else 0,
            total_tokens=session.total_tokens if session else 0,
            total_cost=session.total_cost if session else 0.0,
            active_tools=tuple(core.config.tool_registry.keys()),
            active_connections=(),  # populated by builder if MCP connected
            sandbox_status="none",
            memory_node_count=0,
        )
