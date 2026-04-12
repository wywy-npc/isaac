"""Unified harness state — frozen snapshot for external consumers.

The orchestrator mutates SessionState internally. External consumers
(CLI, SDK, dashboard) see immutable HarnessState snapshots.

Includes harness pillar health indicators:
  - guardrail_blocks: total blocked actions
  - recovery_count: total recovery actions taken
  - feedback_loops: total feedback signals generated
  - tool_success_rate: overall tool success rate
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaac.core.harness import HarnessCore


@dataclass(frozen=True)
class HarnessState:
    """Immutable snapshot of harness state, including pillar health."""
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

    # Harness pillar health
    guardrail_blocks: int = 0
    guardrail_warns: int = 0
    recovery_count: int = 0
    feedback_loops: int = 0
    tool_success_rate: float = 1.0
    anomaly_count: int = 0
    freeze_zones: tuple[str, ...] = ()

    @staticmethod
    def from_core(core: HarnessCore) -> HarnessState:
        session = core.session

        # Extract pillar metrics
        guardrail_blocks = 0
        guardrail_warns = 0
        freeze_zones: tuple[str, ...] = ()
        if core.config.guardrails:
            guardrail_warns = len(core.config.guardrails.warnings)
            freeze_zones = tuple(z.path for z in core.config.guardrails.freeze_zones)

        recovery_count = 0
        if core.config.recovery:
            recovery_count = len(core.config.recovery.events)

        feedback_loops = 0
        if core.config.feedback:
            feedback_loops = core.config.feedback.total_signals

        tool_success_rate = 1.0
        anomaly_count = 0
        if core.config.telemetry:
            tool_success_rate = core.config.telemetry.metrics.tool_success_rate
            guardrail_blocks = core.config.telemetry.metrics.total_guardrail_blocks
            anomaly_count = len(core.config.telemetry.detector.anomalies)

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
            guardrail_blocks=guardrail_blocks,
            guardrail_warns=guardrail_warns,
            recovery_count=recovery_count,
            feedback_loops=feedback_loops,
            tool_success_rate=tool_success_rate,
            anomaly_count=anomaly_count,
            freeze_zones=freeze_zones,
        )
