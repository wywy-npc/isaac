"""HarnessCore — the headless orchestration layer.

Zero UI imports. Consumers: CLI, SDK, gateway, delegation, heartbeat.
This is the CC-inspired "QueryEngine" equivalent for ISAAC.

Built on the 5 Harness Engineering pillars:
  1. Tool Orchestration — registry, concurrent execution, permissions
  2. Guardrails — destructive command detection, cost/time budgets, edit locks
  3. Error Recovery — retry with backoff, graceful degradation, LLM fallback
  4. Observability — structured audit trail, metrics, anomaly detection
  5. Feedback Loops — loop detection, self-verification, output validation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from isaac.core.feedback import FeedbackEngine
from isaac.core.guardrails import BudgetConfig, GuardrailEngine
from isaac.core.llm import LLMClient
from isaac.core.permissions import PermissionGate
from isaac.core.recovery import LLMFallbackChain, RecoveryEngine, RetryPolicy
from isaac.core.telemetry import TelemetryEngine
from isaac.core.types import AgentConfig, SessionState, ToolDef


@dataclass
class HarnessConfig:
    """Everything the harness needs to run. Fully injected — no imports, no side effects.

    The 5 harness pillars are all represented here:
      - tool_registry + permission_gate = Tool Orchestration
      - guardrails = Guardrails
      - recovery = Error Recovery
      - telemetry = Observability
      - feedback = Feedback Loops
    """
    agent_config: AgentConfig
    tool_registry: dict[str, tuple[ToolDef, Any]]
    permission_gate: PermissionGate
    llm_client: LLMClient
    memory_fn: Callable | None = None         # async (query: str) -> str
    approval_fn: Callable | None = None       # async (ToolCall) -> bool  (None = auto-approve)
    soul_mode: str = "full"                   # "full" for interactive, "minimal" for heartbeats
    soul_override: str = ""                   # inject custom soul text (e.g. hatch mode)
    event_handler: Callable | None = None     # optional sync callback for events
    sandbox_session: Any = None               # SandboxSession if VM is active
    sandbox_backend: Any = None               # Sandbox backend (Fly, E2B) for lifecycle mgmt
    connector_registry: Any = None            # ConnectorRegistry tracking MCP connector health + tools

    # --- Harness pillars (2-5) ---
    guardrails: GuardrailEngine | None = None
    recovery: RecoveryEngine | None = None
    telemetry: TelemetryEngine | None = None
    feedback: FeedbackEngine | None = None


class HarnessCore:
    """Zero-UI orchestration. The headless engine.

    All consumers (CLI, SDK, gateway, delegation) build a HarnessConfig
    via HarnessBuilder, then create a HarnessCore to run agents.
    """

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.session: SessionState | None = None

    async def run(self, message: str, state: SessionState) -> AsyncIterator:
        """Run the agentic loop. Yields Event objects.

        This is a thin wrapper that delegates to Orchestrator.run().
        The Orchestrator now takes an LLMClient instead of creating one.
        All 5 harness pillars are passed through.
        """
        from isaac.core.orchestrator import Orchestrator

        self.session = state
        orch = Orchestrator(
            agent_config=self.config.agent_config,
            tool_registry=self.config.tool_registry,
            permission_gate=self.config.permission_gate,
            llm_client=self.config.llm_client,
            memory_fn=self.config.memory_fn,
            approval_fn=self.config.approval_fn,
            soul_mode=self.config.soul_mode,
            soul_override=self.config.soul_override,
            connector_registry=self.config.connector_registry,
            # Harness pillars
            guardrails=self.config.guardrails,
            recovery=self.config.recovery,
            telemetry=self.config.telemetry,
            feedback=self.config.feedback,
        )

        async for event in orch.run(message, state):
            if self.config.event_handler:
                self.config.event_handler(event)
            yield event

    @property
    def state(self):
        """Immutable snapshot of current harness state."""
        from isaac.core.state import HarnessState
        return HarnessState.from_core(self)
