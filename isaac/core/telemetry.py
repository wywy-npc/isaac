"""Telemetry — structured observability for the harness.

"You cannot improve what you cannot see." Every agent action is logged
to a structured audit trail. Consumers (CLI dashboard, SDK, analytics)
can query the trail for debugging, cost analysis, and anomaly detection.

Three layers:
  1. Action log: every tool call, LLM call, guardrail check (JSONL file)
  2. Metrics: aggregated counters and gauges (in-memory, exportable)
  3. Anomaly detection: flag unusual patterns (cost spikes, stuck loops)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ActionType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    GUARDRAIL = "guardrail"
    ERROR = "error"
    COMPACTION = "compaction"
    RECOVERY = "recovery"
    FEEDBACK = "feedback"
    DELEGATION = "delegation"


@dataclass
class ActionRecord:
    """A single entry in the audit trail."""
    timestamp: float
    action_type: ActionType
    agent_name: str
    session_id: str
    iteration: int = 0

    # LLM fields
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost: float = 0.0
    latency_ms: int = 0

    # Tool fields
    tool_name: str = ""
    tool_input_keys: list[str] = field(default_factory=list)
    tool_success: bool = True
    tool_error: str = ""

    # Guardrail fields
    guardrail_rule: str = ""
    guardrail_verdict: str = ""

    # General
    detail: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action_type"] = self.action_type.value
        return d


@dataclass
class SessionMetrics:
    """Aggregated metrics for a session."""
    session_id: str
    agent_name: str
    start_time: float = field(default_factory=time.time)

    # Counters
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_tool_errors: int = 0
    total_guardrail_blocks: int = 0
    total_guardrail_warns: int = 0
    total_recoveries: int = 0
    total_compactions: int = 0
    total_feedback_loops: int = 0

    # Gauges
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cost: float = 0.0
    peak_iteration: int = 0

    # Tool usage breakdown
    tool_usage: dict[str, int] = field(default_factory=dict)  # tool_name -> call_count
    tool_errors: dict[str, int] = field(default_factory=dict)  # tool_name -> error_count
    tool_latency: dict[str, list[int]] = field(default_factory=dict)  # tool_name -> [ms]

    # Model usage breakdown
    model_usage: dict[str, int] = field(default_factory=dict)  # model -> call_count
    model_cost: dict[str, float] = field(default_factory=dict)  # model -> total_cost

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def avg_cost_per_iteration(self) -> float:
        if self.peak_iteration == 0:
            return 0.0
        return self.total_cost / self.peak_iteration

    @property
    def tool_success_rate(self) -> float:
        total = self.total_tool_calls
        if total == 0:
            return 1.0
        return (total - self.total_tool_errors) / total

    def summary(self) -> dict[str, Any]:
        """Compact summary for dashboards."""
        return {
            "session_id": self.session_id,
            "agent": self.agent_name,
            "elapsed_s": round(self.elapsed_seconds, 1),
            "iterations": self.peak_iteration,
            "llm_calls": self.total_llm_calls,
            "tool_calls": self.total_tool_calls,
            "tool_errors": self.total_tool_errors,
            "guardrail_blocks": self.total_guardrail_blocks,
            "recoveries": self.total_recoveries,
            "tokens_in": self.total_input_tokens,
            "tokens_out": self.total_output_tokens,
            "cache_read": self.total_cache_read_tokens,
            "cost": round(self.total_cost, 6),
            "success_rate": round(self.tool_success_rate, 3),
            "top_tools": sorted(
                self.tool_usage.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }


# ---------------------------------------------------------------------------
# Anomaly detection (lightweight, deterministic)
# ---------------------------------------------------------------------------
@dataclass
class Anomaly:
    """A detected anomaly."""
    type: str       # "cost_spike", "stuck_loop", "error_burst", "slow_tool"
    severity: str   # "info", "warning", "critical"
    message: str
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)


class AnomalyDetector:
    """Lightweight anomaly detection based on rolling windows."""

    def __init__(self) -> None:
        self._recent_costs: list[float] = []        # last N iteration costs
        self._recent_tools: list[str] = []           # last N tool names
        self._recent_errors: list[float] = []        # timestamps of recent errors
        self.anomalies: list[Anomaly] = []

    def on_llm_call(self, cost: float, iteration: int) -> Anomaly | None:
        """Check for cost anomalies after an LLM call."""
        self._recent_costs.append(cost)
        if len(self._recent_costs) > 20:
            self._recent_costs = self._recent_costs[-20:]

        # Cost spike: single call > 3x the rolling average
        if len(self._recent_costs) >= 3:
            avg = sum(self._recent_costs[:-1]) / len(self._recent_costs[:-1])
            if avg > 0 and cost > avg * 3:
                a = Anomaly(
                    type="cost_spike",
                    severity="warning",
                    message=f"Cost spike: ${cost:.4f} is {cost/avg:.1f}x the average ${avg:.4f}",
                    data={"cost": cost, "avg": avg, "iteration": iteration},
                )
                self.anomalies.append(a)
                return a
        return None

    def on_tool_call(self, tool_name: str) -> Anomaly | None:
        """Check for stuck loops (same tool called repeatedly)."""
        self._recent_tools.append(tool_name)
        if len(self._recent_tools) > 10:
            self._recent_tools = self._recent_tools[-10:]

        # Stuck loop: same tool called 6+ times in last 8 calls
        if len(self._recent_tools) >= 8:
            window = self._recent_tools[-8:]
            from collections import Counter
            counts = Counter(window)
            for name, count in counts.items():
                if count >= 6:
                    a = Anomaly(
                        type="stuck_loop",
                        severity="warning",
                        message=f"Possible stuck loop: '{name}' called {count}/8 recent times",
                        data={"tool": name, "count": count},
                    )
                    self.anomalies.append(a)
                    return a
        return None

    def on_error(self) -> Anomaly | None:
        """Check for error bursts."""
        now = time.time()
        self._recent_errors.append(now)
        # Keep last 60 seconds
        self._recent_errors = [t for t in self._recent_errors if now - t < 60]

        # Error burst: 5+ errors in 60 seconds
        if len(self._recent_errors) >= 5:
            a = Anomaly(
                type="error_burst",
                severity="critical",
                message=f"Error burst: {len(self._recent_errors)} errors in the last 60 seconds",
                data={"error_count": len(self._recent_errors)},
            )
            self.anomalies.append(a)
            return a
        return None


# ---------------------------------------------------------------------------
# TelemetryEngine — the unified observer
# ---------------------------------------------------------------------------
class TelemetryEngine:
    """Central telemetry engine. Observes the orchestrator and records everything.

    Lightweight — all operations are O(1) amortized. The JSONL log is append-only.
    """

    def __init__(
        self,
        session_id: str,
        agent_name: str,
        log_dir: str | None = None,
    ) -> None:
        self.metrics = SessionMetrics(session_id=session_id, agent_name=agent_name)
        self.detector = AnomalyDetector()
        self._log_path: Path | None = None

        if log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            self._log_path = log_dir_path / f"{session_id}.audit.jsonl"

    def _write(self, record: ActionRecord) -> None:
        """Append record to audit log."""
        if self._log_path:
            try:
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(record.to_dict(), default=str) + "\n")
            except Exception:
                pass  # telemetry must never crash the agent

    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read: int,
        cost: float,
        latency_ms: int,
        iteration: int,
    ) -> Anomaly | None:
        """Record an LLM API call."""
        self.metrics.total_llm_calls += 1
        self.metrics.total_input_tokens += input_tokens
        self.metrics.total_output_tokens += output_tokens
        self.metrics.total_cache_read_tokens += cache_read
        self.metrics.total_cost += cost
        self.metrics.peak_iteration = max(self.metrics.peak_iteration, iteration)

        # Model breakdown
        self.metrics.model_usage[model] = self.metrics.model_usage.get(model, 0) + 1
        self.metrics.model_cost[model] = self.metrics.model_cost.get(model, 0.0) + cost

        self._write(ActionRecord(
            timestamp=time.time(),
            action_type=ActionType.LLM_CALL,
            agent_name=self.metrics.agent_name,
            session_id=self.metrics.session_id,
            iteration=iteration,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cost=cost,
            latency_ms=latency_ms,
        ))

        return self.detector.on_llm_call(cost, iteration)

    def record_tool_call(
        self,
        tool_name: str,
        tool_input: dict,
        iteration: int,
    ) -> Anomaly | None:
        """Record a tool invocation."""
        self.metrics.total_tool_calls += 1
        self.metrics.tool_usage[tool_name] = self.metrics.tool_usage.get(tool_name, 0) + 1

        self._write(ActionRecord(
            timestamp=time.time(),
            action_type=ActionType.TOOL_CALL,
            agent_name=self.metrics.agent_name,
            session_id=self.metrics.session_id,
            iteration=iteration,
            tool_name=tool_name,
            tool_input_keys=list(tool_input.keys()),
        ))

        return self.detector.on_tool_call(tool_name)

    def record_tool_result(
        self,
        tool_name: str,
        success: bool,
        error: str = "",
        latency_ms: int = 0,
        iteration: int = 0,
    ) -> Anomaly | None:
        """Record a tool result."""
        if not success:
            self.metrics.total_tool_errors += 1
            self.metrics.tool_errors[tool_name] = self.metrics.tool_errors.get(tool_name, 0) + 1

        if tool_name not in self.metrics.tool_latency:
            self.metrics.tool_latency[tool_name] = []
        self.metrics.tool_latency[tool_name].append(latency_ms)

        self._write(ActionRecord(
            timestamp=time.time(),
            action_type=ActionType.TOOL_RESULT,
            agent_name=self.metrics.agent_name,
            session_id=self.metrics.session_id,
            iteration=iteration,
            tool_name=tool_name,
            tool_success=success,
            tool_error=error,
            latency_ms=latency_ms,
        ))

        if not success:
            return self.detector.on_error()
        return None

    def record_guardrail(
        self, rule: str, verdict: str, reason: str, iteration: int = 0,
    ) -> None:
        """Record a guardrail check."""
        if verdict == "block":
            self.metrics.total_guardrail_blocks += 1
        elif verdict == "warn":
            self.metrics.total_guardrail_warns += 1

        self._write(ActionRecord(
            timestamp=time.time(),
            action_type=ActionType.GUARDRAIL,
            agent_name=self.metrics.agent_name,
            session_id=self.metrics.session_id,
            iteration=iteration,
            guardrail_rule=rule,
            guardrail_verdict=verdict,
            detail=reason,
        ))

    def record_recovery(self, detail: str, iteration: int = 0) -> None:
        """Record a recovery action."""
        self.metrics.total_recoveries += 1
        self._write(ActionRecord(
            timestamp=time.time(),
            action_type=ActionType.RECOVERY,
            agent_name=self.metrics.agent_name,
            session_id=self.metrics.session_id,
            iteration=iteration,
            detail=detail,
        ))

    def record_compaction(self, summary: str, iteration: int = 0) -> None:
        """Record a compaction event."""
        self.metrics.total_compactions += 1
        self._write(ActionRecord(
            timestamp=time.time(),
            action_type=ActionType.COMPACTION,
            agent_name=self.metrics.agent_name,
            session_id=self.metrics.session_id,
            iteration=iteration,
            detail=summary[:500],
        ))

    def record_feedback(self, feedback_type: str, detail: str, iteration: int = 0) -> None:
        """Record a feedback loop event."""
        self.metrics.total_feedback_loops += 1
        self._write(ActionRecord(
            timestamp=time.time(),
            action_type=ActionType.FEEDBACK,
            agent_name=self.metrics.agent_name,
            session_id=self.metrics.session_id,
            iteration=iteration,
            detail=f"[{feedback_type}] {detail}",
        ))
