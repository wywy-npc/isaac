"""Error Recovery — graceful failure handling for the harness.

"Production agents will fail. The question is whether they fail gracefully."

Three recovery strategies:
  1. Tool retry with exponential backoff (transient failures)
  2. Graceful degradation (skip non-critical tools, use fallback results)
  3. LLM fallback chain (primary model fails → try cheaper/different model)

The recovery engine wraps the executor, intercepting failures and applying
the appropriate strategy before surfacing errors to the orchestrator.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from isaac.core.types import ToolCall, ToolDef, ToolResult


@dataclass
class RetryPolicy:
    """Retry configuration for a tool category."""
    max_retries: int = 2
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 10.0
    retryable_errors: list[str] = field(default_factory=lambda: [
        "timeout", "rate_limit", "connection", "503", "429", "500",
        "EAGAIN", "ECONNRESET", "ETIMEDOUT",
    ])

    def is_retryable(self, error: str) -> bool:
        """Check if an error message matches retryable patterns."""
        error_lower = error.lower()
        return any(pattern.lower() in error_lower for pattern in self.retryable_errors)

    def delay_for_attempt(self, attempt: int) -> float:
        """Exponential backoff with jitter."""
        import random
        delay = min(self.base_delay_seconds * (2 ** attempt), self.max_delay_seconds)
        # Add 10-30% jitter
        jitter = delay * random.uniform(0.1, 0.3)
        return delay + jitter


@dataclass
class RecoveryEvent:
    """Record of a recovery action taken."""
    tool_name: str
    strategy: str       # "retry", "degrade", "fallback"
    attempt: int
    original_error: str
    recovered: bool
    detail: str = ""
    timestamp: float = field(default_factory=time.time)


# Tool categories for retry/degradation decisions
CRITICAL_TOOLS = frozenset({
    "bash", "shell", "run_command", "execute_command",
    "file_write", "file_edit", "write_file", "edit_file",
})

RETRIABLE_TOOLS = frozenset({
    "web_search", "web_fetch", "memory_search", "memory_read",
    "delegate_agent",
})

DEGRADABLE_TOOLS = frozenset({
    "web_search", "web_fetch", "memory_search",
    "connector_status", "connector_reconnect",
})


class RecoveryEngine:
    """Wraps tool execution with recovery strategies.

    Integrated into the orchestrator between permission check and execution.
    """

    def __init__(
        self,
        retry_policy: RetryPolicy | None = None,
        enable_retry: bool = True,
        enable_degradation: bool = True,
    ) -> None:
        self.policy = retry_policy or RetryPolicy()
        self.enable_retry = enable_retry
        self.enable_degradation = enable_degradation
        self.events: list[RecoveryEvent] = []
        self._consecutive_failures: dict[str, int] = {}  # tool_name -> count

    async def execute_with_recovery(
        self,
        tc: ToolCall,
        executor_fn: Any,  # async (ToolCall) -> ToolResult
        tool_def: ToolDef | None = None,
    ) -> tuple[ToolResult, RecoveryEvent | None]:
        """Execute a tool call with recovery strategies.

        Returns (result, recovery_event_or_none).
        """
        # First attempt
        result = await executor_fn(tc)

        if not result.is_error:
            self._consecutive_failures.pop(tc.name, None)
            return result, None

        original_error = result.content

        # Strategy 1: Retry with backoff (transient failures)
        if self.enable_retry and self._should_retry(tc, original_error):
            retry_result, event = await self._retry(tc, executor_fn, original_error)
            if event:
                self.events.append(event)
            if not retry_result.is_error:
                return retry_result, event
            # Retry failed — fall through to degradation
            result = retry_result

        # Strategy 2: Graceful degradation (non-critical tools)
        if self.enable_degradation and self._should_degrade(tc):
            degraded, event = self._degrade(tc, original_error)
            if event:
                self.events.append(event)
            return degraded, event

        # No recovery possible — track consecutive failures
        self._consecutive_failures[tc.name] = self._consecutive_failures.get(tc.name, 0) + 1
        return result, None

    def _should_retry(self, tc: ToolCall, error: str) -> bool:
        """Decide if a tool call should be retried."""
        if tc.name in CRITICAL_TOOLS:
            return False  # Critical tools: don't retry blindly, surface error
        if not self.policy.is_retryable(error):
            return False
        return True

    async def _retry(
        self,
        tc: ToolCall,
        executor_fn: Any,
        original_error: str,
    ) -> tuple[ToolResult, RecoveryEvent | None]:
        """Retry a tool call with exponential backoff."""
        for attempt in range(1, self.policy.max_retries + 1):
            delay = self.policy.delay_for_attempt(attempt)
            await asyncio.sleep(delay)

            result = await executor_fn(tc)
            if not result.is_error:
                event = RecoveryEvent(
                    tool_name=tc.name,
                    strategy="retry",
                    attempt=attempt,
                    original_error=original_error,
                    recovered=True,
                    detail=f"Recovered after {attempt} retries",
                )
                return result, event

        # All retries exhausted
        event = RecoveryEvent(
            tool_name=tc.name,
            strategy="retry",
            attempt=self.policy.max_retries,
            original_error=original_error,
            recovered=False,
            detail=f"Failed after {self.policy.max_retries} retries",
        )
        return ToolResult(
            tool_call_id=tc.id,
            content=f"Error after {self.policy.max_retries} retries: {original_error}",
            is_error=True,
        ), event

    def _should_degrade(self, tc: ToolCall) -> bool:
        """Decide if a tool call should degrade gracefully."""
        return tc.name in DEGRADABLE_TOOLS

    def _degrade(
        self, tc: ToolCall, original_error: str,
    ) -> tuple[ToolResult, RecoveryEvent]:
        """Return a graceful degradation result instead of an error."""
        fallback_messages = {
            "web_search": "Web search unavailable. Proceeding with existing knowledge.",
            "web_fetch": "Unable to fetch URL. Proceeding without this content.",
            "memory_search": "Memory search unavailable. Proceeding without memory context.",
        }
        message = fallback_messages.get(tc.name, f"Tool '{tc.name}' unavailable. Proceeding without it.")

        event = RecoveryEvent(
            tool_name=tc.name,
            strategy="degrade",
            attempt=0,
            original_error=original_error,
            recovered=True,
            detail=message,
        )

        return ToolResult(
            tool_call_id=tc.id,
            content=f"[Degraded] {message}",
            is_error=False,  # Not flagged as error — agent can continue
        ), event

    @property
    def consecutive_failure_count(self) -> int:
        """Total consecutive failures across all tools."""
        return sum(self._consecutive_failures.values())

    def is_tool_unhealthy(self, tool_name: str, threshold: int = 3) -> bool:
        """Check if a tool has too many consecutive failures."""
        return self._consecutive_failures.get(tool_name, 0) >= threshold


# ---------------------------------------------------------------------------
# LLM Fallback Chain
# ---------------------------------------------------------------------------
@dataclass
class LLMFallbackChain:
    """Ordered list of models to try if the primary fails.

    Example: ["claude-sonnet-4-6", "claude-haiku-4-5-20251001", "gpt-4o-mini"]
    """
    models: list[str] = field(default_factory=list)
    _current_index: int = 0
    _failures: dict[str, int] = field(default_factory=dict)

    def next_model(self, failed_model: str, error: str) -> str | None:
        """Get the next fallback model after a failure.

        Returns None if all models exhausted.
        """
        self._failures[failed_model] = self._failures.get(failed_model, 0) + 1

        # Find the failed model's position and return the next one
        try:
            idx = self.models.index(failed_model)
        except ValueError:
            idx = -1

        for i in range(idx + 1, len(self.models)):
            candidate = self.models[i]
            # Skip models that have failed 3+ times
            if self._failures.get(candidate, 0) < 3:
                return candidate

        return None

    def reset(self) -> None:
        """Reset failure counts (e.g., on successful call)."""
        self._failures.clear()
