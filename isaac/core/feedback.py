"""Feedback Loops — self-correction and verification for the harness.

"LangChain's coding agent jumped from 52.8% to 66.5% by only changing
the harness, not the model — by adding a self-verification loop and
loop detection."

Three feedback mechanisms:
  1. Loop detection: detect when the agent is stuck repeating actions
  2. Self-verification: verify tool outputs meet expectations
  3. Output validation: check final responses for quality signals

Feedback is injected as system messages into the conversation, steering
the agent without requiring model changes.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeedbackSignal:
    """A feedback message to inject into the agent's context."""
    type: str           # "loop_detected", "verification_failed", "quality_warning"
    message: str        # human-readable message for the agent
    severity: str       # "info", "warning", "critical"
    iteration: int = 0
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loop Detection
# ---------------------------------------------------------------------------
class LoopDetector:
    """Detects when the agent is stuck in repetitive patterns.

    Tracks tool call sequences and content patterns to identify loops:
    - Exact tool sequence repeats (e.g., read → edit → read → edit → read → edit)
    - Same tool with same inputs repeated
    - Same error repeated across iterations
    """

    def __init__(self, window_size: int = 12, repeat_threshold: int = 3) -> None:
        self.window_size = window_size
        self.repeat_threshold = repeat_threshold
        self._tool_sequence: list[str] = []
        self._tool_inputs: list[tuple[str, str]] = []  # (tool_name, input_hash)
        self._errors: list[str] = []
        self._loop_count = 0

    def record_tool_call(self, tool_name: str, tool_input: dict) -> FeedbackSignal | None:
        """Record a tool call and check for loop patterns."""
        # Track sequence
        self._tool_sequence.append(tool_name)
        if len(self._tool_sequence) > self.window_size * 2:
            self._tool_sequence = self._tool_sequence[-self.window_size * 2:]

        # Track inputs (hash for privacy)
        input_key = _stable_hash(tool_name, tool_input)
        self._tool_inputs.append((tool_name, input_key))
        if len(self._tool_inputs) > self.window_size * 2:
            self._tool_inputs = self._tool_inputs[-self.window_size * 2:]

        # Check 1: Same tool+input repeated N times
        signal = self._check_exact_repeats()
        if signal:
            return signal

        # Check 2: Sequence pattern repeats (e.g., A-B-C-A-B-C)
        signal = self._check_sequence_repeats()
        if signal:
            return signal

        return None

    def record_error(self, error: str) -> FeedbackSignal | None:
        """Record an error and check for repeated error patterns."""
        # Normalize error
        normalized = re.sub(r'\d+', 'N', error[:200])
        self._errors.append(normalized)
        if len(self._errors) > 10:
            self._errors = self._errors[-10:]

        # Same error 3+ times
        counts = Counter(self._errors[-6:])
        for err, count in counts.items():
            if count >= 3:
                self._loop_count += 1
                return FeedbackSignal(
                    type="loop_detected",
                    severity="warning",
                    message=(
                        f"You've encountered the same error {count} times: '{err[:100]}'. "
                        "Try a fundamentally different approach instead of retrying the same action."
                    ),
                    data={"error": err, "count": count, "loop_number": self._loop_count},
                )
        return None

    def _check_exact_repeats(self) -> FeedbackSignal | None:
        """Check if the same tool+input was called N+ times recently."""
        if len(self._tool_inputs) < self.repeat_threshold:
            return None

        recent = self._tool_inputs[-self.window_size:]
        counts = Counter(recent)
        for (tool, _), count in counts.items():
            if count >= self.repeat_threshold:
                self._loop_count += 1
                return FeedbackSignal(
                    type="loop_detected",
                    severity="warning",
                    message=(
                        f"You've called '{tool}' with the same input {count} times. "
                        "This looks like a loop. Step back, reconsider your approach, "
                        "and try something different."
                    ),
                    data={"tool": tool, "count": count, "loop_number": self._loop_count},
                )
        return None

    def _check_sequence_repeats(self) -> FeedbackSignal | None:
        """Check if a tool sequence pattern repeats (e.g., A-B-A-B-A-B)."""
        seq = self._tool_sequence[-self.window_size:]
        if len(seq) < 6:
            return None

        # Check for repeating subsequences of length 2-4
        for pattern_len in range(2, 5):
            if len(seq) < pattern_len * self.repeat_threshold:
                continue

            # Extract the last `pattern_len` entries as the candidate pattern
            candidate = tuple(seq[-pattern_len:])
            repeat_count = 0

            for i in range(len(seq) - pattern_len, -1, -pattern_len):
                window = tuple(seq[i:i + pattern_len])
                if window == candidate:
                    repeat_count += 1
                else:
                    break

            if repeat_count >= self.repeat_threshold:
                self._loop_count += 1
                pattern_str = " → ".join(candidate)
                return FeedbackSignal(
                    type="loop_detected",
                    severity="warning",
                    message=(
                        f"Detected repeating pattern ({pattern_str}) x{repeat_count}. "
                        "You're stuck in a loop. Stop, think about what's different this time, "
                        "and choose a new strategy."
                    ),
                    data={"pattern": list(candidate), "repeats": repeat_count, "loop_number": self._loop_count},
                )

        return None

    @property
    def total_loops_detected(self) -> int:
        return self._loop_count


# ---------------------------------------------------------------------------
# Output Validation
# ---------------------------------------------------------------------------
class OutputValidator:
    """Validates tool outputs and agent responses for quality signals.

    Lightweight checks that don't require LLM calls:
    - Empty/trivial responses
    - Truncated output indicators
    - Error patterns in "successful" results
    - Suspicious content patterns
    """

    def validate_tool_result(
        self, tool_name: str, result: str, is_error: bool,
    ) -> FeedbackSignal | None:
        """Validate a tool result for quality issues."""
        if is_error:
            return None  # Errors are handled by recovery, not validation

        # Empty result from tools that should return content
        content_tools = {"web_search", "web_fetch", "memory_search", "memory_read", "file_read"}
        if tool_name in content_tools and len(result.strip()) < 5:
            return FeedbackSignal(
                type="quality_warning",
                severity="info",
                message=f"Tool '{tool_name}' returned empty/minimal content. The resource may not exist or may be empty.",
                data={"tool": tool_name, "result_length": len(result)},
            )

        # Truncation indicators
        truncation_patterns = [
            r"\.\.\.\s*\(truncated\)",
            r"\[output truncated\]",
            r"--- output cut off ---",
        ]
        for pattern in truncation_patterns:
            if re.search(pattern, result, re.IGNORECASE):
                return FeedbackSignal(
                    type="quality_warning",
                    severity="info",
                    message=f"Tool '{tool_name}' output was truncated. You may be missing important content at the end.",
                    data={"tool": tool_name, "pattern": "truncated"},
                )

        return None

    def validate_response(self, response_text: str) -> FeedbackSignal | None:
        """Validate the agent's final text response for quality."""
        if not response_text or len(response_text.strip()) < 10:
            return None  # Don't flag tool-only turns

        # Detect hedging/uncertainty that might indicate hallucination
        hedge_phrases = [
            "I'm not sure but",
            "I think it might be",
            "This could potentially",
            "I believe it should",
        ]
        hedge_count = sum(1 for phrase in hedge_phrases if phrase.lower() in response_text.lower())
        if hedge_count >= 2:
            return FeedbackSignal(
                type="quality_warning",
                severity="info",
                message="Response contains multiple hedging phrases. Consider using tools to verify claims before presenting them.",
                data={"hedge_count": hedge_count},
            )

        return None


# ---------------------------------------------------------------------------
# Iteration Budget Feedback
# ---------------------------------------------------------------------------
def iteration_budget_check(
    iteration: int, max_iterations: int
) -> FeedbackSignal | None:
    """Inject urgency feedback as the agent approaches iteration limits."""
    if max_iterations <= 0:
        return None  # unlimited

    remaining = max_iterations - iteration
    pct_used = iteration / max_iterations

    if pct_used >= 0.9 and remaining <= 3:
        return FeedbackSignal(
            type="budget_warning",
            severity="critical",
            message=(
                f"URGENT: Only {remaining} iterations remaining out of {max_iterations}. "
                "Wrap up immediately — summarize progress and present what you have."
            ),
            data={"remaining": remaining, "used_pct": round(pct_used, 2)},
        )
    elif pct_used >= 0.75:
        return FeedbackSignal(
            type="budget_warning",
            severity="warning",
            message=(
                f"{remaining} iterations remaining ({int(pct_used * 100)}% used). "
                "Start wrapping up — focus on the most important remaining items."
            ),
            data={"remaining": remaining, "used_pct": round(pct_used, 2)},
        )

    return None


# ---------------------------------------------------------------------------
# FeedbackEngine — unified controller
# ---------------------------------------------------------------------------
class FeedbackEngine:
    """Central feedback engine. Observes agent behavior and injects
    corrective signals into the conversation.

    Signals are surfaced as system-context messages in the orchestrator,
    giving the agent information to self-correct without external intervention.
    """

    def __init__(self) -> None:
        self.loop_detector = LoopDetector()
        self.output_validator = OutputValidator()
        self._signals: list[FeedbackSignal] = []
        self._pending: list[FeedbackSignal] = []  # signals to inject next iteration

    def on_tool_call(
        self, tool_name: str, tool_input: dict, iteration: int,
    ) -> FeedbackSignal | None:
        """Process a tool call through feedback checks."""
        signal = self.loop_detector.record_tool_call(tool_name, tool_input)
        if signal:
            signal.iteration = iteration
            self._signals.append(signal)
            self._pending.append(signal)
            return signal
        return None

    def on_tool_result(
        self, tool_name: str, result: str, is_error: bool, iteration: int,
    ) -> FeedbackSignal | None:
        """Process a tool result through feedback checks."""
        # Loop detection for errors
        if is_error:
            signal = self.loop_detector.record_error(result)
            if signal:
                signal.iteration = iteration
                self._signals.append(signal)
                self._pending.append(signal)
                return signal

        # Output validation
        signal = self.output_validator.validate_tool_result(tool_name, result, is_error)
        if signal:
            signal.iteration = iteration
            self._signals.append(signal)
            # Quality warnings are informational, don't always inject
            return signal

        return None

    def on_response(self, text: str, iteration: int) -> FeedbackSignal | None:
        """Process agent response through validation."""
        signal = self.output_validator.validate_response(text)
        if signal:
            signal.iteration = iteration
            self._signals.append(signal)
            return signal
        return None

    def on_iteration(self, iteration: int, max_iterations: int) -> FeedbackSignal | None:
        """Check iteration budget at the start of each loop."""
        signal = iteration_budget_check(iteration, max_iterations)
        if signal:
            signal.iteration = iteration
            self._signals.append(signal)
            self._pending.append(signal)
            return signal
        return None

    def drain_pending(self) -> list[FeedbackSignal]:
        """Get and clear pending feedback signals for injection."""
        signals = list(self._pending)
        self._pending.clear()
        return signals

    @property
    def total_signals(self) -> int:
        return len(self._signals)

    @property
    def total_loops_detected(self) -> int:
        return self.loop_detector.total_loops_detected

    @property
    def all_signals(self) -> list[FeedbackSignal]:
        return list(self._signals)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stable_hash(tool_name: str, tool_input: dict) -> str:
    """Create a stable hash of tool name + input for loop detection."""
    import hashlib
    import json
    raw = f"{tool_name}:{json.dumps(tool_input, sort_keys=True, default=str)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]
