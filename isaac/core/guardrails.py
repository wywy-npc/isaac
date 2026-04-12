"""Guardrails — deterministic safety layer for the harness.

Prevents agents from taking harmful actions. Operates at multiple levels:
  1. Command guardrails: detect destructive bash commands before execution
  2. Cost guardrails: enforce per-turn and per-session budgets
  3. Time guardrails: enforce wall-clock limits
  4. Freeze zones: lock directories from edits during debugging

Inspired by the harness engineering principle: "Guardrails are deterministic
rules that prevent agents from taking harmful actions."
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GuardrailVerdict(str, Enum):
    ALLOW = "allow"
    WARN = "warn"       # proceed but flag to user
    BLOCK = "block"     # stop execution, return error


@dataclass
class GuardrailResult:
    verdict: GuardrailVerdict
    reason: str = ""
    rule: str = ""      # which rule triggered


# ---------------------------------------------------------------------------
# Destructive command patterns
# ---------------------------------------------------------------------------
DESTRUCTIVE_PATTERNS: list[tuple[str, str]] = [
    # File system destruction
    (r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|.*--force).*(/|~|\$HOME|\.\.|\.)", "rm with force flag on broad path"),
    (r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+(/|~|\$HOME)", "recursive rm on root/home"),
    (r"\bmkfs\b", "filesystem format"),
    (r"\bdd\b.*\bof=/dev/", "dd writing to device"),
    (r"\bshred\b", "secure file deletion"),
    # Git destruction
    (r"\bgit\s+push\s+.*--force\b", "git force push"),
    (r"\bgit\s+push\s+-f\b", "git force push"),
    (r"\bgit\s+reset\s+--hard\b", "git reset hard"),
    (r"\bgit\s+clean\s+-[a-zA-Z]*f", "git clean with force"),
    (r"\bgit\s+checkout\s+\.\s*$", "git checkout . (discard all changes)"),
    # Database destruction
    (r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b", "SQL drop"),
    (r"\bTRUNCATE\s+TABLE\b", "SQL truncate"),
    (r"\bDELETE\s+FROM\b(?!.*\bWHERE\b)", "SQL delete without WHERE"),
    # Process/system
    (r"\bkill\s+-9\s+(-1|0)\b", "kill all processes"),
    (r"\bkillall\b", "kill all by name"),
    (r"\bchmod\s+-R\s+777\b", "recursive world-writable"),
    (r"\bchown\s+-R\b.*\b(root|/)\b", "recursive chown to root"),
    # Container/infra
    (r"\bdocker\s+system\s+prune\s+-a", "docker prune all"),
    (r"\bkubectl\s+delete\s+(namespace|ns)\b", "kubectl delete namespace"),
    # Environment
    (r"\bunset\s+(PATH|HOME|USER)\b", "unset critical env var"),
    (r'\bexport\s+PATH\s*=\s*["\']?\s*["\']?\s*$', "clear PATH"),
]

DESTRUCTIVE_COMPILED = [(re.compile(p, re.IGNORECASE), desc) for p, desc in DESTRUCTIVE_PATTERNS]


def check_destructive_command(command: str) -> GuardrailResult:
    """Check a bash command for destructive patterns."""
    for pattern, description in DESTRUCTIVE_COMPILED:
        if pattern.search(command):
            return GuardrailResult(
                verdict=GuardrailVerdict.WARN,
                reason=f"Potentially destructive: {description}",
                rule="destructive_command",
            )
    return GuardrailResult(verdict=GuardrailVerdict.ALLOW)


# ---------------------------------------------------------------------------
# Freeze zones — lock directories from edits
# ---------------------------------------------------------------------------
@dataclass
class FreezeZone:
    """A directory locked from edits. Used during debugging to prevent
    accidental changes outside the investigation scope."""
    path: str
    reason: str = ""
    created_at: float = field(default_factory=time.time)


def check_freeze_zones(file_path: str, zones: list[FreezeZone]) -> GuardrailResult:
    """Check if a file path falls within any frozen zone."""
    import os
    abs_path = os.path.abspath(file_path)
    for zone in zones:
        abs_zone = os.path.abspath(zone.path)
        if abs_path.startswith(abs_zone):
            return GuardrailResult(
                verdict=GuardrailVerdict.BLOCK,
                reason=f"Path is in frozen zone: {zone.path}" + (f" ({zone.reason})" if zone.reason else ""),
                rule="freeze_zone",
            )
    return GuardrailResult(verdict=GuardrailVerdict.ALLOW)


# ---------------------------------------------------------------------------
# Budget guardrails
# ---------------------------------------------------------------------------
@dataclass
class BudgetConfig:
    """Configurable budget limits for an agent session."""
    max_cost_per_turn: float = 0.0    # 0 = unlimited
    max_cost_per_session: float = 0.0  # 0 = unlimited
    max_time_seconds: int = 0          # 0 = unlimited
    max_iterations: int = 0            # 0 = unlimited (already in AgentConfig)


def check_cost_budget(
    current_cost: float,
    turn_cost: float,
    budget: BudgetConfig,
) -> GuardrailResult:
    """Check if cost budgets are exceeded."""
    if budget.max_cost_per_turn > 0 and turn_cost >= budget.max_cost_per_turn:
        return GuardrailResult(
            verdict=GuardrailVerdict.BLOCK,
            reason=f"Turn cost ${turn_cost:.4f} exceeds budget ${budget.max_cost_per_turn:.4f}",
            rule="cost_per_turn",
        )
    if budget.max_cost_per_session > 0 and current_cost >= budget.max_cost_per_session:
        return GuardrailResult(
            verdict=GuardrailVerdict.BLOCK,
            reason=f"Session cost ${current_cost:.4f} exceeds budget ${budget.max_cost_per_session:.4f}",
            rule="cost_per_session",
        )
    return GuardrailResult(verdict=GuardrailVerdict.ALLOW)


def check_time_budget(elapsed_seconds: float, budget: BudgetConfig) -> GuardrailResult:
    """Check if time budget is exceeded."""
    if budget.max_time_seconds > 0 and elapsed_seconds >= budget.max_time_seconds:
        return GuardrailResult(
            verdict=GuardrailVerdict.BLOCK,
            reason=f"Elapsed {elapsed_seconds:.0f}s exceeds time budget {budget.max_time_seconds}s",
            rule="time_budget",
        )
    return GuardrailResult(verdict=GuardrailVerdict.ALLOW)


# ---------------------------------------------------------------------------
# GuardrailEngine — unified entry point
# ---------------------------------------------------------------------------
class GuardrailEngine:
    """Central guardrail engine. Integrated into the orchestrator loop.

    Checks are fast and deterministic (no LLM calls). They run BEFORE
    each tool execution, acting as a safety net between the agent's
    intent and the real world.
    """

    def __init__(self, budget: BudgetConfig | None = None) -> None:
        self.budget = budget or BudgetConfig()
        self.freeze_zones: list[FreezeZone] = []
        self._session_start = time.time()
        self._turn_cost = 0.0
        self._warnings: list[GuardrailResult] = []

    def reset_turn(self) -> None:
        """Reset per-turn state (called at start of each user turn)."""
        self._turn_cost = 0.0

    def add_freeze(self, path: str, reason: str = "") -> FreezeZone:
        """Lock a directory from edits."""
        zone = FreezeZone(path=path, reason=reason)
        self.freeze_zones.append(zone)
        return zone

    def remove_freeze(self, path: str) -> bool:
        """Unlock a directory."""
        before = len(self.freeze_zones)
        self.freeze_zones = [z for z in self.freeze_zones if z.path != path]
        return len(self.freeze_zones) < before

    def check_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        session_cost: float,
    ) -> GuardrailResult:
        """Run all applicable guardrails on a tool call.

        Returns the most severe result (BLOCK > WARN > ALLOW).
        """
        results: list[GuardrailResult] = []

        # 1. Destructive command check (bash, shell tools)
        if tool_name in ("bash", "shell", "run_command", "execute_command"):
            command = tool_input.get("command", "") or tool_input.get("cmd", "")
            if command:
                results.append(check_destructive_command(command))

        # 2. Freeze zone check (file write tools)
        if tool_name in ("file_write", "file_edit", "write_file", "edit_file", "patch_file"):
            path = tool_input.get("path", "") or tool_input.get("file_path", "")
            if path and self.freeze_zones:
                results.append(check_freeze_zones(path, self.freeze_zones))

        # 3. Cost budget
        results.append(check_cost_budget(session_cost, self._turn_cost, self.budget))

        # 4. Time budget
        elapsed = time.time() - self._session_start
        results.append(check_time_budget(elapsed, self.budget))

        # Return most severe
        for r in results:
            if r.verdict == GuardrailVerdict.BLOCK:
                return r
        for r in results:
            if r.verdict == GuardrailVerdict.WARN:
                self._warnings.append(r)
                return r
        return GuardrailResult(verdict=GuardrailVerdict.ALLOW)

    def record_cost(self, cost: float) -> None:
        """Track cost for budget enforcement."""
        self._turn_cost += cost

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self._session_start

    @property
    def warnings(self) -> list[GuardrailResult]:
        return list(self._warnings)
