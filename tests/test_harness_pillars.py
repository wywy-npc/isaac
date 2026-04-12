"""Tests for the 5 harness engineering pillars.

Covers: Guardrails, Telemetry, Recovery, Feedback.
(Pillar 1 — Tool Orchestration — already tested in test_core.py)
"""
import asyncio
import json
import os
import time

import pytest

from isaac.core.guardrails import (
    BudgetConfig,
    FreezeZone,
    GuardrailEngine,
    GuardrailVerdict,
    check_cost_budget,
    check_destructive_command,
    check_freeze_zones,
    check_time_budget,
)
from isaac.core.telemetry import (
    ActionType,
    AnomalyDetector,
    SessionMetrics,
    TelemetryEngine,
)
from isaac.core.recovery import (
    LLMFallbackChain,
    RecoveryEngine,
    RetryPolicy,
)
from isaac.core.feedback import (
    FeedbackEngine,
    LoopDetector,
    OutputValidator,
    iteration_budget_check,
)
from isaac.core.types import ToolCall, ToolDef, ToolResult, PermissionLevel


# ==========================================================================
# PILLAR 2: GUARDRAILS
# ==========================================================================

class TestDestructiveCommandDetection:
    def test_safe_commands_allowed(self):
        assert check_destructive_command("ls -la").verdict == GuardrailVerdict.ALLOW
        assert check_destructive_command("cat foo.txt").verdict == GuardrailVerdict.ALLOW
        assert check_destructive_command("python script.py").verdict == GuardrailVerdict.ALLOW
        assert check_destructive_command("git status").verdict == GuardrailVerdict.ALLOW
        assert check_destructive_command("npm install").verdict == GuardrailVerdict.ALLOW

    def test_rm_rf_detected(self):
        result = check_destructive_command("rm -rf /")
        assert result.verdict == GuardrailVerdict.WARN
        assert "destructive" in result.reason.lower()

    def test_rm_force_on_home(self):
        result = check_destructive_command("rm -rf ~/*")
        assert result.verdict == GuardrailVerdict.WARN

    def test_git_force_push(self):
        result = check_destructive_command("git push --force origin main")
        assert result.verdict == GuardrailVerdict.WARN
        assert "force push" in result.reason.lower()

    def test_git_reset_hard(self):
        result = check_destructive_command("git reset --hard HEAD~5")
        assert result.verdict == GuardrailVerdict.WARN

    def test_sql_drop(self):
        result = check_destructive_command("psql -c 'DROP TABLE users;'")
        assert result.verdict == GuardrailVerdict.WARN

    def test_sql_delete_without_where(self):
        result = check_destructive_command("DELETE FROM users")
        assert result.verdict == GuardrailVerdict.WARN

    def test_sql_delete_with_where_allowed(self):
        result = check_destructive_command("DELETE FROM users WHERE id = 5")
        assert result.verdict == GuardrailVerdict.ALLOW

    def test_docker_prune_all(self):
        result = check_destructive_command("docker system prune -a")
        assert result.verdict == GuardrailVerdict.WARN

    def test_kubectl_delete_namespace(self):
        result = check_destructive_command("kubectl delete namespace production")
        assert result.verdict == GuardrailVerdict.WARN

    def test_chmod_777_recursive(self):
        result = check_destructive_command("chmod -R 777 /var/www")
        assert result.verdict == GuardrailVerdict.WARN


class TestFreezeZones:
    def test_no_zones_allows_all(self):
        result = check_freeze_zones("/any/path.py", [])
        assert result.verdict == GuardrailVerdict.ALLOW

    def test_frozen_path_blocked(self):
        zones = [FreezeZone(path="/home/user/src", reason="debugging")]
        result = check_freeze_zones("/home/user/src/main.py", zones)
        assert result.verdict == GuardrailVerdict.BLOCK
        assert "frozen zone" in result.reason.lower()

    def test_outside_zone_allowed(self):
        zones = [FreezeZone(path="/home/user/src")]
        result = check_freeze_zones("/home/user/docs/readme.md", zones)
        assert result.verdict == GuardrailVerdict.ALLOW


class TestBudgetGuardrails:
    def test_within_budget(self):
        budget = BudgetConfig(max_cost_per_turn=1.0, max_cost_per_session=5.0)
        result = check_cost_budget(current_cost=2.0, turn_cost=0.5, budget=budget)
        assert result.verdict == GuardrailVerdict.ALLOW

    def test_turn_budget_exceeded(self):
        budget = BudgetConfig(max_cost_per_turn=0.10)
        result = check_cost_budget(current_cost=0.5, turn_cost=0.15, budget=budget)
        assert result.verdict == GuardrailVerdict.BLOCK
        assert "turn cost" in result.reason.lower()

    def test_session_budget_exceeded(self):
        budget = BudgetConfig(max_cost_per_session=1.0)
        result = check_cost_budget(current_cost=1.5, turn_cost=0.05, budget=budget)
        assert result.verdict == GuardrailVerdict.BLOCK
        assert "session cost" in result.reason.lower()

    def test_time_budget_exceeded(self):
        budget = BudgetConfig(max_time_seconds=60)
        result = check_time_budget(elapsed_seconds=90, budget=budget)
        assert result.verdict == GuardrailVerdict.BLOCK
        assert "time budget" in result.reason.lower()

    def test_time_budget_ok(self):
        budget = BudgetConfig(max_time_seconds=60)
        result = check_time_budget(elapsed_seconds=30, budget=budget)
        assert result.verdict == GuardrailVerdict.ALLOW

    def test_unlimited_budget(self):
        budget = BudgetConfig()  # all zeros = unlimited
        result = check_cost_budget(current_cost=100.0, turn_cost=50.0, budget=budget)
        assert result.verdict == GuardrailVerdict.ALLOW


class TestGuardrailEngine:
    def test_engine_checks_bash(self):
        engine = GuardrailEngine()
        result = engine.check_tool_call("bash", {"command": "rm -rf /"}, 0.0)
        assert result.verdict == GuardrailVerdict.WARN

    def test_engine_checks_freeze_zones(self):
        engine = GuardrailEngine()
        engine.add_freeze("/home/user/locked", "testing")
        result = engine.check_tool_call(
            "file_write", {"path": "/home/user/locked/file.py"}, 0.0,
        )
        assert result.verdict == GuardrailVerdict.BLOCK

    def test_engine_remove_freeze(self):
        engine = GuardrailEngine()
        engine.add_freeze("/tmp/locked")
        assert len(engine.freeze_zones) == 1
        removed = engine.remove_freeze("/tmp/locked")
        assert removed
        assert len(engine.freeze_zones) == 0

    def test_engine_cost_tracking(self):
        engine = GuardrailEngine(budget=BudgetConfig(max_cost_per_turn=0.10))
        engine.record_cost(0.05)
        engine.record_cost(0.06)
        # Turn cost now 0.11, exceeding 0.10 budget
        result = engine.check_tool_call("bash", {"command": "ls"}, 0.5)
        assert result.verdict == GuardrailVerdict.BLOCK

    def test_engine_reset_turn(self):
        engine = GuardrailEngine(budget=BudgetConfig(max_cost_per_turn=0.10))
        engine.record_cost(0.15)
        engine.reset_turn()
        result = engine.check_tool_call("bash", {"command": "ls"}, 0.5)
        assert result.verdict == GuardrailVerdict.ALLOW


# ==========================================================================
# PILLAR 3: ERROR RECOVERY
# ==========================================================================

class TestRetryPolicy:
    def test_retryable_errors(self):
        policy = RetryPolicy()
        assert policy.is_retryable("Connection timeout error")
        assert policy.is_retryable("Rate limit exceeded (429)")
        assert policy.is_retryable("503 Service Unavailable")
        assert not policy.is_retryable("File not found")
        assert not policy.is_retryable("Permission denied")

    def test_exponential_backoff(self):
        policy = RetryPolicy(base_delay_seconds=1.0, max_delay_seconds=10.0)
        d0 = policy.delay_for_attempt(0)
        d1 = policy.delay_for_attempt(1)
        d2 = policy.delay_for_attempt(2)
        # Each delay should be roughly 2x the previous (plus jitter)
        assert d1 > d0
        assert d2 > d1
        # Should not exceed max
        d10 = policy.delay_for_attempt(10)
        assert d10 <= 10.0 * 1.3  # max + jitter


class TestRecoveryEngine:
    @pytest.mark.asyncio
    async def test_no_recovery_on_success(self):
        engine = RecoveryEngine()

        async def good_executor(tc):
            return ToolResult(tool_call_id=tc.id, content="ok")

        tc = ToolCall(id="1", name="file_read", input={"path": "test.py"})
        result, event = await engine.execute_with_recovery(tc, good_executor)
        assert not result.is_error
        assert event is None

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self):
        engine = RecoveryEngine(retry_policy=RetryPolicy(
            max_retries=2, base_delay_seconds=0.01,
        ))
        call_count = 0

        async def flaky_executor(tc):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return ToolResult(tool_call_id=tc.id, content="Connection timeout", is_error=True)
            return ToolResult(tool_call_id=tc.id, content="ok")

        tc = ToolCall(id="1", name="web_search", input={"query": "test"})
        result, event = await engine.execute_with_recovery(tc, flaky_executor)
        assert not result.is_error
        assert event is not None
        assert event.strategy == "retry"
        assert event.recovered

    @pytest.mark.asyncio
    async def test_degrade_on_non_critical_failure(self):
        engine = RecoveryEngine()

        async def failing_executor(tc):
            return ToolResult(tool_call_id=tc.id, content="Some error", is_error=True)

        tc = ToolCall(id="1", name="web_search", input={"query": "test"})
        result, event = await engine.execute_with_recovery(tc, failing_executor)
        assert not result.is_error  # degraded, not error
        assert "Degraded" in result.content
        assert event is not None
        assert event.strategy == "degrade"

    @pytest.mark.asyncio
    async def test_no_retry_on_critical_tool(self):
        engine = RecoveryEngine()

        async def failing_executor(tc):
            return ToolResult(tool_call_id=tc.id, content="Permission denied", is_error=True)

        tc = ToolCall(id="1", name="bash", input={"command": "rm test"})
        result, event = await engine.execute_with_recovery(tc, failing_executor)
        assert result.is_error  # critical tools don't get retried or degraded
        assert event is None


class TestLLMFallbackChain:
    def test_fallback_order(self):
        chain = LLMFallbackChain(models=[
            "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "gpt-4o-mini",
        ])
        next_model = chain.next_model("claude-sonnet-4-6", "API error")
        assert next_model == "claude-haiku-4-5-20251001"

        next_model = chain.next_model("claude-haiku-4-5-20251001", "API error")
        assert next_model == "gpt-4o-mini"

        next_model = chain.next_model("gpt-4o-mini", "API error")
        assert next_model is None  # exhausted

    def test_skip_repeatedly_failed_models(self):
        chain = LLMFallbackChain(models=["a", "b", "c"])
        for _ in range(3):
            chain.next_model("a", "error")
        # 'b' should still be available
        assert chain.next_model("a", "error") == "b"
        for _ in range(3):
            chain.next_model("b", "error")
        assert chain.next_model("a", "error") == "c"

    def test_reset(self):
        chain = LLMFallbackChain(models=["a", "b"])
        for _ in range(3):
            chain.next_model("a", "error")
        chain.reset()
        assert chain.next_model("a", "error") == "b"  # b available again


# ==========================================================================
# PILLAR 4: OBSERVABILITY / TELEMETRY
# ==========================================================================

class TestSessionMetrics:
    def test_defaults(self):
        m = SessionMetrics(session_id="s1", agent_name="test")
        assert m.total_llm_calls == 0
        assert m.tool_success_rate == 1.0
        assert m.avg_cost_per_iteration == 0.0

    def test_success_rate(self):
        m = SessionMetrics(session_id="s1", agent_name="test")
        m.total_tool_calls = 10
        m.total_tool_errors = 2
        assert m.tool_success_rate == 0.8

    def test_summary(self):
        m = SessionMetrics(session_id="s1", agent_name="test")
        m.total_llm_calls = 5
        m.total_tool_calls = 20
        m.total_cost = 0.05
        s = m.summary()
        assert s["llm_calls"] == 5
        assert s["tool_calls"] == 20


class TestAnomalyDetector:
    def test_cost_spike(self):
        d = AnomalyDetector()
        for _ in range(5):
            d.on_llm_call(0.01, 0)  # build baseline
        anomaly = d.on_llm_call(0.10, 5)  # 10x spike
        assert anomaly is not None
        assert anomaly.type == "cost_spike"

    def test_stuck_loop(self):
        d = AnomalyDetector()
        for _ in range(7):
            d.on_tool_call("bash")
        anomaly = d.on_tool_call("bash")
        assert anomaly is not None
        assert anomaly.type == "stuck_loop"

    def test_error_burst(self):
        d = AnomalyDetector()
        for _ in range(4):
            d.on_error()
        anomaly = d.on_error()
        assert anomaly is not None
        assert anomaly.type == "error_burst"

    def test_no_anomaly_on_normal_behavior(self):
        d = AnomalyDetector()
        assert d.on_llm_call(0.01, 0) is None
        assert d.on_tool_call("file_read") is None
        assert d.on_tool_call("bash") is None
        assert d.on_error() is None


class TestTelemetryEngine:
    def test_record_llm_call(self, tmp_path):
        engine = TelemetryEngine("s1", "test", str(tmp_path))
        engine.record_llm_call(
            model="claude-sonnet-4-6", input_tokens=1000, output_tokens=500,
            cache_read=100, cost=0.01, latency_ms=500, iteration=0,
        )
        assert engine.metrics.total_llm_calls == 1
        assert engine.metrics.total_cost == 0.01
        assert engine.metrics.model_usage["claude-sonnet-4-6"] == 1

        # Check audit log written
        log_file = tmp_path / "s1.audit.jsonl"
        assert log_file.exists()
        line = json.loads(log_file.read_text().strip())
        assert line["action_type"] == "llm_call"

    def test_record_tool_call(self, tmp_path):
        engine = TelemetryEngine("s1", "test", str(tmp_path))
        engine.record_tool_call("bash", {"command": "ls"}, 0)
        assert engine.metrics.total_tool_calls == 1
        assert engine.metrics.tool_usage["bash"] == 1

    def test_record_tool_error(self, tmp_path):
        engine = TelemetryEngine("s1", "test", str(tmp_path))
        engine.record_tool_result("bash", success=False, error="fail", iteration=0)
        assert engine.metrics.total_tool_errors == 1
        assert engine.metrics.tool_errors["bash"] == 1

    def test_record_guardrail(self, tmp_path):
        engine = TelemetryEngine("s1", "test", str(tmp_path))
        engine.record_guardrail("destructive_command", "block", "rm -rf /", 0)
        assert engine.metrics.total_guardrail_blocks == 1

    def test_no_log_dir(self):
        """Telemetry works without a log dir (in-memory only)."""
        engine = TelemetryEngine("s1", "test")
        engine.record_llm_call("model", 100, 50, 0, 0.001, 100, 0)
        assert engine.metrics.total_llm_calls == 1


# ==========================================================================
# PILLAR 5: FEEDBACK LOOPS
# ==========================================================================

class TestLoopDetector:
    def test_no_loop_on_varied_calls(self):
        d = LoopDetector()
        assert d.record_tool_call("a", {"x": 1}) is None
        assert d.record_tool_call("b", {"x": 2}) is None
        assert d.record_tool_call("c", {"x": 3}) is None
        assert d.record_tool_call("a", {"x": 4}) is None

    def test_exact_repeat_detected(self):
        d = LoopDetector(repeat_threshold=3)
        d.record_tool_call("bash", {"command": "cat /dev/null"})
        d.record_tool_call("bash", {"command": "cat /dev/null"})
        signal = d.record_tool_call("bash", {"command": "cat /dev/null"})
        assert signal is not None
        assert signal.type == "loop_detected"
        assert "same input" in signal.message.lower()

    def test_sequence_repeat_detected(self):
        d = LoopDetector(repeat_threshold=3)
        for _ in range(4):
            d.record_tool_call("file_read", {"path": "a.py"})
            d.record_tool_call("file_write", {"path": "a.py", "content": "x"})
        # Should detect A-B-A-B-A-B pattern
        # (may trigger on exact repeat first)
        assert d.total_loops_detected >= 1

    def test_error_loop_detected(self):
        d = LoopDetector()
        d.record_error("Error: file not found at /tmp/foo.txt")
        d.record_error("Error: file not found at /tmp/foo.txt")
        signal = d.record_error("Error: file not found at /tmp/foo.txt")
        assert signal is not None
        assert signal.type == "loop_detected"
        assert "same error" in signal.message.lower()


class TestOutputValidator:
    def test_empty_result_flagged(self):
        v = OutputValidator()
        signal = v.validate_tool_result("file_read", "", False)
        assert signal is not None
        assert signal.type == "quality_warning"

    def test_truncated_result_flagged(self):
        v = OutputValidator()
        signal = v.validate_tool_result("bash", "data... (truncated)", False)
        assert signal is not None
        assert "truncated" in signal.message.lower()

    def test_normal_result_ok(self):
        v = OutputValidator()
        signal = v.validate_tool_result("bash", "Hello world, this is output", False)
        assert signal is None

    def test_error_result_not_flagged(self):
        v = OutputValidator()
        signal = v.validate_tool_result("bash", "", True)
        assert signal is None  # errors handled by recovery


class TestIterationBudget:
    def test_early_iterations_no_warning(self):
        assert iteration_budget_check(5, 100) is None

    def test_75pct_warning(self):
        signal = iteration_budget_check(76, 100)
        assert signal is not None
        assert signal.severity == "warning"
        assert "remaining" in signal.message.lower()

    def test_90pct_critical(self):
        signal = iteration_budget_check(98, 100)
        assert signal is not None
        assert signal.severity == "critical"
        assert "urgent" in signal.message.lower()

    def test_unlimited_no_warning(self):
        assert iteration_budget_check(1000, 0) is None


class TestFeedbackEngine:
    def test_drain_pending(self):
        engine = FeedbackEngine()
        # Trigger a loop
        for _ in range(4):
            engine.on_tool_call("bash", {"cmd": "same"}, 0)
        signals = engine.drain_pending()
        assert len(signals) >= 1
        # Second drain should be empty
        assert len(engine.drain_pending()) == 0

    def test_iteration_budget_feedback(self):
        engine = FeedbackEngine()
        signal = engine.on_iteration(95, 100)
        assert signal is not None
        assert signal.type == "budget_warning"

    def test_response_validation(self):
        engine = FeedbackEngine()
        # Normal response
        signal = engine.on_response("Here is the definitive answer to your question.", 0)
        assert signal is None

    def test_total_signals_tracked(self):
        engine = FeedbackEngine()
        engine.on_iteration(95, 100)
        engine.on_iteration(96, 100)
        assert engine.total_signals == 2
