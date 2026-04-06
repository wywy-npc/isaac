"""Tests for the model router — heuristic-based model selection."""
import pytest

from isaac.core.router import (
    HAIKU, SONNET, OPUS,
    route_model,
    _classify_message,
    _cap_model,
)


class TestClassifyMessage:
    def test_simple_yes(self):
        assert _classify_message("yes") == "simple"

    def test_simple_ok(self):
        assert _classify_message("ok") == "simple"

    def test_simple_do_it(self):
        assert _classify_message("do it") == "simple"

    def test_simple_y(self):
        assert _classify_message("y") == "simple"

    def test_standard_normal_request(self):
        assert _classify_message("add a login button to the sidebar") == "standard"

    def test_complex_architecture(self):
        assert _classify_message("architect a new caching system for the API layer") == "complex"

    def test_complex_security_audit(self):
        assert _classify_message("run a security audit on the auth module") == "complex"

    def test_complex_long_question(self):
        long_msg = "Can you explain why " + "the system behaves differently " * 20 + "?"
        assert _classify_message(long_msg) == "complex"

    def test_standard_medium_request(self):
        assert _classify_message("fix the bug in the login form validation") == "standard"


class TestCapModel:
    def test_cap_opus_to_sonnet_ceiling(self):
        assert _cap_model(OPUS, SONNET) == SONNET

    def test_haiku_under_sonnet_ceiling(self):
        assert _cap_model(HAIKU, SONNET) == HAIKU

    def test_sonnet_under_opus_ceiling(self):
        assert _cap_model(SONNET, OPUS) == SONNET

    def test_same_model_passes(self):
        assert _cap_model(SONNET, SONNET) == SONNET


class TestRouteModel:
    def test_first_iteration_simple_gets_haiku(self):
        model = route_model(OPUS, iteration=0, user_message="yes")
        assert model == HAIKU

    def test_first_iteration_standard_gets_sonnet(self):
        model = route_model(OPUS, iteration=0, user_message="add a button to the form")
        assert model == SONNET

    def test_first_iteration_complex_gets_opus(self):
        model = route_model(OPUS, iteration=0, user_message="architect the caching layer")
        assert model == OPUS

    def test_tool_use_iteration_gets_sonnet(self):
        model = route_model(OPUS, iteration=3, user_message="anything", has_pending_tool_calls=True)
        assert model == SONNET

    def test_sonnet_ceiling_blocks_opus(self):
        """Agent configured with Sonnet should never get Opus even for complex queries."""
        model = route_model(SONNET, iteration=0, user_message="architect the entire system")
        assert model == SONNET

    def test_haiku_ceiling_blocks_everything(self):
        """Agent configured with Haiku stays on Haiku."""
        model = route_model(HAIKU, iteration=0, user_message="architect the system")
        assert model == HAIKU

    def test_later_iterations_without_tools_get_sonnet(self):
        model = route_model(OPUS, iteration=2, user_message="add a button")
        assert model == SONNET

    def test_simple_with_sonnet_ceiling(self):
        model = route_model(SONNET, iteration=0, user_message="ok")
        assert model == HAIKU  # Haiku is below Sonnet ceiling, so allowed
