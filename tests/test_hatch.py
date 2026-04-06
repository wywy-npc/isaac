"""Tests for hatch onboarding and CLI UI components."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestHatch:
    def test_is_hatched_false(self):
        with tempfile.TemporaryDirectory() as td:
            with patch("isaac.cli.hatch.USER_FILE", Path(td) / "user.md"):
                from isaac.cli.hatch import is_hatched
                assert is_hatched() is False

    def test_is_hatched_true(self):
        with tempfile.TemporaryDirectory() as td:
            user_file = Path(td) / "user.md"
            user_file.write_text("name: Test\n")
            with patch("isaac.cli.hatch.USER_FILE", user_file):
                from isaac.cli.hatch import is_hatched
                assert is_hatched() is True

    def test_load_user_context_empty(self):
        with tempfile.TemporaryDirectory() as td:
            with patch("isaac.cli.hatch.USER_FILE", Path(td) / "user.md"):
                from isaac.cli.hatch import load_user_context
                assert load_user_context() == ""

    def test_load_user_context(self):
        with tempfile.TemporaryDirectory() as td:
            user_file = Path(td) / "user.md"
            user_file.write_text("# User Profile\nname: Wyatt\nrole: Engineer\n")
            with patch("isaac.cli.hatch.USER_FILE", user_file):
                from isaac.cli.hatch import load_user_context
                ctx = load_user_context()
                assert "Wyatt" in ctx
                assert "Engineer" in ctx

    def test_hatch_soul_contains_user_file_path(self):
        from isaac.cli.hatch import HATCH_SOUL, USER_FILE
        assert str(USER_FILE) in HATCH_SOUL

    def test_hatch_soul_instructs_file_write(self):
        from isaac.cli.hatch import HATCH_SOUL
        assert "file_write" in HATCH_SOUL
        assert "memory_write" in HATCH_SOUL

    def test_save_env(self):
        with tempfile.TemporaryDirectory() as td:
            env_file = Path(td) / ".env"
            with patch("isaac.cli.hatch.ENV_FILE", env_file):
                from isaac.cli.hatch import _save_env
                _save_env("sk-ant-test123")
                content = env_file.read_text()
                assert "ANTHROPIC_API_KEY=sk-ant-test123" in content

    def test_save_env_preserves_existing(self):
        with tempfile.TemporaryDirectory() as td:
            env_file = Path(td) / ".env"
            env_file.write_text("BRAVE_API_KEY=brave123\n")
            with patch("isaac.cli.hatch.ENV_FILE", env_file):
                from isaac.cli.hatch import _save_env
                _save_env("sk-ant-new")
                content = env_file.read_text()
                assert "ANTHROPIC_API_KEY=sk-ant-new" in content
                assert "BRAVE_API_KEY=brave123" in content

    def test_models_list(self):
        from isaac.cli.hatch import MODELS
        assert len(MODELS) == 3
        names = [m[1] for m in MODELS]
        assert "Sonnet" in names
        assert "Opus" in names
        assert "Haiku" in names


class TestUI:
    def test_banner_renders(self):
        from isaac.cli.ui import banner
        panel = banner(
            agent_name="lead",
            model="claude-sonnet-4-6",
            tool_count=15,
            session_id="abc123",
            services=3,
        )
        assert panel is not None

    def test_hatch_banner_renders(self):
        from isaac.cli.ui import hatch_banner
        panel = hatch_banner()
        assert panel is not None

    def test_hatch_complete_banner(self):
        from isaac.cli.ui import hatch_complete_banner
        panel = hatch_complete_banner("ISAAC", "Wyatt")
        assert panel is not None

    def test_status_panel(self):
        from isaac.cli.ui import status_panel
        panel = status_panel("lead", "claude-sonnet-4-6", 10000, 0.05, 5, 15)
        assert panel is not None

    def test_tool_table(self):
        from isaac.cli.ui import tool_table
        from isaac.core.types import ToolDef, PermissionLevel
        tools = {
            "test_tool": (
                ToolDef(name="test_tool", description="A test", input_schema={}, permission=PermissionLevel.AUTO),
                None,
            ),
        }
        panel = tool_table(tools)
        assert panel is not None

    def test_model_badge(self):
        from isaac.cli.ui import model_badge
        badge = model_badge("claude-sonnet-4-6")
        assert "SONNET" in badge

    def test_first_run_notice(self):
        from isaac.cli.ui import first_run_notice
        panel = first_run_notice()
        assert panel is not None

    def test_role_colors(self):
        from isaac.cli.ui import ROLE_COLORS
        assert "lead" in ROLE_COLORS
        assert "research" in ROLE_COLORS
        assert "default" in ROLE_COLORS
