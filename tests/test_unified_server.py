"""Tests for the unified MCP server."""
import json
import tempfile
from pathlib import Path

import pytest

from isaac.mcp.unified_server import create_server, _registered_tools
from isaac.mcp.tool_loader import ToolLoader


class TestUnifiedServer:
    def test_create_server_empty_tools(self):
        """Server should start with just meta tools when no tool files exist."""
        with tempfile.TemporaryDirectory() as td:
            server = create_server(tools_dir=Path(td))
            assert server is not None
            # Should have the meta tools (reload_tools, list_tools)
            assert "reload_tools" in _registered_tools
            assert "list_tools" in _registered_tools

    def test_create_server_loads_tools(self):
        """Server should discover and register tool files."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # Write a test tool file
            (td_path / "test_tool.py").write_text('''
async def greet(name: str = "world") -> dict:
    return {"greeting": f"hello {name}"}

TOOLS = [
    {
        "name": "greet",
        "description": "Say hello",
        "params": {"name": str},
        "handler": greet,
    }
]
''')
            server = create_server(tools_dir=td_path)
            assert "greet" in _registered_tools

    def test_skips_underscore_prefixed_files(self):
        """Files starting with _ should be skipped."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "_private.py").write_text('''
TOOLS = [{"name": "hidden", "description": "Should not load", "params": {}, "handler": lambda: None}]
''')
            server = create_server(tools_dir=td_path)
            assert "hidden" not in _registered_tools

    def test_meta_tools_registered(self):
        """reload_tools and list_tools should always be available."""
        with tempfile.TemporaryDirectory() as td:
            server = create_server(tools_dir=Path(td))
            assert "reload_tools" in _registered_tools
            assert "list_tools" in _registered_tools
