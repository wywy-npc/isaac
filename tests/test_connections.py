"""Tests for service connections — connections.yaml management."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from isaac.mcp.connections import (
    ServiceConnection,
    add_connection,
    list_connections,
    load_connections,
    remove_connection,
    save_connections,
)


class TestConnections:
    def test_load_empty(self):
        with tempfile.TemporaryDirectory() as td:
            with patch("isaac.mcp.connections.CONNECTIONS_FILE", Path(td) / "connections.yaml"):
                conns = load_connections()
                assert conns == {}

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as td:
            fpath = Path(td) / "connections.yaml"
            with patch("isaac.mcp.connections.CONNECTIONS_FILE", fpath):
                conns = {
                    "google": ServiceConnection(
                        name="google", command="npx", args=["@google/mcp"],
                        description="Google services",
                    ),
                    "slack": ServiceConnection(
                        name="slack", command="npx", args=["slack-mcp"],
                        enabled=False,
                    ),
                }
                save_connections(conns)
                assert fpath.exists()

                loaded = load_connections()
                assert "google" in loaded
                assert "slack" in loaded
                assert loaded["google"].command == "npx"
                assert loaded["google"].args == ["@google/mcp"]
                assert loaded["slack"].enabled is False

    def test_add_connection(self):
        with tempfile.TemporaryDirectory() as td:
            fpath = Path(td) / "connections.yaml"
            with patch("isaac.mcp.connections.CONNECTIONS_FILE", fpath):
                conn = add_connection(
                    name="granola",
                    command="npx",
                    args=["granola-mcp-server"],
                    description="Meeting notes",
                )
                assert conn.name == "granola"
                assert conn.enabled is True

                loaded = load_connections()
                assert "granola" in loaded

    def test_remove_connection(self):
        with tempfile.TemporaryDirectory() as td:
            fpath = Path(td) / "connections.yaml"
            with patch("isaac.mcp.connections.CONNECTIONS_FILE", fpath):
                add_connection(name="test", command="echo", args=["hello"])
                assert remove_connection("test") is True
                assert remove_connection("nonexistent") is False
                assert "test" not in load_connections()

    def test_list_connections(self):
        with tempfile.TemporaryDirectory() as td:
            fpath = Path(td) / "connections.yaml"
            with patch("isaac.mcp.connections.CONNECTIONS_FILE", fpath):
                add_connection(name="svc1", command="cmd1", description="First")
                add_connection(name="svc2", command="cmd2", description="Second")

                listed = list_connections()
                assert len(listed) == 2
                names = {s["name"] for s in listed}
                assert "svc1" in names
                assert "svc2" in names

    def test_env_var_expansion(self):
        with tempfile.TemporaryDirectory() as td:
            fpath = Path(td) / "connections.yaml"
            fpath.write_text("""
services:
  test:
    type: mcp
    command: npx
    args: ["some-server"]
    env:
      API_KEY: "${TEST_CONN_KEY}"
""")
            with patch("isaac.mcp.connections.CONNECTIONS_FILE", fpath):
                with patch.dict("os.environ", {"TEST_CONN_KEY": "secret123"}):
                    conns = load_connections()
                    assert conns["test"].env["API_KEY"] == "secret123"

    def test_http_transport(self):
        with tempfile.TemporaryDirectory() as td:
            fpath = Path(td) / "connections.yaml"
            with patch("isaac.mcp.connections.CONNECTIONS_FILE", fpath):
                conn = add_connection(
                    name="remote",
                    url="http://localhost:8200/mcp",
                    transport="http",
                )
                assert conn.transport == "http"
                assert conn.url == "http://localhost:8200/mcp"


class TestConnectionTools:
    def test_service_tools_registered(self):
        """connect_service, disconnect_service, list_services should be in registry."""
        from isaac.memory.store import MemoryStore
        from isaac.agents.tools import build_builtin_tools
        with tempfile.TemporaryDirectory() as td:
            store = MemoryStore(Path(td))
            registry = build_builtin_tools(store)
            assert "list_services" in registry
            assert "connect_service" in registry
            assert "disconnect_service" in registry
