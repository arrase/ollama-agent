from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from ollama_agent.mcp.commands import (
    MCPServerStatus,
    check_mcp_server,
    list_mcp_servers,
)
from ollama_agent.mcp.loader import (
    _build_mcp_connection,
    _resolve_env,
    get_mcp_config_path,
    load_main_mcp_tools,
    load_subagent_mcp_tools,
)
from ollama_agent.settings.config import Settings, SubAgentMCPServer, SubAgentSettings


class TestMCPLoader(unittest.IsolatedAsyncioTestCase):
    """Unit tests for MCP tool loading and connection builders."""

    def test_get_mcp_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mcp_json = tmp_path / "mcp.json"
            mcp_servers_json = tmp_path / "mcp_servers.json"

            with patch("ollama_agent.mcp.loader.MCP_PATH", mcp_json), \
                 patch("ollama_agent.mcp.loader.MCP_SERVERS_PATH", mcp_servers_json):
                # When neither exists, default to mcp.json
                self.assertEqual(get_mcp_config_path(), mcp_json)

                # When mcp_servers.json exists
                mcp_servers_json.write_text("{}")
                self.assertEqual(get_mcp_config_path(), mcp_servers_json)

                # When mcp.json exists, prefer mcp.json
                mcp_json.write_text("{}")
                self.assertEqual(get_mcp_config_path(), mcp_json)

    def test_resolve_env_empty(self) -> None:
        self.assertEqual(_resolve_env({}), {})

    def test_resolve_env_success(self) -> None:
        with patch.dict("os.environ", {"API_KEY": "secret123", "PORT": "8080", "USERPROFILE": "C:\\Users\\Test"}):
            env = {
                "AUTH": "${API_KEY}",
                "URL": "http://localhost:${PORT}",
                "HOME_DIR": "%USERPROFILE%",
            }
            resolved = _resolve_env(env)
            self.assertEqual(
                resolved,
                {
                    "AUTH": "secret123",
                    "URL": "http://localhost:8080",
                    "HOME_DIR": "C:\\Users\\Test",
                },
            )

    def test_resolve_env_missing_returns_none(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            env = {"AUTH": "${NONEXISTENT_KEY_XYZ}"}
            self.assertIsNone(_resolve_env(env))
            env_win = {"AUTH": "%NONEXISTENT_KEY_XYZ%"}
            self.assertIsNone(_resolve_env(env_win))

    def test_build_mcp_connection_stdio_with_env(self) -> None:
        with patch.dict("os.environ", {"TOKEN": "mytoken"}):
            cfg = {
                "command": "npx",
                "args": ["-y", "my-server"],
                "env": {"API_TOKEN": "${TOKEN}"},
            }
            conn = _build_mcp_connection(cfg)
            self.assertIsNotNone(conn)
            assert conn is not None
            self.assertEqual(conn["transport"], "stdio")
            self.assertEqual(conn["command"], "npx")
            self.assertEqual(conn["args"], ["-y", "my-server"])
            self.assertEqual(conn["env"], {"API_TOKEN": "mytoken"})

    def test_build_mcp_connection_stdio_without_env(self) -> None:
        cfg = {
            "command": "uvx",
            "args": ["mcp-server-fetch"],
        }
        conn = _build_mcp_connection(cfg)
        self.assertIsNotNone(conn)
        assert conn is not None
        self.assertEqual(conn["transport"], "stdio")
        self.assertEqual(conn["command"], "uvx")
        self.assertEqual(conn["args"], ["mcp-server-fetch"])
        self.assertNotIn("env", conn)

    def test_build_mcp_connection_stdio_missing_env_returns_none(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            cfg = {"command": "npx", "env": {"VAR": "${UNDEFINED_VAR}"}}
            self.assertIsNone(_build_mcp_connection(cfg))

    def test_build_mcp_connection_http_and_types(self) -> None:
        cfg_http = {
            "type": "http",
            "url": "https://mcp.example.com/mcp",
            "headers": {"Authorization": "Bearer 123"},
            "timeout": 10,
        }
        conn = _build_mcp_connection(cfg_http)
        self.assertIsNotNone(conn)
        assert conn is not None
        self.assertEqual(conn["transport"], "http")
        self.assertEqual(conn["url"], "https://mcp.example.com/mcp")
        self.assertEqual(conn["headers"], {"Authorization": "Bearer 123"})
        self.assertEqual(conn["timeout"], 10)

        cfg_sse = {
            "type": "sse",
            "url": "http://localhost:8000/sse",
        }
        conn_sse = _build_mcp_connection(cfg_sse)
        self.assertIsNotNone(conn_sse)
        assert conn_sse is not None
        self.assertEqual(conn_sse["transport"], "sse")
        self.assertEqual(conn_sse["url"], "http://localhost:8000/sse")

    def test_build_mcp_connection_invalid_returns_none(self) -> None:
        self.assertIsNone(_build_mcp_connection({"unknown": "value"}))

    async def test_load_main_mcp_tools_missing_file_returns_empty(self) -> None:
        with patch("ollama_agent.mcp.loader.get_mcp_config_path", return_value=Path("/tmp/nonexistent_mcp_path.json")):
            tools = await load_main_mcp_tools()
            self.assertEqual(tools, [])

    async def test_load_main_mcp_tools_invalid_json_returns_empty(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            tmp.write("{invalid-json")
            tmp_path = Path(tmp.name)

        try:
            with patch("ollama_agent.mcp.loader.get_mcp_config_path", return_value=tmp_path):
                tools = await load_main_mcp_tools()
                self.assertEqual(tools, [])
        finally:
            tmp_path.unlink(missing_ok=True)

    async def test_load_main_mcp_tools_success(self) -> None:
        config_data = {
            "mcpServers": {
                "fetch": {
                    "command": "uvx",
                    "args": ["mcp-server-fetch"],
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            json.dump(config_data, tmp)
            tmp_path = Path(tmp.name)

        try:
            mock_tool = MagicMock(name="fetch_tool")
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=[mock_tool])

            with patch("ollama_agent.mcp.loader.get_mcp_config_path", return_value=tmp_path), \
                 patch("ollama_agent.mcp.loader.MultiServerMCPClient", return_value=mock_client):
                tools = await load_main_mcp_tools()
                self.assertEqual(tools, [mock_tool])
        finally:
            tmp_path.unlink(missing_ok=True)

    async def test_load_subagent_mcp_tools_empty_servers(self) -> None:
        tools = await load_subagent_mcp_tools("test-agent", [])
        self.assertEqual(tools, [])

    async def test_load_subagent_mcp_tools_success(self) -> None:
        servers = [
            SubAgentMCPServer(
                name="git",
                command="git-mcp",
                args=["--verbose"],
                env={"GIT_USER": "test"},
            )
        ]
        mock_tool = MagicMock(name="git_tool")
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])

        with patch("ollama_agent.mcp.loader.MultiServerMCPClient", return_value=mock_client):
            tools = await load_subagent_mcp_tools("coder", servers)
            self.assertEqual(tools, [mock_tool])

    async def test_check_mcp_server_active(self) -> None:
        cfg = {"type": "http", "url": "https://mcp.example.com"}
        mock_tool = MagicMock()
        mock_tool.name = "search_tool"
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])

        with patch("ollama_agent.mcp.commands.MultiServerMCPClient", return_value=mock_client):
            status = await check_mcp_server("test-server", cfg)
            self.assertEqual(status.name, "test-server")
            self.assertEqual(status.status, "active")
            self.assertEqual(status.tools, ["search_tool"])
            self.assertEqual(status.error, "")

    async def test_check_mcp_server_failed(self) -> None:
        cfg = {"command": "invalid-cmd"}
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=RuntimeError("Command not found"))

        with patch("ollama_agent.mcp.commands.MultiServerMCPClient", return_value=mock_client):
            status = await check_mcp_server("bad-server", cfg)
            self.assertEqual(status.name, "bad-server")
            self.assertEqual(status.status, "failed")
            self.assertIn("Command not found", status.error)

    async def test_check_mcp_server_invalid_config(self) -> None:
        status = await check_mcp_server("invalid-server", {"unknown": "bad"})
        self.assertEqual(status.status, "failed")
        self.assertIn("Invalid configuration", status.error)

    async def test_list_mcp_servers_no_config(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        with patch("ollama_agent.mcp.commands.get_mcp_config_path", return_value=Path("/tmp/nonexistent.json")):
            await list_mcp_servers(console)
            out = console.export_text()
            self.assertIn("No MCP servers configured", out)

    async def test_list_mcp_servers_with_servers_and_subagents(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        config_data = {
            "mcpServers": {
                "tavily": {
                    "type": "http",
                    "url": "https://mcp.tavily.com",
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            json.dump(config_data, tmp)
            tmp_path = Path(tmp.name)

        try:
            settings = Settings()
            settings.subagents = [
                SubAgentSettings(
                    name="researcher",
                    description="Researcher",
                    mcp_servers=[
                        SubAgentMCPServer(name="sub-mcp", command="uvx", args=["sub-tool"])
                    ],
                )
            ]

            mock_tool = MagicMock()
            mock_tool.name = "tavily_search"
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=[mock_tool])

            with patch("ollama_agent.mcp.commands.get_mcp_config_path", return_value=tmp_path), \
                 patch("ollama_agent.mcp.commands.MultiServerMCPClient", return_value=mock_client):
                await list_mcp_servers(console, settings=settings)
                out = console.export_text()
                self.assertIn("Model Context Protocol (MCP) Servers", out)
                self.assertIn("tavily", out)
                self.assertIn("Active", out)
                self.assertIn("tavily_search", out)
        finally:
            tmp_path.unlink(missing_ok=True)
