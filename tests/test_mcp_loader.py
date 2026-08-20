from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


from ollama_agent.mcp.loader import (
    _build_mcp_connection,
    _resolve_env,
    load_main_mcp_tools,
    load_subagent_mcp_tools,
)
from ollama_agent.settings.config import SubAgentMCPServer


class TestMCPLoader(unittest.IsolatedAsyncioTestCase):
    """Unit tests for MCP tool loading and connection builders."""

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

    def test_build_mcp_connection_stdio(self) -> None:
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

    def test_build_mcp_connection_stdio_missing_env_returns_none(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            cfg = {"command": "npx", "env": {"VAR": "${UNDEFINED_VAR}"}}
            self.assertIsNone(_build_mcp_connection(cfg))

    def test_build_mcp_connection_http(self) -> None:
        cfg = {
            "url": "http://localhost:8000/sse",
            "headers": {"Authorization": "Bearer 123"},
            "timeout": 10,
        }
        conn = _build_mcp_connection(cfg)
        self.assertIsNotNone(conn)
        assert conn is not None
        self.assertEqual(conn["transport"], "http")
        self.assertEqual(conn["url"], "http://localhost:8000/sse")
        self.assertEqual(conn["headers"], {"Authorization": "Bearer 123"})
        self.assertEqual(conn["timeout"], 10)


    def test_build_mcp_connection_invalid_returns_none(self) -> None:
        self.assertIsNone(_build_mcp_connection({"unknown": "value"}))

    async def test_load_main_mcp_tools_missing_file_returns_empty(self) -> None:
        with patch("ollama_agent.mcp.loader.MCP_SERVERS_PATH", Path("/tmp/nonexistent_mcp_path.json")):
            tools = await load_main_mcp_tools()
            self.assertEqual(tools, [])

    async def test_load_main_mcp_tools_invalid_json_returns_empty(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            tmp.write("{invalid-json")
            tmp_path = Path(tmp.name)

        try:
            with patch("ollama_agent.mcp.loader.MCP_SERVERS_PATH", tmp_path):
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

            with patch("ollama_agent.mcp.loader.MCP_SERVERS_PATH", tmp_path), \
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

