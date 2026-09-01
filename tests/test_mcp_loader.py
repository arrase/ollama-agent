from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from ollama_agent.mcp.commands import (
    check_mcp_server,
    list_mcp_servers,
    reload_mcp_servers,
)
from ollama_agent.mcp.loader import (
    MCPConfigError,
    _build_mcp_connection,
    _read_main_config,
    _resolve_env,
    load_main_mcp_tools,
    load_subagent_mcp_tools,
)
from ollama_agent.settings.config import Settings, SubAgentMCPServer, SubAgentSettings


class TestMCPLoader(unittest.IsolatedAsyncioTestCase):
    """Unit tests for MCP tool loading and connection builders."""

    def test_resolve_env_empty(self) -> None:
        self.assertEqual(_resolve_env({}, "srv"), {})

    def test_resolve_env_success(self) -> None:
        with patch.dict("os.environ", {"API_KEY": "secret123", "PORT": "8080", "USERPROFILE": "C:\\Users\\Test"}):
            env = {
                "AUTH": "${API_KEY}",
                "URL": "http://localhost:${PORT}",
                "HOME_DIR": "%USERPROFILE%",
            }
            resolved = _resolve_env(env, "srv")
            self.assertEqual(
                resolved,
                {
                    "AUTH": "secret123",
                    "URL": "http://localhost:8080",
                    "HOME_DIR": "C:\\Users\\Test",
                },
            )

    def test_resolve_env_missing_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(MCPConfigError, "NONEXISTENT_KEY_XYZ"):
                _resolve_env({"AUTH": "${NONEXISTENT_KEY_XYZ}"}, "srv")
            with self.assertRaisesRegex(MCPConfigError, "NONEXISTENT_KEY_XYZ"):
                _resolve_env({"AUTH": "%NONEXISTENT_KEY_XYZ%"}, "srv")

    def test_build_mcp_connection_stdio_with_env(self) -> None:
        with patch.dict("os.environ", {"TOKEN": "mytoken"}):
            cfg = {
                "command": "npx",
                "args": ["-y", "my-server"],
                "env": {"API_TOKEN": "${TOKEN}"},
            }
            conn = _build_mcp_connection("srv", cfg)
            self.assertEqual(conn["transport"], "stdio")
            self.assertEqual(conn["command"], "npx")
            self.assertEqual(conn["args"], ["-y", "my-server"])
            self.assertEqual(conn["env"], {"API_TOKEN": "mytoken"})

    def test_build_mcp_connection_stdio_without_env(self) -> None:
        cfg = {
            "command": "uvx",
            "args": ["mcp-server-fetch"],
        }
        conn = _build_mcp_connection("srv", cfg)
        self.assertEqual(conn["transport"], "stdio")
        self.assertEqual(conn["command"], "uvx")
        self.assertEqual(conn["args"], ["mcp-server-fetch"])
        self.assertNotIn("env", conn)

    def test_build_mcp_connection_stdio_missing_env_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            cfg = {"command": "npx", "env": {"VAR": "${UNDEFINED_VAR}"}}
            with self.assertRaisesRegex(MCPConfigError, "UNDEFINED_VAR"):
                _build_mcp_connection("srv", cfg)

    def test_build_mcp_connection_null_args_raises(self) -> None:
        with self.assertRaisesRegex(MCPConfigError, "'args' must be a list"):
            _build_mcp_connection("srv", {"command": "x", "args": None})

    def test_build_mcp_connection_http_and_types(self) -> None:
        cfg_http = {
            "type": "http",
            "url": "https://mcp.example.com/mcp",
            "headers": {"Authorization": "Bearer 123"},
            "timeout": 10,
        }
        conn = _build_mcp_connection("srv", cfg_http)
        self.assertEqual(conn["transport"], "http")
        self.assertEqual(conn["url"], "https://mcp.example.com/mcp")
        self.assertEqual(conn["headers"], {"Authorization": "Bearer 123"})
        self.assertEqual(conn["timeout"], 10)

        cfg_sse = {
            "type": "sse",
            "url": "http://localhost:8000/sse",
        }
        conn_sse = _build_mcp_connection("srv", cfg_sse)
        self.assertEqual(conn_sse["transport"], "sse")
        self.assertEqual(conn_sse["url"], "http://localhost:8000/sse")

    def test_build_mcp_connection_default_transport_is_http_when_key_absent(self) -> None:
        conn = _build_mcp_connection("srv", {"url": "https://mcp.example.com/mcp"})
        self.assertEqual(conn["transport"], "http")

    def test_build_mcp_connection_unknown_transport_raises(self) -> None:
        with self.assertRaisesRegex(MCPConfigError, "unsupported transport 'ws'"):
            _build_mcp_connection("srv", {"url": "wss://mcp.example.com", "transport": "ws"})

    def test_build_mcp_connection_invalid_raises(self) -> None:
        with self.assertRaisesRegex(MCPConfigError, "requires either 'command' or 'url'"):
            _build_mcp_connection("srv", {"unknown": "value"})

    async def test_load_main_mcp_tools_missing_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "mcp.json"
            with patch("ollama_agent.mcp.loader.MCP_PATH", missing):
                tools = await load_main_mcp_tools()
                self.assertEqual(tools, [])

    async def test_load_main_mcp_tools_invalid_json_raises(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            tmp.write("{invalid-json")
            tmp_path = Path(tmp.name)

        try:
            with patch("ollama_agent.mcp.loader.MCP_PATH", tmp_path):
                with self.assertRaises(MCPConfigError):
                    await load_main_mcp_tools()
        finally:
            tmp_path.unlink(missing_ok=True)

    async def test_load_main_mcp_tools_non_object_root_raises(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            json.dump(["not", "an", "object"], tmp)
            tmp_path = Path(tmp.name)

        try:
            with patch("ollama_agent.mcp.loader.MCP_PATH", tmp_path):
                with self.assertRaisesRegex(MCPConfigError, "expected a JSON object"):
                    await _read_main_config()
        finally:
            tmp_path.unlink(missing_ok=True)

    async def test_load_main_mcp_tools_legacy_servers_key_ignored(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            json.dump({"servers": {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}}}, tmp)
            tmp_path = Path(tmp.name)

        try:
            with patch("ollama_agent.mcp.loader.MCP_PATH", tmp_path):
                servers_cfg = await _read_main_config()
                self.assertEqual(servers_cfg, {})
        finally:
            tmp_path.unlink(missing_ok=True)

    async def test_load_main_mcp_tools_malformed_entry_raises(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            json.dump({"mcpServers": {"broken": "not-an-object"}}, tmp)
            tmp_path = Path(tmp.name)

        try:
            with patch("ollama_agent.mcp.loader.MCP_PATH", tmp_path):
                with self.assertRaisesRegex(MCPConfigError, "'broken'"):
                    await load_main_mcp_tools()
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

            with patch("ollama_agent.mcp.loader.MCP_PATH", tmp_path), \
                 patch("ollama_agent.mcp.loader.MultiServerMCPClient", return_value=mock_client):
                tools = await load_main_mcp_tools()
                self.assertEqual(tools, [mock_tool])
        finally:
            tmp_path.unlink(missing_ok=True)

    async def test_load_main_mcp_tools_connection_failure_raises(self) -> None:
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
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(side_effect=RuntimeError("boom"))

            with patch("ollama_agent.mcp.loader.MCP_PATH", tmp_path), \
                 patch("ollama_agent.mcp.loader.MultiServerMCPClient", return_value=mock_client):
                with self.assertRaisesRegex(MCPConfigError, "fetch.*boom"):
                    await load_main_mcp_tools()
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

    async def test_load_subagent_mcp_tools_failure_raises(self) -> None:
        servers = [
            SubAgentMCPServer(
                name="git",
                command="git-mcp",
                args=["--verbose"],
                env={"GIT_USER": "${MISSING_VAR_XYZ}"},
            )
        ]
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(MCPConfigError, "MISSING_VAR_XYZ"):
                await load_subagent_mcp_tools("coder", servers)

    async def test_check_mcp_server_active(self) -> None:
        cfg = {"type": "http", "url": "https://mcp.example.com"}
        mock_tool = MagicMock()
        mock_tool.name = "search_tool"
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("ollama_agent.mcp.commands.MultiServerMCPClient", return_value=mock_client):
            status = await check_mcp_server("test-server", cfg)
            self.assertEqual(status.name, "test-server")
            self.assertEqual(status.status, "active")
            self.assertEqual(status.transport, "http")
            self.assertEqual(status.target, "https://mcp.example.com")
            self.assertEqual(status.tools, ["search_tool"])
            self.assertEqual(status.error, "")

    async def test_check_mcp_server_failed(self) -> None:
        cfg = {"command": "invalid-cmd"}
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=RuntimeError("Command not found"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("ollama_agent.mcp.commands.MultiServerMCPClient", return_value=mock_client):
            status = await check_mcp_server("bad-server", cfg)
            self.assertEqual(status.name, "bad-server")
            self.assertEqual(status.status, "failed")
            self.assertEqual(status.transport, "stdio")
            self.assertEqual(status.target, "invalid-cmd")
            self.assertIn("Command not found", status.error)

    async def test_check_mcp_server_invalid_config(self) -> None:
        status = await check_mcp_server("invalid-server", {"unknown": "bad"})
        self.assertEqual(status.status, "failed")
        self.assertIn("requires either 'command' or 'url'", status.error)

    async def test_list_mcp_servers_no_config(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "mcp.json"
            with patch("ollama_agent.mcp.loader.MCP_PATH", missing):
                await list_mcp_servers(console)
                out = console.export_text()
                self.assertIn("No MCP servers configured", out)

    async def test_list_mcp_servers_invalid_json_reports_error(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            tmp.write("{invalid-json")
            tmp_path = Path(tmp.name)

        try:
            with patch("ollama_agent.mcp.loader.MCP_PATH", tmp_path):
                await list_mcp_servers(console)
                out = console.export_text()
                self.assertIn("Failed to load MCP config", out)
        finally:
            tmp_path.unlink(missing_ok=True)

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
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            with patch("ollama_agent.mcp.loader.MCP_PATH", tmp_path), \
                 patch("ollama_agent.mcp.commands.MultiServerMCPClient", return_value=mock_client):
                await list_mcp_servers(console, settings=settings)
                out = console.export_text()
                self.assertIn("Model Context Protocol (MCP) Servers", out)
                self.assertIn("tavily", out)
                self.assertIn("Active", out)
                self.assertIn("tavily_search", out)
        finally:
            tmp_path.unlink(missing_ok=True)

    async def test_reload_mcp_servers_success(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime = MagicMock()
        runtime.reload = AsyncMock()
        runtime.settings = Settings()

        with patch("ollama_agent.mcp.commands.list_mcp_servers", AsyncMock()) as mock_list:
            await reload_mcp_servers(console, runtime)
            runtime.reload.assert_awaited_once()
            mock_list.assert_awaited_once_with(console, settings=runtime.settings)

    async def test_reload_mcp_servers_config_error(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime = MagicMock()
        runtime.reload = AsyncMock(side_effect=MCPConfigError("Invalid JSON in mcp.json"))
        runtime.settings = Settings()

        with self.assertRaisesRegex(MCPConfigError, "Invalid JSON in mcp.json"):
            await reload_mcp_servers(console, runtime)
        runtime.reload.assert_awaited_once()

    async def test_reload_mcp_servers_generic_error(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime = MagicMock()
        runtime.reload = AsyncMock(side_effect=RuntimeError("Connection failed"))
        runtime.settings = Settings()

        with self.assertRaises(RuntimeError):
            await reload_mcp_servers(console, runtime)
        runtime.reload.assert_awaited_once()
