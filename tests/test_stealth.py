from __future__ import annotations

import io
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from ollama_agent.agent import AgentRuntime
from ollama_agent.interfaces.tui import OllamaREPL
from ollama_agent.interfaces.tui.commands import _is_immediate_command
from ollama_agent.interfaces.tui.completion import slash_completions
from ollama_agent.interfaces.tui.widgets import AgentHeader


class TestStealthMode(unittest.IsolatedAsyncioTestCase):
    """Unit tests for stealth mode functionality and visual indicators."""

    async def test_stealth_checkpointer_selection(self) -> None:
        sqlite_cp = MagicMock()
        with (
            patch("ollama_agent.agent.agent.load_instructions", return_value="base"),
            patch("ollama_agent.agent.agent.ensure_model_supports_tools", AsyncMock()),
            patch("ollama_agent.agent.agent.create_ollama_chat_model", AsyncMock(return_value=MagicMock(num_ctx=8192))),
            patch("ollama_agent.agent.agent.create_summarization_tool_middleware", return_value=MagicMock()),
            patch("ollama_agent.agent.agent.load_main_mcp_tools", AsyncMock(return_value=[])),
            patch("ollama_agent.agent.agent._ensure_agent_dirs", MagicMock()),
            patch("ollama_agent.agent.agent._locate_agents_files", MagicMock(return_value=(False, None))),
            patch("ollama_agent.agent.agent.create_deep_agent") as mock_cda,
            patch.object(AgentRuntime, "_sqlite_checkpointer", AsyncMock(return_value=sqlite_cp)) as mock_sql_cp,
        ):
            runtime_normal = AgentRuntime(stealth_mode=False)
            await runtime_normal._build_graph()
            mock_sql_cp.assert_awaited_once()
            assert mock_cda.call_args.kwargs["checkpointer"] is sqlite_cp

            mock_sql_cp.reset_mock()
            runtime_stealth = AgentRuntime(stealth_mode=True)
            await runtime_stealth._build_graph()
            mock_sql_cp.assert_not_awaited()
            cp1 = mock_cda.call_args.kwargs["checkpointer"]
            assert isinstance(cp1, MemorySaver)

            await runtime_stealth._build_graph()
            cp2 = mock_cda.call_args.kwargs["checkpointer"]
            assert cp1 is cp2

    async def test_handle_stealth_command_toggle_and_values(self) -> None:
        out = io.StringIO()
        console = Console(file=out)
        runtime: Any = AgentRuntime(stealth_mode=False)
        repl = OllamaREPL(runtime=runtime)
        repl.console = console

        with patch.object(AgentRuntime, "reload", AsyncMock()) as mock_reload:
            # Toggle on
            await repl._handle_stealth_cmd([])
            assert runtime.stealth_mode
            mock_reload.assert_awaited_once()
            assert "Stealth mode is now on" in out.getvalue()

            # Toggle off
            mock_reload.reset_mock()
            await repl._handle_stealth_cmd([])
            assert not runtime.stealth_mode
            mock_reload.assert_awaited_once()
            assert "Stealth mode is now off" in out.getvalue()

            # Explicit on (with different truthy values)
            for val in ("on", "true", "yes", "1"):
                mock_reload.reset_mock()
                await repl._handle_stealth_cmd([val])
                assert runtime.stealth_mode
                mock_reload.assert_awaited_once()

            # Explicit off (with different falsy values)
            for val in ("off", "false", "no", "0"):
                mock_reload.reset_mock()
                await repl._handle_stealth_cmd([val])
                assert not runtime.stealth_mode
                mock_reload.assert_awaited_once()

            # Invalid argument
            mock_reload.reset_mock()
            out.seek(0)
            out.truncate(0)
            await repl._handle_stealth_cmd(["invalid_arg"])
            assert not runtime.stealth_mode
            mock_reload.assert_not_awaited()
            assert "Usage: /stealth [on|off]" in out.getvalue()

    def test_immediate_command_and_autocomplete(self) -> None:
        assert _is_immediate_command("/stealth")
        assert _is_immediate_command("/stealth on")
        assert _is_immediate_command("/stealth off")

        root_cmds = [cmd for cmd, _ in slash_completions("/stealth", MagicMock(), ())]
        assert "/stealth" in root_cmds

        sub_cmds = [cmd for cmd, _ in slash_completions("/stealth ", MagicMock(), ())]
        assert "/stealth on" in sub_cmds
        assert "/stealth off" in sub_cmds

    def test_agent_header_badges(self) -> None:
        runtime = AgentRuntime(stealth_mode=False, yolo_mode=False)
        repl = MagicMock()
        repl.runtime = runtime
        repl._rag_ctx = None

        header = AgentHeader(repl=repl)
        header.update = MagicMock()  # type: ignore[method-assign]

        # 1. Normal (both off)
        header.update_header()
        rendered_text = header.update.call_args[0][0]
        assert "YOLO: OFF" in rendered_text
        assert "STEALTH: OFF" in rendered_text

        # 2. YOLO on, Stealth off
        runtime.yolo_mode = True
        runtime.stealth_mode = False
        header.update_header()
        rendered_text = header.update.call_args[0][0]
        assert "YOLO: ON" in rendered_text
        assert "STEALTH: OFF" in rendered_text

        # 3. YOLO off, Stealth on
        runtime.yolo_mode = False
        runtime.stealth_mode = True
        header.update_header()
        rendered_text = header.update.call_args[0][0]
        assert "YOLO: OFF" in rendered_text
        assert "STEALTH: ON" in rendered_text

        # 4. Both on
        runtime.yolo_mode = True
        runtime.stealth_mode = True
        header.update_header()
        rendered_text = header.update.call_args[0][0]
        assert "YOLO: ON" in rendered_text
        assert "STEALTH: ON" in rendered_text
