from __future__ import annotations

import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from ollama_agent.agent import AgentRuntime
from ollama_agent.interfaces.tui import OllamaAgentApp, OllamaREPL
from ollama_agent.interfaces.tui.commands import _is_immediate_command
from ollama_agent.interfaces.tui.completion import (
    _get_root_commands,
    _get_subcommands,
)
from ollama_agent.interfaces.tui.widgets import AgentHeader


class TestStealthMode(unittest.IsolatedAsyncioTestCase):
    """Unit tests for stealth mode functionality and visual indicators."""

    async def test_stealth_checkpointer_selection(self) -> None:
        runtime_normal = AgentRuntime(stealth_mode=False)
        assert not runtime_normal.stealth_mode

        runtime_stealth = AgentRuntime(stealth_mode=True)
        assert runtime_stealth.stealth_mode
        mem_checkpointer = runtime_stealth._get_memory_checkpointer()
        assert isinstance(mem_checkpointer, MemorySaver)
        assert runtime_stealth._get_memory_checkpointer() is mem_checkpointer

    async def test_handle_stealth_command_toggle_and_values(self) -> None:
        out = io.StringIO()
        console = Console(file=out)
        runtime = AgentRuntime(stealth_mode=False)
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

        root_cmds = dict(_get_root_commands())
        assert "/stealth" in root_cmds

        subcmds = _get_subcommands()
        assert "/stealth" in subcmds
        sub_names = [name for name, _ in subcmds["/stealth"]]
        assert "on" in sub_names
        assert "off" in sub_names

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

    def test_update_mode_ui_four_states(self) -> None:
        runtime = AgentRuntime(stealth_mode=False, yolo_mode=False)
        repl = MagicMock()
        repl.runtime = runtime
        app = OllamaAgentApp(repl=repl)

        prompt_char = MagicMock()
        input_container = MagicMock()
        header = MagicMock()

        def mock_query_one(selector_or_type: object) -> MagicMock:
            if selector_or_type == "#prompt-char":
                return prompt_char
            if selector_or_type == "#input-container":
                return input_container
            if selector_or_type == AgentHeader:
                return header
            return MagicMock()

        app.query_one = mock_query_one  # type: ignore[method-assign]

        enabled = True
        disabled = False

        # 1. Normal (neither)
        runtime.yolo_mode = False
        runtime.stealth_mode = False
        app.update_mode_ui()
        input_container.set_class.assert_any_call(disabled, "yolo-mode")
        input_container.set_class.assert_any_call(disabled, "stealth-mode")
        assert prompt_char.styles.color == "#38bdf8"

        # 2. YOLO only
        runtime.yolo_mode = True
        runtime.stealth_mode = False
        app.update_mode_ui()
        input_container.set_class.assert_any_call(enabled, "yolo-mode")
        input_container.set_class.assert_any_call(disabled, "stealth-mode")
        assert prompt_char.styles.color == "#f87171"

        # 3. Stealth only
        runtime.yolo_mode = False
        runtime.stealth_mode = True
        app.update_mode_ui()
        input_container.set_class.assert_any_call(disabled, "yolo-mode")
        input_container.set_class.assert_any_call(enabled, "stealth-mode")
        assert prompt_char.styles.color == "#c084fc"

        # 4. Both YOLO and Stealth
        runtime.yolo_mode = True
        runtime.stealth_mode = True
        app.update_mode_ui()
        input_container.set_class.assert_any_call(enabled, "yolo-mode")
        input_container.set_class.assert_any_call(enabled, "stealth-mode")
        assert prompt_char.styles.color == "#fbbf24"
