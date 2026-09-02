from __future__ import annotations

import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console
from langgraph.checkpoint.memory import MemorySaver

from ollama_agent.agent import AgentRuntime
from ollama_agent.interfaces.repl import (
    OllamaAgentApp,
    OllamaREPL,
    _get_root_commands,
    _get_subcommands,
    _is_immediate_command,
)
from ollama_agent.interfaces.tui_components import AgentHeader


class TestStealthMode(unittest.IsolatedAsyncioTestCase):
    """Unit tests for stealth mode functionality and visual indicators."""

    def test_stealth_checkpointer_selection(self) -> None:
        runtime_normal = AgentRuntime(stealth_mode=False)
        self.assertFalse(runtime_normal.stealth_mode)

        runtime_stealth = AgentRuntime(stealth_mode=True)
        self.assertTrue(runtime_stealth.stealth_mode)
        mem_checkpointer = runtime_stealth._get_memory_checkpointer()
        self.assertIsInstance(mem_checkpointer, MemorySaver)
        self.assertIs(runtime_stealth._get_memory_checkpointer(), mem_checkpointer)

    async def test_handle_stealth_command_toggle_and_values(self) -> None:
        out = io.StringIO()
        console = Console(file=out)
        runtime = AgentRuntime(stealth_mode=False)
        repl = OllamaREPL(runtime=runtime)
        repl.console = console

        with patch.object(AgentRuntime, "reload", AsyncMock()) as mock_reload:
            # Toggle on
            await repl._handle_stealth_cmd([])
            self.assertTrue(runtime.stealth_mode)
            mock_reload.assert_awaited_once()
            self.assertIn("Stealth mode is now on", out.getvalue())

            # Toggle off
            mock_reload.reset_mock()
            await repl._handle_stealth_cmd([])
            self.assertFalse(runtime.stealth_mode)
            mock_reload.assert_awaited_once()
            self.assertIn("Stealth mode is now off", out.getvalue())

            # Explicit on (with different truthy values)
            for val in ("on", "true", "yes", "1"):
                mock_reload.reset_mock()
                await repl._handle_stealth_cmd([val])
                self.assertTrue(runtime.stealth_mode)
                mock_reload.assert_awaited_once()

            # Explicit off (with different falsy values)
            for val in ("off", "false", "no", "0"):
                mock_reload.reset_mock()
                await repl._handle_stealth_cmd([val])
                self.assertFalse(runtime.stealth_mode)
                mock_reload.assert_awaited_once()

            # Invalid argument
            mock_reload.reset_mock()
            out.seek(0)
            out.truncate(0)
            await repl._handle_stealth_cmd(["invalid_arg"])
            self.assertFalse(runtime.stealth_mode)
            mock_reload.assert_not_awaited()
            self.assertIn("Usage: /stealth [on|off]", out.getvalue())

    def test_immediate_command_and_autocomplete(self) -> None:
        self.assertTrue(_is_immediate_command("/stealth"))
        self.assertTrue(_is_immediate_command("/stealth on"))
        self.assertTrue(_is_immediate_command("/stealth off"))

        root_cmds = dict(_get_root_commands())
        self.assertIn("/stealth", root_cmds)

        subcmds = _get_subcommands()
        self.assertIn("/stealth", subcmds)
        sub_names = [name for name, _ in subcmds["/stealth"]]
        self.assertIn("on", sub_names)
        self.assertIn("off", sub_names)

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
        self.assertIn("YOLO: OFF", rendered_text)
        self.assertIn("STEALTH: OFF", rendered_text)

        # 2. YOLO on, Stealth off
        runtime.yolo_mode = True
        runtime.stealth_mode = False
        header.update_header()
        rendered_text = header.update.call_args[0][0]
        self.assertIn("YOLO: ON", rendered_text)
        self.assertIn("STEALTH: OFF", rendered_text)

        # 3. YOLO off, Stealth on
        runtime.yolo_mode = False
        runtime.stealth_mode = True
        header.update_header()
        rendered_text = header.update.call_args[0][0]
        self.assertIn("YOLO: OFF", rendered_text)
        self.assertIn("STEALTH: ON", rendered_text)

        # 4. Both on
        runtime.yolo_mode = True
        runtime.stealth_mode = True
        header.update_header()
        rendered_text = header.update.call_args[0][0]
        self.assertIn("YOLO: ON", rendered_text)
        self.assertIn("STEALTH: ON", rendered_text)

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

        # 1. Normal (neither)
        runtime.yolo_mode = False
        runtime.stealth_mode = False
        app.update_mode_ui()
        input_container.set_class.assert_any_call(False, "yolo-mode")
        input_container.set_class.assert_any_call(False, "stealth-mode")
        self.assertEqual(prompt_char.styles.color, "#38bdf8")

        # 2. YOLO only
        runtime.yolo_mode = True
        runtime.stealth_mode = False
        app.update_mode_ui()
        input_container.set_class.assert_any_call(True, "yolo-mode")
        input_container.set_class.assert_any_call(False, "stealth-mode")
        self.assertEqual(prompt_char.styles.color, "#f87171")

        # 3. Stealth only
        runtime.yolo_mode = False
        runtime.stealth_mode = True
        app.update_mode_ui()
        input_container.set_class.assert_any_call(False, "yolo-mode")
        input_container.set_class.assert_any_call(True, "stealth-mode")
        self.assertEqual(prompt_char.styles.color, "#c084fc")

        # 4. Both YOLO and Stealth
        runtime.yolo_mode = True
        runtime.stealth_mode = True
        app.update_mode_ui()
        input_container.set_class.assert_any_call(True, "yolo-mode")
        input_container.set_class.assert_any_call(True, "stealth-mode")
        self.assertEqual(prompt_char.styles.color, "#fbbf24")
