from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock, patch

from rich.console import Console

from ollama_agent.interfaces.repl import (
    IMMEDIATE_COMMANDS,
    OllamaREPL,
    _is_immediate_command,
    _list_models_sync,
)
from ollama_agent.interfaces.tui_components import AgentHeader, PromptQueueWidget, SystemOutputWidget


class TestREPLImmediateCommands(unittest.TestCase):
    """Unit tests for immediate commands constant set and predicate."""

    def test_immediate_commands_set(self) -> None:
        self.assertIn(("/exit", "*"), IMMEDIATE_COMMANDS)
        self.assertIn(("/quit", "*"), IMMEDIATE_COMMANDS)
        self.assertIn(("/queue", "*"), IMMEDIATE_COMMANDS)
        self.assertIn(("/yolo", "*"), IMMEDIATE_COMMANDS)
        self.assertIn(("/stealth", "*"), IMMEDIATE_COMMANDS)
        self.assertIn(("/model", "list"), IMMEDIATE_COMMANDS)
        self.assertIn(("/effort", ""), IMMEDIATE_COMMANDS)
        self.assertIn(("/context", ""), IMMEDIATE_COMMANDS)
        self.assertIn(("/session", "list"), IMMEDIATE_COMMANDS)

    def test_is_immediate_command_matches(self) -> None:
        self.assertTrue(_is_immediate_command("/exit"))
        self.assertTrue(_is_immediate_command("/quit"))
        self.assertTrue(_is_immediate_command("/queue"))
        self.assertTrue(_is_immediate_command("/queue clear"))
        self.assertTrue(_is_immediate_command("/queue rm 1"))
        self.assertTrue(_is_immediate_command("/yolo"))
        self.assertTrue(_is_immediate_command("/yolo on"))
        self.assertTrue(_is_immediate_command("/stealth"))
        self.assertTrue(_is_immediate_command("/stealth off"))
        self.assertTrue(_is_immediate_command("/model"))
        self.assertTrue(_is_immediate_command("/model list"))
        self.assertTrue(_is_immediate_command("/effort"))
        self.assertTrue(_is_immediate_command("/context"))
        self.assertTrue(_is_immediate_command("/session list"))
        self.assertTrue(_is_immediate_command("/session search foo"))
        self.assertTrue(_is_immediate_command("/task list"))
        self.assertTrue(_is_immediate_command("/skill list"))
        self.assertTrue(_is_immediate_command("/rag list"))
        self.assertTrue(_is_immediate_command("/mcp list"))
        self.assertTrue(_is_immediate_command("/agents list"))

    def test_is_immediate_command_non_immediate(self) -> None:
        self.assertFalse(_is_immediate_command(""))
        self.assertFalse(_is_immediate_command("hello agent"))
        self.assertFalse(_is_immediate_command("/model set llama3:8b"))
        self.assertFalse(_is_immediate_command("/effort high"))
        self.assertFalse(_is_immediate_command("/context 16384"))
        self.assertFalse(_is_immediate_command("/task run my-task"))
        self.assertFalse(_is_immediate_command("/session new"))


class TestREPLQueueCommands(unittest.TestCase):
    """Unit tests for REPL prompt queue handling."""

    def test_queue_cmd_list_empty(self) -> None:
        runtime = MagicMock()
        repl = OllamaREPL(runtime=runtime)
        repl.console = Console(file=io.StringIO(), record=True)
        repl.app = None

        repl._handle_queue_cmd(["list"])
        self.assertIn("Prompt queue is empty", repl.console.export_text())

    def test_queue_cmd_unknown_subcommand(self) -> None:
        runtime = MagicMock()
        repl = OllamaREPL(runtime=runtime)
        repl.console = Console(file=io.StringIO(), record=True)
        repl.app = None

        repl._handle_queue_cmd(["invalid"])
        self.assertIn("Unknown queue subcommand 'invalid'", repl.console.export_text())

    def test_queue_cmd_rm_empty(self) -> None:
        runtime = MagicMock()
        repl = OllamaREPL(runtime=runtime)
        repl.console = Console(file=io.StringIO(), record=True)
        repl.app = None

        repl._handle_queue_cmd(["rm", "1"])
        self.assertIn("Prompt queue is empty", repl.console.export_text())


class TestREPLHelpers(unittest.TestCase):
    """Unit tests for REPL helper functions and models list."""

    @patch("ollama.Client")
    def test_list_models_sync(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock(models=[MagicMock(model="gemma4:26b"), MagicMock(model="qwen3:32b")])
        mock_client.list.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        models = _list_models_sync("http://localhost:11434")
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].model, "gemma4:26b")
        self.assertEqual(models[1].model, "qwen3:32b")


class TestTUIComponentsUnit(unittest.TestCase):
    """Unit tests for TUI components without full app run."""

    def test_prompt_queue_widget_renders_string(self) -> None:
        widget = PromptQueueWidget()
        widget.update_queue(["Test prompt 1", "Test prompt 2"])
        self.assertTrue(widget.display)
        out = str(widget.render())
        self.assertIn("Test prompt 1", out)
        self.assertIn("Test prompt 2", out)

    def test_system_output_widget_show_and_clear(self) -> None:
        widget = SystemOutputWidget()
        self.assertFalse(widget.display)
        widget.show_output("Hello world", title="Greetings")
        self.assertTrue(widget.display)
        self.assertIn("Greetings", str(widget.render()))
        self.assertIn("Hello world", str(widget.render()))
        widget.clear_output()
        self.assertFalse(widget.display)

    def test_agent_header_update(self) -> None:
        runtime = MagicMock()
        runtime.settings.model.name = "gemma4:26b"
        runtime.settings.model.context_window = 16384
        runtime.effective_context_window = 16384
        runtime.last_context_tokens = 8192
        repl = OllamaREPL(runtime=runtime)

        header = AgentHeader(repl)
        header.update_header()
        self.assertIn("8.2k/16.4k", str(header.render()))
