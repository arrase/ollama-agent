from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from ollama_agent.interfaces.tui import OllamaAgentApp, OllamaREPL
from ollama_agent.interfaces.tui.completion import _list_models_sync, slash_completions
from ollama_agent.interfaces.tui.widgets import AgentHeader, PromptQueueWidget, SystemOutputWidget


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
        runtime.settings.model.reasoning_effort = "medium"
        runtime.settings.model.context_window = 16384
        runtime.effective_context_window = 16384
        runtime.last_context_tokens = 8192
        repl = OllamaREPL(runtime=runtime)

        header = AgentHeader(repl)
        header.update_header()
        self.assertIn("8.2k/16.4k", str(header.render()))


class TestREPLEffortCompletions(unittest.TestCase):
    """Unit tests for /effort slash command completions."""

    def test_effort_slash_completions(self) -> None:
        runtime = MagicMock()
        runtime.model.show_info = MagicMock(thinking={"values": ["low", "high", "max"], "default": "max"})
        repl = OllamaREPL(runtime=runtime)
        app = OllamaAgentApp(repl)

        comps = slash_completions("/effort ", app.repl, list(app._prompt_queue))
        comp_map = {cmd: text.plain for cmd, text in comps}

        self.assertIn("/effort default", comp_map)
        self.assertIn("(model default)", comp_map["/effort default"])

        self.assertIn("/effort max", comp_map)
        self.assertIn("(default)", comp_map["/effort max"])

        self.assertIn("/effort low", comp_map)
        self.assertIn("/effort high", comp_map)
