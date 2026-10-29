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
        assert len(models) == 2
        assert models[0].model == "gemma4:26b"
        assert models[1].model == "qwen3:32b"


class TestTUIComponentsUnit(unittest.TestCase):
    """Unit tests for TUI components without full app run."""

    def test_prompt_queue_widget_renders_string(self) -> None:
        widget = PromptQueueWidget()
        widget.update_queue(["Test prompt 1", "Test prompt 2"])
        assert widget.display
        out = str(widget.render())
        assert "Test prompt 1" in out
        assert "Test prompt 2" in out

    def test_system_output_widget_show_and_clear(self) -> None:
        widget = SystemOutputWidget()
        assert not widget.display
        widget.show_output("Hello world", title="Greetings")
        assert widget.display
        assert "Greetings" in str(widget.render())
        assert "Hello world" in str(widget.render())
        widget.clear_output()
        assert not widget.display

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
        assert "8.2k/16.4k" in str(header.render())


class TestREPLEffortCompletions(unittest.TestCase):
    """Unit tests for /effort slash command completions."""

    def test_effort_slash_completions(self) -> None:
        runtime = MagicMock()
        runtime.model.show_info = MagicMock(thinking={"values": ["low", "high", "max"], "default": "max"})
        repl = OllamaREPL(runtime=runtime)
        app = OllamaAgentApp(repl)

        comps = slash_completions("/effort ", app.repl, list(app._prompt_queue))
        comp_map = {cmd: text.plain for cmd, text in comps}

        assert "/effort default" in comp_map
        assert "(model default)" in comp_map["/effort default"]

        assert "/effort max" in comp_map
        assert "(default)" in comp_map["/effort max"]

        assert "/effort low" in comp_map
        assert "/effort high" in comp_map
