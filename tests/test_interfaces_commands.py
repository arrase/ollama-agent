from __future__ import annotations

import argparse
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from ollama_agent.interfaces.cli import handle_cli_commands
from ollama_agent.interfaces.dispatch import build_cli_handlers, build_repl_handlers, render_repl_help, safe_call
from ollama_agent.interfaces.model_commands import list_models, set_model
from ollama_agent.interfaces.session_commands import new_session
from ollama_agent.settings.config import Settings
from ollama_agent.skills.commands import SkillError


class TestInterfacesCommands(unittest.IsolatedAsyncioTestCase):
    """Unit tests for CLI, model, session, and dispatch handlers."""

    def test_new_session_generates_uuid_hex(self) -> None:
        console = Console(record=True)
        session_id = new_session(console)
        self.assertEqual(len(session_id), 8)
        self.assertIn("New session started", console.export_text())

    async def test_safe_call_sync_and_async(self) -> None:
        # Sync fn
        called_sync = False
        def sync_fn() -> None:
            nonlocal called_sync
            called_sync = True
        await safe_call(sync_fn)
        self.assertTrue(called_sync)

        # Async fn
        called_async = False
        async def async_fn() -> None:
            nonlocal called_async
            called_async = True
        await safe_call(async_fn)
        self.assertTrue(called_async)

    async def test_safe_call_silences_expected_errors(self) -> None:
        def raise_exit() -> None:
            raise SystemExit(1)
        await safe_call(raise_exit)  # Should not raise

        def raise_skill_error() -> None:
            raise SkillError("Skill failed")
        await safe_call(raise_skill_error)  # Should not raise

    async def test_list_models_empty(self) -> None:
        console = Console(record=True)
        with patch("ollama_agent.interfaces.model_commands._list_models", AsyncMock(return_value=[])):
            await list_models(console, current_model="gemma4:26b", base_url="http://localhost:11434")
            self.assertIn("No models found", console.export_text())

    async def test_list_models_with_models(self) -> None:
        console = Console(record=True)
        mock_m1 = MagicMock(model="gemma4:26b", size=1024**3 * 15)
        mock_m2 = MagicMock(model="llama3:8b", size=1024**3 * 5)
        with patch("ollama_agent.interfaces.model_commands._list_models", AsyncMock(return_value=[mock_m1, mock_m2])), \
             patch("ollama_agent.interfaces.model_commands.model_supports_tools", AsyncMock(return_value=True)):
            await list_models(console, current_model="gemma4:26b", base_url="http://localhost:11434")
            out = console.export_text()
            self.assertIn("gemma4:26b", out)
            self.assertIn("llama3:8b", out)
            self.assertIn("current", out)

    async def test_set_model_not_found(self) -> None:
        console = Console(record=True)
        runtime = MagicMock()
        runtime.settings.model.name = "gemma4:26b"
        runtime.settings.model.base_url = "http://localhost:11434"

        with patch("ollama_agent.interfaces.model_commands._list_models", AsyncMock(return_value=[])):
            res = await set_model(console, "nonexistent_model", runtime=runtime)
            self.assertEqual(res, "gemma4:26b")
            self.assertIn("not found", console.export_text())

    async def test_set_model_same_model(self) -> None:
        console = Console(record=True)
        runtime = MagicMock()
        runtime.settings.model.name = "gemma4:26b"
        runtime.settings.model.base_url = "http://localhost:11434"

        mock_m = MagicMock(model="gemma4:26b")
        with patch("ollama_agent.interfaces.model_commands._list_models", AsyncMock(return_value=[mock_m])):
            res = await set_model(console, "gemma4:26b", runtime=runtime)
            self.assertEqual(res, "gemma4:26b")
            self.assertIn("Already using model", console.export_text())

    async def test_set_model_success(self) -> None:
        console = Console(record=True)
        runtime = MagicMock()
        runtime.settings.model.name = "gemma4:26b"
        runtime.settings.model.base_url = "http://localhost:11434"
        runtime.set_model = AsyncMock()

        mock_m1 = MagicMock(model="gemma4:26b")
        mock_m2 = MagicMock(model="qwen3:32b")
        with patch("ollama_agent.interfaces.model_commands._list_models", AsyncMock(return_value=[mock_m1, mock_m2])), \
             patch("ollama_agent.interfaces.model_commands.model_supports_tools", AsyncMock(return_value=True)):
            res = await set_model(console, "qwen3:32b", runtime=runtime)
            self.assertEqual(res, "qwen3:32b")
            runtime.set_model.assert_awaited_once_with("qwen3:32b")
            self.assertIn("Switched", console.export_text())

    def test_render_repl_help(self) -> None:
        console = Console(record=True)
        handlers = build_repl_handlers(
            task_ctx=MagicMock(),
            skills_ctx=MagicMock(),
            get_rag_ctx=MagicMock(),
            console=console,
            current_model=lambda: "gemma4:26b",
            base_url=lambda: "http://localhost:11434",
            switch_model=AsyncMock(),
            handle_exit=lambda _: None,
            handle_clear=lambda _: None,
            handle_new=AsyncMock(),
            handle_task_create=lambda _: None,
            handle_skill_create=lambda _: None,
            handle_yolo=lambda _: None,
        )
        render_repl_help(console, handlers)
        out = console.export_text()
        self.assertIn("Available Commands", out)
        self.assertIn("/help", out)
        self.assertIn("/yolo", out)
