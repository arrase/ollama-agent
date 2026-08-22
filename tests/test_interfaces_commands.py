from __future__ import annotations

import argparse
import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from ollama_agent.core.models import ModelCapabilityError
from ollama_agent.interfaces.cli import handle_cli_commands
from ollama_agent.interfaces.dispatch import build_repl_handlers, render_repl_help, safe_call
from ollama_agent.interfaces.model_commands import list_models, set_model, set_model_param, show_model_params
from ollama_agent.interfaces.session_commands import new_session
from ollama_agent.settings.config import Settings
from ollama_agent.skills.commands import SkillError


class TestInterfacesCommands(unittest.IsolatedAsyncioTestCase):
    """Unit tests for CLI, model, session, and dispatch handlers."""

    def test_new_session_generates_uuid_hex(self) -> None:
        console = Console(file=io.StringIO(), record=True)
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
        console = Console(file=io.StringIO(), record=True)
        with patch("ollama_agent.interfaces.model_commands._list_models", AsyncMock(return_value=[])):
            await list_models(console, current_model="gemma4:26b", base_url="http://localhost:11434")
            self.assertIn("No models found", console.export_text())

    async def test_list_models_with_models(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        mock_m1 = MagicMock(model="gemma4:26b", size=1024**3 * 15)
        mock_m2 = MagicMock(model="llama3:8b", size=1024**3 * 5)
        with patch("ollama_agent.interfaces.model_commands._list_models", AsyncMock(return_value=[mock_m1, mock_m2])), \
             patch("ollama_agent.interfaces.model_commands.model_supports_tools", AsyncMock(return_value=True)):
            await list_models(console, current_model="gemma4:26b", base_url="http://localhost:11434")
            out = console.export_text()
            self.assertIn("gemma4:26b", out)
            self.assertIn("llama3:8b", out)
            self.assertIn("current", out)

    async def test_list_models_error_handled(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        with patch("ollama_agent.interfaces.model_commands._list_models", AsyncMock(side_effect=ConnectionError("Cannot connect"))):
            await list_models(console, current_model="gemma4:26b", base_url="http://localhost:11434")
            out = console.export_text()
            self.assertIn("Error listing models", out)

    async def test_set_model_not_found(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime = MagicMock()
        runtime.settings.model.name = "gemma4:26b"
        runtime.settings.model.base_url = "http://localhost:11434"

        with patch("ollama_agent.interfaces.model_commands._list_models", AsyncMock(return_value=[])):
            res = await set_model(console, "nonexistent_model", runtime=runtime)
            self.assertEqual(res, "gemma4:26b")
            self.assertIn("not found", console.export_text())

    async def test_set_model_same_model(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime = MagicMock()
        runtime.settings.model.name = "gemma4:26b"
        runtime.settings.model.base_url = "http://localhost:11434"

        mock_m = MagicMock(model="gemma4:26b")
        with patch("ollama_agent.interfaces.model_commands._list_models", AsyncMock(return_value=[mock_m])):
            res = await set_model(console, "gemma4:26b", runtime=runtime)
            self.assertEqual(res, "gemma4:26b")
            self.assertIn("Already using model", console.export_text())

    async def test_set_model_success(self) -> None:
        console = Console(file=io.StringIO(), record=True)
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

    def test_show_model_params(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime = MagicMock()
        runtime.settings.model.name = "llama3.2:3b"
        runtime.effective_model_params = {
            "temperature": (0.7, "user"),
            "top_p": (0.85, "modelfile"),
            "top_k": (40, "default"),
        }

        show_model_params(console, runtime)
        out = console.export_text()
        self.assertIn("Active Model Parameters", out)
        self.assertIn("llama3.2:3b", out)
        self.assertIn("temperature", out)
        self.assertIn("0.7", out)
        self.assertIn("User Config", out)
        self.assertIn("Modelfile", out)
        self.assertIn("Ollama Default", out)

    async def test_set_model_param_success(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime = MagicMock()
        runtime.settings.model.temperature = 0.8
        runtime.reload = AsyncMock()

        with patch("ollama_agent.interfaces.model_commands.save_settings"):
            await set_model_param(console, "temperature", "0.5", runtime=runtime)
            self.assertEqual(runtime.settings.model.temperature, 0.5)
            runtime.reload.assert_awaited_once()
            out = console.export_text()
            self.assertIn("Set", out)
            self.assertIn("temperature", out)
            self.assertIn("0.5", out)

    async def test_set_model_param_invalid(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime = MagicMock()

        # Unknown param
        await set_model_param(console, "invalid_param", "0.5", runtime=runtime)
        self.assertIn("Unknown parameter", console.export_text())

        # Invalid type
        console = Console(file=io.StringIO(), record=True)
        await set_model_param(console, "temperature", "not_a_number", runtime=runtime)
        self.assertIn("Invalid value", console.export_text())

    def test_params_dispatch(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime = MagicMock()
        runtime.settings.model.name = "llama3.2:3b"
        runtime.effective_model_params = {
            "temperature": (0.8, "default"),
        }

        handlers = build_repl_handlers(
            task_ctx=MagicMock(),
            skills_ctx=MagicMock(),
            get_rag_ctx=MagicMock(),
            console=console,
            current_model=lambda: "llama3.2:3b",
            base_url=lambda: "http://localhost:11434",
            switch_model=AsyncMock(),
            handle_exit=lambda _: None,
            handle_clear=lambda _: None,
            handle_new=AsyncMock(),
            handle_task_create=lambda _: None,
            handle_skill_create=lambda _: None,
            handle_yolo=lambda _: None,
            get_runtime=lambda: runtime,
        )

        # /params
        handlers["/params"].callback([])
        out = console.export_text()
        self.assertIn("Active Model Parameters", out)

        # /params list
        console = Console(file=io.StringIO(), record=True)
        handlers["/params"].callback(["list"])
        out_list = console.export_text()
        self.assertIn("Active Model Parameters", out_list)

        # /params set with missing args
        console = Console(file=io.StringIO(), record=True)
        handlers["/params"].callback(["set", "temperature"])
        out_missing = console.export_text()
        self.assertIn("Usage: /params set", out_missing)

        # /model does not handle params
        console = Console(file=io.StringIO(), record=True)
        handlers["/model"].callback(["params"])
        out_model = console.export_text()
        self.assertIn("Usage: /model", out_model)

    def test_render_repl_help(self) -> None:
        console = Console(file=io.StringIO(), record=True)
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
        self.assertIn("/model", out)
        self.assertIn("/task", out)
        self.assertIn("/skill", out)
        self.assertIn("/rag", out)
        self.assertIn("/yolo", out)

    async def test_unified_repl_handlers(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        task_ctx = MagicMock()
        skills_ctx = MagicMock()
        rag_ctx = MagicMock()
        switch_model = AsyncMock()
        handle_task_create = MagicMock()
        handle_skill_create = MagicMock()

        handlers = build_repl_handlers(
            task_ctx=task_ctx,
            skills_ctx=skills_ctx,
            get_rag_ctx=lambda: rag_ctx,
            console=console,
            current_model=lambda: "gemma4:26b",
            base_url=lambda: "http://localhost:11434",
            switch_model=switch_model,
            handle_exit=lambda _: None,
            handle_clear=lambda _: None,
            handle_new=AsyncMock(),
            handle_task_create=handle_task_create,
            handle_skill_create=handle_skill_create,
            handle_yolo=lambda _: None,
        )

        # 1. /model handler
        with patch("ollama_agent.interfaces.dispatch.list_models", AsyncMock()) as mock_list_models:
            await safe_call(handlers["/model"].handler, ["list"])
            mock_list_models.assert_awaited_once()

        await safe_call(handlers["/model"].handler, ["set", "llama3:8b"])
        switch_model.assert_awaited_with("llama3:8b")

        # 2. /task handler
        with patch("ollama_agent.interfaces.dispatch.list_tasks") as mock_list_tasks:
            handlers["/task"].handler([])
            mock_list_tasks.assert_called_once()

        handlers["/task"].handler(["create", "my-task"])
        handle_task_create.assert_called_once_with(["my-task"])

        with patch("ollama_agent.interfaces.dispatch.run_task", MagicMock(return_value=None)) as mock_run_task:
            handlers["/task"].handler(["run", "my-task", "-y"])
            mock_run_task.assert_called_once_with(task_ctx, "my-task", yolo=True)

        with patch("ollama_agent.interfaces.dispatch.delete_task") as mock_del_task:
            handlers["/task"].handler(["delete", "my-task"])
            mock_del_task.assert_called_once_with(task_ctx, "my-task")

        # 3. /skill handler
        with patch("ollama_agent.interfaces.dispatch.list_skills") as mock_list_skills:
            handlers["/skill"].handler([])
            mock_list_skills.assert_called_once()

        with patch("ollama_agent.interfaces.dispatch.show_skill") as mock_show_skill:
            handlers["/skill"].handler(["show", "my-skill"])
            mock_show_skill.assert_called_once_with(skills_ctx, "my-skill")

        handlers["/skill"].handler(["create", "my-skill"])
        handle_skill_create.assert_called_once_with(["my-skill"])

        with patch("ollama_agent.interfaces.dispatch.delete_skill") as mock_del_skill:
            handlers["/skill"].handler(["delete", "my-skill"])
            mock_del_skill.assert_called_once_with(skills_ctx, "my-skill")

        # 4. /rag handler
        with patch("ollama_agent.interfaces.dispatch.show_rag_status") as mock_show_status:
            handlers["/rag"].handler(["status"])
            mock_show_status.assert_called_once_with(rag_ctx)

        with patch("ollama_agent.interfaces.dispatch.list_rag_databases") as mock_list_dbs:
            handlers["/rag"].handler(["list"])
            mock_list_dbs.assert_called_once_with(rag_ctx)

        with patch("ollama_agent.interfaces.dispatch.create_rag_database") as mock_create_db:
            handlers["/rag"].handler(["create", "my-db"])
            mock_create_db.assert_called_once_with(rag_ctx, "my-db")

        with patch("ollama_agent.interfaces.dispatch.load_rag_database") as mock_load_db:
            handlers["/rag"].handler(["load", "my-db"])
            mock_load_db.assert_called_once_with(rag_ctx, "my-db")

        with patch("ollama_agent.interfaces.dispatch.unload_rag_database") as mock_unload_db:
            handlers["/rag"].handler(["unload"])
            mock_unload_db.assert_called_once_with(rag_ctx)

        # 5. /session handler
        with patch("ollama_agent.interfaces.dispatch.list_sessions") as mock_list_sess:
            handlers["/session"].handler([])
            mock_list_sess.assert_called_once()

        await safe_call(handlers["/session"].handler, ["new"])
        await safe_call(handlers["/session"].handler, ["resume", "session-1234"])

        with patch("ollama_agent.interfaces.dispatch.search_sessions") as mock_search_sess:
            handlers["/session"].handler(["search", "my-query"])
            mock_search_sess.assert_called_once()

        with patch("ollama_agent.interfaces.dispatch.delete_session") as mock_del_sess:
            handlers["/session"].handler(["delete", "session-1234"])
            mock_del_sess.assert_called_once_with(console, "session-1234")

    def test_handle_cli_commands_subcommand(self) -> None:
        args = argparse.Namespace(command="task-list", prompt=None, yolo=False, rag=None)
        settings = Settings()
        with patch("ollama_agent.interfaces.dispatch.list_tasks") as mock_list:
            handled = handle_cli_commands(args, settings)
            self.assertTrue(handled)
            mock_list.assert_called_once()

    def test_handle_cli_commands_session_list(self) -> None:
        args = argparse.Namespace(command="session-list", prompt=None, yolo=False, rag=None)
        settings = Settings()
        with patch("ollama_agent.interfaces.dispatch.list_sessions") as mock_list:
            handled = handle_cli_commands(args, settings)
            self.assertTrue(handled)
            mock_list.assert_called_once()

    def test_handle_cli_commands_session_search(self) -> None:
        args = argparse.Namespace(command="session-search", query="fastapi", prompt=None, yolo=False, rag=None)
        settings = Settings()
        with patch("ollama_agent.interfaces.dispatch.search_sessions") as mock_search:
            handled = handle_cli_commands(args, settings)
            self.assertTrue(handled)
            mock_search.assert_called_once()

    def test_handle_cli_commands_session_delete(self) -> None:
        args = argparse.Namespace(command="session-delete", session_id="session-123", prompt=None, yolo=False, rag=None)
        settings = Settings()
        with patch("ollama_agent.interfaces.dispatch.delete_session") as mock_del:
            handled = handle_cli_commands(args, settings)
            self.assertTrue(handled)
            mock_del.assert_called_once()

    def test_handle_cli_commands_prompt(self) -> None:
        args = argparse.Namespace(command=None, prompt="hello world", yolo=True, rag=None)
        settings = Settings()
        with patch("ollama_agent.interfaces.cli.run_non_interactive", AsyncMock()) as mock_run:
            with patch("ollama_agent.agent.agent.AgentRuntime.reload", AsyncMock()):
                handled = handle_cli_commands(args, settings)
                self.assertTrue(handled)
                mock_run.assert_awaited_once()

    def test_handle_cli_commands_prompt_with_rag(self) -> None:
        args = argparse.Namespace(command=None, prompt="query with rag", yolo=False, rag="my_db")
        settings = Settings()
        with patch("ollama_agent.interfaces.cli.run_non_interactive", AsyncMock()) as mock_run:
            with patch("ollama_agent.agent.agent.AgentRuntime.reload", AsyncMock()), \
                 patch("ollama_agent.interfaces.cli.load_rag_database") as mock_load_rag:
                handled = handle_cli_commands(args, settings)
                self.assertTrue(handled)
                mock_load_rag.assert_called_once()
                mock_run.assert_awaited_once()

    def test_handle_cli_commands_unhandled(self) -> None:
        args = argparse.Namespace(command=None, prompt=None)
        settings = Settings()
        self.assertFalse(handle_cli_commands(args, settings))

