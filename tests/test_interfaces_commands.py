from __future__ import annotations

import argparse
import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from ollama_agent.agent.episodic_memory import HistoryError
from ollama_agent.core.models import ModelCapabilityError
from ollama_agent.interfaces.cli import handle_cli_commands
from ollama_agent.interfaces.dispatch import build_repl_handlers, safe_call
from ollama_agent.interfaces.model_commands import (
    ensure_model_configured,
    list_models,
    set_effort,
    set_model,
    set_model_param,
    show_effort,
    show_model_params,
)
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
        def raise_skill_error() -> None:
            raise SkillError("Skill failed")
        await safe_call(raise_skill_error)  # Should not raise

        def raise_history_error() -> None:
            raise HistoryError("History DB broken")
        console = Console(file=io.StringIO(), record=True)
        await safe_call(raise_history_error, console=console)  # Should not raise
        self.assertIn("History DB broken", console.export_text())

    async def test_safe_call_propagates_unexpected_errors(self) -> None:
        def raise_exit() -> None:
            raise SystemExit(1)
        with self.assertRaises(SystemExit):
            await safe_call(raise_exit)

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

    def test_show_effort(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime = MagicMock()
        runtime.settings.model.reasoning_effort = "medium"
        runtime.settings.model.name = "llama3.2:3b"

        show_effort(console, runtime)
        out = console.export_text()
        self.assertIn("Current reasoning effort", out)
        self.assertIn("medium", out)
        self.assertIn("llama3.2:3b", out)

    async def test_set_effort_invalid(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime = MagicMock()
        runtime.settings.model.reasoning_effort = "medium"

        res = await set_effort(console, "super_extreme", runtime=runtime)
        self.assertEqual(res, "medium")
        self.assertIn("Invalid reasoning effort", console.export_text())

    async def test_set_effort_same(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime = MagicMock()
        runtime.settings.model.reasoning_effort = "high"

        res = await set_effort(console, "high", runtime=runtime)
        self.assertEqual(res, "high")
        self.assertIn("Already using reasoning effort", console.export_text())

    async def test_set_effort_success(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime = MagicMock()
        runtime.settings.model.reasoning_effort = "medium"
        runtime.set_reasoning_effort = AsyncMock()

        res = await set_effort(console, "high", runtime=runtime)
        self.assertEqual(res, "high")
        runtime.set_reasoning_effort.assert_awaited_once_with("high")
        self.assertIn("Switched reasoning effort", console.export_text())

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
        switch_mock = AsyncMock()

        handlers = build_repl_handlers(
            task_ctx=MagicMock(),
            skills_ctx=MagicMock(),
            get_rag_ctx=MagicMock(),
            console=console,
            current_model=lambda: "llama3.2:3b",
            base_url=lambda: "http://localhost:11434",
            switch_model=switch_mock,
            handle_exit=lambda _: None,
            handle_new=AsyncMock(),
            handle_task_create=lambda _: None,
            handle_skill_create=lambda _: None,
            handle_yolo=lambda _: None,
            get_runtime=lambda: runtime,
        )

        # /params
        handlers["/params"].handler([])
        out = console.export_text()
        self.assertIn("Active Model Parameters", out)

        # /params list (handlers print to the console they were built with,
        # so all following assertions check the accumulated output)
        handlers["/params"].handler(["list"])
        self.assertIn("Active Model Parameters", console.export_text())

        # /params set with missing args
        handlers["/params"].handler(["set", "temperature"])
        self.assertIn("Usage: /params set", console.export_text())

        # /model <name> routes to switch_model (single arg is treated as a model name)
        handlers["/model"].handler(["some-model"])
        switch_mock.assert_called_once_with("some-model")

    def test_effort_dispatch(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        runtime = MagicMock()
        runtime.settings.model.reasoning_effort = "medium"
        runtime.settings.model.name = "llama3.2:3b"

        handlers = build_repl_handlers(
            task_ctx=MagicMock(),
            skills_ctx=MagicMock(),
            get_rag_ctx=MagicMock(),
            console=console,
            current_model=lambda: "llama3.2:3b",
            base_url=lambda: "http://localhost:11434",
            switch_model=AsyncMock(),
            handle_exit=lambda _: None,
            handle_new=AsyncMock(),
            handle_task_create=lambda _: None,
            handle_skill_create=lambda _: None,
            handle_yolo=lambda _: None,
            get_runtime=lambda: runtime,
            current_effort=lambda: "medium",
            switch_effort=AsyncMock(),
        )

        # /effort without args
        handlers["/effort"].handler([])
        out = console.export_text()
        self.assertIn("Current reasoning effort", out)
        self.assertIn("medium", out)

    async def test_unified_repl_handlers(self) -> None:
        console = Console(file=io.StringIO(), record=True)
        task_ctx = MagicMock()
        skills_ctx = MagicMock()
        rag_ctx = MagicMock()
        switch_model = AsyncMock()
        switch_effort = AsyncMock()
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
            handle_new=AsyncMock(),
            handle_task_create=handle_task_create,
            handle_skill_create=handle_skill_create,
            handle_yolo=lambda _: None,
            current_effort=lambda: "medium",
            switch_effort=switch_effort,
        )

        # 1. /model handler
        with patch("ollama_agent.interfaces.dispatch.list_models", AsyncMock()) as mock_list_models:
            await safe_call(handlers["/model"].handler, ["list"])
            mock_list_models.assert_awaited_once()

        await safe_call(handlers["/model"].handler, ["set", "llama3:8b"])
        switch_model.assert_awaited_with("llama3:8b")

        # 1b. /effort handler
        with patch("ollama_agent.interfaces.dispatch.show_effort"):
            handlers["/effort"].handler([])
            out_effort = console.export_text()
            self.assertIn("Current reasoning effort", out_effort)

        await safe_call(handlers["/effort"].handler, ["set", "high"])
        switch_effort.assert_awaited_with("high")

        await safe_call(handlers["/effort"].handler, ["low"])
        switch_effort.assert_awaited_with("low")

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
        args = argparse.Namespace(command="task", subcommand="list", prompt=None, yolo=False, rag=None)
        settings = Settings()
        with patch("ollama_agent.interfaces.dispatch.list_tasks") as mock_list:
            handled = handle_cli_commands(args, settings)
            self.assertTrue(handled)
            mock_list.assert_called_once()

    def test_handle_cli_commands_session_list(self) -> None:
        args = argparse.Namespace(command="session", subcommand="list", prompt=None, yolo=False, rag=None)
        settings = Settings()
        with patch("ollama_agent.interfaces.dispatch.list_sessions") as mock_list:
            handled = handle_cli_commands(args, settings)
            self.assertTrue(handled)
            mock_list.assert_called_once()

    def test_handle_cli_commands_session_search(self) -> None:
        args = argparse.Namespace(command="session", subcommand="search", query="fastapi", prompt=None, yolo=False, rag=None)
        settings = Settings()
        with patch("ollama_agent.interfaces.dispatch.search_sessions") as mock_search:
            handled = handle_cli_commands(args, settings)
            self.assertTrue(handled)
            mock_search.assert_called_once()

    def test_handle_cli_commands_session_delete(self) -> None:
        args = argparse.Namespace(command="session", subcommand="delete", session_id="session-123", prompt=None, yolo=False, rag=None)
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

    def test_ensure_model_configured_already_available(self) -> None:
        settings = Settings()
        settings.model.name = "qwen3:32b"
        mock_m1 = MagicMock(model="qwen3:32b", size=1024**3 * 10)
        mock_m2 = MagicMock(model="llama3:8b", size=1024**3 * 5)
        mock_resp = MagicMock(models=[mock_m1, mock_m2])

        with patch("ollama.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.list.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            res = ensure_model_configured(settings)
            self.assertEqual(res, "qwen3:32b")
            self.assertEqual(settings.model.name, "qwen3:32b")

    def test_ensure_model_configured_latest_alias(self) -> None:
        settings = Settings()
        settings.model.name = "llama3"
        mock_m = MagicMock(model="llama3:latest", size=1024**3 * 5)
        mock_resp = MagicMock(models=[mock_m])

        with patch("ollama.Client") as mock_client_cls, \
             patch("ollama_agent.interfaces.model_commands.save_settings") as mock_save:
            mock_client = MagicMock()
            mock_client.list.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            res = ensure_model_configured(settings)
            self.assertEqual(res, "llama3:latest")
            self.assertEqual(settings.model.name, "llama3:latest")
            mock_save.assert_called_once_with(settings)

    def test_ensure_model_configured_prompt_numeric_selection(self) -> None:
        settings = Settings()
        settings.model.name = "qwen3.8:2b"
        mock_m1 = MagicMock(model="qwen3.8:27b", size=1024**3 * 17)
        mock_m2 = MagicMock(model="ornith-1.5:9b", size=1024**3 * 6)
        mock_resp = MagicMock(models=[mock_m1, mock_m2])

        console = Console(file=io.StringIO(), record=True)
        with patch("ollama.Client") as mock_client_cls, \
             patch("ollama_agent.interfaces.model_commands.save_settings") as mock_save:
            mock_client = MagicMock()
            mock_client.list.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            inputs = iter(["1"])
            res = ensure_model_configured(settings, console=console, input_func=lambda _: next(inputs))
            self.assertEqual(res, "qwen3.8:27b")
            self.assertEqual(settings.model.name, "qwen3.8:27b")
            mock_save.assert_called_once_with(settings)
            out = console.export_text()
            self.assertIn("not available in Ollama", out)
            self.assertIn("qwen3.8:27b", out)

    def test_ensure_model_configured_prompt_name_selection(self) -> None:
        settings = Settings()
        settings.model.name = ""
        mock_m1 = MagicMock(model="qwen3.8:27b", size=1024**3 * 17)
        mock_m2 = MagicMock(model="ornith-1.5:9b", size=1024**3 * 6)
        mock_resp = MagicMock(models=[mock_m1, mock_m2])

        console = Console(file=io.StringIO(), record=True)
        with patch("ollama.Client") as mock_client_cls, \
             patch("ollama_agent.interfaces.model_commands.save_settings") as mock_save:
            mock_client = MagicMock()
            mock_client.list.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            inputs = iter(["invalid_name", "ornith-1.5:9b"])
            res = ensure_model_configured(settings, console=console, input_func=lambda _: next(inputs))
            self.assertEqual(res, "ornith-1.5:9b")
            self.assertEqual(settings.model.name, "ornith-1.5:9b")
            mock_save.assert_called_once_with(settings)
            out = console.export_text()
            self.assertIn("No model is currently configured", out)
            self.assertIn("Invalid selection", out)

    def test_ensure_model_configured_no_models_raises(self) -> None:
        settings = Settings()
        mock_resp = MagicMock(models=[])

        with patch("ollama.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.list.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            with self.assertRaises(ModelCapabilityError) as cm:
                ensure_model_configured(settings)
            self.assertIn("No models found in Ollama", str(cm.exception))

    def test_ensure_model_configured_connection_error_raises(self) -> None:
        settings = Settings()

        with patch("ollama.Client", side_effect=ConnectionError("Refused")):
            with self.assertRaises(ModelCapabilityError) as cm:
                ensure_model_configured(settings)
            self.assertIn("Could not connect to Ollama", str(cm.exception))


