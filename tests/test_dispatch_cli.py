from __future__ import annotations

import argparse
import asyncio
import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from ollama_agent.core import ModelCapabilityError
from ollama_agent.interfaces.cli import create_argument_parser
from ollama_agent.interfaces.dispatch import (
    build_cli_handlers,
    build_repl_handlers,
)
from ollama_agent.main import main
from ollama_agent.rag.commands import RAGContext
from ollama_agent.rag import RAGManager, RAGSettings
from ollama_agent.settings.config import Settings
from ollama_agent.skills.commands import SkillsContext
from ollama_agent.tasks.commands import TasksContext


def _console() -> Console:
    return Console(file=io.StringIO())


def _build_cli_handlers(args: argparse.Namespace) -> dict[tuple[str, str], object]:
    return build_cli_handlers(
        args,
        task_ctx=TasksContext(console=_console()),
        rag_ctx=RAGContext(rag_manager=RAGManager(RAGSettings()), console=_console()),
        skills_ctx=SkillsContext(console=_console()),
        settings=Settings(),
    )


class TestDispatchAndCLI(unittest.TestCase):
    """Unit tests for CLI argument parsing and dispatch handlers."""

    def test_argument_parser_flags(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["-m", "llama3:8b", "-e", "high", "-y", "--prompt", "hello", "--rag", "docs_db"])
        self.assertEqual(args.model, "llama3:8b")
        self.assertEqual(args.effort, "high")
        self.assertTrue(args.yolo)
        self.assertEqual(args.prompt, "hello")
        self.assertEqual(args.rag, "docs_db")

    def test_argument_parser_runtime_flags(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["--builtin-tool-timeout", "45", "--allow-traversal"])
        self.assertEqual(args.builtin_tool_timeout, 45)
        self.assertTrue(args.allow_traversal)

    def test_argument_parser_config_reset(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["--config-reset", "system-prompt"])
        self.assertEqual(args.config_reset, "system-prompt")

    def test_task_run_yolo_flag_parsing(self) -> None:
        parser = create_argument_parser()
        args1 = parser.parse_args(["task", "run", "my-task", "-y"])
        self.assertEqual(args1.task_id, "my-task")
        self.assertTrue(args1.yolo)

        args2 = parser.parse_args(["-y", "task", "run", "my-task"])
        self.assertEqual(args2.task_id, "my-task")
        self.assertTrue(args2.yolo)

        args3 = parser.parse_args(["task", "run", "my-task"])
        self.assertEqual(args3.task_id, "my-task")
        self.assertFalse(args3.yolo)

    def test_argument_parser_language_choices(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["-l", "es"])
        self.assertEqual(args.language, "es")
        with self.assertRaises(SystemExit):
            parser.parse_args(["-l", "klingon"])

    def test_build_cli_handlers_registry(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["task", "list"])
        handlers = _build_cli_handlers(args)
        self.assertIn(("task", "list"), handlers)
        self.assertIn(("task", "create"), handlers)
        self.assertIn(("rag", "list"), handlers)
        self.assertIn(("skill", "list"), handlers)
        self.assertIn(("mcp", "list"), handlers)
        self.assertIn(("agents", "list"), handlers)

    def test_build_cli_handlers_requires_settings(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["task", "list"])
        with self.assertRaises(TypeError):
            build_cli_handlers(
                args,
                task_ctx=TasksContext(console=_console()),
                rag_ctx=RAGContext(rag_manager=RAGManager(RAGSettings()), console=_console()),
                skills_ctx=SkillsContext(console=_console()),
            )

    def test_argument_parser_agents_subcommand(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["agents", "list"])
        self.assertEqual(args.command, "agents")
        self.assertEqual(args.subcommand, "list")

    def test_build_repl_handlers_registry(self) -> None:
        async def dummy_async_str(_: str) -> None:
            pass

        handlers = build_repl_handlers(
            task_ctx=TasksContext(console=_console()),
            skills_ctx=SkillsContext(console=_console()),
            get_rag_ctx=lambda: RAGContext(rag_manager=RAGManager(RAGSettings()), console=_console()),
            console=_console(),
            current_model=lambda: "gemma4:26b",
            base_url=lambda: "http://localhost:11434",
            switch_model=dummy_async_str,
            handle_yolo=lambda _: None,
            get_runtime=lambda: MagicMock(),
            current_thread_id=lambda: "",
            switch_effort=dummy_async_str,
        )

        self.assertNotIn("/help", handlers)
        self.assertNotIn("/exit", handlers)
        self.assertNotIn("/quit", handlers)
        self.assertNotIn("/clear", handlers)
        self.assertNotIn("/new", handlers)
        self.assertNotIn("/compact", handlers)
        self.assertIn("/model", handlers)
        self.assertIn("/effort", handlers)
        self.assertIn("/yolo", handlers)
        self.assertIn("/session", handlers)
        self.assertIn("/task", handlers)
        self.assertIn("/skill", handlers)
        self.assertIn("/rag", handlers)
        self.assertIn("/mcp", handlers)
        self.assertIn("/agents", handlers)

    def test_build_repl_handlers_intercepted_subcommands_removed(self) -> None:
        async def dummy_async(_: object) -> None:
            pass

        out = io.StringIO()
        console = Console(file=out)
        handlers = build_repl_handlers(
            task_ctx=TasksContext(console=_console()),
            skills_ctx=SkillsContext(console=_console()),
            get_rag_ctx=lambda: RAGContext(rag_manager=RAGManager(RAGSettings()), console=_console()),
            console=console,
            current_model=lambda: "gemma4:26b",
            base_url=lambda: "http://localhost:11434",
            switch_model=dummy_async,
            handle_yolo=lambda _: None,
            get_runtime=lambda: MagicMock(),
            current_thread_id=lambda: "",
            switch_effort=dummy_async,
        )

        # Inline-intercepted branches are gone from the registry handlers.
        task_handler = handlers["/task"].handler
        skill_handler = handlers["/skill"].handler
        session_handler = handlers["/session"].handler

        # /task create and /task run now report unknown subcommand.
        task_handler(["create", "my-task"])
        self.assertIn("Unknown task subcommand 'create'", out.getvalue())
        task_handler(["run", "my-task"])
        self.assertIn("Unknown task subcommand 'run'", out.getvalue())

        # /skill create reports unknown subcommand.
        skill_handler(["create", "my-skill"])
        self.assertIn("Unknown skill subcommand 'create'", out.getvalue())

        # /session new/resume/export report unknown subcommand.
        for sub in ("new", "resume", "switch", "export"):
            session_handler([sub, "x"])
        rendered = out.getvalue()
        for sub in ("new", "resume", "switch", "export"):
            self.assertIn(f"Unknown session subcommand '{sub}'", rendered)

    def test_main_config_reset(self) -> None:
        # Patch set_locale so the system language does not leak into other tests.
        with patch("sys.argv", ["ollama-agent", "--config-reset", "config-file"]), \
             patch("ollama_agent.main.set_locale", return_value="en"), \
             patch("ollama_agent.main.reset_config", return_value=["Reset successful"]) as mock_reset, \
             patch("builtins.print") as mock_print:
            main()
            mock_reset.assert_called_once_with("config-file")
            mock_print.assert_called_once_with("Reset successful")

    def test_main_cli_commands_handled(self) -> None:
        with patch("sys.argv", ["ollama-agent", "task", "list"]), \
             patch("ollama_agent.main.set_locale", return_value="en"), \
             patch("ollama_agent.main.load_settings", return_value=Settings()), \
             patch("ollama_agent.main.handle_cli_commands", return_value=True) as mock_handle:
            main()
            mock_handle.assert_called_once()

    def test_main_rejects_prompt_with_subcommand(self) -> None:
        with patch("sys.argv", ["ollama-agent", "task", "list", "-p", "hello"]), \
             patch("ollama_agent.main.set_locale", return_value="en"):
            with self.assertRaises(SystemExit) as cm:
                main()
        self.assertEqual(cm.exception.code, 2)

    def test_main_repl_flow(self) -> None:
        mock_settings = Settings()
        with patch("sys.argv", ["ollama-agent", "-m", "qwen3:32b", "-e", "high", "--builtin-tool-timeout", "40"]), \
             patch("ollama_agent.main.set_locale", return_value="en"), \
             patch("ollama_agent.main.load_settings", return_value=mock_settings), \
             patch("ollama_agent.main.ensure_model_configured", return_value="qwen3:32b") as mock_ensure, \
             patch("ollama_agent.main.handle_cli_commands", return_value=False), \
             patch("ollama_agent.main.AgentRuntime") as mock_runtime_cls, \
             patch("ollama_agent.main.OllamaREPL") as mock_repl_cls, \
             patch("asyncio.run") as mock_asyncio_run:
            main()
            mock_ensure.assert_called_once_with(mock_settings)
            self.assertEqual(mock_settings.model.name, "qwen3:32b")
            self.assertEqual(mock_settings.model.reasoning_effort, "high")
            self.assertEqual(mock_settings.runtime.builtin_tool_timeout, 40)
            mock_runtime_cls.assert_called_once()
            mock_repl_cls.assert_called_once()
            mock_asyncio_run.assert_called_once()

    def test_main_model_capability_error_exits(self) -> None:
        with patch("sys.argv", ["ollama-agent"]), \
             patch("ollama_agent.main.set_locale", return_value="en"), \
             patch("ollama_agent.main.Console"), \
             patch("ollama_agent.main.load_settings", return_value=Settings()), \
             patch("ollama_agent.main.ensure_model_configured", side_effect=ModelCapabilityError("Model unsupported")):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 1)

    def test_cli_rag_add_file_and_directory(self) -> None:
        parser = create_argument_parser()
        args_file = parser.parse_args(["rag", "add", "docs_db", "file.md"])
        mock_load = MagicMock()
        mock_add_file = AsyncMock()
        mock_add_dir = AsyncMock()

        with patch("ollama_agent.interfaces.dispatch.load_rag_database", mock_load), \
             patch("ollama_agent.interfaces.dispatch.add_rag_file", mock_add_file), \
             patch("ollama_agent.interfaces.dispatch.add_rag_directory", mock_add_dir):
            handlers = _build_cli_handlers(args_file)
            asyncio.run(handlers[("rag", "add")]())
            mock_load.assert_called_once()
            mock_add_file.assert_called_once()
            mock_add_dir.assert_not_called()

        mock_load.reset_mock()
        mock_add_file.reset_mock()
        mock_add_dir.reset_mock()

        args_dir = parser.parse_args(["rag", "add", "docs_db", "./docs", "--dir"])
        with patch("ollama_agent.interfaces.dispatch.load_rag_database", mock_load), \
             patch("ollama_agent.interfaces.dispatch.add_rag_file", mock_add_file), \
             patch("ollama_agent.interfaces.dispatch.add_rag_directory", mock_add_dir):
            handlers = _build_cli_handlers(args_dir)
            asyncio.run(handlers[("rag", "add")]())
            mock_load.assert_called_once()
            mock_add_file.assert_not_called()
            mock_add_dir.assert_called_once()

    def test_cli_session_export(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["session", "export", "sess-123", "-o", "export.md"])
        args._runtime = AsyncMock()

        with patch("ollama_agent.interfaces.dispatch.export_session", AsyncMock()) as mock_export:
            handlers = _build_cli_handlers(args)
            asyncio.run(handlers[("session", "export")]())
            mock_export.assert_called_once()


if __name__ == "__main__":
    unittest.main()
