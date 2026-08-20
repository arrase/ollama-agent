from __future__ import annotations

import argparse
import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from ollama_agent.core import ModelCapabilityError, ModelContextWindowError
from ollama_agent.interfaces.cli import create_argument_parser
from ollama_agent.interfaces.dispatch import (
    REPL_SECTIONS,
    build_cli_handlers,
    build_repl_handlers,
)
from ollama_agent.main import main
from ollama_agent.rag.commands import RAGContext
from ollama_agent.rag.manager import RAGManager
from ollama_agent.rag.settings import RAGSettings
from ollama_agent.settings.config import Settings
from ollama_agent.skills.commands import SkillsContext
from ollama_agent.tasks.commands import CLIContext


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
        args1 = parser.parse_args(["task-run", "my-task", "-y"])
        self.assertEqual(args1.task_id, "my-task")
        self.assertTrue(args1.yolo)

        args2 = parser.parse_args(["-y", "task-run", "my-task"])
        self.assertEqual(args2.task_id, "my-task")
        self.assertTrue(args2.yolo)

        args3 = parser.parse_args(["task-run", "my-task"])
        self.assertEqual(args3.task_id, "my-task")
        self.assertFalse(args3.yolo)

    def test_build_cli_handlers_registry(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["task-list"])
        handlers = build_cli_handlers(
            args,
            task_ctx=CLIContext(console=Console(file=io.StringIO())),
            rag_ctx=RAGContext(rag_manager=RAGManager(RAGSettings()), console=Console(file=io.StringIO())),
            skills_ctx=SkillsContext(console=Console(file=io.StringIO())),
        )
        self.assertIn("task-list", handlers)
        self.assertIn("task-create", handlers)
        self.assertIn("rag-list", handlers)
        self.assertIn("skill-list", handlers)

    def test_build_repl_handlers_registry(self) -> None:
        async def dummy_async_str(_: str) -> None:
            pass

        async def dummy_async_list(_: list[str]) -> None:
            pass

        handlers = build_repl_handlers(
            task_ctx=CLIContext(console=Console(file=io.StringIO())),
            skills_ctx=SkillsContext(console=Console(file=io.StringIO())),
            get_rag_ctx=lambda: RAGContext(rag_manager=RAGManager(RAGSettings()), console=Console(file=io.StringIO())),
            console=Console(file=io.StringIO()),
            current_model=lambda: "gemma4:26b",
            base_url=lambda: "http://localhost:11434",
            switch_model=dummy_async_str,
            handle_exit=lambda _: None,
            handle_clear=lambda _: None,
            handle_new=dummy_async_list,
            handle_task_create=lambda _: None,
            handle_skill_create=lambda _: None,
            handle_yolo=lambda _: None,
        )

        self.assertIn("/help", handlers)
        self.assertIn("/model", handlers)
        self.assertIn("/yolo", handlers)
        self.assertIn("/new", handlers)
        self.assertIn("/task", handlers)
        self.assertIn("/skill", handlers)
        self.assertIn("/rag", handlers)
        self.assertIn("/clear", handlers)
        self.assertIn("/exit", handlers)

        for cmd in handlers.values():
            self.assertIn(cmd.section, REPL_SECTIONS)

    def test_main_config_reset(self) -> None:
        with patch("sys.argv", ["ollama-agent", "--config-reset", "config-file"]), \
             patch("ollama_agent.main.reset_config", return_value=["Reset successful"]) as mock_reset, \
             patch("builtins.print") as mock_print:
            main()
            mock_reset.assert_called_once_with("config-file")
            mock_print.assert_called_once_with("Reset successful")

    def test_main_cli_commands_handled(self) -> None:
        with patch("sys.argv", ["ollama-agent", "task-list"]), \
             patch("ollama_agent.main.load_settings", return_value=Settings()), \
             patch("ollama_agent.main.handle_cli_commands", return_value=True) as mock_handle:
            main()
            mock_handle.assert_called_once()

    def test_main_repl_flow(self) -> None:
        mock_settings = Settings()
        with patch("sys.argv", ["ollama-agent", "-m", "qwen3:32b", "-e", "high", "--builtin-tool-timeout", "40"]), \
             patch("ollama_agent.main.load_settings", return_value=mock_settings), \
             patch("ollama_agent.main.handle_cli_commands", return_value=False), \
             patch("ollama_agent.main.AgentRuntime") as mock_runtime_cls, \
             patch("ollama_agent.main.OllamaREPL") as mock_repl_cls, \
             patch("asyncio.run") as mock_asyncio_run:
            main()
            self.assertEqual(mock_settings.model.name, "qwen3:32b")
            self.assertEqual(mock_settings.model.reasoning_effort, "high")
            self.assertEqual(mock_settings.runtime.builtin_tool_timeout, 40)
            mock_runtime_cls.assert_called_once()
            mock_repl_cls.assert_called_once()
            mock_asyncio_run.assert_called_once()

    def test_main_model_capability_error_exits(self) -> None:
        with patch("sys.argv", ["ollama-agent"]), \
             patch("ollama_agent.main.Console"), \
             patch("ollama_agent.main.load_settings", return_value=Settings()), \
             patch("ollama_agent.main.handle_cli_commands", side_effect=ModelCapabilityError("Model unsupported")):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
