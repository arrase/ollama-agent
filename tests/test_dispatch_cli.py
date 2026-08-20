from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from rich.console import Console

from ollama_agent.interfaces.cli import create_argument_parser
from ollama_agent.interfaces.dispatch import (
    REPL_SECTIONS,
    build_cli_handlers,
    build_repl_handlers,
)
from ollama_agent.rag.commands import RAGContext
from ollama_agent.rag.manager import RAGManager
from ollama_agent.rag.settings import RAGSettings
from ollama_agent.skills.commands import SkillsContext
from ollama_agent.tasks.commands import CLIContext


class TestDispatchAndCLI(unittest.TestCase):
    """Unit tests for CLI argument parsing and dispatch handlers."""

    def test_argument_parser_flags(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["-m", "llama3:8b", "-e", "high", "-y", "--prompt", "hello"])
        self.assertEqual(args.model, "llama3:8b")
        self.assertEqual(args.effort, "high")
        self.assertTrue(args.yolo)
        self.assertEqual(args.prompt, "hello")

    def test_build_cli_handlers_registry(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["task-list"])
        handlers = build_cli_handlers(
            args,
            task_ctx=CLIContext(console=Console()),
            rag_ctx=RAGContext(rag_manager=RAGManager(RAGSettings())),
            skills_ctx=SkillsContext(console=Console()),
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
            task_ctx=CLIContext(console=Console()),
            skills_ctx=SkillsContext(console=Console()),
            get_rag_ctx=lambda: RAGContext(rag_manager=RAGManager(RAGSettings())),
            console=Console(),
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
        self.assertIn("/models", handlers)
        self.assertIn("/yolo", handlers)
        self.assertIn("/new", handlers)
        self.assertIn("/tasks", handlers)
        self.assertIn("/skills", handlers)
        self.assertIn("/rag", handlers)

        for cmd in handlers.values():
            self.assertIn(cmd.section, REPL_SECTIONS)


if __name__ == "__main__":
    unittest.main()
