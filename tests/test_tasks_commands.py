from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rich.console import Console

from ollama_agent.tasks.commands import (
    AmbiguousTaskError,
    TasksContext,
    TaskNotFoundError,
    ValidationError,
    create_task,
    delete_task,
    list_tasks,
    run_task,
)
from ollama_agent.tasks.manager import TaskManager


class TestTasksCommands(unittest.IsolatedAsyncioTestCase):
    """Unit tests for tasks command operations and resolution."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mgr = TaskManager(tasks_dir=Path(self.temp_dir.name))
        self.console = Console(file=io.StringIO(), record=True)
        self.ctx = TasksContext(console=self.console, task_manager=self.mgr)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_require_validation(self) -> None:
        self.assertEqual(self.ctx._require("  valid title  ", "Title"), "valid title")
        with self.assertRaises(ValidationError):
            self.ctx._require("   ", "Title")

    def test_create_and_list_task(self) -> None:
        create_task(
            self.ctx,
            "run-tests",
            title="Run Unit Tests",
            prompt="pytest -v",
            model="gemma4:26b",
            reasoning_effort="high",
        )
        task = self.mgr.get("run-tests")
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.title, "Run Unit Tests")

        list_tasks(self.ctx)
        out = self.console.export_text()
        self.assertIn("Run Unit Tests", out)

    def test_delete_task(self) -> None:
        create_task(
            self.ctx,
            "cleanup-task",
            title="Cleanup",
            prompt="rm -rf /tmp/test",
            model="gemma4:26b",
        )
        delete_task(self.ctx, "cleanup-task")
        self.assertIsNone(self.mgr.get("cleanup-task"))

    async def test_run_task(self) -> None:
        create_task(
            self.ctx,
            "quick-task",
            title="Quick Task",
            prompt="summarize file",
            model="gemma4:26b",
        )
        with patch("ollama_agent.tasks.commands.run_non_interactive", AsyncMock()) as mock_run:
            with patch("ollama_agent.agent.agent.AgentRuntime.reload", AsyncMock()):
                with patch("ollama_agent.tasks.commands.AgentRuntime") as mock_runtime_cls:
                    mock_instance = AsyncMock()
                    mock_runtime_cls.return_value = mock_instance
                    mock_instance.__aenter__.return_value = mock_instance
                    mock_instance.__aexit__.return_value = None

                    await run_task(self.ctx, "quick-task", yolo=True)
                    mock_runtime_cls.assert_called_once()
                    self.assertTrue(mock_runtime_cls.call_args.kwargs.get("yolo_mode"))
                    mock_run.assert_awaited_once()

    def test_find_or_exit_errors(self) -> None:
        with self.assertRaises(TaskNotFoundError):
            self.ctx._find_or_exit("missing_task")

        create_task(self.ctx, "deploy-prod", title="Prod Deploy", prompt="deploy", model="gemma4:26b")
        create_task(self.ctx, "deploy-staging", title="Staging Deploy", prompt="deploy", model="gemma4:26b")

        with self.assertRaises(AmbiguousTaskError):
            self.ctx._find_or_exit("deploy")

