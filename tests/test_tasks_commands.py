from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rich.console import Console

from ollama_agent.settings import Settings, load_settings
from ollama_agent.tasks.commands import (
    AmbiguousTaskError,
    TasksContext,
    TaskError,
    TaskNotFoundError,
    ValidationError,
    apply_task_settings,
    create_task,
    delete_task,
    list_tasks,
    parse_var_assignments,
    run_task,
)
from ollama_agent.tasks.manager import Task, TaskInput, TaskManager


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

    def test_apply_task_settings(self) -> None:
        task = Task(title="T", prompt="P", model="llama3:8b", reasoning_effort="high")
        settings = load_settings(settings_path=Path(self.temp_dir.name) / "settings.yaml")
        apply_task_settings(settings, task)
        self.assertEqual(settings.model.name, "llama3:8b")
        self.assertEqual(settings.model.reasoning_effort, "high")

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
        with self.assertRaises(FileNotFoundError):
            self.mgr.get("cleanup-task")

    async def test_run_task(self) -> None:
        create_task(
            self.ctx,
            "quick-task",
            title="Quick Task",
            prompt="summarize file",
            model="gemma4:26b",
        )
        with patch("ollama_agent.tasks.commands.run_non_interactive", AsyncMock(return_value=True)) as mock_run:
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

    async def test_run_task_raises_when_execution_fails(self) -> None:
        create_task(
            self.ctx,
            "failing-task",
            title="Failing Task",
            prompt="boom",
            model="gemma4:26b",
        )
        with patch("ollama_agent.tasks.commands.run_non_interactive", AsyncMock(return_value=False)):
            with patch("ollama_agent.agent.agent.AgentRuntime.reload", AsyncMock()):
                with patch("ollama_agent.tasks.commands.AgentRuntime") as mock_runtime_cls:
                    mock_instance = AsyncMock()
                    mock_runtime_cls.return_value = mock_instance
                    mock_instance.__aenter__.return_value = mock_instance
                    mock_instance.__aexit__.return_value = None

                    with self.assertRaises(TaskError):
                        await run_task(self.ctx, "failing-task")

    def test_create_task_rejects_empty_reasoning_effort(self) -> None:
        with self.assertRaises(ValidationError):
            create_task(
                self.ctx,
                "bad-effort",
                title="Bad Effort",
                prompt="p",
                model="gemma4:26b",
                reasoning_effort="",
            )

    def test_create_existing_task_raises(self) -> None:
        create_task(
            self.ctx,
            "dupe-task",
            title="Dupe",
            prompt="p",
            model="gemma4:26b",
        )
        with self.assertRaises(TaskError):
            create_task(
                self.ctx,
                "dupe-task",
                title="Dupe",
                prompt="p",
                model="gemma4:26b",
            )

    def test_create_task_default_reasoning_effort(self) -> None:
        create_task(
            self.ctx,
            "default-effort-task",
            title="Default Effort Task",
            prompt="summarize",
            model="gemma4:26b",
        )
        task = self.mgr.get("default-effort-task")
        self.assertEqual(task.reasoning_effort, "medium")

    def test_tasks_context_default_settings(self) -> None:
        self.assertIsInstance(self.ctx.settings, Settings)

    def test_resolve_task_errors(self) -> None:
        with self.assertRaises(TaskNotFoundError):
            self.ctx._resolve_task("missing_task")

        create_task(self.ctx, "deploy-prod", title="Prod Deploy", prompt="deploy", model="gemma4:26b")
        create_task(self.ctx, "deploy-staging", title="Staging Deploy", prompt="deploy", model="gemma4:26b")

        with self.assertRaises(AmbiguousTaskError):
            self.ctx._resolve_task("deploy")

    def test_parse_var_assignments_valid(self) -> None:
        raw = ["file=src/main.py", "strict=true", "count=42", "expr=a=b=c"]
        parsed = parse_var_assignments(raw)
        self.assertEqual(
            parsed,
            {
                "file": "src/main.py",
                "strict": "true",
                "count": "42",
                "expr": "a=b=c",
            },
        )

    def test_parse_var_assignments_empty(self) -> None:
        self.assertEqual(parse_var_assignments([]), {})

    def test_parse_var_assignments_edge_cases(self) -> None:
        raw = [
            "query=SELECT a=b WHERE c=d",
            "msg=hello world with spaces",
            "empty_val=",
            "  key_with_spaces  =value_with_spaces ",
        ]
        parsed = parse_var_assignments(raw)
        self.assertEqual(
            parsed,
            {
                "query": "SELECT a=b WHERE c=d",
                "msg": "hello world with spaces",
                "empty_val": "",
                "key_with_spaces": "value_with_spaces ",
            },
        )

    def test_parse_var_assignments_invalid_no_equals(self) -> None:
        with self.assertRaises(ValidationError) as cm:
            parse_var_assignments(["invalid_assignment"])
        self.assertIn("invalid_assignment", str(cm.exception))

    def test_parse_var_assignments_empty_key_raises(self) -> None:
        with self.assertRaises(ValidationError) as cm1:
            parse_var_assignments(["=value"])
        self.assertIn("=value", str(cm1.exception))

        with self.assertRaises(ValidationError) as cm2:
            parse_var_assignments(["   =value"])
        self.assertIn("=value", str(cm2.exception))

    async def test_run_task_with_variables(self) -> None:
        task = Task(
            title="Render Task",
            prompt="Analyze {{ file }} with mode={{ mode }}",
            model="gemma4:26b",
            inputs={
                "file": TaskInput(description="target file", required=True),
                "mode": TaskInput(description="run mode", default="fast"),
            },
        )
        self.mgr.save("render-task", task)

        with patch("ollama_agent.tasks.commands.run_non_interactive", AsyncMock(return_value=True)) as mock_run:
            with patch("ollama_agent.agent.agent.AgentRuntime.reload", AsyncMock()):
                with patch("ollama_agent.tasks.commands.AgentRuntime") as mock_runtime_cls:
                    mock_instance = AsyncMock()
                    mock_runtime_cls.return_value = mock_instance
                    mock_instance.__aenter__.return_value = mock_instance
                    mock_instance.__aexit__.return_value = None

                    await run_task(self.ctx, "render-task", variables={"file": "app.py"})
                    mock_run.assert_awaited_once_with(mock_instance, "Analyze app.py with mode=fast")

    async def test_run_task_missing_required_variable_raises_validation_error(self) -> None:
        task = Task(
            title="Required Var Task",
            prompt="Process {{ file }}",
            model="gemma4:26b",
            inputs={"file": TaskInput(description="target file", required=True)},
        )
        self.mgr.save("req-task", task)

        with self.assertRaises(ValidationError) as cm:
            await run_task(self.ctx, "req-task", variables={})
        self.assertIn("Missing required input: file", str(cm.exception))

    async def test_run_task_template_syntax_error_raises_validation_error(self) -> None:
        task = Task(
            title="Syntax Error Task",
            prompt="Bad syntax {{ unclosed",
            model="gemma4:26b",
        )
        self.mgr.save("syntax-task", task)

        with self.assertRaises(ValidationError):
            await run_task(self.ctx, "syntax-task")

