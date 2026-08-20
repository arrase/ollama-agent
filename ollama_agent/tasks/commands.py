"""Shared task management commands used by CLI and REPL."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from ..agent import AgentRuntime
from ..core import DEFAULT_REASONING_EFFORT, validate_reasoning_effort
from ..settings import load_settings
from ..streaming import run_non_interactive
from .manager import Task, TaskManager


class TaskError(Exception):
    """Base exception for task command failures."""


class TaskNotFoundError(TaskError):
    """Raised when a task cannot be resolved by its ID/prefix."""


class AmbiguousTaskError(TaskError):
    """Raised when a task ID prefix matches multiple tasks."""


class ValidationError(TaskError):
    """Raised when a validation rule fails for task parameters."""


@dataclass
class TasksContext:
    """Holds shared resources for task-related commands."""

    console: Console = field(default_factory=Console)
    task_manager: TaskManager = field(default_factory=TaskManager)

    def _find_or_exit(self, task_id: str) -> tuple[str, Task]:
        matches = self.task_manager.find_matches(task_id)
        if len(matches) == 1:
            return matches[0]
        msg = (
            f"Task not found: {task_id}"
            if not matches
            else f"Ambiguous prefix: {task_id} -> {', '.join(t[0] for t in matches)}"
        )
        if not matches:
            raise TaskNotFoundError(msg)
        raise AmbiguousTaskError(msg)

    def _require(self, value: str, name: str) -> str:
        if not (cleaned := value.strip()):
            raise ValidationError(f"{name} cannot be empty.")
        return cleaned


# Backward-compatible alias
CLIContext = TasksContext


def list_tasks(ctx: TasksContext) -> None:
    if not (tasks := ctx.task_manager.list_all()):
        ctx.console.print("[yellow]No tasks found.[/yellow]")
        return
    table = Table(title="Saved Tasks", show_header=True, header_style="bold magenta")
    for col, style in [
        ("ID", "cyan"),
        ("Title", "green"),
        ("Model", "blue"),
        ("Effort", "yellow"),
    ]:
        table.add_column(col, style=style)
    for tid, t in tasks:
        table.add_row(tid, t.title, t.model, t.reasoning_effort)
    ctx.console.print(table)


async def run_task(ctx: TasksContext, task_id: str, *, yolo: bool = False) -> None:
    tid, t = ctx._find_or_exit(task_id)
    ctx.console.print(
        f"[bold cyan]Executing:[/bold cyan] {t.title} ({tid})\n"
        f"[bold blue]Prompt:[/bold blue] {t.prompt}\n"
        f"[bold]Model:[/bold] {t.model} | [bold]Effort:[/bold] {t.reasoning_effort}\n"
    )
    settings = load_settings()
    settings.model.name = t.model
    settings.model.reasoning_effort = t.reasoning_effort
    runtime = AgentRuntime(settings=settings, yolo_mode=yolo)
    async with runtime:
        await runtime.reload()
        await run_non_interactive(runtime, t.prompt)


def delete_task(ctx: TasksContext, task_id: str) -> None:
    tid, t = ctx._find_or_exit(task_id)
    if not ctx.task_manager.delete(tid):
        ctx.console.print(f"[red]Error deleting task: {tid}[/red]")
        raise TaskError(f"Error deleting task: {tid}")
    ctx.console.print(f"[green]Task deleted:[/green] {t.title} ({tid})")


def create_task(
    ctx: TasksContext,
    task_id: str,
    *,
    title: str,
    prompt: str,
    model: str,
    reasoning_effort: str | None = None,
    force: bool = False,
) -> None:
    task = Task(
        ctx._require(title, "Title"),
        ctx._require(prompt, "Prompt"),
        ctx._require(model, "Model"),
        validate_reasoning_effort(reasoning_effort or DEFAULT_REASONING_EFFORT),
    )
    try:
        ctx.console.print(
            f"[green]Task created:[/green] {task.title} ({ctx.task_manager.save(task_id, task, overwrite=force)})"
        )
    except FileExistsError as exc:
        ctx.console.print(
            f"[red]Task already exists:[/red] {task_id} (use --force to overwrite)"
        )
        raise TaskError(f"Task already exists: {task_id}") from exc
    except ValueError as e:
        ctx.console.print(f"[red]{e}[/red]")
        raise ValidationError(str(e)) from e

