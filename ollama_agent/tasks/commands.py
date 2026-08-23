"""Shared task management commands used by CLI and REPL."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from ..agent import AgentRuntime
from ..core import DEFAULT_REASONING_EFFORT, validate_reasoning_effort
from ..i18n import _
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
            _("Task not found: {task_id}", task_id=task_id)
            if not matches
            else _("Ambiguous prefix: {name} -> {matches}", name=task_id, matches=", ".join(t[0] for t in matches))
        )
        if not matches:
            raise TaskNotFoundError(msg)
        raise AmbiguousTaskError(msg)

    def _require(self, value: str, name: str) -> str:
        if not (cleaned := value.strip()):
            raise ValidationError(_("{name} cannot be empty.", name=name))
        return cleaned


# Backward-compatible alias
CLIContext = TasksContext


def list_tasks(ctx: TasksContext) -> None:
    if not (tasks := ctx.task_manager.list_all()):
        ctx.console.print(f"[yellow]{_('No tasks found.')}[/yellow]")
        return
    table = Table(title=_("Saved Tasks"), show_header=True, header_style="bold magenta")
    for col, style in [
        (_("ID"), "cyan"),
        (_("Title"), "green"),
        (_("Model"), "blue"),
        (_("Effort"), "yellow"),
    ]:
        table.add_column(col, style=style)
    for tid, t in tasks:
        table.add_row(tid, t.title, t.model, t.reasoning_effort)
    ctx.console.print(table)


async def run_task(ctx: TasksContext, task_id: str, *, yolo: bool = False) -> None:
    tid, t = ctx._find_or_exit(task_id)
    ctx.console.print(
        _("Executing: {title} ({tid})\nPrompt: {prompt}\nModel: {model} | Effort: {effort}", title=t.title, tid=tid, prompt=t.prompt, model=t.model, effort=t.reasoning_effort)
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
        ctx.console.print(f"[red]{_('Error deleting task: {tid}', tid=tid)}[/red]")
        raise TaskError(_("Error deleting task: {tid}", tid=tid))
    ctx.console.print(f"[green]✓ {_('Task deleted: {title} ({tid})', title=t.title, tid=tid)}[/green]")


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
        ctx._require(title, _("Title")),
        ctx._require(prompt, _("Prompt")),
        ctx._require(model, _("Model")),
        validate_reasoning_effort(reasoning_effort or DEFAULT_REASONING_EFFORT),
    )
    try:
        saved_id = ctx.task_manager.save(task_id, task, overwrite=force)
        ctx.console.print(
            f"[green]✓ {_('Task created: {title} ({task_id})', title=task.title, task_id=saved_id)}[/green]"
        )
    except FileExistsError as exc:
        ctx.console.print(
            f"[red]{_('Task already exists: {task_id} (use --force to overwrite)', task_id=task_id)}[/red]"
        )
        raise TaskError(_("Task already exists: {task_id}", task_id=task_id)) from exc
    except ValueError as e:
        ctx.console.print(f"[red]{e}[/red]")
        raise ValidationError(str(e)) from e
