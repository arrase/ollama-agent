"""Shared task management commands used by CLI and REPL."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ..agent import AgentRuntime
from ..core import DEFAULT_REASONING_EFFORT
from ..core.resource_manager import require_text, resolve_unique_match
from ..i18n import _
from ..settings import Settings, load_settings
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
    settings: Settings | None = None

    def _find_or_exit(self, task_id: str) -> tuple[str, Task]:
        try:
            matches = self.task_manager.find_matches(task_id)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        return resolve_unique_match(
            matches,
            task_id,
            label=_("Task"),
            not_found_error=TaskNotFoundError,
            ambiguous_error=AmbiguousTaskError,
        )

    def _require(self, value: str, name: str) -> str:
        return require_text(value, name, ValidationError)


def apply_task_settings(settings: Settings, task: Task) -> None:
    """Apply the task's model and reasoning effort onto *settings*."""
    settings.model.name = task.model
    settings.model.reasoning_effort = task.reasoning_effort


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
        table.add_row(escape(tid), escape(t.title), escape(t.model), escape(t.reasoning_effort))
    ctx.console.print(table)


async def run_task(ctx: TasksContext, task_id: str, *, yolo: bool = False) -> None:
    tid, t = ctx._find_or_exit(task_id)
    ctx.console.print(
        _("Executing: {title} ({tid})\nPrompt: {prompt}\nModel: {model} | Effort: {effort}", title=escape(t.title), tid=escape(tid), prompt=escape(t.prompt), model=escape(t.model), effort=escape(t.reasoning_effort))
    )
    settings = ctx.settings if ctx.settings is not None else load_settings()
    apply_task_settings(settings, t)
    runtime = AgentRuntime(settings=settings, yolo_mode=yolo)
    async with runtime:
        await runtime.reload()
        if not await run_non_interactive(runtime, t.prompt):
            raise TaskError(_("Task execution failed: {tid}", tid=tid))


def delete_task(ctx: TasksContext, task_id: str) -> None:
    tid, t = ctx._find_or_exit(task_id)
    ctx.task_manager.delete(tid)
    ctx.console.print(f"[green]✓ {_('Task deleted: {title} ({tid})', title=escape(t.title), tid=escape(tid))}[/green]")


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
    title = ctx._require(title, _("Title"))
    prompt_text = ctx._require(prompt, _("Prompt"))
    model_name = ctx._require(model, _("Model"))
    effort = DEFAULT_REASONING_EFFORT if reasoning_effort is None else reasoning_effort
    try:
        task = Task(title, prompt_text, model_name, reasoning_effort=effort)
        saved_id = ctx.task_manager.save(task_id, task, overwrite=force)
    except FileExistsError as exc:
        raise TaskError(
            _("Task already exists: {task_id} (use --force to overwrite)", task_id=task_id)
        ) from exc
    except ValueError as e:
        raise ValidationError(str(e)) from e
    ctx.console.print(
        f"[green]✓ {_('Task created: {title} ({task_id})', title=escape(task.title), task_id=escape(saved_id))}[/green]"
    )
