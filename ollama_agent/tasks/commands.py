"""Shared task management commands used by CLI and REPL."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jinja2.exceptions import TemplateError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ..agent import AgentRuntime
from ..agent.builtin_tools import set_rag_manager
from ..core import DEFAULT_REASONING_EFFORT, ReasoningEffortValue, check_ollama_version, require_text
from ..i18n import _
from ..rag import RAGContext, RAGManager, load_rag_database
from ..settings import Settings
from ..settings.config import _task_model_overrides
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
    settings: Settings = field(default_factory=Settings)

    def resolve_task(self, task_id: str) -> tuple[str, Task]:
        return self.task_manager.resolve(
            task_id,
            label=_("Task"),
            not_found_error=TaskNotFoundError,
            ambiguous_error=AmbiguousTaskError,
            validation_error=ValidationError,
        )


def parse_var_assignments(args: list[str]) -> dict[str, str]:
    """Parse 'key=value' argument strings into a dictionary."""
    result: dict[str, str] = {}
    for item in args:
        if "=" not in item:
            raise ValidationError(_("Invalid variable assignment '{item}'. Expected 'key=value'.", item=item))
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValidationError(_("Invalid variable assignment '{item}'. Expected 'key=value'.", item=item))
        result[key] = value
    return result


def apply_task_settings(
    settings: Settings,
    task: Task,
    *,
    preserve_model: bool = False,
    preserve_effort: bool = False,
) -> Callable[[], None]:
    """Apply the task's model and reasoning effort onto *settings* and return a restore callback."""
    model_settings = settings.model
    key = id(model_settings)
    if key in _task_model_overrides and _task_model_overrides[key][0]() is model_settings:
        orig_name, orig_effort = _task_model_overrides[key][1], _task_model_overrides[key][2]
    else:
        orig_name, orig_effort = model_settings.name, model_settings.reasoning_effort
    if not preserve_model and task.model:
        model_settings.name = task.model
    if not preserve_effort and task.reasoning_effort:
        model_settings.reasoning_effort = task.reasoning_effort
    applied_name = model_settings.name
    applied_effort = model_settings.reasoning_effort
    ref = weakref.ref(model_settings, lambda _: _task_model_overrides.pop(key, None))
    _task_model_overrides[key] = (ref, orig_name, orig_effort, applied_name, applied_effort)

    def _restore() -> None:
        _task_model_overrides.pop(key, None)
        if model_settings.name == applied_name:
            model_settings.name = orig_name
        if model_settings.reasoning_effort == applied_effort:
            model_settings.reasoning_effort = orig_effort

    return _restore


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


def prepare_task_run(
    ctx: TasksContext,
    task_id: str,
    variables: dict[str, Any] | None = None,
) -> tuple[str, Task, str]:
    """Resolve *task_id* and render its prompt template with *variables*."""
    tid, task = ctx.resolve_task(task_id)
    try:
        rendered_prompt = task.render(variables)
    except (ValueError, TemplateError) as exc:
        raise ValidationError(str(exc)) from exc
    return tid, task, rendered_prompt


async def run_task(
    ctx: TasksContext,
    task_id: str,
    *,
    variables: dict[str, Any] | None = None,
    yolo: bool = False,
    stealth: bool = False,
    rag: str | None = None,
    preserve_model: bool = False,
    preserve_effort: bool = False,
) -> None:
    tid, t, rendered_prompt = prepare_task_run(ctx, task_id, variables)
    check_ollama_version(ctx.settings.model.base_url)
    model_settings = ctx.settings.model
    restore_settings = apply_task_settings(
        ctx.settings,
        t,
        preserve_model=preserve_model,
        preserve_effort=preserve_effort,
    )
    ctx.console.print(
        _(
            "Executing: {title} ({tid})\nPrompt: {prompt}\nModel: {model} | Effort: {effort}",
            title=escape(t.title),
            tid=escape(tid),
            prompt=escape(rendered_prompt),
            model=escape(model_settings.name),
            effort=escape(model_settings.reasoning_effort),
        )
    )
    runtime = AgentRuntime(settings=ctx.settings, yolo_mode=yolo, stealth_mode=stealth)
    rag_manager: RAGManager | None = None
    try:
        async with runtime:
            if rag:
                rag_manager = RAGManager(ctx.settings.rag)
                rag_ctx = RAGContext(rag_manager=rag_manager, console=ctx.console)
                set_rag_manager(rag_manager)
                load_rag_database(rag_ctx, rag)
            if not await run_non_interactive(runtime, rendered_prompt):
                raise TaskError(_("Task execution failed: {tid}", tid=tid))
    finally:
        if rag_manager is not None:
            set_rag_manager(None)
            await rag_manager.aclose()
        restore_settings()


def delete_task(ctx: TasksContext, task_id: str) -> None:
    tid, t = ctx.resolve_task(task_id)
    ctx.task_manager.delete(tid)
    ctx.console.print(f"[green]✓ {_('Task deleted: {title} ({tid})', title=escape(t.title), tid=escape(tid))}[/green]")


def create_task(
    ctx: TasksContext,
    task_id: str,
    *,
    title: str,
    prompt: str,
    model: str,
    reasoning_effort: ReasoningEffortValue = DEFAULT_REASONING_EFFORT,
    force: bool = False,
) -> None:
    title = require_text(title, _("Title"), ValidationError)
    prompt_text = require_text(prompt, _("Prompt"), ValidationError)
    model_name = require_text(model, _("Model"), ValidationError)
    try:
        task = Task(title, prompt_text, model_name, reasoning_effort=reasoning_effort)
        saved_id = ctx.task_manager.save(task_id, task, overwrite=force)
    except FileExistsError as exc:
        raise TaskError(_("Task already exists: {task_id} (use --force to overwrite)", task_id=task_id)) from exc
    except ValueError as e:
        raise ValidationError(str(e)) from e
    ctx.console.print(
        f"[green]✓ {_('Task created: {title} ({task_id})', title=escape(task.title), task_id=escape(saved_id))}[/green]"
    )
