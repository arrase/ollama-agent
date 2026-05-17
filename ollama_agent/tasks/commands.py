"""Shared task management commands used by CLI and REPL."""

from dataclasses import dataclass, field
from typing import Any, Callable

from rich.console import Console
from rich.table import Table

from ..core import resolve_unique_prefix, validate_reasoning_effort
from ..interfaces.utils import require_or_exit
from ..streaming import run_non_interactive
from .manager import Task, TaskManager


@dataclass
class CLIContext:
    """Holds shared resources for task-related commands."""

    agent_factory: Callable[..., Any]
    console: Console = field(default_factory=Console)
    task_manager: TaskManager = field(default_factory=TaskManager)

    def _find_or_exit(self, task_id: str) -> tuple[str, Task]:
        matches = self.task_manager.find_matches(task_id)
        if len(matches) == 1:
            return matches[0]
        candidates = [p.stem for p in self.task_manager.tasks_dir.glob("*.yaml")]
        resolved = resolve_unique_prefix(task_id, sorted(candidates))
        if resolved and (t := self.task_manager.load(resolved)) is not None:
            return (resolved, t)
        msg = (
            f"Task not found: {task_id}"
            if not matches
            else f"Ambiguous prefix: {task_id} -> {', '.join(t[0] for t in matches)}"
        )
        self.console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)

    def _require(self, value: str, name: str) -> str:
        return require_or_exit(value, name, self.console)


def list_tasks(ctx: CLIContext) -> None:
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


async def run_task(ctx: CLIContext, task_id: str) -> None:
    from ..agent import AgentRuntime
    from ..settings import load_settings

    tid, t = ctx._find_or_exit(task_id)
    ctx.console.print(
        f"[bold cyan]Executing:[/bold cyan] {t.title} ({tid})\n"
        f"[bold blue]Prompt:[/bold blue] {t.prompt}\n"
        f"[bold]Model:[/bold] {t.model} | [bold]Effort:[/bold] {t.reasoning_effort}\n"
    )
    settings = load_settings()
    settings.model.name = t.model
    settings.model.reasoning_effort = t.reasoning_effort
    runtime = AgentRuntime(settings=settings)
    async with runtime:
        await runtime.reload()
        await run_non_interactive(runtime, t.prompt)


def delete_task(ctx: CLIContext, task_id: str) -> None:
    tid, t = ctx._find_or_exit(task_id)
    msg = (
        f"[green]Task deleted:[/green] {t.title} ({tid})"
        if ctx.task_manager.delete(tid)
        else f"[red]Error deleting task: {tid}[/red]"
    )
    ctx.console.print(msg)


def create_task(
    ctx: CLIContext,
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
        validate_reasoning_effort(reasoning_effort or "medium"),
    )
    try:
        ctx.console.print(
            f"[green]Task created:[/green] {task.title} ({ctx.task_manager.save(task_id, task, overwrite=force)})"
        )
    except FileExistsError:
        ctx.console.print(
            f"[red]Task already exists:[/red] {task_id} (use --force to overwrite)"
        )
        raise SystemExit(1)
    except ValueError as e:
        ctx.console.print(f"[red]{e}[/red]")
        raise SystemExit(1)
