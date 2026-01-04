"""Shared task management commands used by CLI and REPL."""

from dataclasses import dataclass, field
from typing import Callable

from rich.console import Console
from rich.table import Table

from ..agent import OllamaAgent
from ..execution import run_non_interactive
from .manager import Task, TaskManager


@dataclass
class CLIContext:
    """Holds shared resources for task-related commands."""
    agent_factory: Callable[..., OllamaAgent]
    console: Console = field(default_factory=Console)
    task_manager: TaskManager = field(default_factory=TaskManager)


def find_task_or_exit(ctx: CLIContext, task_id: str) -> tuple[str, Task]:
    matches = ctx.task_manager.find_matches(task_id)
    if not matches:
        ctx.console.print(f"[red]Task not found: {task_id}[/red]")
        raise SystemExit(1)
    if len(matches) > 1:
        ctx.console.print(f"[red]Ambiguous task prefix:[/red] {task_id} -> {', '.join(tid for tid, _ in matches)}")
        raise SystemExit(1)
    return matches[0]


def list_tasks(ctx: CLIContext) -> None:
    tasks = ctx.task_manager.list_all()
    if not tasks:
        ctx.console.print("[yellow]No tasks found.[/yellow]")
        return

    table = Table(title="Saved Tasks", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Title", style="green")
    table.add_column("Model", style="blue")
    table.add_column("Effort", style="yellow")

    for task_id, task in tasks:
        table.add_row(task_id, task.title, task.model, task.reasoning_effort)

    ctx.console.print(table)


async def run_task(ctx: CLIContext, task_id: str) -> None:
    found_id, task = find_task_or_exit(ctx, task_id)
    ctx.console.print(f"[bold cyan]Executing task:[/bold cyan] {task.title} ({found_id})")
    ctx.console.print(f"[bold blue]Prompt:[/bold blue] {task.prompt}")
    ctx.console.print(
        f"[bold]Model:[/bold] {task.model} | [bold]Effort:[/bold] {task.reasoning_effort}"
    )
    ctx.console.print("")

    agent = ctx.agent_factory(model=task.model, reasoning_effort=task.reasoning_effort)
    await run_non_interactive(agent, task.prompt)


def delete_task(ctx: CLIContext, task_id: str) -> None:
    found_id, task = find_task_or_exit(ctx, task_id)
    if ctx.task_manager.delete(found_id):
        ctx.console.print(f"[green]Task deleted:[/green] {task.title} ({found_id})")
    else:
        ctx.console.print(f"[red]Error deleting task: {found_id}[/red]")


def _exit_if_empty(ctx: CLIContext, value: str, field: str) -> str:
    if not (value := (value or "").strip()):
        ctx.console.print(f"[red]{field} cannot be empty.[/red]")
        raise SystemExit(1)
    return value


def create_task(
    ctx: CLIContext, task_id: str, *, title: str, prompt: str, model: str,
    reasoning_effort: str | None = None, force: bool = False,
) -> None:
    title = _exit_if_empty(ctx, title, "Title")
    prompt = _exit_if_empty(ctx, prompt.strip("\n") if prompt else "", "Prompt")
    model = _exit_if_empty(ctx, model, "Model")

    task = Task(title=title, prompt=prompt, model=model,
                reasoning_effort=reasoning_effort or "medium")
    try:
        saved_id = ctx.task_manager.save(task_id, task, overwrite=force)
        ctx.console.print(f"[green]Task created:[/green] {task.title} ({saved_id})")
    except FileExistsError:
        ctx.console.print(f"[red]Task already exists:[/red] {task_id} (use --force to overwrite)")
        raise SystemExit(1)
    except ValueError as e:
        ctx.console.print(f"[red]{e}[/red]")
        raise SystemExit(1)
