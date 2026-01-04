"""Shared task management commands used by CLI and REPL."""

from typing import Callable

from rich.console import Console
from rich.table import Table

from ..agent import OllamaAgent
from ..execution import run_non_interactive
from .manager import Task, TaskManager


class CLIContext:
    """Holds shared resources for task-related commands."""

    def __init__(self, agent_factory: Callable[..., OllamaAgent]) -> None:
        self.agent_factory = agent_factory
        self.console = Console()
        self.task_manager = TaskManager()


def find_task_or_exit(ctx: CLIContext, task_id: str) -> tuple[str, Task]:
    result = ctx.task_manager.find_by_prefix(task_id)
    if not result:
        ctx.console.print(f"[red]Task not found: {task_id}[/red]")
        raise SystemExit(1)
    return result


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
