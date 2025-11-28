"""Command-line interface for the application."""

import argparse
import asyncio
from typing import Callable

from rich.console import Console
from rich.table import Table

from .agent import OllamaAgent
from .core import ALLOWED_REASONING_EFFORTS
from .tasks import Task, TaskManager
from .runner import run_non_interactive


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Ollama Agent - AI agent to interact with local models"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Specify the AI model to use"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        help="Non-interactive mode, provide a prompt directly from the command line"
    )
    parser.add_argument(
        "-e", "--effort",
        type=str,
        choices=list(ALLOWED_REASONING_EFFORTS),
        help="Set reasoning effort level (low, medium, high, disabled)"
    )
    parser.add_argument(
        "-t", "--builtin-tool-timeout",
        type=int,
        help="Set built-in tool execution timeout in seconds"
    )

    # Task management subcommands
    subparsers = parser.add_subparsers(
        dest="command", help="Task management commands")

    # task-list command
    subparsers.add_parser("task-list", help="List all saved tasks")

    # task-run command
    task_run = subparsers.add_parser("task-run", help="Execute a saved task")
    task_run.add_argument("task_id", type=str,
                          help="Task ID or prefix to execute")

    # task-delete command
    task_delete = subparsers.add_parser(
        "task-delete", help="Delete a saved task")
    task_delete.add_argument(
        "task_id", type=str, help="Task ID or prefix to delete")

    return parser




def find_task_or_exit(task_manager: TaskManager, task_id: str, console: Console) -> tuple[str, Task]:
    """Find a task by ID or prefix, exit if not found."""
    result = task_manager.find_task_by_prefix(task_id)

    if not result:
        console.print(f"[red]Task not found: {task_id}[/red]")
        raise SystemExit(1)

    return result


def list_tasks_command(task_manager: TaskManager, console: Console) -> None:
    """List all saved tasks."""
    tasks = task_manager.list_tasks()

    if not tasks:
        console.print("[yellow]No tasks found.[/yellow]")
        return

    table = Table(title="Saved Tasks", show_header=True,
                  header_style="bold magenta")
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Title", style="green")
    table.add_column("Model", style="blue")
    table.add_column("Effort", style="yellow")

    for task_id, task in tasks:
        table.add_row(task_id, task.title, task.model, task.reasoning_effort)

    console.print(table)


async def run_task_command(
    task_id: str,
    agent_factory: Callable[..., OllamaAgent],
    task_manager: TaskManager,
    console: Console
) -> None:
    """Execute a saved task."""
    found_id, task = find_task_or_exit(task_manager, task_id, console)

    console.print(
        f"[bold cyan]Executing task:[/bold cyan] {task.title} ({found_id})")
    console.print(f"[bold blue]Prompt:[/bold blue] {task.prompt}")
    console.print(
        f"[bold]Model:[/bold] {task.model} | [bold]Effort:[/bold] {task.reasoning_effort}")
    console.print("")

    agent = agent_factory(
        model=task.model, reasoning_effort=task.reasoning_effort)
    await run_non_interactive(agent, task.prompt)


def delete_task_command(task_id: str, task_manager: TaskManager, console: Console) -> None:
    """Delete a saved task."""
    found_id, task = find_task_or_exit(task_manager, task_id, console)

    if task_manager.delete_task(found_id):
        console.print(
            f"[green]Task deleted:[/green] {task.title} ({found_id})")
    else:
        console.print(f"[red]Error deleting task: {found_id}[/red]")


def handle_cli_commands(args: argparse.Namespace, agent_factory: Callable[..., OllamaAgent]) -> bool:
    """Handle CLI commands and return True if a command was handled."""
    console = Console()
    task_manager = TaskManager()

    commands = {
        "task-list": lambda: list_tasks_command(task_manager, console),
        "task-delete": lambda: delete_task_command(args.task_id, task_manager, console),
        "task-run": lambda: asyncio.run(run_task_command(args.task_id, agent_factory, task_manager, console)),
    }

    if args.command in commands:
        commands[args.command]()
        return True

    if args.prompt:
        agent = agent_factory(model=args.model, reasoning_effort=args.effort)
        asyncio.run(run_non_interactive(agent, args.prompt))
        return True

    return False
