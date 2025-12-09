"""Command-line interface for the application."""

import argparse
import asyncio
from typing import Awaitable, Callable

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

    subparsers = parser.add_subparsers(dest="command", help="Task management commands")
    subparsers.add_parser("task-list", help="List all saved tasks")

    task_run = subparsers.add_parser("task-run", help="Execute a saved task")
    task_run.add_argument("task_id", type=str, help="Task ID or prefix to execute")

    task_delete = subparsers.add_parser("task-delete", help="Delete a saved task")
    task_delete.add_argument("task_id", type=str, help="Task ID or prefix to delete")

    return parser


class CLIRunner:
    """Encapsulates CLI command handling and shared state."""

    def __init__(self, agent_factory: Callable[..., OllamaAgent]) -> None:
        self.agent_factory = agent_factory
        self.console = Console()
        self.task_manager = TaskManager()

    def _find_task_or_exit(self, task_id: str) -> tuple[str, Task]:
        result = self.task_manager.find_by_prefix(task_id)
        if not result:
            self.console.print(f"[red]Task not found: {task_id}[/red]")
            raise SystemExit(1)
        return result

    def list_tasks(self) -> None:
        tasks = self.task_manager.list_all()
        if not tasks:
            self.console.print("[yellow]No tasks found.[/yellow]")
            return

        table = Table(title="Saved Tasks", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=10)
        table.add_column("Title", style="green")
        table.add_column("Model", style="blue")
        table.add_column("Effort", style="yellow")

        for task_id, task in tasks:
            table.add_row(task_id, task.title, task.model, task.reasoning_effort)

        self.console.print(table)

    async def run_task(self, task_id: str) -> None:
        found_id, task = self._find_task_or_exit(task_id)
        self.console.print(f"[bold cyan]Executing task:[/bold cyan] {task.title} ({found_id})")
        self.console.print(f"[bold blue]Prompt:[/bold blue] {task.prompt}")
        self.console.print(
            f"[bold]Model:[/bold] {task.model} | [bold]Effort:[/bold] {task.reasoning_effort}"
        )
        self.console.print("")

        agent = self.agent_factory(model=task.model, reasoning_effort=task.reasoning_effort)
        await run_non_interactive(agent, task.prompt)

    def delete_task(self, task_id: str) -> None:
        found_id, task = self._find_task_or_exit(task_id)
        if self.task_manager.delete(found_id):
            self.console.print(f"[green]Task deleted:[/green] {task.title} ({found_id})")
        else:
            self.console.print(f"[red]Error deleting task: {found_id}[/red]")

    async def run_prompt(self, prompt: str, model: str | None, effort: str | None) -> None:
        agent = self.agent_factory(model=model, reasoning_effort=effort)
        await run_non_interactive(agent, prompt)

    def _run(self, coro: Awaitable[None]) -> None:
        asyncio.run(coro)  # Thin wrapper for symmetry in command mapping

    def handle(self, args: argparse.Namespace) -> bool:
        commands = {
            "task-list": lambda: self.list_tasks(),
            "task-delete": lambda: self.delete_task(args.task_id),
            "task-run": lambda: self._run(self.run_task(args.task_id)),
        }

        if args.command in commands:
            commands[args.command]()
            return True

        if args.prompt:
            self._run(self.run_prompt(args.prompt, args.model, args.effort))
            return True

        return False


def handle_cli_commands(args: argparse.Namespace, agent_factory: Callable[..., OllamaAgent]) -> bool:
    """Handle CLI commands and return True if a command was handled."""
    return CLIRunner(agent_factory).handle(args)
