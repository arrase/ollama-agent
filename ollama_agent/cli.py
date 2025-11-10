"""Command-line interface for the application."""

import argparse
import asyncio
from typing import Any, Callable, Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table

from .agent import OllamaAgent
from .tasks import Task, TaskManager
from .streaming import EventHandler, stream_agent_events
from .utils import ALLOWED_REASONING_EFFORTS


class _StreamingConsole:
    """Stateful renderer for non-interactive streaming output."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.live = Live(console=console, refresh_per_second=10)
        self._text: list[str] = []
        self._agent_banner_shown = False
        self._reasoning = False
        self._live_active = False

    def close(self) -> None:
        self._stop_live()
        self.console.print()

    def handlers(self) -> dict[str, EventHandler]:
        return {
            "text_delta": self._on_text_delta,
            "reasoning_delta": self._on_reasoning_delta,
            "tool_call": self._on_tool_call,
            "tool_output": self._on_tool_output,
        }

    def on_error(self, event: dict[str, Any]) -> None:
        self._stop_live()
        self.console.print(
            f"\n[red]❌ Error: {event.get('content', 'Unknown error')}[/red]"
        )

    def _start_live(self) -> None:
        if not self._live_active:
            self.live.start()
            self._live_active = True

    def _stop_live(self) -> None:
        if self._live_active:
            self.live.stop()
            self._live_active = False

    def _ensure_agent_banner(self) -> None:
        if not self._agent_banner_shown:
            self.console.print("\n[bold green]Agent:[/bold green]")
            self._agent_banner_shown = True

    def _conclude_reasoning(self) -> None:
        if self._reasoning:
            self._reasoning = False
            self.console.print()

    def _on_text_delta(self, event: dict[str, Any]) -> None:
        self._conclude_reasoning()
        self._ensure_agent_banner()
        self._start_live()
        self._text.append(event.get("content", ""))
        self.live.update(Markdown("".join(self._text)))

    def _on_reasoning_delta(self, event: dict[str, Any]) -> None:
        if not self._reasoning:
            self._stop_live()
            self.console.print("\n[bold magenta]🧠 Thinking:[/bold magenta] ", end="")
            self._reasoning = True
        self.console.print(event.get("content", ""), end="", style="dim italic magenta")

    def _on_tool_call(self, event: dict[str, Any]) -> None:
        self._conclude_reasoning()
        self._stop_live()
        self.console.print(
            f"\n[yellow]🔧 Calling tool: {event.get('name', 'unknown')}[/yellow]"
        )

    def _on_tool_output(self, event: dict[str, Any]) -> None:
        self._stop_live()
        output = event.get("output", "")
        preview = f"{output[:100]}..." if len(output) > 100 else output
        self.console.print(f"[cyan]📤 Tool output: {preview}[/cyan]\n")


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


async def run_non_interactive(
    agent: OllamaAgent,
    prompt: str,
    model: Optional[str] = None,
    effort: Optional[str] = None,
) -> None:
    """Stream agent output to the console."""
    renderer = _StreamingConsole(Console())
    try:
        await stream_agent_events(
            agent,
            prompt,
            renderer.handlers(),
            model=model,
            reasoning_effort=effort,
            on_error=renderer.on_error,
            ignore={"agent_update"},
        )
    finally:
        renderer.close()
        await agent.cleanup()


def find_task_or_exit(task_manager: TaskManager, task_id: str, console: Console) -> tuple[str, Task]:
    """Find a task by ID or prefix, exit if not found."""
    result = task_manager.find_task_by_prefix(task_id)

    if not result:
        console.print(f"[red]Task not found: {task_id}[/red]")
        raise SystemExit(1)

    return result


def list_tasks_command() -> None:
    """List all saved tasks."""
    console = Console()
    task_manager = TaskManager()
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


async def run_task_command(task_id: str, create_agent_func: Callable[..., OllamaAgent]) -> None:
    """Execute a saved task."""
    console = Console()
    task_manager = TaskManager()

    found_id, task = find_task_or_exit(task_manager, task_id, console)

    console.print(
        f"[bold cyan]Executing task:[/bold cyan] {task.title} ({found_id})")
    console.print(f"[bold blue]Prompt:[/bold blue] {task.prompt}")
    console.print(
        f"[bold]Model:[/bold] {task.model} | [bold]Effort:[/bold] {task.reasoning_effort}")
    console.print("")

    agent = create_agent_func(
        model=task.model, reasoning_effort=task.reasoning_effort)
    await run_non_interactive(agent, task.prompt)


def delete_task_command(task_id: str) -> None:
    """Delete a saved task."""
    console = Console()
    task_manager = TaskManager()

    found_id, task = find_task_or_exit(task_manager, task_id, console)

    if task_manager.delete_task(found_id):
        console.print(
            f"[green]Task deleted:[/green] {task.title} ({found_id})")
    else:
        console.print(f"[red]Error deleting task: {found_id}[/red]")


def handle_cli_commands(args: argparse.Namespace, create_agent_func: Callable[..., OllamaAgent]) -> bool:
    """Handle CLI commands and return True if a command was handled."""
    if args.command == "task-list":
        list_tasks_command()
        return True
    if args.command == "task-delete":
        delete_task_command(args.task_id)
        return True
    if args.command == "task-run":
        asyncio.run(run_task_command(args.task_id, create_agent_func))
        return True
    if args.prompt:
        agent = create_agent_func(
            model=args.model, reasoning_effort=args.effort)
        asyncio.run(run_non_interactive(agent, args.prompt))
        return True
    return False
