"""Main entry point of the application."""

import argparse
import asyncio
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from . import config
from .agent import ALLOWED_REASONING_EFFORTS, OllamaAgent
from .tasks import Task, TaskManager
from .tools import set_timeout
from .tui import ChatInterface


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Ollama Agent - AI agent to interact with local models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specify the AI model to use"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Non-interactive mode, provide a prompt directly from the command line"
    )
    parser.add_argument(
        "--effort",
        type=str,
        choices=list(ALLOWED_REASONING_EFFORTS),
        help="Set reasoning effort level (low, medium, high)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Set command execution timeout in seconds"
    )
    
    # Task management subcommands
    subparsers = parser.add_subparsers(dest="command", help="Task management commands")
    
    # task-list command
    subparsers.add_parser("task-list", help="List all saved tasks")
    
    # task-run command
    task_run = subparsers.add_parser("task-run", help="Execute a saved task")
    task_run.add_argument("task_id", type=str, help="Task ID or prefix to execute")
    
    # task-delete command
    task_delete = subparsers.add_parser("task-delete", help="Delete a saved task")
    task_delete.add_argument("task_id", type=str, help="Task ID or prefix to delete")
    
    return parser


async def run_non_interactive(agent: OllamaAgent, prompt: str, model: Optional[str] = None, effort: Optional[str] = None) -> None:
    """
    Run the agent in non-interactive mode.
    
    Args:
        agent: The OllamaAgent instance.
        prompt: User prompt to process.
        model: Optional model override.
        effort: Optional reasoning effort override.
    """
    console = Console()
    
    console.print(f"[bold blue]User:[/bold blue] {prompt}")
    console.print("[italic yellow]Agent: thinking...[/italic yellow]")
    
    response = await agent.run_async(prompt, model=model, reasoning_effort=effort)
    
    # Render the response as Markdown
    console.print("[bold green]Agent:[/bold green]")
    markdown = Markdown(response)
    console.print(markdown)


def create_agent_from_args(args: argparse.Namespace) -> OllamaAgent:
    """
    Create OllamaAgent instance with CLI args overriding config.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        Configured OllamaAgent instance.
    """
    cfg = config.get_config()
    
    return OllamaAgent(
        model=args.model or cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        reasoning_effort=args.effort or cfg.reasoning_effort,
        database_path=cfg.database_path
    )


def create_agent_from_config(model: Optional[str] = None, reasoning_effort: Optional[str] = None) -> OllamaAgent:
    """
    Create OllamaAgent instance from config with optional overrides.
    
    Args:
        model: Optional model override.
        reasoning_effort: Optional reasoning effort override.
        
    Returns:
        Configured OllamaAgent instance.
    """
    cfg = config.get_config()
    
    return OllamaAgent(
        model=model or cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        reasoning_effort=reasoning_effort or cfg.reasoning_effort,
        database_path=cfg.database_path
    )


def find_task_or_exit(task_manager: TaskManager, task_id: str, console: Console) -> tuple[str, Task]:
    """
    Find a task by ID or prefix, exit if not found.
    
    Args:
        task_manager: TaskManager instance.
        task_id: Task ID or prefix to search.
        console: Console instance for error output.
        
    Returns:
        Tuple of (task_id, task) if found.
        
    Raises:
        SystemExit: If task not found.
    """
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
    
    table = Table(title="Saved Tasks", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Title", style="green")
    table.add_column("Model", style="blue")
    table.add_column("Effort", style="yellow")
    
    for task_id, task in tasks:
        table.add_row(task_id, task.title, task.model, task.reasoning_effort)
    
    console.print(table)


async def run_task_command(task_id: str) -> None:
    """
    Execute a saved task.
    
    Args:
        task_id: Task ID or prefix to execute.
    """
    console = Console()
    task_manager = TaskManager()
    
    found_id, task = find_task_or_exit(task_manager, task_id, console)
    
    console.print(f"[bold cyan]Executing task:[/bold cyan] {task.title} ({found_id})")
    console.print(f"[bold blue]Prompt:[/bold blue] {task.prompt}")
    console.print(f"[bold]Model:[/bold] {task.model} | [bold]Effort:[/bold] {task.reasoning_effort}")
    console.print("")
    
    agent = create_agent_from_config(model=task.model, reasoning_effort=task.reasoning_effort)
    await run_non_interactive(agent, task.prompt)


def delete_task_command(task_id: str) -> None:
    """
    Delete a saved task.
    
    Args:
        task_id: Task ID or prefix to delete.
    """
    console = Console()
    task_manager = TaskManager()
    
    found_id, task = find_task_or_exit(task_manager, task_id, console)
    
    if task_manager.delete_task(found_id):
        console.print(f"[green]Task deleted:[/green] {task.title} ({found_id})")
    else:
        console.print(f"[red]Error deleting task: {found_id}[/red]")


def main() -> None:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure timeout from args or config
    cfg = config.get_config()
    timeout = args.timeout if args.timeout is not None else cfg.timeout
    set_timeout(timeout)
    
    # Handle task commands
    if args.command == "task-list":
        list_tasks_command()
        return
    elif args.command == "task-run":
        asyncio.run(run_task_command(args.task_id))
        return
    elif args.command == "task-delete":
        delete_task_command(args.task_id)
        return
    
    # Normal agent execution
    agent = create_agent_from_args(args)
    
    if args.prompt:
        asyncio.run(run_non_interactive(agent, args.prompt))
    else:
        ChatInterface(agent, timeout=timeout).run()


if __name__ == "__main__":
    main()
