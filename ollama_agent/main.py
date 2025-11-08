"""Main entry point of the application."""

import argparse
import asyncio
from typing import Callable, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.live import Live

from .settings import configini as config
from .agent import OllamaAgent
from .tasks import Task, TaskManager
from .tools import set_builtin_tool_timeout
from .tui.app import ChatInterface
from .utils import ALLOWED_REASONING_EFFORTS


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
        help="Set reasoning effort level (low, medium, high)"
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


async def run_non_interactive(agent: OllamaAgent, prompt: str, model: Optional[str] = None, effort: Optional[str] = None) -> None:
    """Run the agent in non-interactive mode with streaming."""
    console = Console()

    try:
        # Buffers for different content types
        text_buffer = ""
        reasoning_buffer = ""
        reasoning_active = False
        live_active = False
        shown_thinking_header = False
        shown_agent_header = False

        live = Live(console=console, refresh_per_second=10)

        async for event in agent.run_async_streamed(prompt, model=model, reasoning_effort=effort):
            if event["type"] == "text_delta":
                # Regular text output
                if reasoning_active:
                    # Flush reasoning before showing text
                    reasoning_active = False
                    shown_thinking_header = False
                    console.print()  # New line after reasoning
                    reasoning_buffer = ""
                
                # Show "Agent:" header before first text
                if not shown_agent_header:
                    console.print("\n[bold green]Agent:[/bold green]")
                    shown_agent_header = True
                
                if not live_active:
                    live.start()
                    live_active = True
                
                text_buffer += event["content"]
                markdown = Markdown(text_buffer)
                live.update(markdown)
            
            elif event["type"] == "reasoning_delta":
                # Reasoning tokens (thinking process)
                if not shown_thinking_header:
                    if live_active:
                        live.stop()
                        live_active = False
                    console.print("\n[bold magenta]🧠 Thinking:[/bold magenta] ", end="")
                    shown_thinking_header = True
                    reasoning_active = True
                
                reasoning_buffer += event["content"]
                # Show reasoning in real-time
                console.print(event["content"], end="", style="dim italic magenta")
            
            elif event["type"] == "reasoning_summary":
                # Full reasoning summary (after streaming - usually not needed)
                pass
            
            elif event["type"] == "tool_call":
                if reasoning_active:
                    reasoning_active = False
                    shown_thinking_header = False
                    console.print()  # New line after reasoning
                if live_active:
                    live.stop()
                    live_active = False
                console.print(f"\n[yellow]🔧 Calling tool: {event['name']}[/yellow]")
            
            elif event["type"] == "tool_output":
                if live_active:
                    live.stop()
                    live_active = False
                output_preview = event["output"][:100] + "..." if len(event["output"]) > 100 else event["output"]
                console.print(f"[cyan]📤 Tool output: {output_preview}[/cyan]\n")
            
            elif event["type"] == "agent_update":
                # Skip the initial agent update message
                pass
            
            elif event["type"] == "error":
                if live_active:
                    live.stop()
                    live_active = False
                console.print(f"\n[red]❌ Error: {event['content']}[/red]")
                break

        if live_active:
            live.stop()

        # Final newline after streaming completes
        console.print()
    finally:
        # Cleanup MCP servers
        await agent.cleanup()


def create_agent(model: Optional[str] = None, reasoning_effort: Optional[str] = None) -> OllamaAgent:
    """Create OllamaAgent instance from config with optional overrides."""
    cfg = config.get_config()

    return OllamaAgent(
        model=model or cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        reasoning_effort=reasoning_effort or cfg.reasoning_effort,
        database_path=cfg.database_path,
        mcp_config_path=cfg.mcp_config_path
    )


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


async def run_task_command(task_id: str) -> None:
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

    agent = create_agent(
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


def main() -> None:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure built-in tool timeout from args or config
    cfg = config.get_config()
    builtin_tool_timeout = args.builtin_tool_timeout if args.builtin_tool_timeout is not None else cfg.builtin_tool_timeout
    set_builtin_tool_timeout(builtin_tool_timeout)

    command_handlers: dict[str, Callable[[], None]] = {
        "task-list": list_tasks_command,
        "task-delete": lambda: delete_task_command(args.task_id),
        "task-run": lambda: asyncio.run(run_task_command(args.task_id)),
    }

    handler = command_handlers.get(args.command or "")
    if handler:
        handler()
        return

    # Normal agent execution
    agent = create_agent(model=args.model, reasoning_effort=args.effort)

    if args.prompt:
        asyncio.run(run_non_interactive(agent, args.prompt))
    else:
        ChatInterface(agent, builtin_tool_timeout=builtin_tool_timeout).run()


if __name__ == "__main__":
    main()
