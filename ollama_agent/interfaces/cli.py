"""Command-line interface for the application."""

import argparse
import asyncio
from typing import Callable

from ..agent import OllamaAgent
from ..core import ALLOWED_REASONING_EFFORTS
from ..execution import run_non_interactive
from ..tasks.commands import CLIContext, delete_task, list_tasks, run_task


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-m", "--model", type=str, help="Specify the AI model to use")
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Non-interactive mode, provide a prompt directly from the command line",
    )
    parser.add_argument(
        "-e",
        "--effort",
        type=str,
        choices=list(ALLOWED_REASONING_EFFORTS),
        help="Set reasoning effort level (low, medium, high, disabled)",
    )
    parser.add_argument(
        "-t",
        "--builtin-tool-timeout",
        type=int,
        help="Set built-in tool execution timeout in seconds",
    )


def _add_task_subcommands(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="command", help="Task management commands")
    subparsers.add_parser("task-list", help="List all saved tasks")

    task_run_parser = subparsers.add_parser("task-run", help="Execute a saved task")
    task_run_parser.add_argument("task_id", type=str, help="Task ID or prefix to execute")

    task_delete_parser = subparsers.add_parser("task-delete", help="Delete a saved task")
    task_delete_parser.add_argument("task_id", type=str, help="Task ID or prefix to delete")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Ollama Agent - AI agent to interact with local models"
    )

    _add_common_args(parser)
    _add_task_subcommands(parser)

    return parser


def handle_cli_commands(args: argparse.Namespace, agent_factory: Callable[..., OllamaAgent]) -> bool:
    """Handle CLI commands and return True if a command was handled."""
    ctx = CLIContext(agent_factory)

    handlers = {
        "task-list": lambda: list_tasks(ctx),
        "task-delete": lambda: delete_task(ctx, args.task_id),
        "task-run": lambda: asyncio.run(run_task(ctx, args.task_id)),
    }
    if args.command in handlers:
        handlers[args.command]()
        return True

    if args.prompt:
        agent = ctx.agent_factory(model=args.model, reasoning_effort=args.effort)
        asyncio.run(run_non_interactive(agent, args.prompt))
        return True

    return False
