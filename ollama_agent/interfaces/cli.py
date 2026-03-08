"""Command-line interface for the application."""

import argparse
import asyncio
import inspect
from typing import Callable

from ..agent import OllamaAgent
from ..core import ALLOWED_REASONING_EFFORTS
from ..streaming import run_non_interactive
from ..rag import RAGContext, RAGManager
from ..settings import get_config
from ..skills import SkillManager, SkillsContext
from ..tasks.commands import CLIContext
from .dispatch import build_cli_handlers


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-m", "--model", type=str,
                        help="Specify the AI model to use")
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
        help="Set tool-call timeout in seconds (includes shell middleware and built-in tools)",
    )
    parser.add_argument(
        "--rag",
        type=str,
        metavar="DATABASE",
        help="Load a RAG database for the session",
    )
    parser.add_argument(
        "--skills-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Additional skills directory (repeatable)",
    )
    parser.add_argument(
        "--config-reset",
        choices=["all", "system-prompt", "config-file"],
        help="Reset configuration or system prompt to defaults",
    )


def _add_task_subcommands(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(
        dest="command", help="Task management commands")
    subparsers.add_parser("task-list", help="List all saved tasks").set_defaults(_handler="task-list")

    task_create_parser = subparsers.add_parser(
        "task-create", help="Create a new task")
    task_create_parser.set_defaults(_handler="task-create")
    task_create_parser.add_argument(
        "task_id", type=str, help="Task ID (filename stem)")
    task_create_parser.add_argument(
        "--title", type=str, required=True, help="Task title")
    task_create_parser.add_argument(
        "--task-prompt",
        type=str,
        required=True,
        help="Task prompt text (use quotes; REPL supports multiline)",
    )
    task_create_parser.add_argument(
        "-m",
        "--task-model",
        type=str,
        required=False,
        help="Model to save with the task",
    )
    task_create_parser.add_argument(
        "-e",
        "--task-effort",
        type=str,
        choices=list(ALLOWED_REASONING_EFFORTS),
        required=False,
        help="Reasoning effort to save with the task",
    )
    task_create_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite task if it already exists",
    )

    task_run_parser = subparsers.add_parser(
        "task-run", help="Execute a saved task")
    task_run_parser.set_defaults(_handler="task-run")
    task_run_parser.add_argument(
        "task_id", type=str, help="Task ID or prefix to execute")

    task_delete_parser = subparsers.add_parser(
        "task-delete", help="Delete a saved task")
    task_delete_parser.set_defaults(_handler="task-delete")
    task_delete_parser.add_argument(
        "task_id", type=str, help="Task ID or prefix to delete")

    # RAG subcommands
    subparsers.add_parser("rag-list", help="List all RAG databases").set_defaults(_handler="rag-list")

    rag_create_parser = subparsers.add_parser(
        "rag-create", help="Create a new RAG database")
    rag_create_parser.set_defaults(_handler="rag-create")
    rag_create_parser.add_argument(
        "name", type=str, help="Name for the new RAG database")

    rag_delete_parser = subparsers.add_parser(
        "rag-delete", help="Delete a RAG database")
    rag_delete_parser.set_defaults(_handler="rag-delete")
    rag_delete_parser.add_argument(
        "name", type=str, help="Name or prefix of the database to delete")

    rag_add_parser = subparsers.add_parser(
        "rag-add", help="Add file(s) to a RAG database")
    rag_add_parser.set_defaults(_handler="rag-add")
    rag_add_parser.add_argument(
        "database", type=str, help="Name of the RAG database")
    rag_add_parser.add_argument(
        "path", type=str, help="File or directory path to add")
    rag_add_parser.add_argument(
        "--dir", action="store_true",
        help="Treat path as directory and add all files recursively")

    # Skill subcommands
    subparsers.add_parser("skill-list", help="List all skills").set_defaults(_handler="skill-list")

    skill_show_parser = subparsers.add_parser("skill-show", help="Show skill details")
    skill_show_parser.set_defaults(_handler="skill-show")
    skill_show_parser.add_argument("skill_id", type=str, help="Skill ID or prefix")

    skill_create_parser = subparsers.add_parser("skill-create", help="Create a new skill")
    skill_create_parser.set_defaults(_handler="skill-create")
    skill_create_parser.add_argument("skill_id", type=str, help="Skill ID (directory name)")
    skill_create_parser.add_argument("--name", type=str, required=True, help="Skill name")
    skill_create_parser.add_argument("--description", type=str, required=True, help="Skill description")
    skill_create_parser.add_argument("--instructions", type=str, required=True, help="Skill instructions (markdown body)")
    skill_create_parser.add_argument("--force", action="store_true", help="Overwrite skill if it already exists")

    skill_delete_parser = subparsers.add_parser("skill-delete", help="Delete a skill")
    skill_delete_parser.set_defaults(_handler="skill-delete")
    skill_delete_parser.add_argument("skill_id", type=str, help="Skill ID or prefix to delete")

    # NOTE: Manual RAG query subcommand intentionally removed.


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
    cfg = get_config()
    rag_ctx = RAGContext(rag_manager=RAGManager(cfg.rag))
    skills_ctx = SkillsContext(skill_manager=SkillManager())

    cmd = getattr(args, "_handler", None) or args.command
    handlers = build_cli_handlers(args, task_ctx=ctx, rag_ctx=rag_ctx, skills_ctx=skills_ctx)
    if cmd in handlers:
        result = handlers[cmd]()
        if inspect.isawaitable(result):
            asyncio.run(result)
        return True

    if args.prompt:
        agent = ctx.agent_factory(
            model=args.model, reasoning_effort=args.effort)
        # Load RAG database if specified
        if getattr(args, 'rag', None):
            try:
                agent.rag_manager.load_database(args.rag)
            except Exception as e:
                rag_ctx.console.print(f"[red]Failed to load RAG database '{args.rag}': {e}[/red]")
                raise SystemExit(1)
        asyncio.run(run_non_interactive(agent, args.prompt))
        return True

    return False
