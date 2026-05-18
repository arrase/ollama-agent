"""Command-line interface for the application."""

import argparse
import asyncio
import inspect

from ..core import ALLOWED_REASONING_EFFORTS
from ..rag import RAGContext, RAGManager, RAGSettings
from ..settings import load_settings
from ..skills import SkillManager, SkillsContext
from ..streaming import run_non_interactive
from ..tasks.commands import CLIContext
from .dispatch import build_cli_handlers


def _noop_factory(**kwargs):
    raise NotImplementedError("Direct agent factory is no longer supported.")


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
        help="Set tool-call timeout in seconds (includes shell middleware and built-in tools)",
    )
    parser.add_argument(
        "--rag",
        type=str,
        metavar="DATABASE",
        help="Load a RAG database for the session",
    )

    parser.add_argument(
        "--config-reset",
        choices=["all", "system-prompt", "config-file"],
        help="Reset configuration or system prompt to defaults",
    )


def _add_subparser(
    subparsers: argparse._SubParsersAction, name: str, help_msg: str
) -> argparse.ArgumentParser:
    """Helper to add a subparser with a default handler."""
    parser = subparsers.add_parser(name, help=help_msg)
    parser.set_defaults(_handler=name)
    return parser


def _add_task_subcommands(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="command", help="Task management commands")

    # Task commands
    _add_subparser(subparsers, "task-list", "List all saved tasks")

    task_create = _add_subparser(subparsers, "task-create", "Create a new task")
    task_create.add_argument("task_id", type=str, help="Task ID (filename stem)")
    task_create.add_argument("--title", type=str, required=True, help="Task title")
    task_create.add_argument(
        "--task-prompt",
        type=str,
        required=True,
        help="Task prompt text (use quotes; REPL supports multiline)",
    )
    task_create.add_argument(
        "-m",
        "--task-model",
        type=str,
        required=False,
        help="Model to save with the task",
    )
    task_create.add_argument(
        "-e",
        "--task-effort",
        type=str,
        choices=list(ALLOWED_REASONING_EFFORTS),
        required=False,
        help="Reasoning effort to save with the task",
    )
    task_create.add_argument(
        "--force", action="store_true", help="Overwrite task if it already exists"
    )

    task_run = _add_subparser(subparsers, "task-run", "Execute a saved task")
    task_run.add_argument("task_id", type=str, help="Task ID or prefix to execute")

    task_delete = _add_subparser(subparsers, "task-delete", "Delete a saved task")
    task_delete.add_argument("task_id", type=str, help="Task ID or prefix to delete")

    # RAG subcommands
    _add_subparser(subparsers, "rag-list", "List all RAG databases")

    rag_create = _add_subparser(subparsers, "rag-create", "Create a new RAG database")
    rag_create.add_argument("name", type=str, help="Name for the new RAG database")

    rag_delete = _add_subparser(subparsers, "rag-delete", "Delete a RAG database")
    rag_delete.add_argument(
        "name", type=str, help="Name or prefix of the database to delete"
    )

    rag_add = _add_subparser(subparsers, "rag-add", "Add file(s) to a RAG database")
    rag_add.add_argument("database", type=str, help="Name of the RAG database")
    rag_add.add_argument("path", type=str, help="File or directory path to add")
    rag_add.add_argument(
        "--dir",
        action="store_true",
        help="Treat path as directory and add all files recursively",
    )

    # Skill subcommands
    _add_subparser(subparsers, "skill-list", "List all skills")

    skill_show = _add_subparser(subparsers, "skill-show", "Show skill details")
    skill_show.add_argument("skill_id", type=str, help="Skill ID or prefix")

    skill_create = _add_subparser(subparsers, "skill-create", "Create a new skill")
    skill_create.add_argument("skill_id", type=str, help="Skill ID (directory name)")
    skill_create.add_argument("--name", type=str, required=True, help="Skill name")
    skill_create.add_argument(
        "--description", type=str, required=True, help="Skill description"
    )
    skill_create.add_argument(
        "--instructions",
        type=str,
        required=True,
        help="Skill instructions (markdown body)",
    )
    skill_create.add_argument(
        "--force", action="store_true", help="Overwrite skill if it already exists"
    )

    skill_delete = _add_subparser(subparsers, "skill-delete", "Delete a skill")
    skill_delete.add_argument("skill_id", type=str, help="Skill ID or prefix to delete")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Ollama Agent - AI agent to interact with local models"
    )

    _add_common_args(parser)
    _add_task_subcommands(parser)

    return parser


def handle_cli_commands(args: argparse.Namespace) -> bool:
    """Handle CLI commands and return True if a command was handled."""
    settings = load_settings()
    rag_settings = RAGSettings(
        rag_dir=settings.rag.rag_dir,
        embedder_model=settings.rag.embedder_model,
        embedder_base_url=settings.rag.embedder_base_url,
        embedding_dims=settings.rag.embedding_dims,
        default_top_k=settings.rag.default_top_k,
        chunk_size=settings.rag.chunk_size,
        chunk_overlap=settings.rag.chunk_overlap,
    )
    ctx = CLIContext(_noop_factory)
    rag_ctx = RAGContext(rag_manager=RAGManager(rag_settings))
    skills_ctx = SkillsContext(skill_manager=SkillManager())

    cmd = getattr(args, "_handler", None) or args.command
    handlers = build_cli_handlers(
        args, task_ctx=ctx, rag_ctx=rag_ctx, skills_ctx=skills_ctx
    )
    if cmd in handlers:
        result = handlers[cmd]()
        if inspect.isawaitable(result):
            asyncio.run(result)  # type: ignore[arg-type]
        return True

    if args.prompt:
        from ..agent import AgentRuntime

        # Apply CLI overrides
        if args.model:
            settings.model.name = args.model
        if args.effort:
            settings.model.reasoning_effort = args.effort
        if args.builtin_tool_timeout:
            settings.runtime.builtin_tool_timeout = args.builtin_tool_timeout

        runtime = AgentRuntime(settings=settings)

        async def _run():
            async with runtime:
                await runtime.reload()
                if getattr(args, "rag", None):
                    try:
                        rag_ctx.rag_manager.load_database(args.rag)
                    except Exception as e:
                        rag_ctx.console.print(
                            f"[red]Failed to load RAG database '{args.rag}': {e}[/red]"
                        )
                        raise SystemExit(1)
                await run_non_interactive(runtime, args.prompt)

        asyncio.run(_run())
        return True

    return False
