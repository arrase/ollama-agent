"""Command-line interface for the application."""

from __future__ import annotations

import argparse
import asyncio
import inspect

from ..agent import AgentRuntime
from ..agent.builtin_tools import set_rag_manager
from ..core import ALLOWED_REASONING_EFFORTS
from ..i18n import _
from ..rag import RAGContext, RAGManager, RAGError, load_rag_database
from ..settings import Settings
from ..skills import SkillManager, SkillsContext, SkillError
from ..streaming import run_non_interactive
from ..tasks.commands import CLIContext, TaskError
from .dispatch import build_cli_handlers


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-m", "--model", type=str, help=_("Specify the AI model to use"))
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help=_("Non-interactive mode, provide a prompt directly from the command line"),
    )
    parser.add_argument(
        "-e",
        "--effort",
        type=str,
        choices=list(ALLOWED_REASONING_EFFORTS),
        help=_("Set reasoning effort level (low, medium, high, xhigh, disabled, hide, enabled)"),
    )
    parser.add_argument(
        "-t",
        "--builtin-tool-timeout",
        type=int,
        help=_("Set tool-call timeout in seconds (includes shell middleware and built-in tools)"),
    )
    parser.add_argument(
        "--rag",
        type=str,
        metavar="DATABASE",
        help=_("Load a RAG database for the session"),
    )
    parser.add_argument(
        "--allow-traversal",
        action="store_true",
        default=None,
        help=_("Allow virtual filesystem traversal to OS directories"),
    )
    parser.add_argument(
        "--no-allow-traversal",
        action="store_false",
        dest="allow_traversal",
        help=_("Sandbox agent to project directory"),
    )
    parser.add_argument(
        "-l",
        "--lang",
        "--language",
        type=str,
        dest="language",
        help=_("Set interface language (e.g. en, es, fr, de, it, pt, zh, ja, ru, hi)"),
    )
    parser.add_argument(
        "--config-reset",
        choices=["all", "system-prompt", "config-file"],
        help=_("Reset configuration or system prompt to defaults"),
    )
    parser.add_argument(
        "-y",
        "--yolo",
        action="store_true",
        help=_("Enable YOLO mode (bypasses all tool execution confirmation prompts)"),
    )


def _add_subparser(
    subparsers: argparse._SubParsersAction, name: str, help_msg: str
) -> argparse.ArgumentParser:
    """Helper to add a subparser."""
    return subparsers.add_parser(name, help=help_msg)


def _add_task_subcommands(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="command", help=_("Task management commands"))

    # Task commands
    _add_subparser(subparsers, "task-list", _("List all saved tasks"))

    task_create = _add_subparser(subparsers, "task-create", _("Create a new task"))
    task_create.add_argument("task_id", type=str, help=_("Task ID (filename stem)"))
    task_create.add_argument("--title", type=str, required=True, help=_("Task title"))
    task_create.add_argument(
        "--task-prompt",
        type=str,
        required=True,
        help=_("Task prompt text (use quotes; REPL supports multiline)"),
    )
    task_create.add_argument(
        "-m",
        "--task-model",
        type=str,
        required=False,
        help=_("Model to save with the task"),
    )
    task_create.add_argument(
        "-e",
        "--task-effort",
        type=str,
        choices=list(ALLOWED_REASONING_EFFORTS),
        required=False,
        help=_("Reasoning effort to save with the task (low, medium, high, xhigh, disabled, hide, enabled)"),
    )
    task_create.add_argument(
        "--force", action="store_true", help=_("Overwrite task if it already exists")
    )

    task_run = _add_subparser(subparsers, "task-run", _("Execute a saved task"))
    task_run.add_argument("task_id", type=str, help=_("Task ID or prefix to execute"))
    task_run.add_argument(
        "-y",
        "--yolo",
        action="store_true",
        default=argparse.SUPPRESS,
        help=_("Enable YOLO mode (bypasses all tool execution confirmation prompts)"),
    )

    task_delete = _add_subparser(subparsers, "task-delete", _("Delete a saved task"))
    task_delete.add_argument("task_id", type=str, help=_("Task ID or prefix to delete"))

    # RAG subcommands
    _add_subparser(subparsers, "rag-list", _("List all RAG databases"))

    rag_create = _add_subparser(subparsers, "rag-create", _("Create a new RAG database"))
    rag_create.add_argument("name", type=str, help=_("Name for the new RAG database"))

    rag_delete = _add_subparser(subparsers, "rag-delete", _("Delete a RAG database"))
    rag_delete.add_argument(
        "name", type=str, help=_("Name or prefix of the database to delete")
    )

    rag_add = _add_subparser(subparsers, "rag-add", _("Add file(s) to a RAG database"))
    rag_add.add_argument("database", type=str, help=_("Name of the RAG database"))
    rag_add.add_argument("path", type=str, help=_("File or directory path to add"))
    rag_add.add_argument(
        "--dir",
        action="store_true",
        help=_("Treat path as directory and add all files recursively"),
    )

    # Skill subcommands
    _add_subparser(subparsers, "skill-list", _("List all skills"))

    skill_show = _add_subparser(subparsers, "skill-show", _("Show skill details"))
    skill_show.add_argument("skill_id", type=str, help=_("Skill ID or prefix"))

    skill_create = _add_subparser(subparsers, "skill-create", _("Create a new skill"))
    skill_create.add_argument("skill_id", type=str, help=_("Skill ID (directory name)"))
    skill_create.add_argument("--name", type=str, required=True, help=_("Skill name"))
    skill_create.add_argument(
        "--description", type=str, required=True, help=_("Skill description")
    )
    skill_create.add_argument(
        "--instructions",
        type=str,
        required=True,
        help=_("Skill instructions (markdown body)"),
    )
    skill_create.add_argument(
        "--force", action="store_true", help=_("Overwrite skill if it already exists")
    )

    skill_delete = _add_subparser(subparsers, "skill-delete", _("Delete a skill"))
    skill_delete.add_argument("skill_id", type=str, help=_("Skill ID or prefix to delete"))

    # Session subcommands
    _add_subparser(subparsers, "session-list", _("List all past chat sessions"))

    session_search = _add_subparser(
        subparsers, "session-search", _("Search past chat sessions by query keyword")
    )
    session_search.add_argument("query", type=str, help=_("Search query string"))

    session_del = _add_subparser(subparsers, "session-delete", _("Delete a chat session from history"))
    session_del.add_argument("session_id", type=str, help=_("Session ID or prefix to delete"))

    session_exp = _add_subparser(subparsers, "session-export", _("Export a chat session to Markdown"))
    session_exp.add_argument("session_id", type=str, help=_("Session ID or prefix to export"))
    session_exp.add_argument("--output", "-o", type=str, required=False, help=_("Target markdown file path"))


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description=_("Ollama Agent - AI agent to interact with local models")
    )

    _add_common_args(parser)
    _add_task_subcommands(parser)

    return parser


def handle_cli_commands(
    args: argparse.Namespace, settings: Settings
) -> bool:
    """Handle CLI commands and return True if a command was handled."""
    ctx = CLIContext()
    rag_ctx = RAGContext(rag_manager=RAGManager(settings.rag))
    skills_ctx = SkillsContext(skill_manager=SkillManager())

    cmd = args.command
    if cmd == "session-export":
        args._runtime = AgentRuntime(settings=settings)
    handlers = build_cli_handlers(
        args, task_ctx=ctx, rag_ctx=rag_ctx, skills_ctx=skills_ctx
    )
    if cmd in handlers:
        try:
            result = handlers[cmd]()
            if inspect.isawaitable(result):
                asyncio.run(result)  # type: ignore[arg-type]
        except (SkillError, TaskError, RAGError):
            raise SystemExit(1)
        return True

    if args.prompt:
        runtime = AgentRuntime(settings=settings, yolo_mode=args.yolo)

        async def _run():
            async with runtime:
                if args.rag:
                    set_rag_manager(rag_ctx.rag_manager)
                    try:
                        load_rag_database(rag_ctx, args.rag)
                    except RAGError:
                        raise SystemExit(1)
                await runtime.reload()
                await run_non_interactive(runtime, args.prompt)

        asyncio.run(_run())
        return True

    return False
