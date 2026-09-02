"""Command-line interface for the application."""

from __future__ import annotations

import argparse
import asyncio
import inspect

from ..agent import AgentRuntime
from ..agent.builtin_tools import set_rag_manager
from ..agent.episodic_memory import HistoryError
from ..core import ALLOWED_REASONING_EFFORTS
from ..i18n import SUPPORTED_LOCALES, _
from ..mcp.loader import MCPConfigError
from ..rag import RAGContext, RAGError, RAGManager, load_rag_database
from ..settings import Settings
from ..skills import SkillError, SkillManager, SkillsContext
from ..streaming import run_non_interactive
from ..tasks.commands import TaskError, TasksContext
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
        "-c",
        "--num-ctx",
        type=str,
        help=_("Set context window size in tokens (num_ctx) or 'max'"),
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
        choices=SUPPORTED_LOCALES,
        help=_("Set interface language"),
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
    parser.add_argument(
        "-s",
        "--stealth",
        action="store_true",
        help=_("Enable stealth mode (do not save conversation to SQLite history)"),
    )


def _add_subcommands(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="command")

    # Task commands
    task_parser = subparsers.add_parser(
        "task",
        help=_("Manage saved tasks"),
        description=_("Manage saved tasks"),
        epilog=_("Run 'ollama-agent task <subcommand> -h' for more details on a subcommand."),
    )
    task_sub = task_parser.add_subparsers(dest="subcommand", required=True)

    task_sub.add_parser("list", help=_("List all saved tasks"))

    task_create = task_sub.add_parser("create", help=_("Create a new task"))
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

    task_run = task_sub.add_parser("run", help=_("Execute a saved task"))
    task_run.add_argument("task_id", type=str, help=_("Task ID or prefix to execute"))
    task_run.add_argument(
        "vars",
        nargs="*",
        help=_("Variable assignments in key=value format (e.g. file=src/app.py strict=true)"),
    )
    task_run.add_argument(
        "--var",
        action="append",
        dest="flag_vars",
        default=[],
        help=_("Variable assignment in key=value format"),
    )
    task_run.add_argument(
        "-y",
        "--yolo",
        action="store_true",
        default=argparse.SUPPRESS,
        help=_("Enable YOLO mode (bypasses all tool execution confirmation prompts)"),
    )

    task_delete = task_sub.add_parser("delete", help=_("Delete a saved task"))
    task_delete.add_argument("task_id", type=str, help=_("Task ID or prefix to delete"))

    # RAG commands
    rag_parser = subparsers.add_parser(
        "rag",
        help=_("Manage RAG databases"),
        description=_("Manage RAG databases"),
        epilog=_("Run 'ollama-agent rag <subcommand> -h' for more details on a subcommand."),
    )
    rag_sub = rag_parser.add_subparsers(dest="subcommand", required=True)

    rag_sub.add_parser("list", help=_("List all RAG databases"))

    rag_create = rag_sub.add_parser("create", help=_("Create a new RAG database"))
    rag_create.add_argument("name", type=str, help=_("Name for the new RAG database"))

    rag_delete = rag_sub.add_parser("delete", help=_("Delete a RAG database"))
    rag_delete.add_argument(
        "name", type=str, help=_("Name or prefix of the database to delete")
    )

    rag_add = rag_sub.add_parser("add", help=_("Add file(s) to a RAG database"))
    rag_add.add_argument("database", type=str, help=_("Name of the RAG database"))
    rag_add.add_argument("path", type=str, help=_("File or directory path to add"))
    rag_add.add_argument(
        "--dir",
        action="store_true",
        help=_("Treat path as directory and add all files recursively"),
    )

    # Skill commands
    skill_parser = subparsers.add_parser(
        "skill",
        help=_("Manage skills"),
        description=_("Manage skills"),
        epilog=_("Run 'ollama-agent skill <subcommand> -h' for more details on a subcommand."),
    )
    skill_sub = skill_parser.add_subparsers(dest="subcommand", required=True)

    skill_sub.add_parser("list", help=_("List all skills"))

    skill_show = skill_sub.add_parser("show", help=_("Show skill details"))
    skill_show.add_argument("skill_id", type=str, help=_("Skill ID or prefix"))

    skill_create = skill_sub.add_parser("create", help=_("Create a new skill"))
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

    skill_delete = skill_sub.add_parser("delete", help=_("Delete a skill"))
    skill_delete.add_argument("skill_id", type=str, help=_("Skill ID or prefix to delete"))

    # Session commands
    session_parser = subparsers.add_parser(
        "session",
        help=_("Manage chat sessions"),
        description=_("Manage chat sessions"),
        epilog=_("Run 'ollama-agent session <subcommand> -h' for more details on a subcommand."),
    )
    session_sub = session_parser.add_subparsers(dest="subcommand", required=True)

    session_sub.add_parser("list", help=_("List all past chat sessions"))

    session_search = session_sub.add_parser(
        "search", help=_("Search past chat sessions by query keyword")
    )
    session_search.add_argument("query", type=str, help=_("Search query string"))

    session_del = session_sub.add_parser("delete", help=_("Delete a chat session from history"))
    session_del.add_argument("session_id", type=str, help=_("Session ID or prefix to delete"))

    session_exp = session_sub.add_parser("export", help=_("Export a chat session to Markdown"))
    session_exp.add_argument("session_id", type=str, help=_("Session ID or prefix to export"))
    session_exp.add_argument("--output", "-o", type=str, required=False, help=_("Target markdown file path"))

    # MCP commands
    mcp_parser = subparsers.add_parser(
        "mcp",
        help=_("Manage and check MCP servers"),
        description=_("Manage and check MCP servers"),
        epilog=_("Run 'ollama-agent mcp <subcommand> -h' for more details on a subcommand."),
    )
    mcp_sub = mcp_parser.add_subparsers(dest="subcommand", required=True)

    mcp_sub.add_parser("list", help=_("List configured MCP servers and check their status"))

    # Agents commands
    agents_parser = subparsers.add_parser(
        "agents",
        help=_("Manage configured subagents"),
        description=_("Manage configured subagents"),
        epilog=_("Run 'ollama-agent agents <subcommand> -h' for more details on a subcommand."),
    )
    agents_sub = agents_parser.add_subparsers(dest="subcommand", required=True)

    agents_sub.add_parser("list", help=_("List all configured subagents"))


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description=_("Ollama Agent - AI agent to interact with local models"),
        epilog=_("Run 'ollama-agent <command> -h' for more details on a specific command."),
    )

    _add_common_args(parser)
    _add_subcommands(parser)

    return parser


def handle_cli_commands(
    args: argparse.Namespace, settings: Settings
) -> bool:
    """Handle CLI commands and return True if a command was handled."""
    ctx = TasksContext(settings=settings)
    rag_ctx = RAGContext(rag_manager=RAGManager(settings.rag))
    skills_ctx = SkillsContext(skill_manager=SkillManager())

    cmd = args.command
    subcmd = getattr(args, "subcommand", None)
    if cmd and subcmd:
        handlers = build_cli_handlers(
            args,
            task_ctx=ctx,
            rag_ctx=rag_ctx,
            skills_ctx=skills_ctx,
            settings=settings,
        )
        handler_key = (cmd, subcmd)
        if handler_key in handlers:
            try:
                result = handlers[handler_key]()
                if inspect.isawaitable(result):
                    asyncio.run(result)  # type: ignore[arg-type]
            except (SkillError, TaskError, RAGError, HistoryError, MCPConfigError) as exc:
                ctx.console.print(f"[red]{exc}[/red]")
                raise SystemExit(1) from exc
            return True

    if args.prompt:
        runtime = AgentRuntime(
            settings=settings,
            yolo_mode=args.yolo,
            stealth_mode=args.stealth,
        )
        completed = True

        async def _run() -> None:
            nonlocal completed
            async with runtime:
                if args.rag:
                    set_rag_manager(rag_ctx.rag_manager)
                    try:
                        load_rag_database(rag_ctx, args.rag)
                    except RAGError as exc:
                        rag_ctx.console.print(f"[red]{exc}[/red]")
                        raise SystemExit(1) from exc
                await runtime.reload()
                completed = await run_non_interactive(runtime, args.prompt)

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            raise SystemExit(130)

        if not completed:
            raise SystemExit(1)
        return True

    return False
