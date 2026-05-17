"""Shared command dispatch helpers for CLI and REPL interfaces."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Awaitable, Callable

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from ..agent import OllamaAgent
from ..mcp import list_mcp_servers, show_mcp_server
from ..rag import (
    RAGContext,
    add_rag_directory,
    add_rag_file,
    create_rag_database,
    delete_rag_database,
    list_rag_databases,
    load_rag_database,
    show_rag_status,
    unload_rag_database,
)
from ..skills import SkillsContext, create_skill, delete_skill, list_skills, show_skill
from ..tasks.commands import CLIContext, create_task, delete_task, list_tasks, run_task
from .model_commands import list_models
from .session_commands import delete_session, list_sessions, load_session

CLIHandler = Callable[[], object]


@dataclass(frozen=True)
class REPLCommand:
    """Declarative REPL command description."""

    summary: str
    section: str
    usage: str | None
    handler: Callable[[list[str]], object]


REPL_SECTIONS: tuple[str, ...] = (
    "General",
    "Model Management",
    "Session Management",
    "Task Management",
    "RAG (Document Retrieval)",
    "Skills Management",
    "MCP Servers",
)


def build_cli_handlers(
    args: argparse.Namespace,
    *,
    task_ctx: CLIContext,
    rag_ctx: RAGContext,
    skills_ctx: SkillsContext,
) -> dict[str, CLIHandler]:
    """Build the CLI command registry for parsed argparse args."""

    def rag_add() -> None:
        load_rag_database(rag_ctx, args.database)
        if args.dir:
            add_rag_directory(rag_ctx, args.path)
            return
        add_rag_file(rag_ctx, args.path)

    return {
        "task-list": lambda: list_tasks(task_ctx),
        "task-delete": lambda: delete_task(task_ctx, args.task_id),
        "task-run": lambda: run_task(task_ctx, args.task_id),
        "task-create": lambda: create_task(
            task_ctx,
            args.task_id,
            title=args.title,
            prompt=args.task_prompt,
            model=(args.task_model or args.model or ""),
            reasoning_effort=(args.task_effort or args.effort),
            force=bool(args.force),
        ),
        "rag-list": lambda: list_rag_databases(rag_ctx),
        "rag-create": lambda: create_rag_database(rag_ctx, args.name),
        "rag-delete": lambda: delete_rag_database(rag_ctx, args.name),
        "rag-add": rag_add,
        "skill-list": lambda: list_skills(skills_ctx),
        "skill-show": lambda: show_skill(skills_ctx, args.skill_id),
        "skill-delete": lambda: delete_skill(skills_ctx, args.skill_id),
        "skill-create": lambda: create_skill(
            skills_ctx,
            args.skill_id,
            name=args.name,
            description=args.description,
            instructions=args.instructions,
            force=bool(args.force),
        ),
    }


def build_repl_handlers(
    *,
    task_ctx: CLIContext,
    skills_ctx: SkillsContext,
    get_rag_ctx: Callable[[], RAGContext],
    ensure_agent: Callable[[], OllamaAgent],
    console: Console,
    current_model: Callable[[], str],
    switch_model: Callable[[str], Awaitable[None]],
) -> dict[str, REPLCommand]:
    """Build the REPL command registry for slash commands."""

    return {
        "/help": REPLCommand("Show this help message", "General", None, lambda _: None),
        "/exit": REPLCommand("Exit the REPL", "General", None, lambda _: None),
        "/quit": REPLCommand("Exit the REPL", "General", None, lambda _: None),
        "/clear": REPLCommand("Clear the screen", "General", None, lambda _: None),
        "/new": REPLCommand(
            "Start a new chat session (clears context)",
            "Session Management",
            None,
            lambda _: None,
        ),
        "/models": REPLCommand(
            "List available Ollama models",
            "Model Management",
            None,
            lambda _: list_models(console, current_model()),
        ),
        "/model-set": REPLCommand(
            "Switch to a different model",
            "Model Management",
            "/model-set <model>",
            lambda args: switch_model(args[0]),
        ),
        "/sessions": REPLCommand(
            "List saved sessions",
            "Session Management",
            "/sessions [page]",
            lambda args: list_sessions(
                ensure_agent(),
                console,
                page=int(args[0]) if args and args[0].isdigit() else 1,
            ),
        ),
        "/session-load": REPLCommand(
            "Load a saved session",
            "Session Management",
            "/session-load <id>",
            lambda args: load_session(ensure_agent(), console, args[0]),
        ),
        "/session-delete": REPLCommand(
            "Delete a saved session",
            "Session Management",
            "/session-delete <id>",
            lambda args: delete_session(ensure_agent(), console, args[0]),
        ),
        "/tasks": REPLCommand(
            "List saved tasks", "Task Management", None, lambda _: list_tasks(task_ctx)
        ),
        "/task-run": REPLCommand(
            "Run a saved task",
            "Task Management",
            "/task-run <id>",
            lambda args: run_task(task_ctx, args[0]),
        ),
        "/task-delete": REPLCommand(
            "Delete a saved task",
            "Task Management",
            "/task-delete <id>",
            lambda args: delete_task(task_ctx, args[0]),
        ),
        "/rag": REPLCommand(
            "Show current RAG status",
            "RAG (Document Retrieval)",
            None,
            lambda _: show_rag_status(get_rag_ctx()),
        ),
        "/rag-list": REPLCommand(
            "List all RAG databases",
            "RAG (Document Retrieval)",
            None,
            lambda _: list_rag_databases(get_rag_ctx()),
        ),
        "/rag-create": REPLCommand(
            "Create a new RAG database",
            "RAG (Document Retrieval)",
            "/rag-create <name>",
            lambda args: create_rag_database(get_rag_ctx(), args[0]),
        ),
        "/rag-delete": REPLCommand(
            "Delete a RAG database",
            "RAG (Document Retrieval)",
            "/rag-delete <name>",
            lambda args: delete_rag_database(get_rag_ctx(), args[0]),
        ),
        "/rag-load": REPLCommand(
            "Load a RAG database",
            "RAG (Document Retrieval)",
            "/rag-load <name>",
            lambda args: load_rag_database(get_rag_ctx(), args[0]),
        ),
        "/rag-unload": REPLCommand(
            "Unload the current RAG database",
            "RAG (Document Retrieval)",
            None,
            lambda _: unload_rag_database(get_rag_ctx()),
        ),
        "/rag-add": REPLCommand(
            "Add file(s) to RAG",
            "RAG (Document Retrieval)",
            "/rag-add <path> [--dir]",
            lambda args: (
                add_rag_directory(get_rag_ctx(), args[0])
                if "--dir" in args[1:]
                else add_rag_file(get_rag_ctx(), args[0])
            ),
        ),
        "/skills": REPLCommand(
            "List all skills",
            "Skills Management",
            None,
            lambda _: list_skills(skills_ctx),
        ),
        "/skill-show": REPLCommand(
            "Show skill details",
            "Skills Management",
            "/skill-show <id>",
            lambda args: show_skill(skills_ctx, args[0]),
        ),
        "/skill-delete": REPLCommand(
            "Delete a skill",
            "Skills Management",
            "/skill-delete <id>",
            lambda args: delete_skill(skills_ctx, args[0]),
        ),
        "/task-create": REPLCommand(
            "Create a task",
            "Task Management",
            "/task-create <id> [--force]",
            lambda _: None,
        ),
        "/skill-create": REPLCommand(
            "Create a skill",
            "Skills Management",
            "/skill-create <id> [--force]",
            lambda _: None,
        ),
        "/mcps": REPLCommand(
            "List MCP server connection status",
            "MCP Servers",
            "/mcps [name]",
            lambda args: (
                show_mcp_server(console, ensure_agent(), args[0])
                if args
                else list_mcp_servers(console, ensure_agent())
            ),
        ),
    }


def render_repl_help(console: Console, commands: dict[str, REPLCommand]) -> None:
    """Render REPL help from command metadata."""
    lines: list[str] = ["[bold]Available Commands:[/bold]"]
    for section in REPL_SECTIONS:
        section_commands = [
            (name, spec)
            for name, spec in commands.items()
            if spec.section == section and name not in {"/quit"}
        ]
        if not section_commands:
            continue
        lines.append("")
        lines.append(f"[bold]{section}:[/bold]")
        for name, spec in section_commands:
            summary = escape(spec.summary)
            detail = f" (Usage: {escape(spec.usage)})" if spec.usage else ""
            lines.append(f"[green]{escape(name)}[/green]  {summary}{detail}")
    console.print(Panel("\n".join(lines), title="Help"))
