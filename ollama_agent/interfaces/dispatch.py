"""Shared command dispatch helpers for CLI and REPL interfaces."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Awaitable, Callable

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

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
)


def build_cli_handlers(
    args: argparse.Namespace,
    *,
    task_ctx: CLIContext,
    rag_ctx: RAGContext,
    skills_ctx: SkillsContext,
) -> dict[str, CLIHandler]:
    """Build the CLI command registry for parsed argparse args."""

    async def rag_add() -> None:
        load_rag_database(rag_ctx, args.database)
        if args.dir:
            await add_rag_directory(rag_ctx, args.path)
            return
        await add_rag_file(rag_ctx, args.path)

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
            force=args.force,
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
            force=args.force,
        ),
    }


def build_repl_handlers(
    *,
    task_ctx: CLIContext,
    skills_ctx: SkillsContext,
    get_rag_ctx: Callable[[], RAGContext],
    console: Console,
    current_model: Callable[[], str],
    base_url: Callable[[], str],
    switch_model: Callable[[str], Awaitable[None]],
    handle_exit: Callable[[list[str]], object],
    handle_clear: Callable[[list[str]], object],
    handle_new: Callable[[list[str]], Awaitable[None]],
    handle_task_create: Callable[[list[str]], Awaitable[None]],
    handle_skill_create: Callable[[list[str]], Awaitable[None]],
    handle_yolo: Callable[[list[str]], object],
) -> dict[str, REPLCommand]:
    """Build the REPL command registry for slash commands."""

    def handle_rag_add(args: list[str]) -> object:
        is_dir = "--dir" in args
        paths = [a for a in args if a != "--dir"]
        if not paths:
            get_rag_ctx().console.print("[red]Error: Missing file or directory path.[/red]")
            return None
        return (
            add_rag_directory(get_rag_ctx(), paths[0])
            if is_dir
            else add_rag_file(get_rag_ctx(), paths[0])
        )

    cmds: dict[str, REPLCommand] = {}
    cmds.update({
        "/help": REPLCommand("Show this help message", "General", None, lambda _: render_repl_help(console, cmds)),
        "/yolo": REPLCommand(
            "Toggle YOLO mode or set it explicitly (on/off)",
            "General",
            "/yolo [on|off]",
            handle_yolo,
        ),
        "/exit": REPLCommand("Exit the REPL", "General", None, handle_exit),
        "/quit": REPLCommand("Exit the REPL", "General", None, handle_exit),
        "/clear": REPLCommand("Clear the screen", "General", None, handle_clear),
        "/new": REPLCommand(
            "Start a new chat session (clears context)",
            "Session Management",
            None,
            handle_new,
        ),
        "/models": REPLCommand(
            "List available Ollama models",
            "Model Management",
            None,
            lambda _: list_models(console, current_model(), base_url()),
        ),
        "/model-set": REPLCommand(
            "Switch to a different model",
            "Model Management",
            "/model-set <model>",
            lambda args: switch_model(args[0]),
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
            handle_rag_add,
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
            handle_task_create,
        ),
        "/skill-create": REPLCommand(
            "Create a skill",
            "Skills Management",
            "/skill-create <id> [--force]",
            handle_skill_create,
        ),
    })
    return cmds



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
