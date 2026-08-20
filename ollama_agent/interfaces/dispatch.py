"""Shared command dispatch helpers for CLI and REPL interfaces."""

from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from typing import Awaitable, Callable, Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from ..rag import (
    RAGContext,
    RAGError,
    add_rag_directory,
    add_rag_file,
    create_rag_database,
    delete_rag_database,
    list_rag_databases,
    load_rag_database,
    show_rag_status,
    unload_rag_database,
)
from ..skills import SkillError, SkillsContext, create_skill, delete_skill, list_skills, show_skill
from ..tasks.commands import CLIContext, TaskError, create_task, delete_task, list_tasks, run_task
from .model_commands import list_models
from .session_commands import (
    compact_session,
    delete_session,
    export_session,
    list_sessions,
    new_session,
    resume_session,
)

CLIHandler = Callable[[], object]


@dataclass(frozen=True)
class REPLCommand:
    """Declarative REPL command description."""

    summary: str
    section: str
    usage: str | None
    handler: Callable[[list[str]], object]


async def safe_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Call *fn*(*args, **kwargs), awaiting if necessary and silencing SystemExit/domain errors."""
    try:
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            await result
    except (SystemExit, SkillError, TaskError, RAGError):
        pass


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
        "task-run": lambda: run_task(task_ctx, args.task_id, yolo=args.yolo),
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
        "session-list": lambda: list_sessions(Console()),
        "session-delete": lambda: delete_session(Console(), args.session_id),
        "session-export": lambda: export_session(
            Console(),
            args._runtime if hasattr(args, "_runtime") else None,  # type: ignore[arg-type]
            args.session_id,
            output_path=getattr(args, "output", None),
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
    handle_task_create: Callable[[list[str]], object],
    handle_skill_create: Callable[[list[str]], object],
    handle_yolo: Callable[[list[str]], object],
    current_thread_id: Callable[[], str] = lambda: "",
    handle_session_resume: Callable[[str], Awaitable[None]] | None = None,
    handle_session_export: Callable[[list[str]], Awaitable[None]] | None = None,
    handle_compact: Callable[[list[str]], Awaitable[None]] | None = None,
) -> dict[str, REPLCommand]:
    """Build the REPL command registry for unified slash commands."""

    def handle_model(args: list[str]) -> object:
        if not args or args[0] == "list":
            return list_models(console, current_model(), base_url())
        if args[0] in ("set", "use", "switch") and len(args) > 1:
            return switch_model(args[1])
        if len(args) == 1 and args[0] != "list":
            return switch_model(args[0])
        console.print("[red]Usage: /model [list | set <model>][/red]")
        return None

    def handle_task(args: list[str]) -> object:
        if not args or args[0] == "list":
            return list_tasks(task_ctx)
        sub = args[0]
        if sub == "create":
            return handle_task_create(args[1:])
        if sub == "run":
            sub_args = args[1:]
            task_id = next((a for a in sub_args if not a.startswith("-")), "")
            if not task_id:
                console.print("[red]Usage: /task run <id> [-y][/red]")
                return None
            return run_task(task_ctx, task_id, yolo=("-y" in sub_args or "--yolo" in sub_args))
        if sub == "delete":
            if len(args) < 2:
                console.print("[red]Usage: /task delete <id>[/red]")
                return None
            return delete_task(task_ctx, args[1])
        console.print(f"[red]Unknown task subcommand '{sub}'. Usage: /task [list | create | run <id> | delete <id>][/red]")
        return None

    def handle_skill(args: list[str]) -> object:
        if not args or args[0] == "list":
            return list_skills(skills_ctx)
        sub = args[0]
        if sub == "show":
            if len(args) < 2:
                console.print("[red]Usage: /skill show <id>[/red]")
                return None
            return show_skill(skills_ctx, args[1])
        if sub == "create":
            return handle_skill_create(args[1:])
        if sub == "delete":
            if len(args) < 2:
                console.print("[red]Usage: /skill delete <id>[/red]")
                return None
            return delete_skill(skills_ctx, args[1])
        console.print(f"[red]Unknown skill subcommand '{sub}'. Usage: /skill [list | show <id> | create | delete <id>][/red]")
        return None

    def handle_rag(args: list[str]) -> object:
        if not args or args[0] == "status":
            return show_rag_status(get_rag_ctx())
        sub = args[0]
        if sub == "list":
            return list_rag_databases(get_rag_ctx())
        if sub == "create":
            if len(args) < 2:
                console.print("[red]Usage: /rag create <name>[/red]")
                return None
            return create_rag_database(get_rag_ctx(), args[1])
        if sub == "delete":
            if len(args) < 2:
                console.print("[red]Usage: /rag delete <name>[/red]")
                return None
            return delete_rag_database(get_rag_ctx(), args[1])
        if sub == "load":
            if len(args) < 2:
                console.print("[red]Usage: /rag load <name>[/red]")
                return None
            return load_rag_database(get_rag_ctx(), args[1])
        if sub == "unload":
            return unload_rag_database(get_rag_ctx())
        if sub == "add":
            sub_args = args[1:]
            is_dir = "--dir" in sub_args
            paths = [a for a in sub_args if a != "--dir"]
            if not paths:
                console.print("[red]Usage: /rag add <path> [--dir][/red]")
                return None
            return (
                add_rag_directory(get_rag_ctx(), paths[0])
                if is_dir
                else add_rag_file(get_rag_ctx(), paths[0])
            )
        console.print(f"[red]Unknown rag subcommand '{sub}'. Usage: /rag [status | list | create | delete | load | unload | add][/red]")
        return None

    def handle_session(args: list[str]) -> object:
        if not args or args[0] == "list":
            return list_sessions(console, current_thread_id=current_thread_id())
        sub = args[0]
        if sub == "new":
            return handle_new([])
        if sub in ("resume", "switch"):
            if len(args) < 2:
                console.print("[red]Usage: /session resume <session_id>[/red]")
                return None
            if handle_session_resume is not None:
                return handle_session_resume(args[1])
            return resume_session(console, args[1])
        if sub == "export":
            if handle_session_export is not None:
                return handle_session_export(args[1:])
            console.print("[red]Export not available in current context.[/red]")
            return None
        if sub == "delete":
            if len(args) < 2:
                console.print("[red]Usage: /session delete <session_id>[/red]")
                return None
            return delete_session(console, args[1])
        console.print(f"[red]Unknown session subcommand '{sub}'. Usage: /session [list | resume <id> | new | export [path] | delete <id>][/red]")
        return None

    cmds: dict[str, REPLCommand] = {
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
        "/compact": REPLCommand(
            "Compact conversation history into a summary",
            "Session Management",
            None,
            (
                handle_compact
                if handle_compact is not None
                else (lambda _: None)
            ),
        ),
        "/session": REPLCommand(
            "Manage chat sessions (list, resume, new, export, delete)",
            "Session Management",
            "/session [list | resume <id> | new | export [path] | delete <id>]",
            handle_session,
        ),
        "/model": REPLCommand(
            "Manage models (list, set)",
            "Model Management",
            "/model [list | set <model>]",
            handle_model,
        ),
        "/task": REPLCommand(
            "Manage saved tasks (list, create, run, delete)",
            "Task Management",
            "/task [list | create | run <id> | delete <id>]",
            handle_task,
        ),
        "/skill": REPLCommand(
            "Manage skills (list, show, create, delete)",
            "Skills Management",
            "/skill [list | show <id> | create | delete <id>]",
            handle_skill,
        ),
        "/rag": REPLCommand(
            "Manage RAG databases (status, list, create, delete, load, unload, add)",
            "RAG (Document Retrieval)",
            "/rag [status | list | create | delete | load | unload | add]",
            handle_rag,
        ),
    }
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
