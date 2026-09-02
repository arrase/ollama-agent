"""Shared command dispatch helpers for CLI and REPL interfaces."""

from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from typing import Awaitable, Callable, Any

from rich.console import Console

from ..agent import AgentRuntime, list_subagents
from ..agent.episodic_memory import HistoryError
from ..core import DEFAULT_REASONING_EFFORT
from ..i18n import _
from ..mcp import list_mcp_servers, reload_mcp_servers
from ..mcp.loader import MCPConfigError
from ..settings import Settings
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
from ..tasks.commands import (
    TaskError,
    TasksContext,
    create_task,
    delete_task,
    list_tasks,
    parse_var_assignments,
    run_task,
)
from .model_commands import (
    list_models,
    set_model_param,
    show_context_window,
    show_effort,
    show_model_params,
)
from .session_commands import (
    delete_session,
    export_session,
    list_sessions,
    search_sessions,
)

CLIHandler = Callable[[], Any]


@dataclass(frozen=True)
class REPLCommand:
    """Declarative REPL command description."""

    summary: str
    section: str
    usage: str | None
    handler: Callable[[list[str]], object]


async def safe_call(
    fn: Callable[..., Any],
    *args: Any,
    console: Console,
    **kwargs: Any,
) -> None:
    """Call *fn*(*args, **kwargs), awaiting if necessary.

    Domain errors (SkillError, TaskError, RAGError, HistoryError, MCPConfigError)
    carry the full user-facing message and are reported here exactly once;
    raisers print nothing themselves.
    """
    try:
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            await result
    except (SkillError, TaskError, RAGError, HistoryError, MCPConfigError) as exc:
        console.print(f"[red]{exc}[/red]")


async def _cli_export_session(
    console: Console,
    settings: Settings,
    session_id: str,
    output_path: str | None = None,
) -> None:
    async with AgentRuntime(settings=settings) as runtime:
        await export_session(console, runtime, session_id, output_path=output_path)


def build_cli_handlers(
    args: argparse.Namespace,
    *,
    task_ctx: TasksContext,
    rag_ctx: RAGContext,
    skills_ctx: SkillsContext,
    settings: Settings,
) -> dict[tuple[str, str], CLIHandler]:
    """Map parsed CLI subcommands to their synchronous or async handler functions."""
    console = Console()

    async def task_run() -> None:
        raw_vars = list(args.vars) + list(args.flag_vars or [])
        variables = parse_var_assignments(raw_vars)
        await run_task(task_ctx, args.task_id, variables=variables, yolo=args.yolo)

    async def rag_add() -> None:
        load_rag_database(rag_ctx, args.database)
        if args.dir:
            await add_rag_directory(rag_ctx, args.path)
            return
        await add_rag_file(rag_ctx, args.path)

    return {
        ("task", "list"): lambda: list_tasks(task_ctx),
        ("task", "delete"): lambda: delete_task(task_ctx, args.task_id),
        ("task", "run"): task_run,
        ("task", "create"): lambda: create_task(
            task_ctx,
            args.task_id,
            title=args.title,
            prompt=args.task_prompt,
            model=(args.task_model or args.model or ""),
            reasoning_effort=(args.task_effort or args.effort or DEFAULT_REASONING_EFFORT),
            force=args.force,
        ),
        ("rag", "list"): lambda: list_rag_databases(rag_ctx),
        ("rag", "create"): lambda: create_rag_database(rag_ctx, args.name),
        ("rag", "delete"): lambda: delete_rag_database(rag_ctx, args.name),
        ("rag", "add"): rag_add,
        ("skill", "list"): lambda: list_skills(skills_ctx),
        ("skill", "show"): lambda: show_skill(skills_ctx, args.skill_id),
        ("skill", "delete"): lambda: delete_skill(skills_ctx, args.skill_id),
        ("skill", "create"): lambda: create_skill(
            skills_ctx,
            args.skill_id,
            name=args.name,
            description=args.description,
            instructions=args.instructions,
            force=args.force,
        ),
        ("session", "list"): lambda: list_sessions(console),
        ("session", "search"): lambda: search_sessions(console, args.query),
        ("session", "delete"): lambda: delete_session(console, args.session_id),
        ("session", "export"): lambda: _cli_export_session(
            console,
            settings,
            args.session_id,
            output_path=args.output,
        ),
        ("mcp", "list"): lambda: list_mcp_servers(
            console,
            settings=settings,
        ),
        ("agents", "list"): lambda: list_subagents(
            console,
            settings=settings,
        ),
    }


def build_repl_handlers(
    *,
    task_ctx: TasksContext,
    skills_ctx: SkillsContext,
    get_rag_ctx: Callable[[], RAGContext],
    console: Console,
    current_model: Callable[[], str],
    base_url: Callable[[], str],
    switch_model: Callable[[str], Awaitable[None]],
    handle_yolo: Callable[[list[str]], object],
    handle_stealth: Callable[[list[str]], object],
    handle_queue: Callable[[list[str]], object],
    get_runtime: Callable[[], Any],
    current_thread_id: Callable[[], str],
    switch_effort: Callable[[str], Awaitable[None]],
    switch_context_window: Callable[[str], Awaitable[None]],
) -> dict[str, REPLCommand]:
    """Build the REPL command registry for unified slash commands.

    Only commands not intercepted inline by the TUI app are registered
    (/exit, /quit, /clear, /new, /compact, /session new|resume|switch|export,
    /task create|run and /skill create are handled by OllamaAgentApp).
    """

    def handle_agents(args: list[str]) -> object:
        if not args or args[0] == "list":
            return list_subagents(console, settings=get_runtime().settings)
        err_msg = _("Unknown agents subcommand '{sub}'. Usage: /agents [list]", sub=args[0])
        console.print(f"[red]{err_msg}[/red]")
        return None

    def handle_mcp(args: list[str]) -> object:
        if not args or args[0] in ("list", "status"):
            return list_mcp_servers(console, settings=get_runtime().settings)
        if args[0] == "reload":
            return reload_mcp_servers(console, runtime=get_runtime())
        err_msg = _("Unknown mcp subcommand '{sub}'. Usage: /mcp [list | reload]", sub=args[0])
        console.print(f"[red]{err_msg}[/red]")
        return None

    def handle_model(args: list[str]) -> object:
        if not args or args[0] == "list":
            return list_models(console, current_model(), base_url())
        if args[0] in ("set", "use", "switch"):
            if len(args) > 1:
                return switch_model(args[1])
            console.print(f"[red]{_('Usage: /model [list | set <model>]')}[/red]")
            return None
        if len(args) == 1:
            return switch_model(args[0])
        console.print(f"[red]{_('Usage: /model [list | set <model>]')}[/red]")
        return None

    def handle_effort(args: list[str]) -> object:
        if not args:
            show_effort(console, get_runtime())
            return None
        if args[0] in ("set", "use", "switch"):
            if len(args) == 2:
                return switch_effort(args[1])
            console.print(f"[red]{_('Usage: /effort [set <level>]')}[/red]")
            return None
        if len(args) == 1:
            return switch_effort(args[0])
        console.print(f"[red]{_('Usage: /effort [set <level>]')}[/red]")
        return None

    def handle_context(args: list[str]) -> object:
        if not args:
            show_context_window(console, get_runtime())
            return None
        if args[0] in ("set", "use", "switch"):
            if len(args) == 2:
                return switch_context_window(args[1])
            console.print(f"[red]{_('Usage: /context [set <size>]')}[/red]")
            return None
        if len(args) == 1:
            return switch_context_window(args[0])
        console.print(f"[red]{_('Usage: /context [set <size>]')}[/red]")
        return None

    def handle_params(args: list[str]) -> object:
        if not args or args[0] == "list":
            show_model_params(console, get_runtime())
            return None
        if args[0] == "set":
            if len(args) < 3:
                console.print(
                    f"[red]{_('Usage: /params set <parameter> <value>')}[/red]\n"
                    f"[dim]{_('Example: /params set temperature 0.7')}[/dim]"
                )
                return None
            return set_model_param(
                console, args[1], args[2], runtime=get_runtime()
            )
        console.print(f"[red]{_('Usage: /params [list | set <parameter> <value>]')}[/red]")
        return None

    def handle_task(args: list[str]) -> object:
        if not args or args[0] == "list":
            list_tasks(task_ctx)
            return None
        sub = args[0]
        if sub == "delete":
            if len(args) < 2:
                console.print(f"[red]{_('Usage: /task delete <id>')}[/red]")
                return None
            delete_task(task_ctx, args[1])
            return None
        err_msg = _("Unknown task subcommand '{sub}'. Usage: /task [list | delete <id>]", sub=sub)
        console.print(f"[red]{err_msg}[/red]")
        return None

    def handle_skill(args: list[str]) -> object:
        if not args or args[0] == "list":
            list_skills(skills_ctx)
            return None
        sub = args[0]
        if sub == "show":
            if len(args) < 2:
                console.print(f"[red]{_('Usage: /skill show <id>')}[/red]")
                return None
            show_skill(skills_ctx, args[1])
            return None
        if sub == "delete":
            if len(args) < 2:
                console.print(f"[red]{_('Usage: /skill delete <id>')}[/red]")
                return None
            delete_skill(skills_ctx, args[1])
            return None
        err_msg = _("Unknown skill subcommand '{sub}'. Usage: /skill [list | show <id> | delete <id>]", sub=sub)
        console.print(f"[red]{err_msg}[/red]")
        return None

    def handle_rag(args: list[str]) -> object:
        if not args or args[0] == "status":
            show_rag_status(get_rag_ctx())
            return None
        sub = args[0]
        if sub == "list":
            list_rag_databases(get_rag_ctx())
            return None
        if sub == "create":
            if len(args) < 2:
                console.print(f"[red]{_('Usage: /rag create <name>')}[/red]")
                return None
            create_rag_database(get_rag_ctx(), args[1])
            return None
        if sub == "delete":
            if len(args) < 2:
                console.print(f"[red]{_('Usage: /rag delete <name>')}[/red]")
                return None
            delete_rag_database(get_rag_ctx(), args[1])
            return None
        if sub == "load":
            if len(args) < 2:
                console.print(f"[red]{_('Usage: /rag load <name>')}[/red]")
                return None
            load_rag_database(get_rag_ctx(), args[1])
            return None
        if sub == "unload":
            unload_rag_database(get_rag_ctx())
            return None
        if sub == "add":
            sub_args = args[1:]
            is_dir = "--dir" in sub_args
            paths = [a for a in sub_args if a != "--dir"]
            if not paths:
                console.print(f"[red]{_('Usage: /rag add <path> [--dir]')}[/red]")
                return None
            target_path = " ".join(paths).strip("\"'")
            return (
                add_rag_directory(get_rag_ctx(), target_path)
                if is_dir
                else add_rag_file(get_rag_ctx(), target_path)
            )
        err_msg = _("Unknown rag subcommand '{sub}'. Usage: /rag [status | list | create | delete | load | unload | add]", sub=sub)
        console.print(f"[red]{err_msg}[/red]")
        return None

    def handle_session(args: list[str]) -> object:
        if not args or args[0] == "list":
            list_sessions(console, current_thread_id=current_thread_id())
            return None
        sub = args[0]
        if sub == "search":
            if len(args) < 2:
                console.print(f"[red]{_('Usage: /session search <query>')}[/red]")
                return None
            search_sessions(
                console,
                " ".join(args[1:]),
                current_thread_id=current_thread_id(),
            )
            return None
        if sub == "delete":
            if len(args) < 2:
                console.print(f"[red]{_('Usage: /session delete <session_id>')}[/red]")
                return None
            delete_session(console, args[1])
            return None
        err_msg = _("Unknown session subcommand '{sub}'. Usage: /session [list | search <query> | delete <id>]", sub=sub)
        console.print(f"[red]{err_msg}[/red]")
        return None

    cmds: dict[str, REPLCommand] = {
        "/queue": REPLCommand(
            _("Show or clear the prompt queue"),
            "General",
            _("Usage: /queue [clear | rm <position>]"),
            handle_queue,
        ),
        "/yolo": REPLCommand(
            _("Toggle YOLO mode or set it explicitly (on/off)"),
            "General",
            _("Usage: /yolo [on|off]"),
            handle_yolo,
        ),
        "/stealth": REPLCommand(
            _("Toggle stealth mode or set it explicitly (on/off)"),
            "General",
            _("Usage: /stealth [on|off]"),
            handle_stealth,
        ),
        "/session": REPLCommand(
            _("Manage chat sessions"),
            "Session Management",
            _("Usage: /session [list | search <query> | delete <id>]"),
            handle_session,
        ),
        "/model": REPLCommand(
            _("Manage models"),
            "Model Management",
            _("Usage: /model [list | set <model>]"),
            handle_model,
        ),
        "/effort": REPLCommand(
            _("Show or set reasoning/thinking effort"),
            "Model Management",
            _("Usage: /effort [set <level>]"),
            handle_effort,
        ),
        "/context": REPLCommand(
            _("Show or set context window size (num_ctx)"),
            "Model Management",
            _("Usage: /context [set <size>]"),
            handle_context,
        ),
        "/params": REPLCommand(
            _("Manage model sampling parameters"),
            "Model Management",
            _("Usage: /params [list | set <parameter> <value>]"),
            handle_params,
        ),
        "/task": REPLCommand(
            _("Manage saved tasks"),
            "Task Management",
            _("Usage: /task [list | delete <id>]"),
            handle_task,
        ),
        "/skill": REPLCommand(
            _("Manage skills"),
            "Skills Management",
            _("Usage: /skill [list | show <id> | delete <id>]"),
            handle_skill,
        ),
        "/rag": REPLCommand(
            _("Manage RAG databases"),
            "RAG (Document Retrieval)",
            _("Usage: /rag [status | list | create | delete | load | unload | add]"),
            handle_rag,
        ),
        "/mcp": REPLCommand(
            _("Manage and check MCP servers"),
            "MCP (Model Context Protocol)",
            _("Usage: /mcp [list | reload]"),
            handle_mcp,
        ),
        "/agents": REPLCommand(
            _("Manage configured subagents"),
            "Subagents Management",
            _("Usage: /agents [list]"),
            handle_agents,
        ),
    }
    return cmds
