"""Shared command dispatch helpers for CLI and REPL interfaces."""

from __future__ import annotations

import argparse
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.markup import escape

from ...agent import AgentRuntime
from ...agent.episodic_memory import HistoryError
from ...i18n import _
from ...mcp import MCPConfigError
from ...mcp.commands import list_mcp_servers, reload_mcp_servers
from ...rag import (
    RAGContext,
    RAGError,
    RAGManager,
    add_rag_directory,
    add_rag_file,
    create_rag_database,
    delete_rag_database,
    list_rag_databases,
    load_rag_database,
    show_rag_status,
    unload_rag_database,
)
from ...settings import Settings
from ...skills import SkillError, SkillsContext, create_skill, delete_skill, list_skills, show_skill
from ...tasks import (
    TaskError,
    TasksContext,
    create_task,
    delete_task,
    list_tasks,
    parse_var_assignments,
    run_task,
)
from .models import list_models, list_subagents, set_model
from .params import (
    set_context_window,
    set_effort,
    set_model_param,
    show_context_window,
    show_effort,
    show_model_params,
)
from .sessions import (
    delete_session,
    export_session,
    list_sessions,
    search_sessions,
)

CLIHandler = Callable[[], Any]
REPLHandler = Callable[[list[str]], object]
DOMAIN_ERRORS = (SkillError, TaskError, RAGError, HistoryError, MCPConfigError)

_log = logging.getLogger(__name__)


async def safe_call(
    fn: Callable[..., Any],
    *args: Any,
    console: Console,
    **kwargs: Any,
) -> None:
    """Call *fn*(*args, **kwargs), awaiting if necessary.

    Domain errors (SkillError, TaskError, RAGError, HistoryError, MCPConfigError)
    carry the full user-facing message and are reported here exactly once;
    raisers print nothing themselves. Unexpected errors are also reported rather
    than propagated: a slash command must never take the TUI down with a traceback.
    """
    try:
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            await result
    except DOMAIN_ERRORS as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
    # A command must never crash the caller.
    except Exception as exc:
        _log.exception("Unhandled error in command %r", fn.__name__)
        console.print(f"[red]{escape(_('Command failed: {exc}', exc=exc))}[/red]")


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
    rag_ctx: RAGContext | None = None,
    skills_ctx: SkillsContext,
    settings: Settings,
) -> dict[tuple[str, str], CLIHandler]:
    """Map parsed CLI subcommands to their synchronous or async handler functions."""
    console = task_ctx.console

    def _get_rag_ctx() -> RAGContext:
        nonlocal rag_ctx
        if rag_ctx is None:
            rag_ctx = RAGContext(rag_manager=RAGManager(settings.rag), console=console)
        return rag_ctx

    async def task_run() -> None:
        raw_vars = list(args.vars) + list(args.flag_vars)
        variables = parse_var_assignments(raw_vars)
        await run_task(
            task_ctx,
            args.task_id,
            variables=variables,
            yolo=args.yolo,
            stealth=args.stealth,
            rag=args.rag,
            preserve_model=bool(args.model),
            preserve_effort=bool(args.effort),
        )

    async def rag_add() -> None:
        ctx = _get_rag_ctx()
        load_rag_database(ctx, args.database)
        try:
            if args.dir:
                await add_rag_directory(ctx, args.path)
                return
            await add_rag_file(ctx, args.path)
        finally:
            await ctx.rag_manager.aclose()

    return {
        ("task", "list"): lambda: list_tasks(task_ctx),
        ("task", "delete"): lambda: delete_task(task_ctx, args.task_id),
        ("task", "run"): task_run,
        ("task", "create"): lambda: create_task(
            task_ctx,
            args.task_id,
            title=args.title,
            prompt=args.task_prompt,
            model=args.task_model,
            reasoning_effort=args.task_effort,
            force=args.force,
        ),
        ("rag", "list"): lambda: list_rag_databases(_get_rag_ctx()),
        ("rag", "create"): lambda: create_rag_database(_get_rag_ctx(), args.name),
        ("rag", "delete"): lambda: delete_rag_database(_get_rag_ctx(), args.name),
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


@dataclass(slots=True)
class REPLEnvironment:
    runtime: AgentRuntime
    task_ctx: TasksContext
    skills_ctx: SkillsContext
    get_rag_ctx: Callable[[], RAGContext]
    console: Console
    handle_yolo: Callable[[list[str]], object]
    handle_stealth: Callable[[list[str]], object]
    handle_queue: Callable[[list[str]], object]

    # ── Argument-parsing helpers ─────────────────────────────────────────
    # Every handler follows the same "missing argument" / "unknown subcommand"
    # shape; centralising it keeps user-supplied text escaped exactly once.

    def _error(self, message: str) -> None:
        """Report a usage or unrecognised-subcommand error, escaping *message*."""
        self.console.print(f"[red]{escape(message)}[/red]")

    def _require_arg(self, args: list[str], index: int, usage: str) -> str | None:
        """Return ``args[index]`` or report *usage* (a full "Usage: ..." line) when absent."""
        if len(args) <= index:
            self._error(usage)
            return None
        return args[index]

    def _dispatch_switch(
        self,
        args: list[str],
        *,
        usage: str,
        on_show: Callable[[], object],
        on_switch: Callable[[str], object],
    ) -> object:
        if not args or args[0] in ("list", "show"):
            return on_show()
        if args[0] in ("set", "use", "switch"):
            val = self._require_arg(args, 1, usage)
            return on_switch(val) if val is not None else None
        if len(args) == 1:
            return on_switch(args[0])
        self._error(usage)
        return None

    # ── Handlers ─────────────────────────────────────────────────────────

    def handle_agents(self, args: list[str]) -> object:
        if not args or args[0] == "list":
            list_subagents(self.console, settings=self.runtime.settings)
            return None
        self._error(_("Unknown agents subcommand '{sub}'. Usage: /agents [list]", sub=args[0]))
        return None

    def handle_mcp(self, args: list[str]) -> object:
        if not args or args[0] in ("list", "status"):
            return list_mcp_servers(self.console, settings=self.runtime.settings)
        if args[0] == "reload":
            return reload_mcp_servers(self.console, runtime=self.runtime)
        self._error(_("Unknown mcp subcommand '{sub}'. Usage: /mcp [list | reload]", sub=args[0]))
        return None

    def handle_model(self, args: list[str]) -> object:
        return self._dispatch_switch(
            args,
            usage=_("Usage: /model [list | set <model>]"),
            on_show=lambda: list_models(
                self.console,
                self.runtime.settings.model.name,
                self.runtime.settings.model.base_url,
            ),
            on_switch=lambda model_name: set_model(self.console, model_name, runtime=self.runtime),
        )

    def handle_effort(self, args: list[str]) -> object:
        return self._dispatch_switch(
            args,
            usage=_("Usage: /effort [set <level>]"),
            on_show=lambda: show_effort(self.console, self.runtime),
            on_switch=lambda effort: set_effort(self.console, effort, runtime=self.runtime),
        )

    def handle_context(self, args: list[str]) -> object:
        return self._dispatch_switch(
            args,
            usage=_("Usage: /context [set <size>]"),
            on_show=lambda: show_context_window(self.console, self.runtime),
            on_switch=lambda size: set_context_window(self.console, size, runtime=self.runtime),
        )

    def handle_params(self, args: list[str]) -> object:
        if not args or args[0] == "list":
            show_model_params(self.console, self.runtime)
            return None
        if args[0] == "set":
            if len(args) < 3:
                self.console.print(
                    f"[red]{escape(_('Usage: /params set <parameter> <value>'))}[/red]\n"
                    f"[dim]{escape(_('Example: /params set temperature 0.7'))}[/dim]"
                )
                return None
            return set_model_param(self.console, args[1], args[2], runtime=self.runtime)
        self._error(_("Usage: /params [list | set <parameter> <value>]"))
        return None

    def handle_task(self, args: list[str]) -> object:
        if not args or args[0] == "list":
            list_tasks(self.task_ctx)
            return None
        if args[0] == "delete":
            task_id = self._require_arg(args, 1, _("Usage: /task delete <id>"))
            if task_id is not None:
                delete_task(self.task_ctx, task_id)
            return None
        self._error(_("Unknown task subcommand '{sub}'. Usage: /task [list | delete <id>]", sub=args[0]))
        return None

    def handle_skill(self, args: list[str]) -> object:
        if not args or args[0] == "list":
            list_skills(self.skills_ctx)
            return None
        if args[0] in ("show", "delete"):
            action = args[0]
            usage = _("Usage: /skill show <id>") if action == "show" else _("Usage: /skill delete <id>")
            skill_id = self._require_arg(args, 1, usage)
            if skill_id is None:
                return None
            if action == "show":
                show_skill(self.skills_ctx, skill_id)
            else:
                delete_skill(self.skills_ctx, skill_id)
            return None
        self._error(_("Unknown skill subcommand '{sub}'. Usage: /skill [list | show <id> | delete <id>]", sub=args[0]))
        return None

    def _handle_rag_add(self, sub_args: list[str]) -> object:
        # "--dir" is accepted in leading or trailing position (as documented by the
        # usage string). Quote-stripping the joined path is deliberately avoided so a
        # path containing quotes is passed through verbatim.
        is_dir = "--dir" in sub_args
        paths = [a for a in sub_args if a != "--dir"]
        if not paths:
            self._error(_("Usage: /rag add <path> [--dir]"))
            return None
        target_path = " ".join(paths)
        if is_dir:
            return add_rag_directory(self.get_rag_ctx(), target_path)
        return add_rag_file(self.get_rag_ctx(), target_path)

    def handle_rag(self, args: list[str]) -> object:
        if not args or args[0] == "status":
            show_rag_status(self.get_rag_ctx())
            return None
        sub = args[0]
        if sub == "list":
            list_rag_databases(self.get_rag_ctx())
            return None
        if sub == "create":
            name = self._require_arg(args, 1, _("Usage: /rag create <name>"))
            if name is not None:
                create_rag_database(self.get_rag_ctx(), name)
            return None
        if sub == "delete":
            name = self._require_arg(args, 1, _("Usage: /rag delete <name>"))
            if name is None:
                return None
            delete_rag_database(self.get_rag_ctx(), name)
            return self.runtime.reload()
        if sub == "load":
            name = self._require_arg(args, 1, _("Usage: /rag load <name>"))
            if name is None:
                return None
            load_rag_database(self.get_rag_ctx(), name)
            return self.runtime.reload()
        if sub == "unload":
            unload_rag_database(self.get_rag_ctx())
            return self.runtime.reload()
        if sub == "add":
            return self._handle_rag_add(args[1:])
        self._error(
            _(
                "Unknown rag subcommand '{sub}'. Usage: /rag [status | list | create | delete | load | unload | add]",
                sub=sub,
            )
        )
        return None

    def handle_session(self, args: list[str]) -> object:
        if not args or args[0] == "list":
            list_sessions(self.console, current_thread_id=self.runtime.thread_id)
            return None
        sub = args[0]
        if sub == "search":
            if len(args) < 2:
                self._error(_("Usage: /session search <query>"))
                return None
            search_sessions(
                self.console,
                " ".join(args[1:]),
                current_thread_id=self.runtime.thread_id,
            )
            return None
        if sub == "delete":
            session_id = self._require_arg(args, 1, _("Usage: /session delete <session_id>"))
            if session_id is not None:
                delete_session(self.console, session_id)
            return None
        self._error(
            _("Unknown session subcommand '{sub}'. Usage: /session [list | search <query> | delete <id>]", sub=sub)
        )
        return None


def build_repl_handlers(env: REPLEnvironment) -> dict[str, REPLHandler]:
    """Build the REPL command registry for unified slash commands.

    Only commands not intercepted inline by the TUI app are registered
    (/exit, /quit, /clear, /new, /session new|resume|switch|export,
    /task create|run and /skill create are handled by OllamaAgentApp).
    """
    return {
        "/queue": env.handle_queue,
        "/yolo": env.handle_yolo,
        "/stealth": env.handle_stealth,
        "/session": env.handle_session,
        "/model": env.handle_model,
        "/effort": env.handle_effort,
        "/context": env.handle_context,
        "/params": env.handle_params,
        "/task": env.handle_task,
        "/skill": env.handle_skill,
        "/rag": env.handle_rag,
        "/mcp": env.handle_mcp,
        "/agents": env.handle_agents,
    }
