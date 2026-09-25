"""Interactive slash command handling and execution in the TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jinja2.exceptions import TemplateError
from rich.text import Text

from ...agent.episodic_memory import HistoryError
from ...core.common import extract_text
from ...i18n import _
from ...tasks import (
    TaskError,
    apply_task_settings,
    parse_var_assignments,
)
from ..commands.dispatch import safe_call
from ..commands.sessions import (
    export_session,
    resume_session,
)
from .completion import (
    CMD_AGENTS,
    CMD_CLEAR,
    CMD_CONTEXT,
    CMD_EFFORT,
    CMD_EXIT,
    CMD_MCP,
    CMD_MODEL,
    CMD_NEW,
    CMD_PARAMS,
    CMD_QUEUE,
    CMD_QUIT,
    CMD_RAG,
    CMD_SESSION,
    CMD_SKILL,
    CMD_STEALTH,
    CMD_TASK,
    CMD_YOLO,
    _CHAT_SCROLL_ID,
)
from .widgets.header import AgentHeader
from .widgets.messages import AgentResponse, UserMessage

if TYPE_CHECKING:
    from .app import OllamaAgentApp

IMMEDIATE_COMMANDS: frozenset[tuple[str, str]] = frozenset(
    {
        (CMD_EXIT, "*"),
        (CMD_QUIT, "*"),
        (CMD_QUEUE, "*"),
        (CMD_YOLO, "*"),
        (CMD_STEALTH, "*"),
        (CMD_MODEL, ""),
        (CMD_MODEL, "list"),
        (CMD_EFFORT, ""),
        (CMD_CONTEXT, ""),
        (CMD_PARAMS, ""),
        (CMD_PARAMS, "list"),
        (CMD_SESSION, ""),
        (CMD_SESSION, "list"),
        (CMD_SESSION, "search"),
        (CMD_SESSION, "export"),
        (CMD_SESSION, "delete"),
        (CMD_TASK, ""),
        (CMD_TASK, "list"),
        (CMD_TASK, "delete"),
        (CMD_SKILL, ""),
        (CMD_SKILL, "list"),
        (CMD_SKILL, "show"),
        (CMD_SKILL, "delete"),
        (CMD_RAG, ""),
        (CMD_RAG, "status"),
        (CMD_RAG, "list"),
        (CMD_RAG, "create"),
        (CMD_RAG, "delete"),
        (CMD_RAG, "load"),
        (CMD_RAG, "unload"),
        (CMD_MCP, ""),
        (CMD_MCP, "list"),
        (CMD_MCP, "status"),
        (CMD_AGENTS, ""),
        (CMD_AGENTS, "list"),
    }
)


def _is_immediate_command(val: str) -> bool:
    parts = val.split()
    if not parts:
        return False
    cmd = parts[0].lower()
    sub = parts[1].lower() if len(parts) > 1 else ""
    return (cmd, "*") in IMMEDIATE_COMMANDS or (cmd, sub) in IMMEDIATE_COMMANDS


async def _resume_slash_session(app: OllamaAgentApp, args: list[str], scroll: Any) -> None:
    if len(args) < 2:
        usage_msg = _("Usage: /session resume <session_id>")
        app.show_system_notice(f"[bold #f87171]✕ {usage_msg}[/bold #f87171]")
        return
    try:
        resolved = resume_session(app.repl.console, args[1])
    except HistoryError as exc:
        app.show_system_notice(f"[red]{exc}[/red]")
        return
    if resolved:
        app.repl.runtime.thread_id = resolved
        await scroll.remove_children()
        messages = await app.repl.runtime.get_thread_messages(resolved)
        app.repl.runtime.last_context_tokens = await app.repl.runtime.count_effective_tokens(resolved)
        for msg in messages:
            role = msg.type
            content = extract_text(msg.content)
            if not content:
                continue
            if role in ("human", "user"):
                scroll.mount(UserMessage(content))
            elif role in ("ai", "assistant"):
                scroll.mount(AgentResponse(initial_text=content))
        resumed_label = f"{resolved[:8]} ({resolved})"
        notice = _("Resumed session: {session_id}", session_id=resumed_label)
        app.show_system_notice(f"[bold #38bdf8]✓ {notice}[/bold #38bdf8]")
        app.query_one(AgentHeader).update_header()
        app._deferred_scroll()
    else:
        not_found = _("Session not found: {session_id}", session_id=args[1])
        app.show_system_notice(f"[bold #f87171]✕ {not_found}[/bold #f87171]")


async def _export_slash_session(app: OllamaAgentApp, args: list[str]) -> None:
    try:
        out_file = await export_session(
            app.repl.console,
            app.repl.runtime,
            app.repl.runtime.thread_id,
            output_path=args[1] if len(args) > 1 else None,
        )
    except HistoryError as exc:
        app.show_system_notice(f"[red]{exc}[/red]")
        return
    if out_file:
        export_msg = _("Session exported to: {path}", path=out_file)
        app.show_system_notice(f"[bold #38bdf8]✓ {export_msg}[/bold #38bdf8]")
    else:
        app.show_system_notice(f"[bold #f87171]✕ {_('Failed to export session.')}[/bold #f87171]")


async def _handle_slash_session(app: OllamaAgentApp, cmd: str, args: list[str], scroll: Any) -> bool:
    if cmd in (CMD_CLEAR, CMD_NEW) or (cmd == CMD_SESSION and args and args[0] == "new"):
        app.repl._handle_new_session()
        await scroll.remove_children()
        notice = _("New session started: {session_id}", session_id=app.repl.runtime.thread_id[:8])
        app.show_system_notice(f"[bold #38bdf8]✓ {notice}[/bold #38bdf8]")
        app.query_one(AgentHeader).update_header()
        return True

    if cmd != CMD_SESSION or not args:
        return False

    if args[0] in ("resume", "switch"):
        await _resume_slash_session(app, args, scroll)
        return True

    if args[0] == "export":
        await _export_slash_session(app, args)
        return True

    return False


async def _handle_slash_task(app: OllamaAgentApp, cmd_line: str, args: list[str], scroll: Any) -> bool:
    cmd = cmd_line.split()[0].lower()
    if cmd != CMD_TASK or not args:
        return False

    if args[0] == "create":
        sub_args = args[1:]
        task_info = " ".join(sub_args)
        if task_info:
            prompt_text = (
                f"[System Instruction: The user executed '/task create {task_info}'. "
                f"Use your 'task-creator' instructions to guide the user or draft the task, "
                f"generate a clear and self-contained YAML task file, and save it in /tasks/<task_id>.yaml.]"
            )
        else:
            prompt_text = (
                "[System Instruction: The user executed '/task create'. "
                "Use your 'task-creator' instructions to ask what repeatable workflow or prompt "
                "they want to save as a task, and guide them through creating it in /tasks/<task_id>.yaml.]"
            )
        scroll.mount(UserMessage(cmd_line))
        agent_msg = AgentResponse()
        scroll.mount(agent_msg)
        app._deferred_scroll()
        await app._run_stream(prompt_text, scroll, agent_msg)
        return True

    if args[0] == "run":
        sub_args = args[1:]
        positional = [a for a in sub_args if not a.startswith("-")]
        if not positional:
            app.show_system_notice(f"[bold #f87171]✕ {_('Usage: /task run <id> [-y]')}[/bold #f87171]")
            return True
        target_id = positional[0]
        var_args = positional[1:]
        try:
            t = app.repl._task_ctx.resolve_task(target_id)[1]
            variables = parse_var_assignments(var_args)
            rendered_prompt = t.render(variables)
        except (TaskError, ValueError, TemplateError) as exc:
            app.show_system_notice(f"[red]{exc}[/red]")
            return True

        scroll.mount(UserMessage(cmd_line))
        agent_msg = AgentResponse()
        scroll.mount(agent_msg)
        app._deferred_scroll()

        settings = app.repl.runtime.settings
        prev_model = settings.model.name
        prev_effort = settings.model.reasoning_effort
        prev_yolo = app.repl.runtime.yolo_mode
        try:
            apply_task_settings(settings, t)
            if "-y" in sub_args or "--yolo" in sub_args:
                app.repl.runtime.yolo_mode = True
            await app.repl.runtime.reload()
            await app._run_stream(rendered_prompt, scroll, agent_msg)
        finally:
            settings.model.name = prev_model
            settings.model.reasoning_effort = prev_effort
            app.repl.runtime.yolo_mode = prev_yolo
            await app.repl.runtime.reload()
            app.update_mode_ui()
        return True

    return False


async def _handle_slash_skill(app: OllamaAgentApp, cmd_line: str, args: list[str], scroll: Any) -> bool:
    cmd = cmd_line.split()[0].lower()
    if cmd != CMD_SKILL or not args or args[0] != "create":
        return False

    sub_args = args[1:]
    skill_info = " ".join(sub_args)
    if skill_info:
        prompt_text = (
            f"[System Instruction: The user executed '/skill create {skill_info}'. "
            f"Use your 'skill-creator' instructions to guide the user, gather requirements, "
            f"evaluate whether helper scripts in scripts/ are needed, write the SKILL.md and any scripts "
            f"to /skills/<skill_id>/, and confirm when created.]"
        )
    else:
        prompt_text = (
            "[System Instruction: The user executed '/skill create'. "
            "Use your 'skill-creator' instructions to ask what capability or workflow they want to teach "
            "the agent, evaluate whether helper scripts are needed, and guide them step-by-step through "
            "creating the skill in /skills/<skill_id>/.]"
        )
    scroll.mount(UserMessage(cmd_line))
    agent_msg = AgentResponse()
    scroll.mount(agent_msg)
    app._deferred_scroll()
    await app._run_stream(prompt_text, scroll, agent_msg)
    return True


async def run_slash_command(app: OllamaAgentApp, cmd_line: str) -> None:
    """Execute a user-typed slash command in the TUI."""
    try:
        parts = cmd_line.split()
        cmd = parts[0].lower()
        args = parts[1:]
        scroll = app.query_one(_CHAT_SCROLL_ID)

        if cmd in (CMD_EXIT, CMD_QUIT):
            app.exit()
            return

        if await _handle_slash_session(app, cmd, args, scroll):
            return

        if await _handle_slash_task(app, cmd_line, args, scroll):
            return

        if await _handle_slash_skill(app, cmd_line, args, scroll):
            return

        commands = app.repl._get_commands()
        if cmd not in commands:
            app.show_system_notice(f"[bold #f87171]✕ {_('Unknown command: {cmd}', cmd=cmd)}[/bold #f87171]")
            return

        handler = commands[cmd]
        scroll_w = scroll.size.width if scroll.size.width > 10 else app.size.width
        app.repl.console.width = max(40, scroll_w - 6)
        app.repl.console.height = 25
        with app.repl.console.capture() as capture:
            await safe_call(handler, args, console=app.repl.console)
        output = capture.get()
        if output:
            app.show_system_output(Text.from_ansi(output), title=cmd_line)

        if cmd in (CMD_YOLO, CMD_STEALTH):
            app.update_mode_ui()
        elif cmd == CMD_QUEUE:
            app._update_queue_ui()
        elif cmd in (CMD_MODEL, CMD_EFFORT, CMD_CONTEXT, CMD_RAG):
            app.query_one(AgentHeader).update_header()
    finally:
        app._process_next_in_queue()
