"""Interactive slash command handling and execution in the TUI."""

from __future__ import annotations

import asyncio
import logging
import shlex
from typing import Any

from rich.markup import escape
from rich.text import Text

from ...agent.episodic_memory import HistoryError
from ...core.common import extract_text
from ...i18n import _
from ...tasks import (
    TaskError,
    apply_task_settings,
    parse_var_assignments,
    prepare_task_run,
)
from ..commands.dispatch import safe_call
from ..commands.sessions import (
    export_session,
    is_current,
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
)
from .widgets.footer import AgentFooter
from .widgets.header import AgentHeader
from .widgets.messages import AgentResponse, UserMessage

_log = logging.getLogger(__name__)
_CHAT_SCROLL_ID = "#chat-scroll"

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


def _mount_history_messages(scroll: Any, messages: list[Any]) -> None:
    for msg in messages:
        content = extract_text(msg.content)
        if not content:
            continue
        if msg.type in ("human", "user"):
            scroll.mount(UserMessage(content))
        elif msg.type in ("ai", "assistant"):
            scroll.mount(AgentResponse(initial_text=content))


async def _resume_slash_session(app: Any, args: list[str], scroll: Any) -> None:
    if len(args) < 2:
        usage_msg = _("Usage: /session resume <session_id>")
        app.show_system_notice(f"[bold #f87171]✕ {usage_msg}[/bold #f87171]")
        return
    try:
        with app.repl.console.capture() as capture:
            resolved = resume_session(app.repl.console, args[1])
    except HistoryError as exc:
        app.show_system_notice(f"[red]{escape(str(exc))}[/red]")
        return
    if not resolved:
        err_output = capture.get().strip()
        if err_output:
            app.show_system_notice(Text.from_ansi(err_output))
        else:
            not_found = _("Session not found: {session_id}", session_id=escape(args[1]))
            app.show_system_notice(f"[bold #f87171]✕ {not_found}[/bold #f87171]")
        return

    app.repl.runtime.thread_id = resolved
    await scroll.remove_children()
    messages = await app.repl.runtime.get_thread_messages(resolved)
    app.repl.runtime.last_context_tokens = await app.repl.runtime.count_effective_tokens(resolved)
    _mount_history_messages(scroll, messages)
    resumed_label = escape(f"{resolved[:8]} ({resolved})")
    notice = _("Resumed session: {session_id}", session_id=resumed_label)
    app.show_system_notice(f"[bold #38bdf8]✓ {notice}[/bold #38bdf8]")
    app.query_one(AgentHeader).update_header()
    app._deferred_scroll()


async def _export_slash_session(app: Any, args: list[str]) -> None:
    try:
        with app.repl.console.capture() as capture:
            out_file = await export_session(
                app.repl.console,
                app.repl.runtime,
                app.repl.runtime.thread_id,
                output_path=args[1] if len(args) > 1 else None,
            )
    except HistoryError as exc:
        app.show_system_notice(f"[red]{escape(str(exc))}[/red]")
        return
    if out_file:
        export_msg = _("Session exported to: {path}", path=escape(str(out_file)))
        app.show_system_notice(f"[bold #38bdf8]✓ {export_msg}[/bold #38bdf8]")
    else:
        err_output = capture.get().strip()
        if err_output:
            app.show_system_notice(Text.from_ansi(err_output))
        else:
            app.show_system_notice(f"[bold #f87171]✕ {_('Failed to export session.')}[/bold #f87171]")


async def _handle_slash_session(app: Any, cmd: str, args: list[str], scroll: Any) -> bool:
    if cmd in (CMD_CLEAR, CMD_NEW) or (cmd == CMD_SESSION and args and args[0] == "new"):
        app.repl._handle_new_session()
        await scroll.remove_children()
        notice = _("New session started: {session_id}", session_id=escape(app.repl.runtime.thread_id[:8]))
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


async def _run_creator_skill(
    app: Any,
    cmd_line: str,
    sub_args: list[str],
    scroll: Any,
    cmd_prefix: str,
    with_args_instructions: str,
    without_args_instructions: str,
) -> None:
    info = " ".join(sub_args)
    if info:
        prompt_text = f"[System Instruction: The user executed '{cmd_prefix} {info}'. {with_args_instructions}]"
    else:
        prompt_text = f"[System Instruction: The user executed '{cmd_prefix}'. {without_args_instructions}]"
    scroll, agent_msg = app.mount_turn_widgets(cmd_line, scroll)
    await app._run_stream(prompt_text, scroll, agent_msg)


async def _handle_task_create(app: Any, cmd_line: str, sub_args: list[str], scroll: Any) -> None:
    await _run_creator_skill(
        app,
        cmd_line,
        sub_args,
        scroll,
        "/task create",
        (
            "Use your 'task-creator' instructions to guide the user or draft the task, "
            "generate a clear and self-contained YAML task file, and save it in /tasks/<task_id>.yaml."
        ),
        (
            "Use your 'task-creator' instructions to ask what repeatable workflow or prompt "
            "they want to save as a task, and guide them through creating it in /tasks/<task_id>.yaml."
        ),
    )


async def _handle_task_run(app: Any, cmd_line: str, sub_args: list[str], scroll: Any) -> None:
    positional: list[str] = []
    yolo_requested = False
    idx = 0
    while idx < len(sub_args):
        arg = sub_args[idx]
        if arg in ("-y", "--yolo"):
            yolo_requested = True
        elif arg == "--var" and idx + 1 < len(sub_args):
            positional.append(sub_args[idx + 1])
            idx += 1
        elif arg.startswith("--var="):
            positional.append(arg.split("=", 1)[1])
        elif not arg.startswith("-"):
            positional.append(arg)
        idx += 1

    if not positional:
        app.show_system_notice(f"[bold #f87171]✕ {_('Usage: /task run <id> [-y]')}[/bold #f87171]")
        return
    target_id = positional[0]
    var_args = positional[1:]
    try:
        variables = parse_var_assignments(var_args)
        _tid, t, rendered_prompt = prepare_task_run(app.repl._task_ctx, target_id, variables)
    except TaskError as exc:
        app.show_system_notice(f"[red]{escape(str(exc))}[/red]")
        return

    scroll, agent_msg = app.mount_turn_widgets(cmd_line, scroll)

    settings = app.repl.runtime.settings
    prev_yolo = app.repl.runtime.yolo_mode
    restore_settings = apply_task_settings(settings, t)

    async def _restore_task_settings() -> None:
        restore_settings()
        app.repl.runtime.yolo_mode = prev_yolo
        await app.repl.runtime.reload()
        app.update_mode_ui()

    app._post_stream_cleanup = _restore_task_settings
    try:
        if yolo_requested:
            app.repl.runtime.yolo_mode = True
        await app.repl.runtime.reload()
        app.update_mode_ui()
        await app._run_stream(rendered_prompt, scroll, agent_msg)
    finally:
        if not app._is_approval_pending:
            await app._run_post_stream_cleanup()


async def _handle_slash_task(app: Any, cmd_line: str, args: list[str], scroll: Any) -> bool:
    cmd = cmd_line.split(maxsplit=1)[0].lower()
    if cmd != CMD_TASK or not args:
        return False

    if args[0] == "create":
        await _handle_task_create(app, cmd_line, args[1:], scroll)
        return True

    if args[0] == "run":
        await _handle_task_run(app, cmd_line, args[1:], scroll)
        return True

    return False


async def _handle_slash_skill(app: Any, cmd_line: str, args: list[str], scroll: Any) -> bool:
    cmd = cmd_line.split(maxsplit=1)[0].lower()
    if cmd != CMD_SKILL or not args or args[0] != "create":
        return False

    await _run_creator_skill(
        app,
        cmd_line,
        args[1:],
        scroll,
        "/skill create",
        (
            "Use your 'skill-creator' instructions to guide the user, gather requirements, "
            "evaluate whether helper scripts in scripts/ are needed, write the SKILL.md and any scripts "
            "to /skills/<skill_id>/, and confirm when created."
        ),
        (
            "Use your 'skill-creator' instructions to ask what capability or workflow they want to teach "
            "the agent, evaluate whether helper scripts are needed, and guide them step-by-step through "
            "creating the skill in /skills/<skill_id>/."
        ),
    )
    return True


async def _execute_repl_command(
    app: Any,
    cmd_line: str,
    cmd: str,
    args: list[str],
    scroll: Any,
) -> None:
    commands = app.repl._get_commands()
    if cmd not in commands:
        notice = escape(_("Unknown command: {cmd}", cmd=cmd))
        app.show_system_notice(f"[bold #f87171]✕ {notice}[/bold #f87171]")
        return

    handler = commands[cmd]
    console = app.repl.console
    prev_width, prev_height = console.width, console.height
    scroll_w = scroll.size.width if scroll.size.width > 10 else app.size.width
    console.width = max(40, scroll_w - 6)
    try:
        with console.capture() as capture:
            await safe_call(handler, args, console=console)
    finally:
        console.width, console.height = prev_width, prev_height
    output = capture.get()
    if output:
        app.show_system_output(Text.from_ansi(output), title=cmd_line)

    if cmd in (CMD_YOLO, CMD_STEALTH):
        app.update_mode_ui()
    elif cmd == CMD_QUEUE:
        app._update_queue_ui()
    elif cmd in (CMD_MODEL, CMD_EFFORT, CMD_CONTEXT, CMD_RAG):
        app.query_one(AgentHeader).update_header()
    elif (
        cmd == CMD_SESSION
        and len(args) > 1
        and args[0] == "delete"
        and is_current(args[1].strip(), app.repl.runtime.thread_id)
        and "✓" in output
    ):
        app.repl._handle_new_session()
        await scroll.remove_children()
        app.query_one(AgentHeader).update_header()


async def run_slash_command(app: Any, cmd_line: str) -> None:
    """Execute a user-typed slash command in the TUI.

    Never propagates: a slash command that fails must surface the error in the UI,
    not terminate the app via Textual's default ``exit_on_error`` worker behaviour.
    """
    is_immediate = _is_immediate_command(cmd_line)
    if not is_immediate:
        app._is_generating = True
        app.query_one(AgentFooter).set_generating(True)
    try:
        try:
            parts = shlex.split(cmd_line)
        except ValueError:
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

        await _execute_repl_command(app, cmd_line, cmd, args, scroll)
    except asyncio.CancelledError:
        raise
    # A slash command must never kill the app.
    except Exception as exc:
        _log.exception("Unhandled error running slash command %r", cmd_line)
        notice = escape(_("Command failed: {exc}", exc=exc))
        app.show_system_notice(f"[bold #f87171]✕ {notice}[/bold #f87171]")
    finally:
        # Only non-immediate commands claim the generation slot via _start_slash.
        if not is_immediate:
            app._is_generating = False
            app.query_one(AgentFooter).set_generating(False)
            if not app._is_approval_pending:
                app._process_next_in_queue()
