"""Interactive wizard helpers for the REPL.

Provides prompt utilities and multi-step creation wizards that are used
by the REPL but decoupled from its main loop.
"""

import inspect

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console

from ..skills import SkillsContext, create_skill
from ..tasks.commands import CLIContext, create_task


async def safe_call(fn, *args, **kwargs):
    """Call *fn*(*args, **kwargs), awaiting if necessary and silencing SystemExit."""
    try:
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            await result
    except SystemExit:
        pass


async def prompt_line(session: PromptSession, label: str, default: str = "") -> str:
    """Prompt the user for a single line of input."""
    prompt_html = f"<b>{label}</b>"
    if default:
        prompt_html += f" (default: {default})"
    prompt_html += "> "
    val = (await session.prompt_async(HTML(prompt_html))).strip()
    return val or default


async def prompt_multiline(
    session: PromptSession,
    console: Console,
    prompt_html: str,
    hint: str,
) -> str:
    """Prompt the user for multiline input with a hint message."""
    console.print(f"[dim]{hint}[/dim]")
    buf = session.default_buffer
    old_multiline = buf.multiline
    try:
        return await session.prompt_async(HTML(prompt_html), multiline=True)
    finally:
        buf.multiline = old_multiline


async def task_create_wizard(
    session: PromptSession,
    console: Console,
    task_ctx: CLIContext,
    default_model: str,
    default_effort: str,
    args: list[str],
) -> None:
    """Interactive wizard for creating a task."""
    task_id, force = args[0], "--force" in args[1:]

    title = await prompt_line(session, "title")
    model = await prompt_line(session, "model", default_model)
    effort = await prompt_line(session, "effort", default_effort)
    task_prompt = await prompt_multiline(
        session,
        console,
        "<b>prompt> </b>",
        "Enter the task prompt (multiline). Finish with Esc+Enter.",
    )
    await safe_call(
        create_task,
        task_ctx,
        task_id,
        title=title,
        prompt=task_prompt,
        model=model,
        reasoning_effort=effort,
        force=force,
    )


async def skill_create_wizard(
    session: PromptSession,
    console: Console,
    skills_ctx: SkillsContext,
    args: list[str],
) -> None:
    """Interactive wizard for creating a skill."""
    skill_id, force = args[0], "--force" in args[1:]

    name = await prompt_line(session, "name")
    description = await prompt_line(session, "description")
    instructions = await prompt_multiline(
        session,
        console,
        "<b>instructions> </b>",
        "Enter skill instructions (multiline markdown). Finish with Esc+Enter.",
    )
    await safe_call(
        create_skill,
        skills_ctx,
        skill_id,
        name=name,
        description=description,
        instructions=instructions,
        force=force,
    )
