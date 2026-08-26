"""Streaming helpers.

The agent runtime is responsible for producing normalized payload dictionaries.
Renderers consume those payloads.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable

from langgraph.types import Command
from rich.console import Console

from ..i18n import _
from .console_renderer import ConsoleStreamingRenderer

if TYPE_CHECKING:
    from ..agent import AgentRuntime
    from .base import StreamingRenderer

logger = logging.getLogger(__name__)


async def stream_agent_events(
    runtime: AgentRuntime,
    prompt: str | Command,
    renderer: StreamingRenderer,
    *,
    thread_id: str = "",
    ignore: Iterable[str] = (),
    auto_close: bool = True,
) -> bool:
    """Stream agent events, returning whether the run completed without aborting."""
    ignored = set(ignore)
    current_prompt = prompt
    completed = True
    try:
        while True:
            interrupted = False
            interrupt_event: dict[str, Any] | None = None
            async for event in runtime.run_streamed(current_prompt, thread_id=thread_id):
                etype = event.get("type")
                if etype == "interrupt":
                    interrupted = True
                    interrupt_event = event
                elif etype and etype not in ignored:
                    renderer.on_event(event)

            if interrupted and interrupt_event is not None:
                decisions = await renderer.handle_interrupt(interrupt_event, runtime)
                if decisions is not None:
                    current_prompt = Command(resume={"decisions": decisions})
                    continue
                completed = False

            break
    except KeyboardInterrupt:
        completed = False
        logger.info("Agent run interrupted by user")
        renderer.on_warning({"type": "warning", "content": _("Execution interrupted by user.")})
    finally:
        if auto_close:
            renderer.close()
    return completed


async def run_non_interactive(
    runtime: AgentRuntime, prompt: str | Command, *, thread_id: str = ""
) -> bool:
    """Stream agent output to the console (non-interactive mode).

    Returns whether the run finished completely (no abort, cancellation or error).
    """
    return await stream_agent_events(
        runtime,
        prompt,
        ConsoleStreamingRenderer(Console()),
        thread_id=thread_id,
    )

