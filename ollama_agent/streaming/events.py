"""Streaming helpers.

The agent runtime is responsible for producing normalized payload dictionaries.
Renderers consume those payloads.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Iterable

from rich.console import Console
from langgraph.types import Command

from .console_renderer import ConsoleStreamingRenderer

if TYPE_CHECKING:
    from ..agent import AgentRuntime
    from .base import StreamingRenderer

logger = logging.getLogger(__name__)


async def stream_agent_events(
    runtime: "AgentRuntime",
    prompt: str | Command,
    renderer: "StreamingRenderer",
    *,
    thread_id: str = "",
    ignore: Iterable[str] = (),
    auto_close: bool = True,
) -> None:
    ignored = set(ignore)
    current_prompt = prompt
    try:
        while True:
            interrupted = False
            interrupt_event = None
            async for event in runtime.run_streamed(current_prompt, thread_id=thread_id):
                etype = event.get("type")
                if etype == "interrupt":
                    interrupted = True
                    interrupt_event = event
                elif etype and etype not in ignored:
                    renderer.on_event(event)

            if interrupted and interrupt_event:
                decisions = await renderer.handle_interrupt(interrupt_event, runtime)
                if decisions is not None:
                    current_prompt = Command(resume={"decisions": decisions})
                    continue

            break
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Error streaming agent events: %s", exc)
        renderer.on_error({"type": "error", "content": str(exc)})
    finally:
        if auto_close:
            renderer.close()


async def run_non_interactive(
    runtime: "AgentRuntime", prompt: str | Command, *, thread_id: str = ""
) -> None:
    """Stream agent output to the console (non-interactive mode)."""
    await stream_agent_events(
        runtime,
        prompt,
        ConsoleStreamingRenderer(Console()),
        thread_id=thread_id,
        ignore={"agent_update"},
    )

