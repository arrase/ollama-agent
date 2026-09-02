"""Streaming helpers.

The agent runtime is responsible for producing normalized payload dictionaries.
Renderers consume those payloads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.types import Command
from rich.console import Console

from .console_renderer import ConsoleStreamingRenderer

if TYPE_CHECKING:
    from ..agent import AgentRuntime
    from .base import StreamingRenderer


async def stream_agent_events(
    runtime: AgentRuntime,
    prompt: str | Command[Any],
    renderer: StreamingRenderer,
    *,
    thread_id: str = "",
) -> bool:
    """Stream agent events, returning whether the run completed without aborting."""
    current_prompt = prompt
    completed = True
    try:
        while True:
            interrupt_event: dict[str, Any] | None = None
            async for event in runtime.run_streamed(current_prompt, thread_id=thread_id):
                etype = event["type"]
                if etype == "interrupt":
                    interrupt_event = event
                    continue
                if etype == "error":
                    completed = False
                renderer.on_event(event)

            if interrupt_event is not None:
                decisions = await renderer.handle_interrupt(interrupt_event, runtime)
                if decisions is not None:
                    current_prompt = Command(resume={"decisions": decisions})
                    continue
                completed = False

            break
    finally:
        renderer.close()
    return completed


async def run_non_interactive(runtime: AgentRuntime, prompt: str | Command[Any], *, thread_id: str = "") -> bool:
    """Stream agent output to the console (non-interactive mode).

    Returns whether the run finished completely (no abort, cancellation or error).
    """
    return await stream_agent_events(
        runtime,
        prompt,
        ConsoleStreamingRenderer(Console()),
        thread_id=thread_id,
    )
