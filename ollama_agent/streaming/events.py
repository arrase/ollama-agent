"""Streaming helpers.

The agent runtime is responsible for producing normalized payload dictionaries.
Renderers consume those payloads.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Iterable

from rich.console import Console

from .console_renderer import ConsoleStreamingRenderer

if TYPE_CHECKING:
    from ..agent import AgentRuntime
    from .base import StreamingRenderer

logger = logging.getLogger(__name__)


async def stream_agent_events(
    runtime: "AgentRuntime",
    prompt: str,
    renderer: "StreamingRenderer",
    *,
    thread_id: str | None = None,
    ignore: Iterable[str] | None = None,
    auto_close: bool = True,
) -> None:
    ignored = set(ignore or ())
    try:
        async for event in runtime.run_streamed(prompt, thread_id=thread_id):
            etype = event.get("type")
            if etype and etype not in ignored:
                renderer.on_event(event)
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.exception("Error streaming agent events: %s", exc)
        renderer.on_error({"type": "error", "content": str(exc)})
    finally:
        if auto_close:
            renderer.close()


async def run_non_interactive(
    runtime: "AgentRuntime", prompt: str, *, thread_id: str | None = None
) -> None:
    """Stream agent output to the console (non-interactive mode)."""
    await stream_agent_events(
        runtime,
        prompt,
        ConsoleStreamingRenderer(Console()),
        thread_id=thread_id,
        ignore={"agent_update"},
    )

