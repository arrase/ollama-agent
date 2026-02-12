"""Streaming helpers.

The agent runtime is responsible for producing normalized payload dictionaries.
Renderers consume those payloads.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from ..agent import OllamaAgent
    from .base import StreamingRenderer

logger = logging.getLogger(__name__)


async def stream_agent_events(
    agent: "OllamaAgent",
    prompt: object,
    renderer: "StreamingRenderer",
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    ignore: Iterable[str] | None = None,
    auto_close: bool = False,
) -> None:
    ignored = set(ignore or ())
    try:
        async for event in agent.run_async_streamed(prompt, model=model, reasoning_effort=reasoning_effort):
            etype = event.get("type")
            if etype and etype not in ignored:
                renderer.on_event(event)
    except Exception as exc:
        logger.exception("Error streaming agent events: %s", exc)
        renderer.on_error({"type": "error", "content": str(exc)})
    finally:
        if auto_close:
            renderer.close()


async def stream_agent_events_with_renderer(
    agent: "OllamaAgent", prompt: object, renderer: "StreamingRenderer", **kwargs: Any
) -> None:
    await stream_agent_events(agent, prompt, renderer, auto_close=True, **kwargs)
