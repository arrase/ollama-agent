"""Shared helpers for streaming agent events."""

from __future__ import annotations

import logging
from typing import Iterable

from ..agent import OllamaAgent
from .renderer import StreamingRenderer

logger = logging.getLogger(__name__)


async def stream_agent_events(
    agent: OllamaAgent,
    prompt: str,
    renderer: StreamingRenderer,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    ignore: Iterable[str] | None = None,
) -> None:
    """Dispatch streamed agent events to the provided renderer."""

    ignored = set(ignore or ())

    try:
        async for event in agent.run_async_streamed(
            prompt,
            model=model,
            reasoning_effort=reasoning_effort,
        ):
            event_type = event.get("type")

            if not isinstance(event_type, str) or event_type in ignored:
                continue

            renderer.on_event(event)

    except Exception as exc:
        logger.exception("Error streaming agent events: %s", exc)
        renderer.on_error({"type": "error", "content": str(exc)})
