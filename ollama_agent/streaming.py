"""Shared helpers for streaming agent events."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .agent import OllamaAgent

EventHandler = Callable[[dict[str, Any]], None]


async def stream_agent_events(
    agent: OllamaAgent,
    prompt: str,
    handlers: dict[str, EventHandler],
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    on_error: Callable[[dict[str, Any]], None] | None = None,
    ignore: Iterable[str] | None = None,
) -> None:
    """Dispatch streamed agent events to the provided handlers."""

    ignored = set(ignore or ())

    async for event in agent.run_async_streamed(
        prompt,
        model=model,
        reasoning_effort=reasoning_effort,
    ):
        event_type = event.get("type")

        if event_type == "error":
            if on_error:
                on_error(event)
            break

        if not isinstance(event_type, str) or event_type in ignored:
            continue

        handler = handlers.get(event_type)
        if handler:
            handler(event)
