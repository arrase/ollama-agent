"""Event payload extraction and streaming dispatch for agent events."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable
from openai.types.responses import ResponseReasoningTextDeltaEvent, ResponseTextDeltaEvent

if TYPE_CHECKING:
    from ..agent import OllamaAgent
    from .base import StreamingRenderer

logger = logging.getLogger(__name__)


def _payloads_raw_response_event(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, ResponseReasoningTextDeltaEvent) and data.delta:
        yield {"type": "reasoning_delta", "content": data.delta}
    elif isinstance(data, ResponseTextDeltaEvent) and data.delta:
        yield {"type": "text_delta", "content": data.delta}


def _payloads_run_item_stream_event(item: Any) -> Iterable[dict[str, Any]]:
    match getattr(item, "type", ""):
        case "tool_call_item": yield {"type": "tool_call", "name": getattr(item, "name", "unknown")}
        case "tool_call_output_item": yield {"type": "tool_output", "output": str(getattr(item, "output", ""))}
        case "reasoning" if (s := getattr(item, "summary", "")): yield {"type": "reasoning_summary", "content": s}


def event_payloads(event: Any) -> Iterable[dict[str, Any]]:
    """Convert an agent streaming event into typed payload dictionaries."""
    match getattr(event, "type", ""):
        case "raw_response_event": yield from _payloads_raw_response_event(getattr(event, "data", None))
        case "run_item_stream_event": yield from _payloads_run_item_stream_event(getattr(event, "item", None))
        case "agent_updated_stream_event": yield {"type": "agent_update", "name": getattr(getattr(event, "new_agent", None), "name", "unknown")}


async def stream_agent_events(
    agent: "OllamaAgent", prompt: object, renderer: "StreamingRenderer", *,
    model: str | None = None, reasoning_effort: str | None = None,
    ignore: Iterable[str] | None = None, auto_close: bool = False,
) -> None:
    """Dispatch streamed agent events to the provided renderer."""
    ignored = set(ignore or ())
    try:
        async for event in agent.run_async_streamed(prompt, model=model, reasoning_effort=reasoning_effort):
            if (etype := event.get("type")) and etype not in ignored:
                renderer.on_event(event)
    except Exception as exc:
        logger.exception("Error streaming agent events: %s", exc)
        renderer.on_error({"type": "error", "content": str(exc)})
    finally:
        if auto_close:
            renderer.close()


async def stream_agent_events_with_renderer(
    agent: "OllamaAgent", prompt: object, renderer: "StreamingRenderer", **kwargs
) -> None:
    await stream_agent_events(agent, prompt, renderer, auto_close=True, **kwargs)
