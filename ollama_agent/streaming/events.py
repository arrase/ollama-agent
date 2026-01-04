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
    item_type = getattr(item, "type", "")
    if item_type == "tool_call_item":
        yield {"type": "tool_call", "name": getattr(item, "name", "unknown")}
    elif item_type == "tool_call_output_item":
        yield {"type": "tool_output", "output": str(getattr(item, "output", ""))}
    elif item_type == "reasoning" and (summary := getattr(item, "summary", "")):
        yield {"type": "reasoning_summary", "content": summary}


def _payloads_agent_updated_stream_event(new_agent: Any) -> Iterable[dict[str, Any]]:
    yield {"type": "agent_update", "name": getattr(new_agent, "name", "unknown")}


def event_payloads(event: Any) -> Iterable[dict[str, Any]]:
    """Convert an agent streaming event into typed payload dictionaries.

    Supported event types:
    - raw_response_event: Text and reasoning deltas
    - run_item_stream_event: Tool calls and outputs
    - agent_updated_stream_event: Agent updates
    """
    match getattr(event, "type", ""):
        case "raw_response_event":
            yield from _payloads_raw_response_event(getattr(event, "data", None))
        case "run_item_stream_event":
            yield from _payloads_run_item_stream_event(getattr(event, "item", None))
        case "agent_updated_stream_event":
            yield from _payloads_agent_updated_stream_event(
                getattr(event, "new_agent", None)
            )


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
    """Dispatch streamed agent events to the provided renderer.
    
    If auto_close=True, renderer.close() is called in finally block.
    """
    ignored = set(ignore or ())
    try:
        async for event in agent.run_async_streamed(
            prompt, model=model, reasoning_effort=reasoning_effort
        ):
            if (etype := event.get("type")) and etype not in ignored:
                renderer.on_event(event)
    except Exception as exc:
        logger.exception("Error streaming agent events: %s", exc)
        renderer.on_error({"type": "error", "content": str(exc)})
    finally:
        if auto_close:
            renderer.close()


# Backwards-compatible alias
async def stream_agent_events_with_renderer(
    agent: "OllamaAgent", prompt: object, renderer: "StreamingRenderer", **kwargs
) -> None:
    await stream_agent_events(agent, prompt, renderer, auto_close=True, **kwargs)
