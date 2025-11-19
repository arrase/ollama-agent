from typing import Any, Iterable

from openai.types.responses import ResponseReasoningTextDeltaEvent, ResponseTextDeltaEvent

from .dispatcher import stream_agent_events
from .renderer import StreamingRenderer, ConsoleStreamingRenderer, TUIStreamingRenderer

__all__ = [
    "stream_agent_events",
    "StreamingRenderer",
    "ConsoleStreamingRenderer",
    "TUIStreamingRenderer",
    "event_payloads",
]


def _raw_event_payloads(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, ResponseReasoningTextDeltaEvent) and data.delta:
        yield {"type": "reasoning_delta", "content": data.delta}
    elif isinstance(data, ResponseTextDeltaEvent) and data.delta:
        yield {"type": "text_delta", "content": data.delta}


def _item_event_payloads(item: Any) -> Iterable[dict[str, Any]]:
    item_type = getattr(item, "type", "")
    if item_type == "tool_call_item":
        yield {"type": "tool_call", "name": getattr(item, "name", "unknown")}
    elif item_type == "tool_call_output_item":
        yield {"type": "tool_output", "output": str(getattr(item, "output", ""))}
    elif item_type == "reasoning" and (summary := getattr(item, "summary", "")):
        yield {"type": "reasoning_summary", "content": summary}


def event_payloads(event: Any) -> Iterable[dict[str, Any]]:
    event_type = getattr(event, "type", "")
    if event_type == "raw_response_event":
        yield from _raw_event_payloads(getattr(event, "data", None))
    elif event_type == "run_item_stream_event":
        yield from _item_event_payloads(getattr(event, "item", None))
    elif event_type == "agent_updated_stream_event":
        yield {"type": "agent_update", "name": getattr(getattr(event, "new_agent", None), "name", "unknown")}
