"""Tool-call middleware for the DeepAgents runtime.

Extracted from :mod:`~ollama_agent.agent.agent` to keep that module focused
on agent initialisation and the main inference workflow.

The public entry point is :data:`stream_tool_events_mw`, a wrapped middleware
compatible with DeepAgents' ``middleware=`` parameter.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ..i18n import _
from .builtin_tools import get_tool_timeout


def _compute_output_len(result: Any) -> int:
    content = getattr(result, "content", None)
    if content is None and isinstance(result, Command) and isinstance(result.update, dict):
        messages = result.update.get("messages")
        if messages:
            content = getattr(messages[-1], "content", None)
    if content is None:
        return len(result) if isinstance(result, str) else 0
    return len(str(content))


async def _invoke_with_timeout(
    request: Any,
    handler: Any,
    tool_name: str,
    tool_call_id: str,
) -> Any:
    if tool_name == "task":
        return await handler(request)

    timeout_s = get_tool_timeout()
    try:
        async with asyncio.timeout(timeout_s) as timer:
            return await handler(request)
    except TimeoutError:
        # Since 3.11 asyncio.TimeoutError is TimeoutError, so a TimeoutError raised by
        # the tool itself would otherwise be misreported as our own deadline expiring.
        if not timer.expired():
            raise
        return ToolMessage(
            content=_("Tool '{tool_name}' timed out after {timeout_s}s", tool_name=tool_name, timeout_s=timeout_s),
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )


async def _stream_tool_events(request: Any, handler: Any) -> Any:
    """Emit tool_call / tool_output events and enforce tool timeout."""
    tool_call = request.tool_call
    tool_name = tool_call["name"]
    tool_call_id = tool_call["id"]
    args = tool_call["args"]
    agent_name = args.get("subagent_type") or (tool_call.get("metadata") or {}).get("lc_agent_name")

    event: dict[str, Any] = {"type": "tool_call", "name": tool_name}
    if agent_name:
        event["agent_name"] = agent_name
    request.runtime.stream_writer(event)

    result = await _invoke_with_timeout(request, handler, tool_name, tool_call_id)

    out_event: dict[str, Any] = {"type": "tool_output", "output_len": _compute_output_len(result)}
    if agent_name:
        out_event["agent_name"] = agent_name
    request.runtime.stream_writer(out_event)
    return result


#: Ready-to-use DeepAgents middleware that wraps :func:`_stream_tool_events`.
stream_tool_events_mw: Any = wrap_tool_call(_stream_tool_events)
