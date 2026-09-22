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

from ..i18n import _
from .builtin_tools import get_tool_timeout


async def _stream_tool_events(request: Any, handler: Any) -> Any:
    """Emit tool_call / tool_output events and enforce tool timeout."""
    tool_call = request.tool_call
    tool_name = tool_call["name"]
    tool_call_id = tool_call["id"]
    agent_name = None
    if tool_name == "task":
        args = tool_call.get("args") or {}
        agent_name = args.get("subagent_type") or args.get("name")
    if not agent_name:
        metadata = tool_call.get("metadata") or {}
        agent_name = metadata.get("lc_agent_name")

    event: dict[str, Any] = {"type": "tool_call", "name": tool_name}
    if agent_name:
        event["agent_name"] = agent_name
    request.runtime.stream_writer(event)

    timeout_s = get_tool_timeout()
    try:
        result = await asyncio.wait_for(handler(request), timeout=timeout_s)
    except asyncio.TimeoutError:
        result = ToolMessage(
            content=_("Tool '{tool_name}' timed out after {timeout_s}s", tool_name=tool_name, timeout_s=timeout_s),
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )

    content = getattr(result, "content", result)
    out_event: dict[str, Any] = {"type": "tool_output", "output_len": len(str(content))}
    if agent_name:
        out_event["agent_name"] = agent_name
    request.runtime.stream_writer(out_event)
    return result


#: Ready-to-use DeepAgents middleware that wraps :func:`_stream_tool_events`.
stream_tool_events_mw: Any = wrap_tool_call(_stream_tool_events)
