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

from ..i18n import _
from .builtin_tools import get_tool_timeout


async def _stream_tool_events(request: Any, handler: Any) -> Any:
    """Emit tool_call / tool_output events and enforce tool timeout."""
    runtime = request.runtime
    tool_name = request.tool_call["name"]
    agent_name = None

    args = request.tool_call.get("args")
    if args and tool_name == "task":
        agent_name = args.get("name")
    meta = request.tool_call.get("metadata")
    if meta and not agent_name:
        agent_name = meta.get("lc_agent_name")

    event: dict[str, Any] = {"type": "tool_call", "name": tool_name}
    if agent_name:
        event["agent_name"] = agent_name
    runtime.stream_writer(event)

    timeout_s = get_tool_timeout()
    try:
        result = await asyncio.wait_for(handler(request), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            _("Tool '{tool_name}' timed out after {timeout_s}s", tool_name=tool_name, timeout_s=timeout_s)
        ) from exc

    content = result.content if hasattr(result, "content") else result
    content_str = str(content)
    out_event: dict[str, Any] = {"type": "tool_output", "output_len": len(content_str)}
    if agent_name:
        out_event["agent_name"] = agent_name
    runtime.stream_writer(out_event)
    return result


#: Ready-to-use DeepAgents middleware that wraps :func:`_stream_tool_events`.
stream_tool_events_mw: Any = wrap_tool_call(_stream_tool_events)
