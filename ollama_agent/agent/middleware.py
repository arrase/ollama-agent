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

from .builtin_tools import get_tool_timeout


def _extract_tool_name(request: Any) -> str:
    """Extract tool name from a middleware request."""
    if name := getattr(request, "name", None):
        return str(name)
    if tool_call := getattr(request, "tool_call", None):
        if isinstance(tool_call, dict) and "name" in tool_call:
            return str(tool_call["name"])
        if name := getattr(tool_call, "name", None):
            return str(name)
    if tool := getattr(request, "tool", None):
        return getattr(tool, "name", getattr(tool, "__name__", "unknown"))
    return "unknown"


async def _stream_tool_events(request: Any, handler: Any) -> Any:
    """Emit tool_call / tool_output events and enforce tool timeout."""
    runtime = getattr(request, "runtime", None)
    tool_name = _extract_tool_name(request)
    agent_name: str | None = None

    tool_call = getattr(request, "tool_call", None)
    if isinstance(tool_call, dict):
        args = tool_call.get("args")
        if isinstance(args, dict) and tool_name == "task":
            agent_name = args.get("name")
        meta = tool_call.get("metadata")
        if isinstance(meta, dict) and not agent_name:
            agent_name = meta.get("lc_agent_name")

    if runtime is not None:
        event: dict[str, Any] = {"type": "tool_call", "name": tool_name}
        if agent_name:
            event["agent_name"] = agent_name
        runtime.stream_writer(event)

    timeout_s = float(get_tool_timeout())
    try:
        if timeout_s > 0:
            result = await asyncio.wait_for(handler(request), timeout=timeout_s)
        else:
            result = await handler(request)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Tool '{tool_name}' timed out after {timeout_s}s") from exc

    if runtime is not None:
        content = getattr(result, "content", result)
        content_str = str(content)
        if not agent_name:
            meta = getattr(result, "response_metadata", None) or getattr(result, "additional_kwargs", None)
            if isinstance(meta, dict):
                agent_name = meta.get("lc_agent_name")

        out_event: dict[str, Any] = {
            "type": "tool_output",
            "output": "",
            "output_len": len(content_str),
        }
        if agent_name:
            out_event["agent_name"] = agent_name
        runtime.stream_writer(out_event)
    return result


#: Ready-to-use DeepAgents middleware that wraps :func:`_stream_tool_events`.
stream_tool_events_mw: Any = wrap_tool_call(_stream_tool_events)
