"""Tool-call middleware for the DeepAgents runtime.

Extracted from :mod:`~ollama_agent.agent.agent` to keep that module focused
on agent initialisation and the main inference workflow.

The public entry point is :data:`stream_tool_events_mw`, a wrapped middleware
compatible with DeepAgents' ``middleware=`` parameter.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from langchain.agents.middleware import wrap_tool_call

from .builtin_tools import get_tool_timeout

logger = logging.getLogger(__name__)


async def _stream_tool_events(request: Any, handler: Any) -> Any:
    """Emit ``tool_call`` / ``tool_output`` events so renderers keep working.

    Also enforces the configured tool timeout.
    """
    runtime = getattr(request, "runtime", None)
    agent_name: str | None = None
    tool_name = next(
        (n for attr in ("name", "tool_name") if (n := getattr(request, attr, None)))
        or (
            n
            for obj in (getattr(request, "tool", None),)
            if obj
            for attr in ("name", "__name__")
            if (n := getattr(obj, attr, None))
        ),
        "",
    )

    tool_call = getattr(request, "tool_call", None)
    tool_args: dict[str, Any] | None = None
    if isinstance(tool_call, dict):
        if not tool_name:
            tool_name = str(tool_call.get("name") or "")
        maybe_args = tool_call.get("args")
        if isinstance(maybe_args, dict):
            tool_args = maybe_args
        maybe_meta = tool_call.get("metadata")
        if isinstance(maybe_meta, dict):
            maybe_agent = maybe_meta.get("lc_agent_name")
            if isinstance(maybe_agent, str) and maybe_agent:
                agent_name = maybe_agent
    elif tool_call is not None:
        maybe_name = getattr(tool_call, "name", None)
        if not tool_name and isinstance(maybe_name, str) and maybe_name:
            tool_name = maybe_name
        maybe_args = getattr(tool_call, "args", None)
        if isinstance(maybe_args, dict):
            tool_args = maybe_args

    if not tool_name:
        tool_name = "unknown"

    if (not agent_name) and tool_name == "task" and isinstance(tool_args, dict):
        maybe_task_agent = tool_args.get("name")
        if isinstance(maybe_task_agent, str) and maybe_task_agent:
            agent_name = maybe_task_agent

    if runtime is not None:
        try:
            event: dict[str, Any] = {"type": "tool_call", "name": tool_name}
            if isinstance(agent_name, str) and agent_name:
                event["agent_name"] = agent_name
            runtime.stream_writer(event)
        except Exception:
            pass

    timeout_s = int(get_tool_timeout())
    try:
        if timeout_s > 0:
            result = await asyncio.wait_for(handler(request), timeout=float(timeout_s))
        else:
            result = await handler(request)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Tool '{tool_name}' timed out after {timeout_s}s") from exc

    if runtime is not None:
        try:
            content = getattr(result, "content", result)
            content_str = str(content)
            if not agent_name:
                for attr in ("response_metadata", "additional_kwargs", "metadata"):
                    maybe_meta = getattr(result, attr, None)
                    if isinstance(maybe_meta, dict):
                        maybe_agent_name = maybe_meta.get("lc_agent_name")
                        if isinstance(maybe_agent_name, str) and maybe_agent_name:
                            agent_name = maybe_agent_name
                            break
            # Tool outputs can be large; emit only small metadata for the UI.
            event = {
                "type": "tool_output",
                "output": "",
                "output_len": len(content_str),
            }
            if isinstance(agent_name, str) and agent_name:
                event["agent_name"] = agent_name
            runtime.stream_writer(event)
        except Exception:
            pass
    return result


#: Ready-to-use DeepAgents middleware that wraps :func:`_stream_tool_events`.
stream_tool_events_mw: Any = cast(Any, wrap_tool_call)(_stream_tool_events)
