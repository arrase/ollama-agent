"""Build subagent specifications for create_deep_agent."""

from __future__ import annotations

import asyncio
import platform
from contextlib import AsyncExitStack
from typing import Any

from ..core import create_ollama_chat_model, validate_reasoning_effort
from ..mcp import load_subagent_mcp_tools
from ..settings import ModelSettings, SubAgentSettings


async def build_subagents(
    subagent_settings: list[SubAgentSettings],
    *,
    model_settings: ModelSettings,
    exit_stack: AsyncExitStack,
) -> list[dict[str, Any]]:
    """Convert ``SubAgentSettings`` into dicts for ``create_deep_agent(subagents=...)``."""
    tasks = [
        _build_spec(sa, model_settings=model_settings, exit_stack=exit_stack)
        for sa in subagent_settings
    ]
    return await asyncio.gather(*tasks)


async def _build_spec(
    sa: SubAgentSettings,
    *,
    model_settings: ModelSettings,
    exit_stack: AsyncExitStack,
) -> dict[str, Any]:
    """Build a single subagent spec dict."""
    if not sa.name:
        raise ValueError("Subagent configuration error: name cannot be empty")
    if not sa.description:
        raise ValueError(f"Subagent '{sa.name}' configuration error: description cannot be empty")

    base_prompt = sa.system_prompt or sa.description
    os_info = f"\n\n# ENVIRONMENT\nOperating System: {platform.system()} ({platform.release()})\n"

    spec: dict[str, Any] = {
        "name": sa.name,
        "description": sa.description,
        "system_prompt": base_prompt + os_info,
    }

    name = sa.model or model_settings.name
    num_ctx = sa.context_window or model_settings.context_window
    spec["model"] = await create_ollama_chat_model(
        model=name,
        base_url=model_settings.base_url,
        context_window=num_ctx,
        reasoning_effort=validate_reasoning_effort(model_settings.reasoning_effort),
    )

    spec["skills"] = ["/skills/"]

    if sa.mcp_servers:
        tools = await load_subagent_mcp_tools(sa.name, sa.mcp_servers, exit_stack)
        if tools:
            spec["tools"] = tools

    return spec
