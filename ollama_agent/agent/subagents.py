"""Build subagent specifications for create_deep_agent."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

from ..core import create_ollama_chat_model, validate_reasoning_effort
from ..mcp import load_subagent_mcp_tools
from ..settings import ModelSettings, SubAgentSettings

_log = logging.getLogger(__name__)


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
    results = await asyncio.gather(*tasks)
    return [spec for spec in results if spec is not None]


async def _build_spec(
    sa: SubAgentSettings,
    *,
    model_settings: ModelSettings,
    exit_stack: AsyncExitStack,
) -> dict[str, Any] | None:
    """Build a single subagent spec dict."""
    if not sa.name:
        _log.warning("Skipping subagent with empty name")
        return None
    if not sa.description:
        _log.warning("Skipping subagent '%s': missing description", sa.name)
        return None

    spec: dict[str, Any] = {
        "name": sa.name,
        "description": sa.description,
        "system_prompt": sa.system_prompt or sa.description,
    }

    name = sa.model or model_settings.name
    num_ctx = sa.context_window or model_settings.context_window
    spec["model"] = await create_ollama_chat_model(
        model=name,
        base_url=model_settings.base_url,
        api_key=None,
        context_window=num_ctx if (num_ctx is not None and num_ctx > 0) else None,
        reasoning_effort=validate_reasoning_effort(model_settings.reasoning_effort),
    )

    spec["skills"] = ["/skills/"]

    if sa.mcp_servers:
        tools = await load_subagent_mcp_tools(sa.name, sa.mcp_servers, exit_stack)
        if tools:
            spec["tools"] = tools

    return spec
