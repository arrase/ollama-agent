"""Build subagent specifications for create_deep_agent."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..core import create_ollama_chat_model, validate_reasoning_effort
from ..i18n import _
from ..mcp import load_subagent_mcp_tools
from ..settings import ModelSettings, SubAgentSettings
from .environment import SKILL_ROOTS, environment_block

_log = logging.getLogger(__name__)


async def build_subagents(
    subagent_settings: list[SubAgentSettings],
    *,
    model_settings: ModelSettings,
) -> list[dict[str, Any]]:
    """Convert ``SubAgentSettings`` into dicts for ``create_deep_agent(subagents=...)``."""
    tasks = [
        _build_spec(sa, model_settings=model_settings)
        for sa in subagent_settings
    ]
    return await asyncio.gather(*tasks)


async def _build_spec(
    sa: SubAgentSettings,
    *,
    model_settings: ModelSettings,
) -> dict[str, Any]:
    """Build a single subagent spec dict."""
    if not sa.name:
        raise ValueError(_("Subagent configuration error: name cannot be empty"))
    if not sa.description:
        raise ValueError(_("Subagent '{name}' configuration error: description cannot be empty", name=sa.name))
    if not sa.system_prompt:
        raise ValueError(
            _("Subagent '{name}' configuration error: system_prompt cannot be empty", name=sa.name)
        )

    spec: dict[str, Any] = {
        "name": sa.name,
        "description": sa.description,
        "system_prompt": sa.system_prompt + environment_block(include_cwd=False),
    }

    name = sa.model or model_settings.name
    num_ctx = sa.context_window or model_settings.context_window
    spec["model"] = await create_ollama_chat_model(
        model=name,
        base_url=model_settings.base_url,
        context_window=num_ctx,
        reasoning_effort=validate_reasoning_effort(model_settings.reasoning_effort),
        temperature=model_settings.temperature,
        top_p=model_settings.top_p,
        top_k=model_settings.top_k,
        min_p=model_settings.min_p,
        presence_penalty=model_settings.presence_penalty,
        repeat_penalty=model_settings.repeat_penalty,
        warn_callback=_log.warning,
    )

    spec["skills"] = SKILL_ROOTS

    if sa.mcp_servers:
        tools = await load_subagent_mcp_tools(sa.name, sa.mcp_servers)
        if tools:
            spec["tools"] = tools

    return spec
