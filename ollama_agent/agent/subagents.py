"""Build subagent specifications for create_deep_agent."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from ..core import create_ollama_chat_model, render_prompt_template
from ..i18n import _
from ..mcp.loader import load_subagent_mcp_tools
from ..settings import (
    ModelSettings,
    SubAgentSettings,
)
from .builtin_tools import BUILTIN_TOOLS
from .environment import SKILL_ROOTS, environment_block
from .middleware import stream_tool_events_mw


def _default_should_interrupt(_request: Any) -> bool:
    return True


def build_interrupt_config(
    mcp_tools: Sequence[BaseTool],
    should_interrupt_tool: Callable[[Any], bool],
) -> dict[str, Any]:
    """Build the ``interrupt_on`` mapping for sensitive built-in and MCP tools."""
    return {
        tool_name: {
            "allowed_decisions": ["approve", "reject"],
            "when": should_interrupt_tool,
        }
        for tool_name in ("execute", "write_file", "edit_file", *(t.name for t in mcp_tools))
    }


async def build_subagents(
    subagent_settings: list[SubAgentSettings],
    *,
    model_settings: ModelSettings,
    should_interrupt_tool: Callable[[Any], bool] = _default_should_interrupt,
) -> list[dict[str, Any]]:
    """Convert ``SubAgentSettings`` into dicts for ``create_deep_agent(subagents=...)``."""
    return [
        await _build_spec(sa, model_settings=model_settings, should_interrupt_tool=should_interrupt_tool)
        for sa in subagent_settings
    ]


async def _build_spec(
    sa: SubAgentSettings,
    *,
    model_settings: ModelSettings,
    should_interrupt_tool: Callable[[Any], bool] = _default_should_interrupt,
) -> dict[str, Any]:
    """Build a single subagent spec dict."""
    if not sa.name:
        raise ValueError(_("Subagent configuration error: name cannot be empty"))
    if not sa.description:
        raise ValueError(_("Subagent '{name}' configuration error: description cannot be empty", name=sa.name))
    if not sa.system_prompt:
        raise ValueError(_("Subagent '{name}' configuration error: system_prompt cannot be empty", name=sa.name))

    rendered_prompt = render_prompt_template(
        sa.system_prompt,
        {"subagent": sa, "model_settings": model_settings},
    )

    spec: dict[str, Any] = {
        "name": sa.name,
        "description": sa.description,
        "system_prompt": rendered_prompt + environment_block(include_cwd=False),
        "middleware": [stream_tool_events_mw],
    }

    spec["model"] = await create_ollama_chat_model(
        model=sa.model or model_settings.name,
        base_url=model_settings.base_url,
        context_window=sa.context_window or model_settings.context_window,
        reasoning_effort=model_settings.reasoning_effort,
        temperature=model_settings.temperature,
        top_p=model_settings.top_p,
        top_k=model_settings.top_k,
        min_p=model_settings.min_p,
        presence_penalty=model_settings.presence_penalty,
        repeat_penalty=model_settings.repeat_penalty,
    )

    spec["skills"] = SKILL_ROOTS

    if sa.mcp_servers:
        tools = await load_subagent_mcp_tools(sa.name, sa.mcp_servers)
        if tools:
            spec["tools"] = [*BUILTIN_TOOLS, *tools]
            spec["interrupt_on"] = build_interrupt_config(tools, should_interrupt_tool)

    return spec
