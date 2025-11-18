"""Agent factory for creating OllamaAgent instances."""

import asyncio
from typing import Optional
from dataclasses import replace

from .settings import configini as config
from .agent import OllamaAgent
from .settings.mcp import initialize_mcp_servers
from .utils import ModelCapabilityError, validate_reasoning_effort


def create_agent(model: Optional[str] = None, reasoning_effort: Optional[str] = None) -> OllamaAgent:
    """Create OllamaAgent instance from config with optional overrides."""
    cfg = config.get_config()
    target_model = model or cfg.model

    effort = validate_reasoning_effort(
        reasoning_effort or cfg.reasoning_effort)

    try:
        # Ensure Mem0 uses the same LLM model as the agent when not explicitly set
        default_mem0 = config.Mem0Settings()
        mem0_settings = cfg.mem0
        if not getattr(mem0_settings, "llm_model", None) or mem0_settings.llm_model == getattr(default_mem0, "llm_model", None):
            mem0_settings = replace(mem0_settings, llm_model=target_model)

        mcp_servers = []
        if cfg.mcp_config_path:
            mcp_servers = asyncio.run(initialize_mcp_servers(
                cfg.mcp_config_path,
                default_model=target_model,
            ))

        return OllamaAgent(
            model=target_model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            reasoning_effort=effort,
            database_path=cfg.database_path,
            mcp_servers=mcp_servers,
            mem0_settings=mem0_settings,
        )
    except ModelCapabilityError as exc:
        raise SystemExit(str(exc)) from exc
