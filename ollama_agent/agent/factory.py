"""Agent factory for creating OllamaAgent instances."""

from dataclasses import replace
from typing import Optional

from ..settings import configini as config
from ..utils import ModelCapabilityError, validate_reasoning_effort
from .agent import OllamaAgent


def create_agent(
    model: Optional[str] = None, reasoning_effort: Optional[str] = None
) -> OllamaAgent:
    """Create OllamaAgent instance from config with optional overrides."""
    cfg = config.get_config()
    target_model = model or cfg.model
    effort = validate_reasoning_effort(reasoning_effort or cfg.reasoning_effort)

    try:
        mem0_settings = _resolve_mem0_settings(cfg.mem0, target_model)
        return OllamaAgent(
            model=target_model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            reasoning_effort=effort,
            database_path=cfg.database_path,
            mcp_config_path=cfg.mcp_config_path,
            mem0_settings=mem0_settings,
        )
    except ModelCapabilityError as exc:
        raise SystemExit(str(exc)) from exc


def _resolve_mem0_settings(
    current_settings: config.Mem0Settings, target_model: str
) -> config.Mem0Settings:
    """Ensure Mem0 uses the same LLM model as the agent if using defaults."""
    default_settings = config.Mem0Settings()

    # If the model is not set or is the default one, use the agent's model
    if (
        not current_settings.llm_model
        or current_settings.llm_model == default_settings.llm_model
    ):
        return replace(current_settings, llm_model=target_model)

    return current_settings
