"""Agent factory for creating OllamaAgent instances."""

from dataclasses import replace

from ..core import ModelCapabilityError, validate_reasoning_effort
from ..memory import Mem0InitializationError, Mem0Settings
from ..settings import get_config
from .agent import OllamaAgent


def create_agent(model: str | None = None, reasoning_effort: str | None = None) -> OllamaAgent:
    """Create OllamaAgent from config with optional overrides."""
    cfg = get_config()
    target_model = model or cfg.model
    effort = validate_reasoning_effort(reasoning_effort or cfg.reasoning_effort)

    # Sync Mem0 LLM model with agent model if using defaults
    mem0 = cfg.mem0
    if not mem0.llm_model or mem0.llm_model == Mem0Settings().llm_model:
        mem0 = replace(mem0, llm_model=target_model)

    try:
        return OllamaAgent(
            model=target_model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            reasoning_effort=effort,
            database_path=cfg.database_path,
            mcp_config_path=cfg.mcp_config_path,
            mem0_settings=mem0,
        )
    except (ModelCapabilityError, Mem0InitializationError) as e:
        raise SystemExit(str(e)) from e
