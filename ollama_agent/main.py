"""Main entry point of the application."""

from typing import Optional
from dataclasses import replace

from .settings import configini as config
from .agent import OllamaAgent
from .agent.tools import set_builtin_tool_timeout
from .memory import Mem0InitializationError, bootstrap_memory_backend
from .tui.app import ChatInterface
from .cli import create_argument_parser, handle_cli_commands
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


def main() -> None:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure built-in tool timeout from args or config
    cfg = config.get_config()
    try:
        bootstrap_memory_backend(cfg.mem0)
    except Mem0InitializationError as exc:
        raise SystemExit(str(exc)) from exc

    builtin_tool_timeout = args.builtin_tool_timeout if args.builtin_tool_timeout is not None else cfg.builtin_tool_timeout
    set_builtin_tool_timeout(builtin_tool_timeout)

    if not handle_cli_commands(args, create_agent):
        # If no CLI command was handled, start the TUI
        agent = create_agent(model=args.model, reasoning_effort=args.effort)
        ChatInterface(agent, builtin_tool_timeout=builtin_tool_timeout).run()


if __name__ == "__main__":
    main()
