"""Main entry point of the application."""

from typing import Optional

from .settings import configini as config
from .agent import OllamaAgent
from .agent.tools import set_builtin_tool_timeout
from .tui.app import ChatInterface
from .cli import create_argument_parser, handle_cli_commands


def create_agent(model: Optional[str] = None, reasoning_effort: Optional[str] = None) -> OllamaAgent:
    """Create OllamaAgent instance from config with optional overrides."""
    cfg = config.get_config()

    return OllamaAgent(
        model=model or cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        reasoning_effort=reasoning_effort or cfg.reasoning_effort,
        database_path=cfg.database_path,
        mcp_config_path=cfg.mcp_config_path
    )


def main() -> None:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure built-in tool timeout from args or config
    cfg = config.get_config()
    builtin_tool_timeout = args.builtin_tool_timeout if args.builtin_tool_timeout is not None else cfg.builtin_tool_timeout
    set_builtin_tool_timeout(builtin_tool_timeout)

    if not handle_cli_commands(args, create_agent):
        # If no CLI command was handled, start the TUI
        agent = create_agent(model=args.model, reasoning_effort=args.effort)
        ChatInterface(agent, builtin_tool_timeout=builtin_tool_timeout).run()


if __name__ == "__main__":
    main()
