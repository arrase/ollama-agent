"""Main entry point of the application."""

from .settings import configini as config
from .agent_factory.factory import create_agent
from .main_agent.tools import set_builtin_tool_timeout
from .memory import Mem0InitializationError, bootstrap_memory_backend
from .tui.app import ChatInterface
from .cli import create_argument_parser, handle_cli_commands


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
