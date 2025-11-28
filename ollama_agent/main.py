"""Main entry point of the application."""

from .agent import create_agent, set_tool_timeout
from .cli import create_argument_parser, handle_cli_commands
from .settings import get_config
from .tui.app import ChatInterface


def main() -> None:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    cfg = get_config()
    timeout = args.builtin_tool_timeout if args.builtin_tool_timeout is not None else cfg.builtin_tool_timeout
    set_tool_timeout(timeout)

    if not handle_cli_commands(args, create_agent):
        agent = create_agent(model=args.model, reasoning_effort=args.effort)
        ChatInterface(agent, tool_timeout=timeout).run()


if __name__ == "__main__":
    main()
