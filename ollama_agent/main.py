import asyncio
from .agent import create_agent, set_tool_timeout
from .interfaces.cli import create_argument_parser, handle_cli_commands
from .settings import get_config
from .interfaces.repl import OllamaREPL


def main() -> None:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    cfg = get_config()
    timeout = args.builtin_tool_timeout if args.builtin_tool_timeout is not None else cfg.builtin_tool_timeout
    set_tool_timeout(timeout)

    if handle_cli_commands(args, create_agent):
        return

    # Resolve model/effort via REPL logic or pass default None to let REPL/agent handle it
    repl = OllamaREPL(
        agent_factory=create_agent,
        model=args.model or cfg.model,
        effort=args.effort or cfg.reasoning_effort
    )
    asyncio.run(repl.run())


if __name__ == "__main__":
    main()

