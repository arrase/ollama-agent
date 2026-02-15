import asyncio
from functools import partial

from .agent import create_agent, set_tool_timeout
from .interfaces.cli import create_argument_parser, handle_cli_commands
from .settings import get_config, reset_config
from .interfaces.repl import OllamaREPL


def main() -> None:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.config_reset:
        reset_config(args.config_reset)
        return

    cfg = get_config()
    set_tool_timeout(
        cfg.builtin_tool_timeout if args.builtin_tool_timeout is None else args.builtin_tool_timeout)

    extra_skills: tuple[str, ...] = tuple(getattr(args, "skills_dir", None) or [])
    agent_factory = partial(create_agent, extra_skills_dirs=extra_skills)

    if handle_cli_commands(args, agent_factory):
        return

    repl = OllamaREPL(agent_factory=agent_factory, model=args.model or cfg.model,
                      effort=args.effort or cfg.reasoning_effort,
                      rag_database=getattr(args, 'rag', None))
    asyncio.run(repl.run())


if __name__ == "__main__":
    main()
