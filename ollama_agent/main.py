import asyncio
import warnings

from rich.console import Console

from .agent import AgentRuntime
from .agent.builtin_tools import set_tool_timeout
from .core import ModelCapabilityError, ModelContextWindowError
from .interfaces.cli import create_argument_parser, handle_cli_commands
from .interfaces.repl import OllamaREPL
from .settings import load_settings, reset_config

try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

    warnings.filterwarnings(
        "ignore",
        category=LangChainPendingDeprecationWarning,
        message=".*allowed_objects.*",
    )
except ImportError:
    pass


def main() -> None:
    """Main entry point."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = create_argument_parser()
    args = parser.parse_args()

    if args.config_reset:
        reset_config(args.config_reset)
        return

    settings = load_settings()
    settings.setup_environment()

    # Apply CLI overrides
    if args.model:
        settings.model.name = args.model
    if args.effort:
        settings.model.reasoning_effort = args.effort
    if args.builtin_tool_timeout is not None:
        settings.runtime.builtin_tool_timeout = args.builtin_tool_timeout

    set_tool_timeout(settings.runtime.builtin_tool_timeout)

    try:
        if handle_cli_commands(args, settings):
            return

        runtime = AgentRuntime(settings=settings)

        repl = OllamaREPL(
            runtime=runtime,
            rag_database=getattr(args, "rag", None),
        )
        try:
            asyncio.run(repl.run())
        except KeyboardInterrupt:
            pass
    except (ModelCapabilityError, ModelContextWindowError) as exc:
        console = Console()
        console.print(f"[red]Error: {exc}[/red]")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
