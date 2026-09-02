from __future__ import annotations

import asyncio
import sys
import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
from rich.console import Console

from .agent import AgentRuntime
from .agent.builtin_tools import set_tool_timeout
from .core import ModelCapabilityError, ModelContextWindowError
from .i18n import SUPPORTED_LOCALES, _, set_locale
from .interfaces.cli import create_argument_parser, handle_subcommand, run_prompt_session
from .interfaces.model_commands import ensure_model_configured
from .interfaces.repl import OllamaREPL
from .settings import load_settings, reset_config

# Silence only the known third-party noise, not all deprecations.
warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
    message=".*allowed_objects.*",
)


def _extract_early_language(argv: list[str]) -> str | None:
    """Extract language code from CLI arguments if present and supported."""
    for i, arg in enumerate(argv):
        if arg in ("-l", "--lang", "--language"):
            if i + 1 < len(argv) and argv[i + 1] in SUPPORTED_LOCALES:
                return argv[i + 1]
        for prefix in ("-l=", "--lang=", "--language="):
            if arg.startswith(prefix):
                val = arg[len(prefix) :]
                if val in SUPPORTED_LOCALES:
                    return val
    return None


def main() -> None:
    """Main entry point."""
    early_lang = _extract_early_language(sys.argv[1:])
    set_locale(early_lang)

    parser = create_argument_parser()
    args = parser.parse_args()

    if args.command and args.prompt:
        parser.error(_("--prompt cannot be used together with a subcommand."))

    if args.config_reset:
        console = Console()
        for msg in reset_config(args.config_reset):
            console.print(msg)
        return

    settings = load_settings()
    settings.setup_environment()

    if args.language:
        settings.runtime.language = args.language
        set_locale(args.language)
    elif settings.runtime.language:
        set_locale(settings.runtime.language)

    # Apply CLI overrides
    if args.model:
        settings.model.name = args.model
    if args.effort:
        settings.model.reasoning_effort = args.effort
    if args.num_ctx:
        settings.model.context_window = int(args.num_ctx) if args.num_ctx.isdigit() else args.num_ctx
    if args.builtin_tool_timeout is not None:
        settings.runtime.builtin_tool_timeout = args.builtin_tool_timeout
    if args.allow_traversal is not None:
        settings.runtime.allow_traversal = args.allow_traversal

    set_tool_timeout(settings.runtime.builtin_tool_timeout)

    try:
        if args.command:
            handle_subcommand(args, settings)
            return

        ensure_model_configured(settings)

        if args.prompt:
            run_prompt_session(args, settings)
            return

        runtime = AgentRuntime(
            settings=settings,
            yolo_mode=args.yolo,
            stealth_mode=args.stealth,
        )
        repl = OllamaREPL(
            runtime=runtime,
            rag_database=args.rag,
        )
        asyncio.run(repl.run())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (ModelCapabilityError, ModelContextWindowError) as exc:
        console = Console()
        console.print(f"[red]{_('Error: {exc}', exc=exc)}[/red]")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
