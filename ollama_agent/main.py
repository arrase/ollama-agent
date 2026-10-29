from __future__ import annotations

import argparse
import asyncio
import sys
import warnings

import yaml
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
from rich.console import Console
from rich.markup import escape

from .agent import AgentRuntime
from .agent.builtin_tools import set_tool_timeout
from .core import (
    ModelCapabilityError,
    ModelContextWindowError,
    OllamaVersionError,
    check_ollama_version,
    validate_reasoning_effort,
)
from .i18n import SUPPORTED_LOCALES, _, set_locale
from .interfaces.cli import create_argument_parser, handle_subcommand, run_prompt_session
from .interfaces.commands.dispatch import DOMAIN_ERRORS
from .interfaces.commands.models import ensure_model_configured
from .interfaces.tui import OllamaREPL
from .settings import Settings, load_settings, reset_config

# Silence only the known third-party noise, not all deprecations.
warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
    message=".*allowed_objects.*",
)


def _extract_early_language(argv: list[str]) -> str | None:
    """Extract language code from CLI arguments if present and supported."""
    flags = ("-l", "--lang", "--language")
    for i, arg in enumerate(argv):
        flag, eq, val = arg.partition("=")
        if eq:
            if flag in flags and val in SUPPORTED_LOCALES:
                return val
        elif arg in flags and i + 1 < len(argv) and argv[i + 1] in SUPPORTED_LOCALES:
            return argv[i + 1]
    return None


def _apply_cli_overrides(settings: Settings, args: argparse.Namespace) -> None:
    """Apply CLI arguments to settings."""
    if args.model:
        settings.model.name = args.model
    if args.effort:
        settings.model.reasoning_effort = validate_reasoning_effort(args.effort)
    if args.num_ctx is not None:
        settings.model.context_window = args.num_ctx
    if args.builtin_tool_timeout is not None:
        settings.runtime.builtin_tool_timeout = args.builtin_tool_timeout
    if args.allow_traversal is not None:
        settings.runtime.allow_traversal = args.allow_traversal


def _load_and_configure_settings(args: argparse.Namespace, console: Console) -> Settings:
    """Load settings from disk and apply CLI/locale overrides."""
    try:
        settings = load_settings()
        settings.setup_environment()

        if args.language:
            settings.runtime.language = args.language
            set_locale(args.language)
        elif settings.runtime.language:
            set_locale(settings.runtime.language)

        _apply_cli_overrides(settings, args)
        settings.model.reasoning_effort = validate_reasoning_effort(settings.model.reasoning_effort)
        set_tool_timeout(settings.runtime.builtin_tool_timeout)
    except (ValueError, yaml.YAMLError) as exc:
        # Invalid locale, settings.yaml, or reasoning effort coming from settings.yaml / CLI.
        console.print(f"[red]{escape(_('Error: {exc}', exc=exc))}[/red]")
        raise SystemExit(1) from None
    return settings


def main() -> None:
    """Main entry point."""
    console = Console()
    try:
        early_lang = _extract_early_language(sys.argv[1:])
        set_locale(early_lang)

        parser = create_argument_parser()
        args = parser.parse_args()

        if args.command and args.prompt:
            parser.error(_("--prompt cannot be used together with a subcommand."))

        if args.config_reset:
            for msg in reset_config(args.config_reset):
                console.print(msg)
            return

        settings = _load_and_configure_settings(args, console)

        if args.command:
            handle_subcommand(args, settings)
            return

        check_ollama_version(settings.model.base_url)
        ensure_model_configured(settings, console=console)

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
    except (ModelCapabilityError, ModelContextWindowError, OllamaVersionError) as exc:
        console.print(f"[red]{escape(_('Error: {exc}', exc=exc))}[/red]")
        raise SystemExit(1) from None
    except SystemExit:
        raise
    except DOMAIN_ERRORS as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
