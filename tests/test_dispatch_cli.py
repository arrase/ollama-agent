from __future__ import annotations

import argparse
import asyncio
import io
import unittest
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from conftest import make_repl_environment, recording_console
from rich.console import Console

from ollama_agent.core import (
    ModelCapabilityError,
    ModelContextWindowError,
    OllamaVersionError,
)
from ollama_agent.interfaces.cli import create_argument_parser, handle_subcommand
from ollama_agent.interfaces.commands.dispatch import (
    build_cli_handlers,
    build_repl_handlers,
)
from ollama_agent.main import _apply_cli_overrides, _extract_early_language, main
from ollama_agent.rag import RAGManager
from ollama_agent.rag.commands import RAGContext
from ollama_agent.settings import RAGSettings, Settings
from ollama_agent.skills.commands import SkillsContext
from ollama_agent.tasks.commands import TasksContext


def _build_cli_handlers(args: argparse.Namespace) -> dict[tuple[str, str], object]:
    return build_cli_handlers(
        args,
        task_ctx=TasksContext(console=recording_console()),
        rag_ctx=RAGContext(rag_manager=RAGManager(RAGSettings()), console=recording_console()),
        skills_ctx=SkillsContext(console=recording_console()),
        settings=Settings(),
    )


class TestDispatchAndCLI(unittest.TestCase):
    """Unit tests for CLI argument parsing and dispatch handlers."""

    def test_argument_parser_flags(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(
            ["-m", "llama3:8b", "-e", "high", "-c", "16384", "-y", "-s", "--prompt", "hello", "--rag", "docs_db"]
        )
        assert args.model == "llama3:8b"
        assert args.effort == "high"
        assert args.num_ctx == 16384
        assert args.yolo
        assert args.stealth
        assert args.prompt == "hello"
        assert args.rag == "docs_db"

    def test_argument_parser_num_ctx(self) -> None:
        parser = create_argument_parser()
        args1 = parser.parse_args(["--num-ctx", "32768"])
        assert args1.num_ctx == 32768
        args2 = parser.parse_args(["-c", "max"])
        assert args2.num_ctx == "max"

    def test_argument_parser_rejects_invalid_numeric_args(self) -> None:
        """Garbage for -c/-t must be an argparse error, not a later traceback."""
        parser = create_argument_parser()
        with patch("sys.stderr", new_callable=io.StringIO):
            for argv in (["-c", "invalid"], ["-c", "-5"], ["-c", "0"], ["-t", "0"], ["-t", "abc"]):
                with self.subTest(argv=argv), pytest.raises(SystemExit) as ctx:
                    parser.parse_args(argv)
                assert ctx.value.code == 2

    def test_argument_parser_runtime_flags(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["--builtin-tool-timeout", "45", "--allow-traversal"])
        assert args.builtin_tool_timeout == 45
        assert args.allow_traversal

    def test_argument_parser_config_reset(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["--config-reset", "system-prompt"])
        assert args.config_reset == "system-prompt"

    def test_task_run_yolo_flag_parsing(self) -> None:
        parser = create_argument_parser()
        args1 = parser.parse_args(["task", "run", "my-task", "-y"])
        assert args1.task_id == "my-task"
        assert args1.yolo

        args2 = parser.parse_args(["-y", "task", "run", "my-task"])
        assert args2.task_id == "my-task"
        assert args2.yolo

        args3 = parser.parse_args(["task", "run", "my-task"])
        assert args3.task_id == "my-task"
        assert not args3.yolo

    def test_task_run_vars_argument_parsing(self) -> None:
        parser = create_argument_parser()
        args1 = parser.parse_args(["task", "run", "my-task", "file=app.py", "strict=true"])
        assert args1.task_id == "my-task"
        assert args1.vars == ["file=app.py", "strict=true"]
        assert args1.flag_vars == []

        args2 = parser.parse_args(["task", "run", "my-task", "--var", "file=app.py", "--var", "mode=fast"])
        assert args2.task_id == "my-task"
        assert args2.vars == []
        assert args2.flag_vars == ["file=app.py", "mode=fast"]

        args3 = parser.parse_args(["task", "run", "my-task", "file=app.py", "--var", "mode=fast", "-y"])
        assert args3.task_id == "my-task"
        assert args3.vars == ["file=app.py"]
        assert args3.flag_vars == ["mode=fast"]
        assert args3.yolo

    def test_argument_parser_language_choices(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["-l", "es"])
        assert args.language == "es"
        with patch("sys.stderr", new_callable=io.StringIO), pytest.raises(SystemExit):
            parser.parse_args(["-l", "klingon"])

    def test_build_cli_handlers_registry(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["task", "list"])
        handlers = _build_cli_handlers(args)
        assert ("task", "list") in handlers
        assert ("task", "create") in handlers
        assert ("rag", "list") in handlers
        assert ("skill", "list") in handlers
        assert ("mcp", "list") in handlers
        assert ("agents", "list") in handlers

    def test_build_cli_handlers_requires_settings(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["task", "list"])
        task_ctx = TasksContext(console=recording_console())
        rag_ctx = RAGContext(rag_manager=RAGManager(RAGSettings()), console=recording_console())
        skills_ctx = SkillsContext(console=recording_console())
        with pytest.raises(TypeError):
            build_cli_handlers(
                args,
                task_ctx=task_ctx,
                rag_ctx=rag_ctx,
                skills_ctx=skills_ctx,
            )

    def test_argument_parser_agents_subcommand(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["agents", "list"])
        assert args.command == "agents"
        assert args.subcommand == "list"

    def test_build_repl_handlers_registry(self) -> None:
        handlers = build_repl_handlers(make_repl_environment())

        assert "/help" not in handlers
        assert "/exit" not in handlers
        assert "/quit" not in handlers
        assert "/clear" not in handlers
        assert "/new" not in handlers
        assert "/compact" not in handlers
        assert "/model" in handlers
        assert "/effort" in handlers
        assert "/context" in handlers
        assert "/ctx" not in handlers
        assert "/queue" in handlers
        assert "/yolo" in handlers
        assert "/stealth" in handlers
        assert "/session" in handlers
        assert "/task" in handlers
        assert "/skill" in handlers
        assert "/rag" in handlers
        assert "/mcp" in handlers
        assert "/agents" in handlers

    def test_build_repl_handlers_intercepted_subcommands_removed(self) -> None:
        out = io.StringIO()
        console = Console(file=out)
        handlers = build_repl_handlers(make_repl_environment(console=console))

        # Inline-intercepted branches are gone from the registry handlers.
        task_handler = handlers["/task"]
        skill_handler = handlers["/skill"]
        session_handler = handlers["/session"]

        # /task create and /task run now report unknown subcommand.
        task_handler(["create", "my-task"])
        assert "Unknown task subcommand 'create'" in out.getvalue()
        task_handler(["run", "my-task"])
        assert "Unknown task subcommand 'run'" in out.getvalue()

        # /skill create reports unknown subcommand.
        skill_handler(["create", "my-skill"])
        assert "Unknown skill subcommand 'create'" in out.getvalue()

        # /session new/resume/export report unknown subcommand.
        for sub in ("new", "resume", "switch", "export"):
            session_handler([sub, "x"])
        rendered = out.getvalue()
        for sub in ("new", "resume", "switch", "export"):
            assert f"Unknown session subcommand '{sub}'" in rendered

    def test_main_config_reset(self) -> None:
        # Patch set_locale so the system language does not leak into other tests.
        mock_console = MagicMock()
        with (
            patch("sys.argv", ["ollama-agent", "--config-reset", "config-file"]),
            patch("ollama_agent.main.set_locale", return_value="en"),
            patch("ollama_agent.main.reset_config", return_value=["Reset successful"]) as mock_reset,
            patch("ollama_agent.main.Console", return_value=mock_console),
        ):
            main()
            mock_reset.assert_called_once_with("config-file")
            mock_console.print.assert_called_once_with("Reset successful")

    def test_main_cli_commands_handled(self) -> None:
        with (
            patch("sys.argv", ["ollama-agent", "task", "list"]),
            patch("ollama_agent.main.set_locale", return_value="en"),
            patch("ollama_agent.main.load_settings", return_value=Settings()),
            patch("ollama_agent.main.handle_subcommand") as mock_handle,
        ):
            main()
            mock_handle.assert_called_once()

    def test_main_rejects_prompt_with_subcommand(self) -> None:
        with (
            patch("sys.argv", ["ollama-agent", "task", "list", "-p", "hello"]),
            patch("ollama_agent.main.set_locale", return_value="en"),
            patch("sys.stderr", new_callable=io.StringIO),
            pytest.raises(SystemExit) as cm,
        ):
            main()
        assert cm.value.code == 2

    def test_main_repl_flow(self) -> None:
        mock_settings = Settings()
        with (
            patch("sys.argv", ["ollama-agent", "-m", "qwen3:32b", "-e", "high", "--builtin-tool-timeout", "40"]),
            patch("ollama_agent.main.set_locale", return_value="en"),
            patch("ollama_agent.main.load_settings", return_value=mock_settings),
            patch("ollama_agent.main.check_ollama_version", return_value="0.34.3") as mock_check_version,
            patch("ollama_agent.main.ensure_model_configured", return_value="qwen3:32b") as mock_ensure,
            patch("ollama_agent.main.AgentRuntime") as mock_runtime_cls,
            patch("ollama_agent.main.OllamaREPL") as mock_repl_cls,
            patch("asyncio.run") as mock_asyncio_run,
        ):
            main()
            mock_check_version.assert_called_once_with(mock_settings.model.base_url)
            mock_ensure.assert_called_once_with(mock_settings, console=ANY)
            assert mock_settings.model.name == "qwen3:32b"
            assert mock_settings.model.reasoning_effort == "high"
            assert mock_settings.runtime.builtin_tool_timeout == 40
            mock_runtime_cls.assert_called_once()
            mock_repl_cls.assert_called_once()
            mock_asyncio_run.assert_called_once()

    def test_main_prompt_flow(self) -> None:
        mock_settings = Settings()
        with (
            patch("sys.argv", ["ollama-agent", "-p", "hello world"]),
            patch("ollama_agent.main.set_locale", return_value="en"),
            patch("ollama_agent.main.load_settings", return_value=mock_settings),
            patch("ollama_agent.main.check_ollama_version", return_value="0.34.3") as mock_check_version,
            patch("ollama_agent.main.ensure_model_configured", return_value="qwen3:32b") as mock_ensure,
            patch("ollama_agent.main.run_prompt_session") as mock_prompt_session,
        ):
            main()
            mock_check_version.assert_called_once_with(mock_settings.model.base_url)
            mock_ensure.assert_called_once_with(mock_settings, console=ANY)
            mock_prompt_session.assert_called_once()

    def test_main_model_capability_error_exits(self) -> None:
        with (
            patch("sys.argv", ["ollama-agent"]),
            patch("ollama_agent.main.set_locale", return_value="en"),
            patch("ollama_agent.main.Console"),
            patch("ollama_agent.main.load_settings", return_value=Settings()),
            patch("ollama_agent.main.check_ollama_version", return_value="0.34.3"),
            patch("ollama_agent.main.ensure_model_configured", side_effect=ModelCapabilityError("Model unsupported")),
        ):
            with pytest.raises(SystemExit) as cm:
                main()
            assert cm.value.code == 1

    def test_apply_cli_overrides_propagates_every_flag(self) -> None:
        """Every CLI flag must reach the settings object, not just be parsed."""
        parser = create_argument_parser()
        settings = Settings()
        _apply_cli_overrides(
            settings,
            parser.parse_args(
                [
                    "-m",
                    "llama3:8b",
                    "-e",
                    "high",
                    "-c",
                    "16384",
                    "-t",
                    "42",
                    "--allow-traversal",
                ]
            ),
        )
        assert settings.model.name == "llama3:8b"
        assert settings.model.reasoning_effort == "high"
        assert settings.model.context_window == 16384
        assert settings.runtime.builtin_tool_timeout == 42
        assert settings.runtime.allow_traversal

    def test_apply_cli_overrides_num_ctx_max_and_traversal_off(self) -> None:
        parser = create_argument_parser()
        settings = Settings()
        _apply_cli_overrides(settings, parser.parse_args(["-c", "max"]))
        assert settings.model.context_window == "max"

        settings2 = Settings()
        settings2.runtime.allow_traversal = True
        _apply_cli_overrides(settings2, parser.parse_args(["--no-allow-traversal"]))
        assert not settings2.runtime.allow_traversal

    def test_main_context_window_error_exits(self) -> None:
        with (
            patch("sys.argv", ["ollama-agent"]),
            patch("ollama_agent.main.set_locale", return_value="en"),
            patch("ollama_agent.main.Console"),
            patch("ollama_agent.main.load_settings", return_value=Settings()),
            patch("ollama_agent.main.check_ollama_version", return_value="0.34.3"),
            patch("ollama_agent.main.ensure_model_configured", return_value="qwen3:32b"),
            patch("ollama_agent.main.AgentRuntime", side_effect=ModelContextWindowError("Invalid context window")),
        ):
            with pytest.raises(SystemExit) as cm:
                main()
            assert cm.value.code == 1

    def test_main_ollama_version_too_low_exits(self) -> None:
        mock_console = MagicMock()
        err_msg = "Ollama version 0.33.0 is lower than required 0.34.3. Please update your Ollama installation."
        with (
            patch("sys.argv", ["ollama-agent"]),
            patch("ollama_agent.main.set_locale", return_value="en"),
            patch("ollama_agent.main.Console", return_value=mock_console),
            patch("ollama_agent.main.load_settings", return_value=Settings()),
            patch("ollama_agent.main.check_ollama_version", side_effect=OllamaVersionError(err_msg)),
            patch("ollama_agent.main.ensure_model_configured") as mock_ensure,
        ):
            with pytest.raises(SystemExit) as cm:
                main()
            assert cm.value.code == 1
            mock_ensure.assert_not_called()
            mock_console.print.assert_called_once()
            printed = mock_console.print.call_args[0][0]
            assert "0.33.0" in printed
            assert "0.34.3" in printed

    def test_main_ollama_connection_error_exits(self) -> None:
        mock_console = MagicMock()
        with (
            patch("sys.argv", ["ollama-agent"]),
            patch("ollama_agent.main.set_locale", return_value="en"),
            patch("ollama_agent.main.Console", return_value=mock_console),
            patch("ollama_agent.main.load_settings", return_value=Settings()),
            patch(
                "ollama_agent.main.check_ollama_version",
                side_effect=ModelCapabilityError(
                    "Could not connect to Ollama at 'http://localhost:11434': Connection refused"
                ),
            ),
            patch("ollama_agent.main.ensure_model_configured") as mock_ensure,
        ):
            with pytest.raises(SystemExit) as cm:
                main()
            assert cm.value.code == 1
            mock_ensure.assert_not_called()
            mock_console.print.assert_called_once()

    def test_cli_rag_add_file_and_directory(self) -> None:
        parser = create_argument_parser()
        args_file = parser.parse_args(["rag", "add", "docs_db", "file.md"])
        mock_load = MagicMock()
        mock_add_file = AsyncMock()
        mock_add_dir = AsyncMock()

        with (
            patch("ollama_agent.interfaces.commands.dispatch.load_rag_database", mock_load),
            patch("ollama_agent.interfaces.commands.dispatch.add_rag_file", mock_add_file),
            patch("ollama_agent.interfaces.commands.dispatch.add_rag_directory", mock_add_dir),
        ):
            handlers = _build_cli_handlers(args_file)
            asyncio.run(handlers[("rag", "add")]())
            mock_load.assert_called_once()
            mock_add_file.assert_called_once()
            mock_add_dir.assert_not_called()

        mock_load.reset_mock()
        mock_add_file.reset_mock()
        mock_add_dir.reset_mock()

        args_dir = parser.parse_args(["rag", "add", "docs_db", "./docs", "--dir"])
        with (
            patch("ollama_agent.interfaces.commands.dispatch.load_rag_database", mock_load),
            patch("ollama_agent.interfaces.commands.dispatch.add_rag_file", mock_add_file),
            patch("ollama_agent.interfaces.commands.dispatch.add_rag_directory", mock_add_dir),
        ):
            handlers = _build_cli_handlers(args_dir)
            asyncio.run(handlers[("rag", "add")]())
            mock_load.assert_called_once()
            mock_add_file.assert_not_called()
            mock_add_dir.assert_called_once()

    def test_cli_session_export(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["session", "export", "sess-123", "-o", "export.md"])

        with (
            patch("ollama_agent.interfaces.commands.dispatch.AgentRuntime") as mock_runtime_cls,
            patch("ollama_agent.interfaces.commands.dispatch.export_session", AsyncMock()) as mock_export,
        ):
            mock_runtime = MagicMock()
            mock_runtime.__aenter__ = AsyncMock(return_value=mock_runtime)
            mock_runtime.__aexit__ = AsyncMock(return_value=None)
            mock_runtime_cls.return_value = mock_runtime

            handlers = _build_cli_handlers(args)
            asyncio.run(handlers[("session", "export")]())
            mock_export.assert_called_once()

    def test_cli_task_run_dispatch(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["task", "run", "my-task", "file=app.py", "--var", "mode=fast", "-y"])

        with patch("ollama_agent.interfaces.commands.dispatch.run_task", AsyncMock()) as mock_run_task:
            handlers = _build_cli_handlers(args)
            asyncio.run(handlers[("task", "run")]())
            mock_run_task.assert_awaited_once()
            call_args = mock_run_task.call_args
            assert call_args.args[1] == "my-task"
            assert call_args.kwargs["variables"] == {"file": "app.py", "mode": "fast"}
            assert call_args.kwargs["yolo"]

    def test_handle_subcommand_task_run_validation_error(self) -> None:
        parser = create_argument_parser()
        args = parser.parse_args(["task", "run", "my-task", "invalid_var"])
        settings = Settings()
        with (
            patch("sys.stdout", new_callable=io.StringIO),
            patch("sys.stderr", new_callable=io.StringIO),
        ):
            with pytest.raises(SystemExit) as cm:
                handle_subcommand(args, settings)
            assert cm.value.code == 1

    def test_extract_early_language(self) -> None:
        assert _extract_early_language(["-l", "es"]) == "es"
        assert _extract_early_language(["--lang", "fr"]) == "fr"
        assert _extract_early_language(["--language=de"]) == "de"
        assert _extract_early_language(["-l=it"]) == "it"
        assert _extract_early_language(["--unknown", "val"]) is None
        assert _extract_early_language(["-l", "unsupported_lang"]) is None
        assert _extract_early_language([]) is None

    def test_main_help_localization(self) -> None:
        stdout = io.StringIO()
        with patch("sys.argv", ["ollama-agent", "-l", "es", "--help"]), patch("sys.stdout", stdout):
            with pytest.raises(SystemExit) as cm:
                main()
            assert cm.value.code == 0
        output = stdout.getvalue()
        assert "Agente de IA" in output


if __name__ == "__main__":
    unittest.main()
