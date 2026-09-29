"""Regression tests for bugs found in the code-quality audit.

Each test here fails against the pre-fix code and guards a specific behaviour
that a plausible future refactor could silently reintroduce.
"""

from __future__ import annotations

import asyncio
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessageChunk
from rich.console import Console

from ollama_agent.agent.episodic_memory import HistoryError, format_iso_timestamp
from ollama_agent.agent.middleware import _stream_tool_events
from ollama_agent.core.common import extract_text, shorten
from ollama_agent.core.models import get_ollama_version, validate_reasoning_effort
from ollama_agent.core.prompt_processor import PromptProcessingError, process_prompt_mentions
from ollama_agent.i18n import _, get_locale, get_text, set_locale
from ollama_agent.interfaces.commands.dispatch import safe_call
from ollama_agent.interfaces.commands.params import set_model_param
from ollama_agent.interfaces.commands.sessions import export_session
from ollama_agent.mcp import MCPConfigError
from ollama_agent.mcp.loader import _build_stdio_connection, _read_main_config
from ollama_agent.rag.manager import RAGError, RAGManager
from ollama_agent.settings import RAGSettings
from ollama_agent.settings.config import Settings
from ollama_agent.streaming.console_renderer import ConsoleStreamingRenderer
from ollama_agent.streaming.parsers import ThinkTagParser
from ollama_agent.tasks.manager import Task, TaskInput

from conftest import make_repl_environment


def _console() -> Console:
    return Console(file=io.StringIO(), record=True)


class TestRichMarkupInjection(unittest.TestCase):
    """Model/tool/user text must never be parsed as Rich markup.

    ``Console.print`` raises ``MarkupError`` on an unmatched closing tag, so any
    unescaped text turns a prompt into a fatal crash.
    """

    def test_reasoning_delta_with_closing_tag_does_not_crash(self) -> None:
        console = _console()
        renderer = ConsoleStreamingRenderer(console=console)
        renderer.on_reasoning_delta({"type": "reasoning_delta", "content": "see [/italic] and foo [/]"})
        self.assertIn("[/italic]", console.export_text())

    def test_error_and_warning_content_does_not_crash(self) -> None:
        console = _console()
        renderer = ConsoleStreamingRenderer(console=console)
        renderer.on_error({"type": "error", "content": "boom [/red]"})
        renderer.on_warning({"type": "warning", "content": "careful [/bold]"})
        self.assertIn("boom", console.export_text())

    def test_tool_call_with_markup_in_name_does_not_crash(self) -> None:
        console = _console()
        renderer = ConsoleStreamingRenderer(console=console)
        renderer.on_tool_call({"type": "tool_call", "name": "weird[/name]", "agent_name": "a[/b]c"})
        self.assertIn("weird", console.export_text())

    def test_unknown_subcommand_with_markup_does_not_crash(self) -> None:
        env = make_repl_environment(console=_console())
        for handler_name in ("handle_agents", "handle_mcp", "handle_task", "handle_skill", "handle_rag"):
            with self.subTest(handler=handler_name):
                getattr(env, handler_name)(["x[/y]z"])
        self.assertTrue(env.console.file.getvalue())  # type: ignore[union-attr]

    def test_safe_call_escapes_domain_error(self) -> None:
        from ollama_agent.tasks.commands import TaskError

        console = _console()
        env = make_repl_environment(console=console)
        with patch("ollama_agent.interfaces.commands.dispatch.list_tasks", side_effect=TaskError("bad [/x]")):
            asyncio.run(safe_call(env.handle_task, ["list"], console=console))
        self.assertIn("bad", console.export_text())

    def test_shorten_collapses_and_truncates(self) -> None:
        self.assertEqual(shorten("a\nb"), "a b")
        self.assertEqual(shorten("x" * 100), "x" * 57 + "...")
        self.assertEqual(len(shorten("x" * 100)), 60)
        self.assertEqual(shorten("short"), "short")


class TestExtractTextMultimodal(unittest.TestCase):
    """An image/audio attachment must not break history reading."""

    def test_non_text_block_yields_empty_string(self) -> None:
        block = {"type": "image", "base64": "AAAA", "mime_type": "image/png"}
        self.assertEqual(extract_text(block), "")

    def test_mixed_blocks_keep_the_text(self) -> None:
        blocks = [
            {"type": "text", "text": "describe this"},
            {"type": "image", "base64": "AAAA", "mime_type": "image/png"},
        ]
        self.assertEqual(extract_text(blocks), "describe this")

    def test_malformed_dict_still_raises(self) -> None:
        with self.assertRaises(TypeError):
            extract_text({"unexpected": 1})


class TestHistoryErrorContract(unittest.TestCase):
    def test_malformed_timestamp_raises_history_error(self) -> None:
        with self.assertRaises(HistoryError):
            format_iso_timestamp("not-a-timestamp")

    def test_valid_timestamp_is_formatted(self) -> None:
        self.assertEqual(format_iso_timestamp("2026-08-20T10:00:00+00:00"), "2026-08-20 10:00 UTC")


class TestStreamingParser(unittest.TestCase):
    """A chunk carrying both thinking and content must yield both."""

    def test_thinking_and_content_in_one_chunk(self) -> None:
        parser = ThinkTagParser()
        chunk = AIMessageChunk(
            content="Hello world",
            additional_kwargs={"reasoning_content": "let me think"},
            response_metadata={},
        )
        events = parser.process_chunk(chunk)
        kinds = [e["type"] for e in events]
        self.assertIn("reasoning_delta", kinds)
        self.assertIn("text_delta", kinds)
        text = "".join(e["content"] for e in events if e["type"] == "text_delta")
        self.assertEqual(text, "Hello world")

    def test_hide_reasoning_keeps_text(self) -> None:
        parser = ThinkTagParser()
        chunk = AIMessageChunk(
            content="Visible",
            additional_kwargs={"reasoning_content": "hidden"},
            response_metadata={},
        )
        events = parser.process_chunk(chunk, hide_reasoning=True)
        self.assertTrue(all(e["type"] != "reasoning_delta" for e in events))
        self.assertEqual("".join(e["content"] for e in events), "Visible")

    def test_hidden_reasoning_does_not_desync_the_think_buffer(self) -> None:
        """With reasoning hidden, text inside a think block must stay hidden.

        Returning early when reasoning was present skipped ``feed()`` entirely, so the
        parser never entered the think state and the following text leaked out.
        """
        parser = ThinkTagParser()
        opener = AIMessageChunk(
            content="<think>",
            additional_kwargs={"reasoning_content": "secret"},
            response_metadata={},
        )
        parser.process_chunk(opener, hide_reasoning=True)
        self.assertTrue(parser.in_think)

        body = AIMessageChunk(content="hidden body", additional_kwargs={}, response_metadata={})
        events = parser.process_chunk(body, hide_reasoning=True)
        self.assertEqual(events, [])

        closer = AIMessageChunk(content="</think>visible", additional_kwargs={}, response_metadata={})
        events = parser.process_chunk(closer, hide_reasoning=True)
        self.assertEqual("".join(e["content"] for e in events), "visible")


class TestMiddlewareTimeout(unittest.IsolatedAsyncioTestCase):
    """A TimeoutError raised by the tool must not be reported as our deadline."""

    async def test_tool_timeout_error_propagates(self) -> None:
        class _Runtime:
            def stream_writer(self, event: dict[str, object]) -> None:
                pass

        request = MagicMock(
            tool_call={"name": "flaky", "args": {}, "id": "c1"},
            runtime=_Runtime(),
        )

        async def handler(_req: object) -> str:
            raise TimeoutError("socket timed out")

        with patch("ollama_agent.agent.middleware.get_tool_timeout", return_value=30):
            with self.assertRaises(TimeoutError):
                await _stream_tool_events(request, handler)

    async def test_real_deadline_is_reported_as_tool_message(self) -> None:
        class _Runtime:
            def stream_writer(self, event: dict[str, object]) -> None:
                pass

        request = MagicMock(
            tool_call={"name": "slow", "args": {}, "id": "c2"},
            runtime=_Runtime(),
        )

        async def handler(_req: object) -> str:
            await asyncio.sleep(1)
            return "never"

        with patch("ollama_agent.agent.middleware.get_tool_timeout", return_value=0.01):
            result = await _stream_tool_events(request, handler)
        self.assertEqual(result.status, "error")
        self.assertIn("timed out", result.content)


class TestOllamaClientLifetime(unittest.TestCase):
    """ollama.AsyncClient defaults to no timeout and must be closed."""

    def test_async_client_has_bounded_timeout(self) -> None:
        from ollama_agent.core.models import create_ollama_async_client

        client = create_ollama_async_client("http://localhost:11434/")
        try:
            self.assertIsNotNone(client._client.timeout.read)
            self.assertGreater(client._client.timeout.read, 0)
        finally:
            asyncio.run(client.close())

    def test_rag_manager_defers_embedder_client(self) -> None:
        with temp_dir() as tmp:
            mgr = RAGManager(RAGSettings(rag_dir=str(tmp)))
            # A manager that only lists databases must not hold a connection pool.
            self.assertIsNone(mgr._ollama_client)

    def test_get_ollama_version_rejects_body_without_version(self) -> None:
        from ollama_agent.core.models import ModelCapabilityError

        response = MagicMock()
        response.json.return_value = {"unexpected": "field"}
        with patch("httpx.get", return_value=response):
            with self.assertRaises(ModelCapabilityError):
                get_ollama_version("http://localhost:11434")


class TestValidationGaps(unittest.TestCase):
    def test_settings_reject_wrong_types(self) -> None:
        # A bool would otherwise reach Ollama as num_ctx=1.
        with self.assertRaises(ValueError):
            Settings.from_dict({"model": {"context_window": True}})
        with self.assertRaises(ValueError):
            Settings.from_dict({"model": {"context_window": 3.5}})
        with self.assertRaises(ValueError):
            Settings.from_dict({"model": {"temperature": "hot"}})
        with self.assertRaises(ValueError):
            Settings.from_dict({"runtime": {"allow_traversal": "yes"}})

    def test_settings_accept_valid_values(self) -> None:
        s = Settings.from_dict(
            {
                "model": {"name": "m", "temperature": 0.7, "context_window": "max"},
                "runtime": {"allow_traversal": False},
            }
        )
        self.assertEqual(s.model.context_window, "max")
        self.assertEqual(s.model.temperature, 0.7)

    def test_reasoning_effort_rejects_arbitrary_text(self) -> None:
        for bad in ["a" * 40, "with space", "bad/slash", "quote'"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_reasoning_effort(bad)
        self.assertEqual(validate_reasoning_effort("high"), "high")
        self.assertEqual(validate_reasoning_effort(True), "true")

    def test_mcp_config_rejects_non_object_server(self) -> None:
        with temp_dir() as tmp:
            cfg = Path(tmp) / "mcp.json"
            cfg.write_text(json.dumps({"mcpServers": {"bad": 5}}), encoding="utf-8")
            with patch("ollama_agent.mcp.loader.MCP_PATH", cfg):
                with self.assertRaises(MCPConfigError):
                    asyncio.run(_read_main_config())

    def test_mcp_args_must_be_strings(self) -> None:
        with self.assertRaises(MCPConfigError):
            _build_stdio_connection("s", {"command": "echo", "args": [1, 2]})
        self.assertEqual(_build_stdio_connection("s", {"command": "echo"})["args"], [])

    def test_mention_expanduser_error_is_a_domain_error(self) -> None:
        with self.assertRaises(PromptProcessingError):
            process_prompt_mentions(
                "@~nosuchuser/notes.md",
                allow_traversal=True,
            )


class TestSamplingParamBounds(unittest.IsolatedAsyncioTestCase):
    async def test_out_of_range_param_is_rejected(self) -> None:
        console = _console()
        runtime = MagicMock()
        runtime.settings.model.temperature = 0.7
        runtime.reload = AsyncMock()
        with patch("ollama_agent.interfaces.commands.params.save_settings"):
            await set_model_param(console, "temperature", "99", runtime=runtime)
            await set_model_param(console, "top_k", "-1", runtime=runtime)
        self.assertEqual(runtime.settings.model.temperature, 0.7)
        runtime.reload.assert_not_awaited()
        self.assertIn("between", console.export_text())


class TestI18nRobustness(unittest.TestCase):
    def tearDown(self) -> None:
        set_locale("en")

    def test_failed_locale_load_does_not_corrupt_state(self) -> None:
        set_locale("en")
        with patch("ollama_agent.i18n.resources") as res:
            res.files.return_value.joinpath.return_value.read_text.side_effect = OSError("boom")
            with self.assertRaises(ValueError):
                set_locale("es")
        self.assertEqual(get_locale(), "en")
        self.assertEqual(_("Usage: /model list"), "Usage: /model list")

    def test_literal_brace_does_not_crash(self) -> None:
        self.assertEqual(get_text("Expected JSON: {}"), "Expected JSON: {}")
        self.assertEqual(get_text("a {missing} b", other=1), "a {missing} b")

    def test_empty_locale_string_rejected(self) -> None:
        with self.assertRaises(ValueError):
            set_locale("   ")


class TestTaskValidation(unittest.TestCase):
    def test_missing_keys_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Task.from_dict({"title": "t", "model": "m"})

    def test_unknown_input_keys_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Task.from_dict(
                {
                    "title": "t",
                    "prompt": "p",
                    "model": "m",
                    "inputs": {"x": {"type": "string", "bogus": 1}},
                }
            )

    def test_optional_absent_input_renders_empty(self) -> None:
        task = Task(
            title="t",
            prompt="val={{ val }}",
            model="m",
            inputs={"val": TaskInput(type="string")},
        )
        self.assertEqual(task.render(), "val=")

    def test_numeric_default_is_coerced(self) -> None:
        task = Task(
            title="t",
            prompt="n={{ n }}",
            model="m",
            inputs={"n": TaskInput(type="number", default="5")},
        )
        self.assertEqual(task.render(), "n=5")


class TestExportSessionErrors(unittest.IsolatedAsyncioTestCase):
    async def test_uncreatable_parent_is_reported_not_raised(self) -> None:
        console = _console()
        session_id = "abc12345"
        runtime = MagicMock()
        runtime.get_thread_messages = AsyncMock(return_value=[MagicMock(type="human", content="hi")])
        # A path under an existing *file* cannot have its parent created.
        with temp_dir() as tmp:
            blocker = Path(tmp) / "blocker"
            blocker.write_text("x", encoding="utf-8")
            target = str(blocker / "sub" / "out.md")
            with patch("ollama_agent.interfaces.commands.sessions.get_available_sessions") as sessions:
                sessions.return_value = [{"thread_id": session_id, "steps": 1, "timestamp": ""}]
                result = await export_session(console, runtime, session_id, output_path=target)
        self.assertIsNone(result)
        self.assertIn("Failed to export", console.export_text())


class TestRAGIsolation(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_payload_raises_rag_error(self) -> None:
        with temp_dir() as tmp:
            mgr = RAGManager(RAGSettings(rag_dir=str(tmp)))
            mgr._current_db = "db"
            client = MagicMock()
            hit = MagicMock()
            hit.payload = {"content": "c"}  # missing source/filename/chunk_index
            hit.id = 1
            hit.score = 0.5
            client.query_points.return_value = MagicMock(points=[hit])
            mgr._client = client
            with patch.object(RAGManager, "_get_embedding", AsyncMock(return_value=[0.0])):
                with self.assertRaises(RAGError):
                    await mgr.search("q")


class TestSkillManagerIsolation(unittest.TestCase):
    def test_broken_skill_does_not_break_listing(self) -> None:
        from ollama_agent.skills.manager import SkillManager

        with temp_dir() as tmp:
            base = Path(tmp) / "skills"
            (base / "broken").mkdir(parents=True)
            (base / "broken" / "SKILL.md").write_text("no frontmatter here", encoding="utf-8")
            (base / "good").mkdir()
            (base / "good" / "SKILL.md").write_text(
                "---\nname: Good\ndescription: d\n---\nbody\n", encoding="utf-8"
            )
            mgr = SkillManager(skills_dir=base, builtin_skills_dir=None)
            names = [sid for sid, _ in mgr.list_all()]
            self.assertIn("good", names)
            self.assertNotIn("broken", names)


# --- small local helpers -----------------------------------------------------


class temp_dir:
    """Context manager yielding a fresh temporary directory path."""

    def __init__(self) -> None:
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="regression-"))
        return self.path

    def __exit__(self, *exc: object) -> None:
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
