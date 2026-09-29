from __future__ import annotations

import atexit
import io
import os
import shutil
import tempfile
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# Must happen BEFORE any `ollama_agent` import: settings/paths.py resolves APP_DIR from
# Path.home() at import time and binds it as a default argument, so redirecting HOME
# afterwards would be too late. Without this the suite creates directories in the
# developer's real ~/.ollama-agent and can reach real MCP servers over the network.
_TMP_HOME = tempfile.mkdtemp(prefix="ollama-agent-tests-")
os.environ["HOME"] = _TMP_HOME
os.environ["USERPROFILE"] = _TMP_HOME
atexit.register(shutil.rmtree, _TMP_HOME, True)

import pytest  # noqa: E402
import pydantic.root_model  # noqa: E402, F401
from rich.console import Console  # noqa: E402

from ollama_agent.interfaces.commands.dispatch import REPLEnvironment  # noqa: E402


def recording_console() -> Console:
    """A Console that records output so assertions can read it back."""
    return Console(file=io.StringIO(), record=True)


@pytest.fixture(autouse=True)
def _reset_global_state() -> Iterator[None]:
    """Restore the process-wide ContextVars and locale after every test.

    ``set_tool_timeout`` and the active locale are module-level state; without this
    a test that changes them silently affects every test that runs after it.
    """
    from ollama_agent.agent.builtin_tools import get_tool_timeout, set_tool_timeout
    from ollama_agent.i18n import get_locale, set_locale

    timeout, locale = get_tool_timeout(), get_locale()
    yield
    set_tool_timeout(timeout)
    set_locale(locale)


def make_mock_runtime(**overrides: Any) -> MagicMock:
    """A mocked AgentRuntime whose async surface is genuinely awaitable.

    ``MagicMock()`` auto-creates *sync* children, so a real ``await runtime.warmup()``
    would raise TypeError. Every coroutine the app awaits is pre-wired here so the
    TUI exercises its real control flow instead of silently skipping it.
    """
    runtime = MagicMock()
    runtime.warmup = AsyncMock()
    runtime.reload = AsyncMock()
    runtime.graph.aget_state = AsyncMock(return_value=MagicMock(interrupts=[]))
    runtime.get_thread_messages = AsyncMock(return_value=[])
    runtime.count_effective_tokens = AsyncMock(return_value=0)
    runtime.settings.model.name = "qwen2.5-coder:32b"
    runtime.settings.model.reasoning_effort = "high"
    runtime.settings.model.context_window = 16384
    runtime.settings.runtime.collapse_thinking = True
    runtime.effective_context_window = 16384
    runtime.thread_id = "session_001"
    runtime.last_context_tokens = 512
    runtime.yolo_mode = False
    runtime.stealth_mode = False
    runtime.auto_approved_tools = set()
    for name, value in overrides.items():
        setattr(runtime, name, value)
    return runtime


def make_mock_repl(**overrides: Any) -> MagicMock:
    """A mocked OllamaREPL wrapping a :func:`make_mock_runtime`."""
    repl = MagicMock()
    repl.runtime = make_mock_runtime()
    repl._rag_ctx = None
    repl._get_commands.return_value = {}
    for name, value in overrides.items():
        setattr(repl, name, value)
    return repl


def make_repl_environment(**overrides: Any) -> REPLEnvironment:
    """Build a fully-stubbed REPLEnvironment for dispatch tests.

    Every collaborator is mocked so tests can override just the one under test
    (``get_runtime``, ``switch_model``, ...) without repeating the whole wiring.
    """
    kwargs: dict[str, Any] = {
        "task_ctx": MagicMock(),
        "skills_ctx": MagicMock(),
        "get_rag_ctx": MagicMock(),
        "console": recording_console(),
        "current_model": lambda: "gemma4:26b",
        "base_url": lambda: "http://localhost:11434",
        "switch_model": AsyncMock(),
        "handle_yolo": lambda _: None,
        "handle_stealth": lambda _: None,
        "handle_queue": lambda _: None,
        "get_runtime": lambda: MagicMock(),
        "current_thread_id": lambda: "",
        "switch_effort": AsyncMock(),
        "switch_context_window": AsyncMock(),
    }
    kwargs.update(overrides)
    return REPLEnvironment(**kwargs)
