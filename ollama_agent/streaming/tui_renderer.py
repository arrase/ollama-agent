"""TUI streaming renderer for Textual applications."""

from __future__ import annotations

from typing import Any

from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import RichLog

from .base import BufferedTokenRenderer, StreamingRenderer


def _make_reasoning_text(content: str) -> Text:
    """Create styled reasoning text."""
    text = Text()
    text.append("🧠 Thinking: ", style="bold magenta")
    text.append(content, style="dim italic magenta")
    return text


class TUIStreamingRenderer(StreamingRenderer):
    """Renderer for streaming to the TUI with markdown and reasoning support."""

    def __init__(self, chat_log: RichLog, update_frequency: int = 5):
        self.chat_log = chat_log
        self._text = BufferedTokenRenderer(chat_log, Markdown, update_frequency)
        self._reasoning = BufferedTokenRenderer(
            chat_log, _make_reasoning_text, update_frequency
        )

    def on_text_delta(self, event: dict[str, Any]) -> None:
        self._end_reasoning()
        if not self._text.is_active:
            self._text.start(Text("Agent:", style="bold green"))
        self._text.append(event.get("content", ""))

    def on_reasoning_delta(self, event: dict[str, Any]) -> None:
        if not self._reasoning.is_active:
            self._reasoning.start()
        self._reasoning.append(event.get("content", ""))

    def on_reasoning_summary(self, event: dict[str, Any]) -> None:
        if self._reasoning.is_active:
            return
        if preview := event.get("content", "")[:100]:
            self.chat_log.write(Text(f"💭 Reasoning: {preview}...", style="dim italic magenta"))

    def on_tool_call(self, event: dict[str, Any]) -> None:
        self._end_reasoning()
        self.chat_log.write(
            Text(f"🔧 Calling tool: {event.get('name', 'unknown')}", style="bold yellow")
        )

    def on_tool_output(self, event: dict[str, Any]) -> None:
        output = event.get("output", "")
        preview = f"{output[:100]}..." if len(output) > 100 else output
        self.chat_log.write(Text(f"📤 Tool output: {preview}", style="cyan"))

    def on_error(self, event: dict[str, Any]) -> None:
        self.chat_log.write(
            Text(f"Error: {event.get('content', 'Unknown error')}", style="bold red")
        )

    def close(self) -> None:
        self._end_reasoning()
        self._text.finalize()
        self.chat_log.write("")
        self.chat_log.scroll_end(animate=False)

    def _end_reasoning(self) -> None:
        """Finalize reasoning if active."""
        if self._reasoning.is_active:
            self._reasoning.finalize()
            self.chat_log.write("")


# Compatibility aliases
StreamingMarkdownRenderer = BufferedTokenRenderer
ReasoningRenderer = BufferedTokenRenderer
