"""Streaming renderers for different output formats."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown


class StreamingRenderer:
    """Base class for streaming renderers."""

    def on_event(self, event: dict[str, Any]) -> None:
        """Handle a streaming event."""
        handler = getattr(self, f"on_{event['type']}", None)
        if handler:
            handler(event)

    def on_error(self, event: dict[str, Any]) -> None:
        """Handle an error event."""
        print(f"Error: {event.get('content', 'Unknown error')}")

    def close(self) -> None:
        """Close the renderer."""


from ..tui.renderers import ReasoningRenderer, StreamingMarkdownRenderer
from rich.text import Text
from textual.widgets import RichLog


class TUIStreamingRenderer(StreamingRenderer):
    """Renderer for streaming to the TUI."""

    def __init__(self, chat_log: RichLog):
        self.chat_log = chat_log
        self.text_renderer = StreamingMarkdownRenderer(self.chat_log)
        self.reasoning_renderer = ReasoningRenderer(self.chat_log)

    def on_text_delta(self, event: dict[str, Any]) -> None:
        if self.reasoning_renderer.is_active:
            self.reasoning_renderer.finalize_reasoning()
        self.text_renderer.append_token(event.get("content", ""))

    def on_reasoning_delta(self, event: dict[str, Any]) -> None:
        self.reasoning_renderer.start_reasoning()
        self.reasoning_renderer.append_reasoning_token(event.get("content", ""))

    def on_reasoning_summary(self, event: dict[str, Any]) -> None:
        if self.reasoning_renderer.is_active:
            return
        preview = event.get("content", "")[:100]
        if preview:
            self.chat_log.write(Text(f"💭 Reasoning: {preview}...",
                                     style="dim italic magenta"))

    def on_tool_call(self, event: dict[str, Any]) -> None:
        if self.reasoning_renderer.is_active:
            self.reasoning_renderer.finalize_reasoning()
        tool_name = event.get("name", "unknown tool")
        self.chat_log.write(Text(f"🔧 Calling tool: {tool_name}",
                                 style="bold yellow"))

    def on_tool_output(self, event: dict[str, Any]) -> None:
        output = event.get("output", "")
        preview = f"{output[:100]}..." if len(output) > 100 else output
        self.chat_log.write(Text(f"📤 Tool output: {preview}",
                                 style="cyan"))

    def on_error(self, event: dict[str, Any]) -> None:
        self.chat_log.write(Text(f"Error: {event.get('content', 'Unknown error')}",
                                 style="bold red"))

    def close(self) -> None:
        if self.reasoning_renderer.is_active:
            self.reasoning_renderer.finalize_reasoning()
        self.text_renderer.finalize()
        self.chat_log.write("")
        self.chat_log.scroll_end(animate=False)


class ConsoleStreamingRenderer(StreamingRenderer):
    """Renderer for streaming to the console."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.live = Live(console=console, refresh_per_second=10)
        self._text: list[str] = []
        self._agent_banner_shown = False
        self._reasoning = False
        self._live_active = False

    def on_text_delta(self, event: dict[str, Any]) -> None:
        self._conclude_reasoning()
        self._ensure_agent_banner()
        self._start_live()
        self._text.append(event.get("content", ""))
        self.live.update(Markdown("".join(self._text)))

    def on_reasoning_delta(self, event: dict[str, Any]) -> None:
        if not self._reasoning:
            self._stop_live()
            self.console.print("\n[bold magenta]🧠 Thinking:[/bold magenta] ", end="")
            self._reasoning = True
        self.console.print(event.get("content", ""), end="", style="dim italic magenta")

    def on_tool_call(self, event: dict[str, Any]) -> None:
        self._conclude_reasoning()
        self._stop_live()
        self.console.print(
            f"\n[yellow]🔧 Calling tool: {event.get('name', 'unknown')}[/yellow]"
        )

    def on_tool_output(self, event: dict[str, Any]) -> None:
        self._stop_live()
        output = event.get("output", "")
        preview = f"{output[:100]}..." if len(output) > 100 else output
        self.console.print(f"[cyan]📤 Tool output: {preview}[/cyan]\n")

    def on_error(self, event: dict[str, Any]) -> None:
        self._stop_live()
        self.console.print(
            f"\n[red]❌ Error: {event.get('content', 'Unknown error')}[/red]"
        )

    def close(self) -> None:
        self._stop_live()
        self.console.print()

    def _start_live(self) -> None:
        if not self._live_active:
            self.live.start()
            self._live_active = True

    def _stop_live(self) -> None:
        if self._live_active:
            self.live.stop()
            self._live_active = False

    def _ensure_agent_banner(self) -> None:
        if not self._agent_banner_shown:
            self.console.print("\n[bold green]Agent:[/bold green]")
            self._agent_banner_shown = True

    def _conclude_reasoning(self) -> None:
        if self._reasoning:
            self._reasoning = False
            self.console.print()
