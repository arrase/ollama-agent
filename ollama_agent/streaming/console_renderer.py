"""Console streaming renderer for CLI output."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from .base import StreamingRenderer


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
        self._end_reasoning()
        self._ensure_banner()
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
        self._end_reasoning()
        self._stop_live()
        self.console.print(f"\n[yellow]🔧 Calling tool: {event.get('name', 'unknown')}[/yellow]")

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

    def _ensure_banner(self) -> None:
        if not self._agent_banner_shown:
            self.console.print()
            self._agent_banner_shown = True

    def _end_reasoning(self) -> None:
        if self._reasoning:
            self._reasoning = False
            self.console.print()
