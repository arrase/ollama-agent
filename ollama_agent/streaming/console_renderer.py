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
        self.console, self.live = console, Live(
            console=console, refresh_per_second=10)
        self._text: list[str] = []
        self._banner_shown = self._reasoning = self._live_active = False

    def _toggle_live(self, start: bool) -> None:
        if start and not self._live_active:
            self.live.start()
            self._live_active = True
        elif not start and self._live_active:
            self.live.stop()
            self._live_active = False

    def _end_reasoning(self) -> None:
        if self._reasoning:
            self._reasoning = False
            self.console.print()

    def on_text_delta(self, event: dict[str, Any]) -> None:
        self._end_reasoning()
        if not self._banner_shown:
            self.console.print()
            self._banner_shown = True
        self._toggle_live(True)
        self._text.append(event.get("content", ""))
        self.live.update(Markdown("".join(self._text)))

    def on_reasoning_delta(self, event: dict[str, Any]) -> None:
        if not self._reasoning:
            self._toggle_live(False)
            self.console.print(
                "\n[bold magenta]🧠 Thinking:[/bold magenta] ", end="")
            self._reasoning = True
        self.console.print(event.get("content", ""), end="",
                           style="dim italic magenta")

    def on_tool_call(self, event: dict[str, Any]) -> None:
        self._end_reasoning()
        self._toggle_live(False)
        self.console.print(
            f"\n[yellow]🔧 Calling tool: {event.get('name', 'unknown')}[/yellow]")

    def on_tool_output(self, event: dict[str, Any]) -> None:
        self._toggle_live(False)
        # Tool outputs are meant for the model; printing them makes the CLI noisy
        # and can interleave with streamed assistant output.
        out_len = event.get("output_len")
        suffix = f" ({out_len} chars)" if isinstance(out_len, int) else ""
        self.console.print(f"[dim cyan]📤 Tool output recibido (oculto){suffix}[/dim cyan]\n")

    def on_error(self, event: dict[str, Any]) -> None:
        self._toggle_live(False)
        self.console.print(
            f"\n[red]❌ Error: {event.get('content', 'Unknown error')}[/red]")

    def close(self) -> None:
        self._toggle_live(False)
        self.console.print()
