"""Console streaming renderer for CLI output."""

from __future__ import annotations
from typing import Any
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from .base import StreamingRenderer


class ConsoleStreamingRenderer(StreamingRenderer):
    """Renderer for streaming to the console."""

    def __init__(self, console: Console) -> None:
        self.console, self.live = console, Live(console=console, refresh_per_second=10)
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
            self.console.print("\n  [dim magenta]└──────────────────────────────────[/dim magenta]\n")

    def on_text_delta(self, event: dict[str, Any]) -> None:
        self._end_reasoning()
        if not self._banner_shown:
            self.console.print("  [bold green]🤖 Agent[/bold green]")
            self._banner_shown = True
        self._toggle_live(True)
        self._text.append(event.get("content", ""))
        self.live.update(Padding(Markdown("".join(self._text)), (0, 0, 0, 4)))

    def on_reasoning_delta(self, event: dict[str, Any]) -> None:
        content = event.get("content", "")
        if not content:
            return
        if not self._reasoning:
            self._toggle_live(False)
            self.console.print("\n  [bold magenta]🧠 Thinking[/bold magenta]")
            self.console.print("  [dim magenta]│[/dim magenta] ", end="")
            self._reasoning = True
            self._rendered_reasoning = ""
        
        # Determine the new delta to print (handles both cumulative and pure delta backends)
        if self._rendered_reasoning and content.startswith(self._rendered_reasoning):
            delta = content[len(self._rendered_reasoning):]
            self._rendered_reasoning = content
        else:
            delta = content
            self._rendered_reasoning += content

        if not delta:
            return

        parts = delta.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                self.console.print("\n  [dim magenta]│[/dim magenta] ", end="")
            if part:
                self.console.print(part, end="", style="dim italic magenta")

    def on_tool_call(self, event: dict[str, Any]) -> None:
        self._end_reasoning()
        self._toggle_live(False)
        agent = event.get("agent_name")
        prefix = f"[{agent}] " if isinstance(agent, str) and agent else ""
        self.console.print(
            f"  [yellow]✦ {prefix}Calling tool:[/yellow] [bold yellow]{event.get('name', 'unknown')}[/bold yellow]"
        )

    def on_tool_output(self, event: dict[str, Any]) -> None:
        self._toggle_live(False)
        # Tool outputs are meant for the model; printing them makes the CLI noisy
        # and can interleave with streamed assistant output.
        out_len = event.get("output_len")
        agent = event.get("agent_name")
        prefix = f"[{agent}] " if isinstance(agent, str) and agent else ""
        suffix = f" ({out_len} chars)" if isinstance(out_len, int) else ""
        self.console.print(
            f"  [dim cyan]✓ {prefix}Tool output received{suffix}[/dim cyan]\n"
        )

    def on_error(self, event: dict[str, Any]) -> None:
        self._toggle_live(False)
        self.console.print(
            f"  [red]❌ Error: {event.get('content', 'Unknown error')}[/red]"
        )

    def on_warning(self, event: dict[str, Any]) -> None:
        self._toggle_live(False)
        self.console.print(
            f"  [yellow]⚠ Warning: {event.get('content', 'Unknown warning')}[/yellow]"
        )

    def close(self) -> None:
        self._toggle_live(False)
        self.console.print()
