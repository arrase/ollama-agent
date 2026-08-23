"""Console streaming renderer for CLI output."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding

from ..i18n import _
from .base import StreamingRenderer

if TYPE_CHECKING:
    from ..agent import AgentRuntime


class ConsoleStreamingRenderer(StreamingRenderer):
    """Renderer for streaming to the console."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.live = Live(console=console, refresh_per_second=10)
        self._text: list[str] = []
        self._banner_shown = False
        self._reasoning = False
        self._live_active = False
        self._rendered_reasoning = ""

    def _toggle_live(self, start: bool) -> None:
        if start and not self._live_active:
            self.live.start()
            self._live_active = True
        elif not start and self._live_active:
            self.live.stop()
            self._live_active = False
            self._text.clear()

    def _end_reasoning(self) -> None:
        if self._reasoning:
            self._reasoning = False
            self.console.print("\n  [dim magenta]└──────────────────────────────────[/dim magenta]\n")

    def on_text_delta(self, event: dict[str, Any]) -> None:
        self._end_reasoning()
        if not self._banner_shown:
            self.console.print(f"  [bold green]🤖 {_('Assistant')}[/bold green]")
            self._banner_shown = True
        self._toggle_live(True)
        self._text.append(event["content"])
        self.live.update(Padding(Markdown("".join(self._text)), (0, 0, 0, 4)))

    def on_reasoning_delta(self, event: dict[str, Any]) -> None:
        content = event["content"]
        if not content:
            return
        if not self._reasoning:
            self._toggle_live(False)
            self.console.print(f"\n  [bold magenta]🧠 {_('Thinking')}[/bold magenta]")
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

    def _agent_prefix(self, event: dict[str, Any]) -> str:
        agent = event.get("agent_name")
        return f"[{agent}] " if agent else ""

    def on_tool_call(self, event: dict[str, Any]) -> None:
        self._end_reasoning()
        self._toggle_live(False)
        prefix = self._agent_prefix(event)
        tool_name = event["name"]
        tool_msg = _("Calling tool: {tool_name}", tool_name=tool_name)
        self.console.print(
            f"  [yellow]✦ {prefix}{tool_msg}[/yellow]"
        )

    def on_tool_output(self, event: dict[str, Any]) -> None:
        self._toggle_live(False)
        out_len = event.get("output_len")
        prefix = self._agent_prefix(event)
        suffix = f" ({_('{output_len} chars', output_len=out_len)})" if out_len is not None else ""
        self.console.print(
            f"  [dim cyan]✓ {prefix}{_('Tool output received')}{suffix}[/dim cyan]\n"
        )

    def on_error(self, event: dict[str, Any]) -> None:
        self._toggle_live(False)
        self.console.print(
            f"  [red]❌ {_('Error:')} {event['content']}[/red]"
        )

    def on_warning(self, event: dict[str, Any]) -> None:
        self._toggle_live(False)
        self.console.print(
            f"  [yellow]⚠ {_('Warning:')} {event['content']}[/yellow]"
        )

    async def handle_interrupt(
        self, event: dict[str, Any], runtime: AgentRuntime
    ) -> list[dict[str, Any]] | None:
        self._toggle_live(False)
        self._end_reasoning()

        interrupts = event.get("interrupts", [])
        if not interrupts:
            return None

        self.console.print(f"\n  [bold yellow]⚠️ {_('Sensitive Tool Approval Required')}[/bold yellow]")

        interrupt_val = interrupts[0].value
        action_requests = interrupt_val.get("action_requests", [])

        for req in action_requests:
            name = req["name"]
            args = req.get("args", {})
            self.console.print(f"  {_('Tool:')} [bold]{name}[/bold]")
            self.console.print(f"  {_('Arguments:')} {args}")

        try:
            while True:
                prompt_msg = f"  {_('Choose action: Approve (y) / Reject (n) / Allow Session (a) / Cancel (c): ')}"
                self.console.print(prompt_msg, end="")
                self.console.file.flush()
                choice = (await asyncio.to_thread(input)).strip().lower()
                if choice == "y":
                    return [{"type": "approve"} for _ in action_requests]
                elif choice == "n":
                    return [{
                        "type": "reject",
                        "message": _("User rejected executing tool '{name}'.", name=r["name"])
                    } for r in action_requests]
                elif choice == "a":
                    for req in action_requests:
                        runtime.auto_approved_tools.add(req["name"])
                    return [{"type": "approve"} for _ in action_requests]
                elif choice == "c":
                    self.console.print(f"  [red]✗ {_('Cancelled')}[/red]\n")
                    raise KeyboardInterrupt()
                else:
                    invalid_msg = _("Invalid choice. Please enter 'y', 'n', 'a', or 'c'.")
                    self.console.print(f"  [red]{invalid_msg}[/red]")
        except (EOFError, KeyboardInterrupt):
            self.console.print(f"  [red]✗ {_('Cancelled')}[/red]\n")
            raise KeyboardInterrupt()

    def close(self) -> None:
        self._toggle_live(False)
        self.console.print()
