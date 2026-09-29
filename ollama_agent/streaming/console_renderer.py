"""Console streaming renderer for CLI output."""

from __future__ import annotations

import asyncio
import sys
import threading
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.padding import Padding

from ..i18n import _
from .base import StreamingRenderer
from .interrupts import build_approval_decisions, extract_action_requests

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

        parts = content.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                self.console.print("\n  [dim magenta]│[/dim magenta] ", end="")
            if part:
                self.console.print(escape(part), end="", style="dim italic magenta")

    def _agent_prefix(self, event: dict[str, Any]) -> str:
        agent = event.get("agent_name")
        return escape(f"[{agent}] ") if agent else ""

    def on_tool_call(self, event: dict[str, Any]) -> None:
        self._end_reasoning()
        self._toggle_live(False)
        prefix = self._agent_prefix(event)
        tool_name = escape(str(event["name"]))
        tool_msg = escape(_("Calling tool: {tool_name}", tool_name=tool_name))
        self.console.print(f"  [yellow]✦ {prefix}{tool_msg}[/yellow]")

    def on_tool_output(self, event: dict[str, Any]) -> None:
        self._toggle_live(False)
        prefix = self._agent_prefix(event)
        suffix = f" ({_('{output_len} chars', output_len=event['output_len'])})"
        self.console.print(f"  [dim cyan]✓ {prefix}{_('Tool output received')}{suffix}[/dim cyan]\n")

    def on_error(self, event: dict[str, Any]) -> None:
        self._end_reasoning()
        self._toggle_live(False)
        self.console.print(f"  [red]❌ {_('Error:')} {escape(str(event['content']))}[/red]")

    def on_warning(self, event: dict[str, Any]) -> None:
        self._end_reasoning()
        self._toggle_live(False)
        self.console.print(f"  [yellow]⚠ {_('Warning:')} {escape(str(event['content']))}[/yellow]")

    @staticmethod
    async def _prompt(prompt: str) -> str:
        """Read one line on a daemon thread so Ctrl-C can never block interpreter shutdown.

        ``asyncio.to_thread`` uses the default ThreadPoolExecutor, whose workers are joined
        (without timeout) at exit, so a pending ``input()`` there hangs the whole process.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()

        def _read() -> None:
            # Every input() failure is forwarded to the awaiting coroutine. KeyboardInterrupt
            # is listed explicitly instead of catching BaseException, which would also swallow
            # SystemExit and stop the interpreter from unwinding.
            try:
                value = input(prompt)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(future.set_exception, exc)
            except KeyboardInterrupt as exc:
                loop.call_soon_threadsafe(future.set_exception, exc)
            else:
                loop.call_soon_threadsafe(future.set_result, value)

        threading.Thread(target=_read, daemon=True, name="ollama-agent-approval").start()
        return await future

    async def handle_interrupt(self, event: dict[str, Any], runtime: AgentRuntime) -> list[dict[str, Any]] | None:
        self._toggle_live(False)
        self._end_reasoning()

        action_requests = extract_action_requests(event)

        self.console.print(f"\n  [bold yellow]⚠️ {_('Sensitive Tool Approval Required')}[/bold yellow]")

        for req in action_requests:
            self.console.print(f"  {_('Tool:')} [bold]{escape(str(req['name']))}[/bold]")
            self.console.print(f"  {_('Arguments:')} {escape(str(req['args']))}")

        if not sys.stdin.isatty():
            hint = _(
                "Cannot request tool approval in a non-interactive session. "
                "Re-run with -y (--yolo) to auto-approve sensitive tools."
            )
            self.console.print(f"  [red]❌ {hint}[/red]")
            return None

        try:
            while True:
                prompt_msg = f"  {_('Choose action: Approve (y) / Reject (n) / Allow Session (a) / Cancel (c): ')}"
                self.console.print(prompt_msg, end="")
                self.console.file.flush()
                choice = (await self._prompt("")).strip().lower()
                if choice == "y":
                    return build_approval_decisions(action_requests, "approve")
                if choice == "n":
                    return build_approval_decisions(action_requests, "reject")
                if choice == "a":
                    return build_approval_decisions(action_requests, "allow", runtime=runtime)
                if choice == "c":
                    break
                invalid_msg = _("Invalid choice. Please enter 'y', 'n', 'a', or 'c'.")
                self.console.print(f"  [red]{invalid_msg}[/red]")
        except (EOFError, KeyboardInterrupt):
            pass

        self.console.print(f"  [red]✗ {_('Cancelled')}[/red]\n")
        return None

    def close(self) -> None:
        self._end_reasoning()
        self._toggle_live(False)
        self.console.print()
