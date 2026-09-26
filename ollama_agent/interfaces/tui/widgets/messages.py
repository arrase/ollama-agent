"""Chat message and event widgets for user prompts and agent turn output."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.timer import Timer
from textual.widgets import Collapsible, Markdown, Static

from ....i18n import _

if TYPE_CHECKING:
    from ..app import OllamaAgentApp


class UserMessage(Container):
    """Rendered user prompt."""

    def __init__(self, text: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.text = text

    def compose(self) -> ComposeResult:
        yield Static(
            f"[bold #38bdf8]❯ {_('you')}[/bold #38bdf8]",
            classes="msg-role user-role",
        )
        yield Static(self.text, markup=False, classes="msg-content user-content")


class AgentResponse(Container):
    """Container representing the agent's turn.

    It dynamically hosts thinking, text responses, and tool calls in order.
    """

    def __init__(self, initial_text: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.initial_text = initial_text
        self._header = Static(
            f"[bold #34d399]◆ {_('agent')}[/bold #34d399]",
            classes="msg-role agent-role",
        )
        self.current_thinking: Collapsible | None = None
        self.current_thinking_text: Static | None = None
        self._thinking_chunks: list[str] = []
        self.current_text_widget: Markdown | None = None
        self._text_chunks: list[str] = [initial_text] if initial_text else []
        self.thinking_timer: Timer | None = None
        self._thinking_dots_count = 3
        self._text_update_timer: Timer | None = None
        self._last_text_update = 0.0

    def compose(self) -> ComposeResult:
        yield self._header
        if self.initial_text:
            self.current_text_widget = Markdown(self.initial_text, classes="msg-content agent-content")
            yield self.current_text_widget

    def _animate_thinking(self) -> None:
        if self.current_thinking is not None:
            self._thinking_dots_count = (self._thinking_dots_count % 3) + 1
            dots = " ·" * self._thinking_dots_count
            self.current_thinking.title = f"{_('Thinking')}{dots}"

    def _stop_thinking_animation(self) -> None:
        if self.thinking_timer is not None:
            self.thinking_timer.stop()
            self.thinking_timer = None
        if self.current_thinking is not None:
            self.current_thinking.title = _("Thought process")

    def append_thinking(self, delta: str) -> None:
        self.current_text_widget = None
        if self.current_thinking is None:
            app = cast("OllamaAgentApp", self.app)
            collapse_default = app.repl.runtime.settings.runtime.collapse_thinking
            self.current_thinking_text = Static("", classes="msg-content thinking-body")
            self._thinking_chunks = []
            self.current_thinking = Collapsible(
                self.current_thinking_text,
                title=_("Thinking ···"),
                collapsed=collapse_default,
            )
            self.mount(self.current_thinking)
            self._thinking_dots_count = 3
            self.thinking_timer = self.set_interval(0.5, self._animate_thinking)

        if self.current_thinking_text is not None:
            self._thinking_chunks.append(delta)
            self.current_thinking_text.update(Text("".join(self._thinking_chunks), style="dim italic #8b949e"))

    def append_text(self, delta: str) -> None:
        self._stop_thinking_animation()
        self.current_thinking = None
        self.current_thinking_text = None
        if self.current_text_widget is None:
            self.current_text_widget = Markdown("", classes="msg-content agent-content")
            self.mount(self.current_text_widget)
            self._text_chunks = []
        self._text_chunks.append(delta)

        now = time.monotonic()
        if now - self._last_text_update > 0.1:
            self.flush_text()
        elif self._text_update_timer is None:
            self._text_update_timer = self.set_timer(0.1, self.flush_text)

    def flush_text(self) -> None:
        if self._text_update_timer is not None:
            self._text_update_timer.stop()
            self._text_update_timer = None
        if self.current_text_widget is not None:
            self.current_text_widget.update("".join(self._text_chunks))
        self._last_text_update = time.monotonic()

    def finish_generation(self) -> None:
        self.flush_text()
        self._stop_thinking_animation()

    def _reset_active_stream(self) -> None:
        self.flush_text()
        self._stop_thinking_animation()
        self.current_thinking = None
        self.current_thinking_text = None
        self.current_text_widget = None

    def add_tool_call(self, name: str, agent: str | None = None) -> None:
        self._reset_active_stream()
        self.mount(ToolCallMessage(tool_name=name, agent_name=agent))

    def add_tool_output(self, agent: str | None = None, output_len: int | None = None) -> None:
        self._reset_active_stream()
        self.mount(ToolOutputMessage(agent_name=agent, output_len=output_len))

    def add_error(self, content: str) -> None:
        self._reset_active_stream()
        self.mount(
            Static(
                f"[bold #f87171]✕ {_('Error:')}[/bold #f87171] [red]{escape(content)}[/red]", classes="system-message"
            )
        )

    def add_warning(self, content: str) -> None:
        self._reset_active_stream()
        self.mount(
            Static(
                f"[bold #fbbf24]⚠ {_('Warning:')}[/bold #fbbf24] [yellow]{escape(content)}[/yellow]",
                classes="system-message",
            )
        )


class ToolCallMessage(Static):
    """One-line tool invocation event."""

    def __init__(self, tool_name: str, agent_name: str | None = None, **kwargs: Any) -> None:
        prefix = f"[dim]{escape(f'[{agent_name}]')}[/dim] " if agent_name else ""
        super().__init__(
            f"  [#fbbf24]⚙[/#fbbf24] {prefix}[bold #e6edf3]{escape(tool_name)}[/bold #e6edf3]",
            **kwargs,
        )


class ToolOutputMessage(Static):
    """One-line tool output acknowledgement."""

    def __init__(self, agent_name: str | None = None, output_len: int | None = None, **kwargs: Any) -> None:
        prefix = f"[dim]{escape(f'[{agent_name}]')}[/dim] " if agent_name else ""
        suffix = f" [dim]({_('{output_len} chars', output_len=output_len)})[/dim]" if output_len is not None else ""
        super().__init__(
            f"  [#34d399]✓[/#34d399] {prefix}[dim]{_('output received')}[/dim]{suffix}",
            **kwargs,
        )
