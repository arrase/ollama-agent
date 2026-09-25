"""Streaming renderer that dynamically updates Textual widgets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...streaming import StreamingRenderer
from .widgets.messages import AgentResponse

if TYPE_CHECKING:
    from .app import OllamaAgentApp


class _TUIStreamingRenderer(StreamingRenderer):
    """Bridge connecting streaming event callbacks to Textual TUI widgets."""

    def __init__(self, app: OllamaAgentApp, scroll: Any, widget: AgentResponse):
        self.app = app
        self.scroll = scroll
        self.widget = widget
        self._auto_scroll = True
        self._last_scroll_y = scroll.scroll_y
        self._last_max_scroll_y = scroll.max_scroll_y
        self._timer = app.set_interval(0.1, self._do_scroll)

    def _do_scroll(self) -> None:
        if self._auto_scroll:
            self.scroll.scroll_end(animate=False)
            self._last_scroll_y = self.scroll.scroll_y
            self._last_max_scroll_y = self.scroll.max_scroll_y

    def _scroll(self) -> None:
        # If scroll_y decreased but max_scroll_y didn't drop, the user scrolled up.
        if self.scroll.scroll_y < self._last_scroll_y and self.scroll.max_scroll_y >= self._last_max_scroll_y:
            self._auto_scroll = False
        elif self.scroll.scroll_y >= self.scroll.max_scroll_y - 4:
            self._auto_scroll = True
        self._last_scroll_y = self.scroll.scroll_y
        self._last_max_scroll_y = self.scroll.max_scroll_y

    def on_text_delta(self, event: dict[str, Any]) -> None:
        self.widget.append_text(event["content"])
        self._scroll()

    def on_reasoning_delta(self, event: dict[str, Any]) -> None:
        self.widget.append_thinking(event["content"])
        self._scroll()

    def on_tool_call(self, event: dict[str, Any]) -> None:
        self.widget.add_tool_call(
            name=event["name"],
            agent=event.get("agent_name"),
        )
        self._scroll()

    def on_tool_output(self, event: dict[str, Any]) -> None:
        self.widget.add_tool_output(
            agent=event.get("agent_name"),
            output_len=event.get("output_len"),
        )
        self._scroll()

    def on_error(self, event: dict[str, Any]) -> None:
        self.widget.add_error(event["content"])
        self._scroll()

    def on_warning(self, event: dict[str, Any]) -> None:
        self.widget.add_warning(event["content"])
        self._scroll()

    def close(self) -> None:
        self._timer.stop()
        self.widget.finish_generation()
        self._do_scroll()
