"""Base classes for streaming renderers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from textual.widgets import RichLog


def clear_log_lines(log: RichLog, start_line: int) -> None:
    """Remove rendered lines after ``start_line`` and clear RichLog caches."""
    while len(log.lines) > start_line:
        log.lines.pop()
    if hasattr(log, "_line_cache"):
        log._line_cache.clear()


class StreamingRenderer(ABC):
    """Abstract base class for streaming renderers."""

    def on_event(self, event: dict[str, Any]) -> None:
        """Dispatch event to type-specific handler (on_<type>)."""
        if handler := getattr(self, f"on_{event.get('type', '')}", None):
            handler(event)

    def on_error(self, event: dict[str, Any]) -> None:
        """Handle an error event."""
        print(f"Error: {event.get('content', 'Unknown error')}")

    @abstractmethod
    def close(self) -> None:
        """Clean up renderer resources."""


class BufferedTokenRenderer:
    """Token-buffered rendering with periodic updates for RichLog widgets.

    Accumulates tokens and refreshes display at configurable frequency
    to avoid excessive re-renders.
    """

    def __init__(
        self,
        log: RichLog,
        render_fn: Callable[[str], Any],
        update_frequency: int = 5,
    ) -> None:
        self.log = log
        self.render_fn = render_fn
        self.update_frequency = max(1, update_frequency)
        self._buffer = ""
        self._count = 0
        self._start_line = 0
        self._active = False

    @property
    def buffer(self) -> str:
        return self._buffer

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self, header: Any | None = None) -> None:
        """Begin buffered rendering, optionally writing a header."""
        if self._active:
            return
        if header is not None:
            self.log.write(header)
        self._start_line = len(self.log.lines)
        self._active = True

    def append(self, token: str) -> None:
        """Append token and refresh display at configured cadence."""
        if not self._active:
            self.start()
        self._buffer += token
        self._count += 1
        if self._count % self.update_frequency == 0:
            self._refresh()

    def finalize(self) -> None:
        """Final render and reset state."""
        if self._active:
            self._refresh()
        self._reset()

    def _refresh(self) -> None:
        """Re-render buffer content."""
        if not self._buffer:
            return
        clear_log_lines(self.log, self._start_line)
        self.log.write(self.render_fn(self._buffer))
        self.log.scroll_end(animate=False)

    def _reset(self) -> None:
        """Reset internal state."""
        self._buffer = ""
        self._count = 0
        self._active = False
