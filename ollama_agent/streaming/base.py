"""Base classes for streaming renderers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StreamingRenderer(ABC):
    """Abstract base class for streaming renderers."""

    def on_event(self, event: dict[str, Any]) -> None:
        """Dispatch event to type-specific handler (on_<type>)."""
        if handler := getattr(self, f"on_{event.get('type', '')}", None):
            handler(event)

    def on_error(self, event: dict[str, Any]) -> None:
        """Handle an error event."""
        print(f"Error: {event.get('content', 'Unknown error')}")

    def on_warning(self, event: dict[str, Any]) -> None:
        """Handle a warning event."""
        print(f"Warning: {event.get('content', 'Unknown warning')}")

    @abstractmethod
    def close(self) -> None:
        """Clean up renderer resources."""
