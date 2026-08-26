"""Base classes for streaming renderers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agent import AgentRuntime


class StreamingRenderer(ABC):
    """Abstract base class for streaming renderers."""

    def on_event(self, event: dict[str, Any]) -> None:
        """Dispatch event to type-specific handler (on_<type>)."""
        etype = event.get("type")
        handler = getattr(self, f"on_{etype}", None) if etype else None
        if handler is None:
            raise ValueError(f"Unhandled event type: {etype}")
        handler(event)

    @abstractmethod
    def on_error(self, event: dict[str, Any]) -> None:
        """Handle an error event."""
        ...

    @abstractmethod
    def on_warning(self, event: dict[str, Any]) -> None:
        """Handle a warning event."""
        ...

    async def handle_interrupt(
        self, event: dict[str, Any], runtime: AgentRuntime
    ) -> list[dict[str, Any]] | None:
        """Handle an interrupt event. Returning None aborts the run."""
        return None

    @abstractmethod
    def close(self) -> None:
        """Clean up renderer resources."""
