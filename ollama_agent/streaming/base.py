"""Base classes for streaming renderers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agent import AgentRuntime

_log = logging.getLogger(__name__)


class StreamingRenderer(ABC):
    """Abstract base class for streaming renderers."""

    def on_event(self, event: dict[str, Any]) -> None:
        """Dispatch event to type-specific handler (on_<type>)."""
        etype = event.get("type")
        if not etype or etype == "event":
            _log.debug("Skipping event with unroutable type: %s", etype)
            return
        handler = getattr(self, f"on_{etype}", None)
        if not callable(handler):
            _log.debug("Unhandled event type skipped: %s", etype)
            return
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
