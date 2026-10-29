"""Base classes for streaming renderers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..agent import AgentRuntime


class StreamingRenderer(ABC):
    """Abstract base class for streaming renderers."""

    def on_event(self, event: dict[str, Any]) -> None:
        """Dispatch event to type-specific handler."""
        handlers = {
            "text_delta": self.on_text_delta,
            "reasoning_delta": self.on_reasoning_delta,
            "tool_call": self.on_tool_call,
            "tool_output": self.on_tool_output,
            "error": self.on_error,
            "warning": self.on_warning,
        }
        handler = handlers.get(event["type"])
        if handler is not None:
            handler(event)

    def on_text_delta(self, event: dict[str, Any]) -> None:
        """Handle a text delta event."""

    def on_reasoning_delta(self, event: dict[str, Any]) -> None:
        """Handle a reasoning delta event."""

    def on_tool_call(self, event: dict[str, Any]) -> None:
        """Handle a tool call event."""

    def on_tool_output(self, event: dict[str, Any]) -> None:
        """Handle a tool output event."""

    @abstractmethod
    def on_error(self, event: dict[str, Any]) -> None:
        """Handle an error event."""

    @abstractmethod
    def on_warning(self, event: dict[str, Any]) -> None:
        """Handle a warning event."""

    async def handle_interrupt(self, event: dict[str, Any], runtime: AgentRuntime) -> list[dict[str, Any]] | None:
        """Handle an interrupt event. Returning None aborts the run."""

    @abstractmethod
    def close(self) -> None:
        """Clean up renderer resources."""
