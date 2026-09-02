"""Interrupt payload extraction for LangGraph interrupt events."""

from __future__ import annotations

from typing import Any, Mapping


from ..i18n import _


def extract_action_requests(interrupt_event: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract action requests from a streaming interrupt event."""
    return interrupt_event["interrupts"][0].value["action_requests"]


def build_approval_decisions(
    action_requests: list[dict[str, Any]],
    action: str,  # "approve", "reject", or "allow"
    *,
    runtime: Any = None,
    reject_message: str | None = None,
) -> list[dict[str, Any]]:
    """Consolidate LangGraph approval/rejection payload construction used across CLI and TUI."""
    if action == "allow" and runtime:
        for req in action_requests:
            runtime.auto_approved_tools.add(req["name"])

    if action in ("approve", "allow"):
        return [{"type": "approve"} for _ in action_requests]

    if action == "reject":
        if reject_message:
            return [{"type": "reject", "message": reject_message} for _ in action_requests]
        return [
            {"type": "reject", "message": _("User rejected executing tool '{name}'.", name=req["name"])}
            for req in action_requests
        ]

    raise ValueError(f"Unknown action: {action}")
