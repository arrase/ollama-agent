"""Interrupt payload extraction for LangGraph interrupt events."""

from __future__ import annotations

from typing import Any, Mapping


def extract_action_requests(interrupt_event: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract validated action requests from a streaming interrupt event.

    This is the single integration point for parsing the ``{"interrupts": [...]}``
    payload emitted by :meth:`AgentRuntime.run_streamed`. Consumers outside the
    streaming package (e.g. ``ollama_agent/interfaces/repl.py``) must import this
    helper instead of re-implementing the parse, so any change to the interrupt
    format only needs to happen here.

    Args:
        interrupt_event: Event mapping with a non-empty ``"interrupts"`` list;
            the first interrupt's ``value`` must be a mapping holding an
            ``"action_requests"`` list of dicts with ``name`` and ``args`` keys.

    Returns:
        The validated action requests list.

    Raises:
        ValueError: If the interrupt payload is malformed.
    """
    interrupts = interrupt_event.get("interrupts")
    if not isinstance(interrupts, (list, tuple)) or not interrupts:
        raise ValueError("Malformed interrupt event: 'interrupts' must be a non-empty list or tuple")

    value = getattr(interrupts[0], "value", None)
    if not isinstance(value, Mapping):
        raise ValueError("Malformed interrupt payload: first interrupt has no mapping 'value'")

    action_requests = value.get("action_requests")
    if not isinstance(action_requests, (list, tuple)) or not action_requests:
        raise ValueError("Malformed interrupt payload: 'action_requests' must be a non-empty list or tuple")

    for req in action_requests:
        if not isinstance(req, dict) or "name" not in req or "args" not in req:
            raise ValueError(f"Malformed action request: {req!r}")
    return list(action_requests)
