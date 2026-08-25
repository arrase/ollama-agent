"""Manual conversation compaction built on deepagents' public contracts.

Interoperates with deepagents' ``SummarizationMiddleware`` through the
documented state contract: the ``_summarization_event`` key holds a
``SummarizationEvent`` TypedDict and the effective conversation the model
sees is ``[summary_message] + messages[cutoff_index:]``. History offload
appends to a single markdown file per session under
``/conversation_history/`` (deepagents' default prefix).
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from deepagents.middleware.summarization import DEEPAGENTS_DEFAULT_SUMMARY_PROMPT
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import get_buffer_string

_log = logging.getLogger(__name__)

#: Path prefix used by deepagents to store conversation history files.
HISTORY_PATH_PREFIX = "/conversation_history"

#: Number of recent messages preserved (not summarized) by manual compaction.
KEEP_RECENT_MESSAGES = 2

_SUMMARY_SOURCE = "summarization"


def is_summary_message(msg: Any) -> bool:
    """Check whether *msg* is a summary HumanMessage produced by compaction."""
    return isinstance(msg, HumanMessage) and msg.additional_kwargs.get("lc_source") == _SUMMARY_SOURCE


def apply_summarization_event(messages: list[Any], event: dict[str, Any] | None) -> list[Any]:
    """Reconstruct the effective message list from raw state and a prior event.

    The effective conversation is ``[summary_message] + messages[cutoff:]``.
    Malformed events fall back to the full raw list (matching deepagents).
    """
    if event is None:
        return list(messages)
    try:
        summary_msg = event["summary_message"]
        cutoff = event["cutoff_index"]
    except (KeyError, TypeError) as exc:
        _log.warning("Malformed summarization event (%s); using full history", exc)
        return list(messages)
    if cutoff > len(messages):
        return [summary_msg]
    return [summary_msg, *messages[cutoff:]]


def compute_state_cutoff(prior_event: dict[str, Any] | None, effective_cutoff: int) -> int:
    """Translate an effective-list cutoff into an absolute state index.

    The summary message occupies effective index 0 but no state slot,
    hence the ``-1`` adjustment when a prior event exists.
    """
    if prior_event is None:
        return effective_cutoff
    prior = prior_event.get("cutoff_index")
    return prior + effective_cutoff - 1 if isinstance(prior, int) else effective_cutoff


def find_safe_cutoff(messages: list[Any], keep: int) -> int:
    """Return the index where messages can be cut keeping the last *keep*.

    Never splits an AI/tool-call request from its tool outputs: if the cut
    lands on a ToolMessage, the cutoff moves back to include the owning
    AIMessage (or forward past the orphaned tool outputs). Returns 0 when
    there are not enough messages to compact.
    """
    if len(messages) <= keep:
        return 0
    target = len(messages) - keep
    if not isinstance(messages[target], ToolMessage):
        return target

    tool_call_ids: set[str] = set()
    idx = target
    while idx < len(messages) and isinstance(messages[idx], ToolMessage):
        if messages[idx].tool_call_id:
            tool_call_ids.add(messages[idx].tool_call_id)
        idx += 1

    for i in range(target - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            ai_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
            if tool_call_ids & ai_ids:
                return i

    return idx


def build_summary_message(summary: str, file_path: str | None) -> HumanMessage:
    """Build the HumanMessage that replaces the summarized history."""
    if file_path is not None:
        content = (
            "You are in the middle of a conversation that has been summarized.\n\n"
            f"The full conversation history has been saved to {file_path} should you "
            "need to refer back to it for details.\n\n"
            "A condensed summary follows:\n\n"
            f"<summary>\n{summary}\n</summary>"
        )
    else:
        content = f"Here is a summary of the conversation to date:\n\n{summary}"
    return HumanMessage(content=content, additional_kwargs={"lc_source": _SUMMARY_SOURCE})


async def generate_summary(model: Any, messages: list[Any]) -> str:
    """Generate a structured summary of *messages* using *model*."""
    formatted = get_buffer_string(messages, format="xml")
    prompt = DEEPAGENTS_DEFAULT_SUMMARY_PROMPT.format(messages=formatted).rstrip()
    response = await model.ainvoke(prompt)
    return response.text.strip()


async def offload_history(backend: Any, messages: list[Any], path: str) -> str | None:
    """Append *messages* to the session history markdown file on *backend*.

    Returns the file path on success, or None when offloading fails
    (non-fatal: compaction proceeds with the summary alone).
    """
    filtered = [m for m in messages if not is_summary_message(m)]
    timestamp = datetime.now(UTC).isoformat()
    section = f"## Summarized at {timestamp}\n\n{get_buffer_string(filtered, format='xml')}\n\n"

    existing = ""
    try:
        responses = await backend.adownload_files([path])
        if responses and responses[0].content is not None and responses[0].error is None:
            existing = responses[0].content.decode("utf-8")
    except Exception as exc:
        _log.debug("No existing history at %s (%s): %s", path, type(exc).__name__, exc)

    combined = existing + section
    result = await backend.aedit(path, existing, combined) if existing else await backend.awrite(path, combined)
    if result is None or getattr(result, "error", None):
        _log.warning("Failed to offload conversation history to %s", path)
        return None
    return path


def new_session_id(state_values: dict[str, Any]) -> str:
    """Reuse the persisted summarization session id or create a fresh one."""
    existing = state_values.get("_summarization_session_id")
    return existing if isinstance(existing, str) and existing else f"session_{uuid.uuid4().hex}"
