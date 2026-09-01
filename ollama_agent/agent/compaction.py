"""Manual conversation compaction interoperable with deepagents' SummarizationMiddleware.

The pure state-arithmetic helpers delegate to a live ``SummarizationMiddleware``
instance so the logic exists in exactly one place (upstream). This module keeps
only what deepagents does not provide or deliberately does differently: strict
fail-loud validation of summarization events (deepagents warns and continues),
history offloading that refuses to lose data (``HistoryOffloadError``), the
tool-safe cutoff search, and summary generation.

Interop contract (guarded by ``tests/test_compaction_interop.py``): state key
``SUMMARIZATION_STATE_KEY`` holds a ``SummarizationEvent`` TypedDict, the
summary ``HumanMessage`` carries ``lc_source='summarization'``, session id
persists under ``SUMMARIZATION_SESSION_ID_KEY``, and history appends to
``/conversation_history/<session_id>.md``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from deepagents.backends.protocol import FILE_NOT_FOUND, BackendProtocol
from deepagents.middleware.summarization import (
    DEEPAGENTS_DEFAULT_SUMMARY_PROMPT,
    SummarizationEvent,
    SummarizationMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import get_buffer_string

from ..core import extract_text
from ..i18n import _

#: Path prefix used by deepagents to store conversation history files.
HISTORY_PATH_PREFIX = "/conversation_history"

#: Number of recent messages preserved (not summarized) by manual compaction.
KEEP_RECENT_MESSAGES = 2

_SUMMARY_SOURCE = "summarization"

#: State keys of deepagents' SummarizationMiddleware private contract.
SUMMARIZATION_STATE_KEY = "_summarization_event"
SUMMARIZATION_SESSION_ID_KEY = "_summarization_session_id"


class HistoryOffloadError(RuntimeError):
    """Raised when conversation history cannot be persisted to the backend."""


def is_summary_message(msg: BaseMessage) -> bool:
    """Check whether *msg* is a summary HumanMessage produced by compaction."""
    return isinstance(msg, HumanMessage) and msg.additional_kwargs.get("lc_source") == _SUMMARY_SOURCE


def apply_summarization_event(
    engine: SummarizationMiddleware,
    messages: list[BaseMessage],
    event: SummarizationEvent | dict[str, Any] | None,
) -> list[BaseMessage]:
    """Reconstruct the effective message list from raw state and a prior event.

    The effective conversation is ``[summary_message] + messages[cutoff:]``.
    The slicing itself is delegated to *engine* so upstream owns the semantics;
    malformed events raise ``ValueError`` before delegation instead of hitting
    deepagents' warn-and-continue path.
    """
    if event is None:
        return list(messages)
    try:
        cutoff = event["cutoff_index"]
        event["summary_message"]
        if not isinstance(cutoff, int):
            raise TypeError(f"invalid cutoff_index {cutoff!r}")
    except (KeyError, TypeError) as exc:
        raise ValueError(_("Malformed summarization event: {detail}", detail=str(exc))) from exc
    return engine._apply_event_to_messages(messages, event)


def compute_state_cutoff(
    engine: SummarizationMiddleware,
    prior_event: SummarizationEvent | dict[str, Any] | None,
    effective_cutoff: int,
) -> int:
    """Translate an effective-list cutoff into an absolute state index.

    The summary message occupies effective index 0 but no state slot,
    hence the ``-1`` adjustment when a prior event exists. Arithmetic is
    delegated to *engine*; malformed priors raise ``ValueError``.
    """
    if prior_event is None:
        return effective_cutoff
    prior = prior_event.get("cutoff_index")
    if not isinstance(prior, int):
        raise ValueError(
            _("Malformed summarization event: {detail}", detail=f"invalid cutoff_index {prior!r}")
        )
    return engine._compute_state_cutoff(prior_event, effective_cutoff)


def find_safe_cutoff(messages: list[BaseMessage], keep: int) -> int:
    """Return the index where messages can be cut keeping the last *keep*.

    Never splits an AI/tool-call request from its tool outputs: if the cut
    lands on a ToolMessage, the cutoff moves back to include the owning
    AIMessage (or forward past the orphaned tool outputs). Returns 0 when
    there are not enough messages to compact.
    """
    if keep <= 0 or len(messages) <= keep:
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


def build_summary_message(engine: SummarizationMiddleware, summary: str, file_path: str) -> HumanMessage:
    """Build the HumanMessage that replaces the summarized history.

    Delegated to *engine*: the wording and the ``lc_source`` marker are
    upstream's contract, so they must not be duplicated here.
    """
    return engine._build_new_messages_with_path(summary, file_path)[0]


async def generate_summary(model: BaseChatModel, messages: list[BaseMessage]) -> str:
    """Generate a structured summary of *messages* using *model*."""
    formatted = get_buffer_string(messages, format="xml")
    prompt = DEEPAGENTS_DEFAULT_SUMMARY_PROMPT.format(messages=formatted).rstrip()
    response = await model.ainvoke(prompt)
    return extract_text(response.content).strip()


async def offload_history(backend: BackendProtocol, messages: list[BaseMessage], path: str) -> str:
    """Append *messages* to the session history markdown file on *backend*.

    Returns the file path on success. Raises ``HistoryOffloadError`` when the
    backend refuses the read-modify-write cycle: manual compaction must never
    proceed against a stale or missing history file, otherwise the summary
    would reference history that was silently lost or overwritten.
    """
    filtered = [m for m in messages if not is_summary_message(m)]
    timestamp = datetime.now(UTC).isoformat()
    section = f"## Summarized at {timestamp}\n\n{get_buffer_string(filtered, format='xml')}\n\n"

    # Backends report recoverable failures through FileDownloadResponse.error
    # instead of raising; only "file_not_found" means there is nothing to append to.
    response = (await backend.adownload_files([path]))[0]
    existing = ""
    if response.error is None:
        existing = response.content.decode("utf-8")
    elif response.error != FILE_NOT_FOUND:
        raise HistoryOffloadError(
            _("Failed to offload conversation history to {path}: {error}", path=path, error=response.error)
        )

    combined = existing + section
    result = await backend.aedit(path, existing, combined) if existing else await backend.awrite(path, combined)
    if result.error:
        raise HistoryOffloadError(
            _("Failed to offload conversation history to {path}: {error}", path=path, error=result.error)
        )
    return path


def new_session_id(state_values: dict[str, Any]) -> str:
    """Reuse the persisted summarization session id or create a fresh one."""
    existing = state_values.get(SUMMARIZATION_SESSION_ID_KEY)
    return existing if isinstance(existing, str) and existing else f"session_{uuid.uuid4().hex}"
