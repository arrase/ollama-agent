"""Built-in tools and shared runtime state."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any, TypedDict

from langchain_core.tools import BaseTool, tool

from ..i18n import _
from ..rag import RAGError, RAGManager, RAGNotLoadedError
from . import episodic_memory


class RAGToolResult(TypedDict, total=False):
    """Result from RAG operations."""

    success: bool
    context: str
    results: list[dict[str, Any]]
    error: str


_UNSET: Any = object()
_default_tool_timeout: int = 30
_default_rag_manager: RAGManager | None = None
_default_active_thread_id: str = ""

_tool_timeout_var: ContextVar[Any] = ContextVar("tool_timeout", default=_UNSET)
_rag_manager_var: ContextVar[Any] = ContextVar("rag_manager", default=_UNSET)
_active_thread_id_var: ContextVar[Any] = ContextVar("active_thread_id", default=_UNSET)


def set_tool_timeout(timeout: int) -> None:
    global _default_tool_timeout
    if timeout <= 0:
        raise ValueError(_("Tool timeout must be greater than 0, got {timeout_s}", timeout_s=timeout))
    _default_tool_timeout = timeout
    _tool_timeout_var.set(timeout)


def get_tool_timeout() -> int:
    val = _tool_timeout_var.get()
    return _default_tool_timeout if val is _UNSET else val


def set_rag_manager(mgr: RAGManager | None) -> None:
    global _default_rag_manager
    _default_rag_manager = mgr
    _rag_manager_var.set(mgr)


def get_rag_manager() -> RAGManager | None:
    val = _rag_manager_var.get()
    return _default_rag_manager if val is _UNSET else val


def set_active_thread_id(thread_id: str) -> None:
    global _default_active_thread_id
    _default_active_thread_id = thread_id
    _active_thread_id_var.set(thread_id)


def get_active_thread_id() -> str:
    val = _active_thread_id_var.get()
    return _default_active_thread_id if val is _UNSET else val


@tool
async def rag_search(query: str, top_k: int | None = None) -> RAGToolResult:
    """Search the loaded RAG database for relevant document chunks."""
    try:
        mgr = get_rag_manager()
        if mgr is None:
            raise RAGNotLoadedError(_("No RAG database loaded."))
        results = await mgr.search(query, top_k)
        context_parts: list[str] = []
        for r in results:
            source = r["filename"]
            context_parts.append(f"[{_('Source:')} {source}]\n{r['content']}")
        context = "\n\n---\n\n".join(context_parts)
        return {"success": True, "context": context}
    except RAGError as exc:
        return {"success": False, "error": str(exc)}


@tool
async def search_past_conversations(query: str, limit: int = 3) -> str:
    """Search past conversation sessions and episodic memory by keywords, topics, or dates
    (e.g. 'yesterday', 'auth', 'database'). Returns timestamped excerpts of previous sessions.
    """
    if limit <= 0:
        raise ValueError(_("Limit must be greater than 0, got {limit}", limit=limit))
    try:
        results = await asyncio.to_thread(
            episodic_memory.search_past_conversations_in_db,
            query=query,
            db_path=episodic_memory.HISTORY_DB_PATH,
            exclude_thread_id=get_active_thread_id(),
            limit=limit,
        )
    except episodic_memory.HistoryError as exc:
        return _("Error searching past conversations: {exc}", exc=exc)
    return episodic_memory.format_past_conversations_context(results)


BUILTIN_TOOLS: list[BaseTool] = [search_past_conversations]

__all__ = [
    "BUILTIN_TOOLS",
    "RAGToolResult",
    "get_active_thread_id",
    "get_rag_manager",
    "get_tool_timeout",
    "rag_search",
    "search_past_conversations",
    "set_active_thread_id",
    "set_rag_manager",
    "set_tool_timeout",
]
