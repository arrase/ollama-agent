"""Built-in tools and shared runtime state."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar

from langchain_core.tools import BaseTool, tool

from ..core import RAGToolResult
from ..i18n import _
from ..rag import RAGError, RAGManager
from . import episodic_memory
from .episodic_memory import (
    HistoryError,
    format_past_conversations_context,
    search_past_conversations_in_db,
)

_tool_timeout: ContextVar[int] = ContextVar("tool_timeout", default=30)
_rag_manager: ContextVar[RAGManager | None] = ContextVar("rag_manager", default=None)
_active_thread_id: ContextVar[str] = ContextVar("active_thread_id", default="")


def set_tool_timeout(timeout: int) -> None:
    if timeout <= 0:
        raise ValueError(_("Tool timeout must be greater than 0, got {timeout_s}", timeout_s=timeout))
    _tool_timeout.set(timeout)


def get_tool_timeout() -> int:
    return _tool_timeout.get()


def set_rag_manager(mgr: RAGManager | None) -> None:
    _rag_manager.set(mgr)


def get_rag_manager() -> RAGManager | None:
    return _rag_manager.get()


def set_active_thread_id(thread_id: str) -> None:
    _active_thread_id.set(thread_id)


def get_active_thread_id() -> str:
    return _active_thread_id.get()


@tool
async def rag_search(query: str, top_k: int | None = None) -> RAGToolResult:
    """Search the loaded RAG database for relevant document chunks."""
    mgr = get_rag_manager()
    if mgr is None:
        return {"success": False, "error": _("RAG manager not initialized")}
    if mgr.current_database is None:
        return {
            "success": False,
            "error": _("No RAG database loaded. Use /rag load <name> first."),
        }
    try:
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
    """Search past conversation sessions and episodic memory by keywords, topics, or dates (e.g. 'yesterday', 'auth', 'database'). Returns timestamped excerpts of previous sessions."""
    safe_limit = max(1, limit)
    try:
        results = await asyncio.to_thread(
            search_past_conversations_in_db,
            query=query,
            db_path=episodic_memory.HISTORY_DB_PATH,
            exclude_thread_id=get_active_thread_id(),
            limit=safe_limit,
        )
    except HistoryError as exc:
        return _("Error searching past conversations: {exc}", exc=exc)
    return format_past_conversations_context(results)


BUILTIN_TOOLS: list[BaseTool] = [search_past_conversations]

__all__ = [
    "BUILTIN_TOOLS",
    "get_active_thread_id",
    "get_rag_manager",
    "get_tool_timeout",
    "rag_search",
    "search_past_conversations",
    "set_active_thread_id",
    "set_rag_manager",
    "set_tool_timeout",
]
