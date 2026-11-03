"""Built-in tools and shared runtime state."""

from __future__ import annotations

import asyncio
from typing import TypedDict

from langchain_core.tools import BaseTool, tool

from ..i18n import _
from ..rag import RAGError, RAGManager
from ..settings import HISTORY_DB_PATH
from . import episodic_memory


class RAGToolResult(TypedDict, total=False):
    """Result from RAG operations."""

    success: bool
    context: str
    error: str


_tool_timeout: int = 30
_rag_manager: RAGManager | None = None
_active_thread_id: str = ""


def set_tool_timeout(timeout: int) -> None:
    global _tool_timeout
    if timeout <= 0:
        raise ValueError(_("Tool timeout must be greater than 0, got {timeout_s}", timeout_s=timeout))
    _tool_timeout = timeout


def get_tool_timeout() -> int:
    return _tool_timeout


def set_rag_manager(mgr: RAGManager | None) -> None:
    global _rag_manager
    _rag_manager = mgr


def get_rag_manager() -> RAGManager | None:
    return _rag_manager


def set_active_thread_id(thread_id: str) -> None:
    global _active_thread_id
    _active_thread_id = thread_id


def get_active_thread_id() -> str:
    return _active_thread_id


@tool
async def rag_search(query: str, top_k: int | None = None) -> RAGToolResult:
    """Search the loaded RAG database for relevant document chunks."""
    try:
        mgr = get_rag_manager()
        if mgr is None:
            raise RAGError(_("No RAG database loaded."))  # noqa: TRY301
        results = await mgr.search(query, top_k)
        context_parts: list[str] = []
        for r in results:
            source = r["filename"]
            context_parts.append(f"[{_('Source:')} {source}]\n{r['content']}")
        context = "\n\n---\n\n".join(context_parts)
    except RAGError as exc:
        return {"success": False, "error": str(exc)}
    return {"success": True, "context": context}


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
            db_path=HISTORY_DB_PATH,
            exclude_thread_id=get_active_thread_id(),
            limit=limit,
        )
    except episodic_memory.HistoryError as exc:
        return _("Error searching past conversations: {exc}", exc=exc)
    return episodic_memory.format_past_conversations_context(results)


BUILTIN_TOOLS: list[BaseTool] = [search_past_conversations]
