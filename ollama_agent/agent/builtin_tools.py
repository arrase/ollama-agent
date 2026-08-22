"""Built-in tools and shared runtime state."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from langchain_core.tools import tool

from ..core import RAGToolResult
from ..rag import RAGError, RAGManager
from .episodic_memory import (
    format_past_conversations_context,
    search_past_conversations_in_db,
)


_tool_timeout: ContextVar[int] = ContextVar("tool_timeout", default=30)
_rag_manager: ContextVar[RAGManager | None] = ContextVar("rag_manager", default=None)
_active_thread_id: ContextVar[str] = ContextVar("active_thread_id", default="")


def set_tool_timeout(timeout: int) -> None:
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
    if not (mgr := get_rag_manager()):
        return {"success": False, "error": "RAG manager not initialized"}
    if mgr.current_database is None:
        return {
            "success": False,
            "error": "No RAG database loaded. Use /rag load <name> first.",
        }
    try:
        results = await mgr.search(query, top_k)
        context_parts: list[str] = []
        for r in results:
            source = r["filename"] or r["source"]
            context_parts.append(f"[Source: {source}]\n{r['content']}")
        context = "\n\n---\n\n".join(context_parts)
        return {"success": True, "context": context, "results": results}
    except RAGError as exc:
        return {"success": False, "error": str(exc)}


@tool
async def search_past_conversations(query: str, limit: int = 3) -> str:
    """Search past conversation sessions and episodic memory by keywords, topics, or dates (e.g. 'yesterday', 'auth', 'database'). Returns timestamped excerpts of previous sessions."""
    results = search_past_conversations_in_db(
        query=query,
        exclude_thread_id=get_active_thread_id(),
        limit=limit,
    )
    return format_past_conversations_context(results)


BUILTIN_TOOLS: list[Any] = [search_past_conversations]


