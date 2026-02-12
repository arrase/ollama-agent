"""Built-in tools and shared runtime state."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from langchain.tools import tool

from ..core import Mem0ToolResult, RAGToolResult
from ..memory import Mem0InitializationError
from ..rag import RAGError, RAGNotLoadedError

if TYPE_CHECKING:
    from ..memory import MemoryManager
    from ..rag import RAGManager


_tool_timeout: ContextVar[int] = ContextVar("tool_timeout", default=30)
_memory_manager: ContextVar["MemoryManager | None"] = ContextVar("memory_manager", default=None)
_rag_manager: ContextVar["RAGManager | None"] = ContextVar("rag_manager", default=None)

set_tool_timeout, get_tool_timeout = _tool_timeout.set, _tool_timeout.get
set_memory_manager, get_memory_manager = _memory_manager.set, _memory_manager.get
set_rag_manager, get_rag_manager = _rag_manager.set, _rag_manager.get


def _mem0_call(fn: str, *args: Any, **kwargs: Any) -> Mem0ToolResult:
    if not (mgr := get_memory_manager()):
        return {"success": False, "error": "Mem0 not initialized"}
    try:
        return {"success": True, "data": getattr(mgr, fn)(*args, **kwargs)}
    except (Mem0InitializationError, Exception) as exc:
        kind = "init failed" if isinstance(exc, Mem0InitializationError) else "error"
        return {"success": False, "error": f"Mem0 {kind}: {exc}"}


@tool
def mem0_add_memory(memory: str) -> Mem0ToolResult:
    """Persist a new memory for the active user."""
    return _mem0_call("add", memory)


@tool
def mem0_search_memory(query: str, limit: int | None = None) -> Mem0ToolResult:
    """Search stored memories relevant to the query."""
    return _mem0_call("search", query, limit=limit)


@tool
def rag_search(query: str, top_k: int | None = None) -> RAGToolResult:
    """Search the loaded RAG database for relevant document chunks."""
    if not (mgr := get_rag_manager()):
        return {"success": False, "error": "RAG manager not initialized"}
    if mgr.current_database is None:
        return {
            "success": False,
            "error": "No RAG database loaded. Use /rag-load <name> first.",
        }
    try:
        results = mgr.search(query, top_k)
        context_parts: list[str] = []
        for r in results:
            source = r.get("filename", r.get("source", "unknown"))
            context_parts.append(f"[Source: {source}]\n{r['content']}")
        context = "\n\n---\n\n".join(context_parts) if context_parts else ""
        return {"success": True, "context": context, "results": results}
    except (RAGNotLoadedError, RAGError) as exc:
        return {"success": False, "error": str(exc)}


BUILTIN_TOOLS = [mem0_add_memory, mem0_search_memory, rag_search]
