"""Agent submodule for AI agent functionality."""

from .agent import OllamaAgent
from .builtin_tools import (
    BUILTIN_TOOLS,
    get_rag_manager,
    get_tool_timeout,
    mem0_add_memory,
    mem0_search_memory,
    rag_search,
    set_rag_manager,
    set_tool_timeout,
)
from .factory import create_agent
from .session_manager import SessionManager

__all__ = [
    "BUILTIN_TOOLS",
    "OllamaAgent",
    "SessionManager",
    "create_agent",
    "get_rag_manager",
    "get_tool_timeout",
    "mem0_add_memory",
    "mem0_search_memory",
    "rag_search",
    "set_rag_manager",
    "set_tool_timeout",
]
