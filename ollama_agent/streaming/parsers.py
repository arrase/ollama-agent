"""Streaming chunk parsers for LangChain / DeepAgents events.

These functions extract structured text from the raw streaming payloads
produced by LangChain chat models and DeepAgents, keeping
:mod:`~ollama_agent.agent.agent` focused solely on agent initialisation and
workflow orchestration.
"""

from __future__ import annotations

from typing import Any


def streaming_text(content: Any) -> str:
    """Extract text from a streaming chunk without altering whitespace."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content["text"] if content.get("type") == "text" and "text" in content else ""
    if isinstance(content, list):
        return "".join(
            b["text"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "text" and "text" in b
        )
    return ""


def streaming_reasoning(content: Any, additional_kwargs: dict[str, Any] | None = None) -> str:
    """Extract reasoning/thinking text from a streaming chunk.

    Supports both OpenAI-style reasoning blocks and ChatOllama's
    ``additional_kwargs['reasoning_content']``.
    """
    if isinstance(additional_kwargs, dict):
        reasoning_content = additional_kwargs.get("reasoning_content")
        if isinstance(reasoning_content, str):
            return reasoning_content

    if not isinstance(content, list):
        return ""
    return "".join(
        entry["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "reasoning"
        for entry in (block.get("summary") or ())
        if isinstance(entry, dict) and entry.get("type") == "summary_text" and "text" in entry
    )
