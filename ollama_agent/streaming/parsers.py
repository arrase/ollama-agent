"""Streaming chunk parsers for LangChain / DeepAgents events.

These functions extract structured text from the raw streaming payloads
produced by :class:`~langchain_openai.ChatOpenAI` and DeepAgents, keeping
:mod:`~ollama_agent.agent.agent` focused solely on agent initialisation and
workflow orchestration.
"""

from __future__ import annotations

from typing import Any


def streaming_text(content: Any) -> str:
    """Extract text from a streaming chunk without altering whitespace.

    Handles the three payload shapes that LangChain / Responses-API can
    produce: plain ``str``, a ``dict`` with ``type='text'``, or a ``list``
    of such dicts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "") if content.get("type") == "text" else ""
    if isinstance(content, list):
        return "".join(
            b["text"]
            for b in content
            if isinstance(b, dict)
            and b.get("type") == "text"
            and isinstance(b.get("text"), str)
        )
    return ""


def streaming_reasoning(content: Any) -> str:
    """Extract reasoning/thinking text from a streaming chunk.

    With ``use_responses_api=True`` reasoning tokens arrive as content blocks
    of ``type='reasoning'`` whose text lives inside ``summary`` entries of
    ``type='summary_text'``.
    """
    if not isinstance(content, list):
        return ""
    return "".join(
        entry["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "reasoning"
        for entry in (block.get("summary") or ())
        if isinstance(entry, dict)
        and entry.get("type") == "summary_text"
        and isinstance(entry.get("text"), str)
    )
