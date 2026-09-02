"""Streaming chunk parsers for LangChain / DeepAgents events.

These functions extract structured text from the raw streaming payloads
produced by LangChain chat models and DeepAgents, keeping
:mod:`~ollama_agent.agent.agent` focused solely on agent initialisation and
workflow orchestration.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessageChunk


def streaming_text(content: Any) -> str:
    """Extract text from a streaming chunk without altering whitespace."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content["text"] if content.get("type") == "text" else ""
    if isinstance(content, list):
        return "".join(b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def streaming_reasoning(content: Any, additional_kwargs: dict[str, Any] | None = None) -> str:
    """Extract reasoning/thinking text from a streaming chunk.

    Supports both OpenAI-style reasoning blocks and ChatOllama's
    ``additional_kwargs['reasoning_content']``.
    """
    if additional_kwargs:
        reasoning_content = additional_kwargs.get("reasoning_content")
        if isinstance(reasoning_content, str):
            return reasoning_content

    if not isinstance(content, list):
        return ""
    return "".join(
        entry["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "reasoning"
        for entry in block.get("summary", ())
        if isinstance(entry, dict) and entry.get("type") == "summary_text"
    )


class ThinkTagParser:
    """Stateful parser for tracking <think> and </think> boundaries across streaming chunks."""

    def __init__(self) -> None:
        self.in_think: bool = False
        self._buffer: str = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        """Feed text into parser and return list of (kind, delta) tuples.

        kind is either 'text' or 'reasoning'.
        """
        combined = self._buffer + text
        self._buffer = ""
        deltas: list[tuple[str, str]] = []
        accumulated: list[str] = []
        i = 0
        n = len(combined)

        while i < n:
            tag = "</think>" if self.in_think else "<think>"
            kind = "reasoning" if self.in_think else "text"

            if combined.startswith(tag, i):
                if accumulated:
                    deltas.append((kind, "".join(accumulated)))
                    accumulated = []
                self.in_think = not self.in_think
                i += len(tag)
                continue

            rem = combined[i:]
            if tag.startswith(rem):
                self._buffer = rem
                break

            accumulated.append(combined[i])
            i += 1

        if accumulated:
            kind = "reasoning" if self.in_think else "text"
            deltas.append((kind, "".join(accumulated)))

        return deltas

    def flush(self, hide_reasoning: bool = False) -> list[dict[str, Any]]:
        """Flush any pending buffer as a final delta."""
        if not self._buffer:
            return []
        buf = self._buffer
        self._buffer = ""
        if self.in_think:
            if hide_reasoning:
                return []
            return [{"type": "reasoning_delta", "content": buf}]
        return [{"type": "text_delta", "content": buf}]

    def process_chunk(
        self,
        chunk: AIMessageChunk,
        hide_reasoning: bool = False,
    ) -> list[dict[str, Any]]:
        """Process a chunk and return reasoning or text delta events."""
        if chunk.type in ("tool", "ToolMessageChunk"):
            return []

        content = chunk.content
        reasoning = streaming_reasoning(content, getattr(chunk, "additional_kwargs", None))
        if reasoning:
            return [] if hide_reasoning else [{"type": "reasoning_delta", "content": reasoning}]

        text = streaming_text(content)
        if not text:
            return []

        return [
            {"type": f"{kind}_delta", "content": delta}
            for kind, delta in self.feed(text)
            if not (kind == "reasoning" and hide_reasoning)
        ]
