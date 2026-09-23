"""Streaming chunk parsers for LangChain events."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessageChunk


def streaming_text(content: Any) -> str:
    """Extract text from a streaming chunk."""
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
    """Extract reasoning text from a streaming chunk."""
    if additional_kwargs and isinstance(additional_kwargs.get("reasoning_content"), str):
        return additional_kwargs["reasoning_content"]
    if not isinstance(content, list):
        return ""
    return "".join(
        entry["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "reasoning"
        for entry in block.get("summary", ())
        if isinstance(entry, dict) and entry.get("type") == "summary_text" and "text" in entry
    )


def _buffer_partial_tag(tag: str, text: str) -> tuple[str, str]:
    """Split text into remaining content and trailing partial tag prefix."""
    for k in range(len(tag) - 1, 0, -1):
        if text.endswith(tag[:k]):
            return text[:-k], tag[:k]
    return text, ""


class ThinkTagParser:
    """Parser tracking <think> and </think> tags in streaming text."""

    def __init__(self) -> None:
        self.in_think: bool = False
        self._buffer: str = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        """Feed incoming text and return parsed segments."""
        text = self._buffer + text
        self._buffer = ""
        deltas: list[tuple[str, str]] = []

        while True:
            tag = "</think>" if self.in_think else "<think>"
            if tag not in text:
                break
            before, _, text = text.partition(tag)
            if before:
                deltas.append(("reasoning" if self.in_think else "text", before))
            self.in_think = not self.in_think

        tag = "</think>" if self.in_think else "<think>"
        text, self._buffer = _buffer_partial_tag(tag, text)

        if text:
            deltas.append(("reasoning" if self.in_think else "text", text))

        return deltas

    def flush(self, hide_reasoning: bool = False) -> list[dict[str, Any]]:
        """Flush pending buffer as delta event."""
        buf, self._buffer = self._buffer, ""
        if buf and not (self.in_think and hide_reasoning):
            kind = "reasoning" if self.in_think else "text"
            return [{"type": f"{kind}_delta", "content": buf}]
        return []

    def process_chunk(self, chunk: AIMessageChunk, hide_reasoning: bool = False) -> list[dict[str, Any]]:
        """Convert incoming chunk into delta events."""
        if chunk.type == "tool":
            return []

        reasoning = streaming_reasoning(chunk.content, chunk.additional_kwargs)
        if reasoning:
            if hide_reasoning:
                return []
            return [{"type": "reasoning_delta", "content": reasoning}]

        text = streaming_text(chunk.content)
        if not text:
            return []

        deltas = self.feed(text)
        return [
            {"type": f"{kind}_delta", "content": delta}
            for kind, delta in deltas
            if not (hide_reasoning and kind == "reasoning")
        ]
