from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk

from ollama_agent.core.parsers import (
    ThinkTagParser,
    streaming_reasoning,
    streaming_text,
)


class TestStreamingParsers(unittest.TestCase):
    """Unit tests for streaming chunk text and reasoning parsers."""

    def test_streaming_text_string(self) -> None:
        assert streaming_text("hello") == "hello"
        assert streaming_text("") == ""

    def test_streaming_text_list_payload(self) -> None:
        payload = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "World!"},
            {"type": "image", "data": "..."},
        ]
        assert streaming_text(payload) == "Hello World!"

    def test_streaming_reasoning_from_additional_kwargs(self) -> None:
        kwargs = {"reasoning_content": "Thinking step by step"}
        assert streaming_reasoning(kwargs) == "Thinking step by step"

    def test_streaming_reasoning_empty(self) -> None:
        assert streaming_reasoning({}) == ""

    def test_streaming_text_edge_cases(self) -> None:
        assert streaming_text(None) == ""
        assert streaming_text(123) == ""
        assert streaming_text(["foo", "bar"]) == ""
        assert streaming_text([{"text": "foo"}]) == ""
        assert streaming_text([{"type": "text"}]) == ""

    def test_streaming_reasoning_edge_cases(self) -> None:
        assert streaming_reasoning({"reasoning_content": 123}) == ""
        assert streaming_reasoning({"reasoning_content": "step 1"}) == "step 1"

    def test_think_tag_parser_chunk_splits(self) -> None:
        parser = ThinkTagParser()
        deltas1 = parser.feed("Hello <th")
        assert deltas1 == [("text", "Hello ")]

        deltas2 = parser.feed("ink>I am thinking...</th")
        assert deltas2 == [("reasoning", "I am thinking...")]

        deltas3 = parser.feed("ink>Result: 42")
        assert deltas3 == [("text", "Result: 42")]

    def test_think_tag_parser_non_tag_brackets(self) -> None:
        parser = ThinkTagParser()
        deltas1 = parser.feed("If x <")
        assert deltas1 == [("text", "If x ")]

        deltas2 = parser.feed(" 5 then True")
        assert deltas2 == [("text", "< 5 then True")]

    def test_think_tag_parser_process_chunk(self) -> None:
        parser = ThinkTagParser()
        chunk1 = MagicMock(type="ai", content="<think>Deep thought</think>Answer", additional_kwargs={})
        events = parser.process_chunk(chunk1, hide_reasoning=False)
        assert events == [
            {"type": "reasoning_delta", "content": "Deep thought"},
            {"type": "text_delta", "content": "Answer"},
        ]

    def test_think_tag_parser_hide_reasoning(self) -> None:
        parser = ThinkTagParser()
        chunk = MagicMock(type="ai", content="<think>Secret</think>Public", additional_kwargs={})
        events = parser.process_chunk(chunk, hide_reasoning=True)
        assert events == [{"type": "text_delta", "content": "Public"}]

    def test_thinking_and_content_in_one_chunk(self) -> None:
        parser = ThinkTagParser()
        chunk = AIMessageChunk(
            content="answer",
            additional_kwargs={"reasoning_content": "thought"},
        )
        events = parser.process_chunk(chunk, hide_reasoning=False)
        assert events == [
            {"type": "reasoning_delta", "content": "thought"},
            {"type": "text_delta", "content": "answer"},
        ]

    def test_hide_reasoning_keeps_text(self) -> None:
        parser = ThinkTagParser()
        chunk = AIMessageChunk(
            content="visible",
            additional_kwargs={"reasoning_content": "secret"},
        )
        events = parser.process_chunk(chunk, hide_reasoning=True)
        assert events == [{"type": "text_delta", "content": "visible"}]

    def test_hidden_reasoning_does_not_desync_the_think_buffer(self) -> None:
        parser = ThinkTagParser()
        c1 = AIMessageChunk(content="<think>secret </th")
        c2 = AIMessageChunk(content="ink>visible")
        assert parser.process_chunk(c1, hide_reasoning=True) == []
        assert parser.process_chunk(c2, hide_reasoning=True) == [{"type": "text_delta", "content": "visible"}]

        parser2 = ThinkTagParser()
        assert parser2.process_chunk(AIMessageChunk(content="<think>trailing </th"), hide_reasoning=True) == []
        assert parser2.flush(hide_reasoning=True) == []

    def test_think_tag_parser_flush(self) -> None:
        parser = ThinkTagParser()
        parser.feed("Trailing text <th")
        flushed = parser.flush(hide_reasoning=False)
        assert flushed == [{"type": "text_delta", "content": "<th"}]

    def test_think_tag_parser_process_chunk_empty_text(self) -> None:
        parser = ThinkTagParser()
        chunk_empty = MagicMock(type="ai", content="", additional_kwargs={})
        assert parser.process_chunk(chunk_empty, hide_reasoning=False) == []

        chunk_tool = MagicMock(type="tool", content="call", additional_kwargs={})
        assert parser.process_chunk(chunk_tool, hide_reasoning=False) == []


if __name__ == "__main__":
    unittest.main()
