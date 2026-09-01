from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from ollama_agent.streaming.parsers import (
    ThinkTagParser,
    streaming_reasoning,
    streaming_text,
)


class TestStreamingParsers(unittest.TestCase):
    """Unit tests for streaming chunk text and reasoning parsers."""

    def test_streaming_text_string(self) -> None:
        self.assertEqual(streaming_text("hello"), "hello")
        self.assertEqual(streaming_text(""), "")

    def test_streaming_text_dict_payload(self) -> None:
        self.assertEqual(streaming_text({"type": "text", "text": "hello"}), "hello")
        self.assertEqual(streaming_text({"type": "other", "text": "skip"}), "")

    def test_streaming_text_list_payload(self) -> None:
        payload = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "World!"},
            {"type": "image", "data": "..."},
        ]
        self.assertEqual(streaming_text(payload), "Hello World!")

    def test_streaming_reasoning_from_additional_kwargs(self) -> None:
        kwargs = {"reasoning_content": "Thinking step by step"}
        self.assertEqual(streaming_reasoning("", kwargs), "Thinking step by step")

    def test_streaming_reasoning_from_content_list(self) -> None:
        payload = [
            {
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": "Step 1: Analyzed prompt. "},
                    {"type": "summary_text", "text": "Step 2: Generating response."},
                ],
            }
        ]
        self.assertEqual(
            streaming_reasoning(payload),
            "Step 1: Analyzed prompt. Step 2: Generating response.",
        )

    def test_streaming_reasoning_empty(self) -> None:
        self.assertEqual(streaming_reasoning("just plain string"), "")
        self.assertEqual(streaming_reasoning(None), "")

    def test_think_tag_parser_chunk_splits(self) -> None:
        parser = ThinkTagParser()
        deltas1 = parser.feed("Hello <th")
        self.assertEqual(deltas1, [("text", "Hello ")])

        deltas2 = parser.feed("ink>I am thinking...</th")
        self.assertEqual(deltas2, [("reasoning", "I am thinking...")])

        deltas3 = parser.feed("ink>Result: 42")
        self.assertEqual(deltas3, [("text", "Result: 42")])

    def test_think_tag_parser_non_tag_brackets(self) -> None:
        parser = ThinkTagParser()
        deltas1 = parser.feed("If x <")
        self.assertEqual(deltas1, [("text", "If x ")])

        deltas2 = parser.feed(" 5 then True")
        self.assertEqual(deltas2, [("text", "< 5 then True")])

    def test_think_tag_parser_process_chunk(self) -> None:
        parser = ThinkTagParser()
        chunk1 = MagicMock(type="ai", content="<think>Deep thought</think>Answer", additional_kwargs={})
        events = parser.process_chunk(chunk1)
        self.assertEqual(
            events,
            [
                {"type": "reasoning_delta", "content": "Deep thought"},
                {"type": "text_delta", "content": "Answer"},
            ],
        )

    def test_think_tag_parser_hide_reasoning(self) -> None:
        parser = ThinkTagParser()
        chunk = MagicMock(type="ai", content="<think>Secret</think>Public", additional_kwargs={})
        events = parser.process_chunk(chunk, hide_reasoning=True)
        self.assertEqual(events, [{"type": "text_delta", "content": "Public"}])

    def test_think_tag_parser_flush(self) -> None:
        parser = ThinkTagParser()
        parser.feed("Trailing text <th")
        flushed = parser.flush()
        self.assertEqual(flushed, [{"type": "text_delta", "content": "<th"}])


if __name__ == "__main__":
    unittest.main()
