from __future__ import annotations

import unittest

from ollama_agent.streaming.parsers import streaming_reasoning, streaming_text


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


if __name__ == "__main__":
    unittest.main()
