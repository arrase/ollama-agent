from __future__ import annotations

import unittest
from types import SimpleNamespace

from ollama_agent.core.common import (
    ALLOWED_REASONING_EFFORTS,
    DEFAULT_REASONING_EFFORT,
    assistant_text_from_messages,
    extract_text,
    final_text_from_state,
    validate_identifier,
)


class TestCommonUtilities(unittest.TestCase):
    """Unit tests for shared core utility functions."""

    def test_extract_text_from_string(self) -> None:
        self.assertEqual(extract_text("hello world"), "hello world")

    def test_extract_text_from_list_of_strings(self) -> None:
        self.assertEqual(extract_text(["hello", "world"]), "hello world")

    def test_extract_text_from_nested_dicts(self) -> None:
        payload = [{"text": "part 1"}, {"content": "part 2"}]
        self.assertEqual(extract_text(payload), "part 1 part 2")

    def test_extract_text_empty_and_unknown_types(self) -> None:
        self.assertEqual(extract_text(123), "")
        self.assertEqual(extract_text(None), "")

    def test_assistant_text_from_messages_finds_latest_ai_message(self) -> None:
        messages = [
            SimpleNamespace(type="human", content="hello"),
            SimpleNamespace(type="ai", content="first response"),
            SimpleNamespace(type="human", content="next question"),
            SimpleNamespace(type="ai", content="latest response"),
        ]
        self.assertEqual(assistant_text_from_messages(messages), "latest response")

    def test_assistant_text_from_messages_returns_empty_when_no_ai_message(self) -> None:
        messages = [
            SimpleNamespace(type="human", content="hello"),
        ]
        self.assertEqual(assistant_text_from_messages(messages), "")

    def test_final_text_from_state_with_assistant_message(self) -> None:
        state = {
            "messages": [
                SimpleNamespace(type="human", content="hello"),
                SimpleNamespace(type="ai", content="agent answer"),
            ]
        }
        self.assertEqual(final_text_from_state(state), "agent answer")

    def test_final_text_from_state_without_messages(self) -> None:
        state = {"raw_output": "data"}
        self.assertEqual(final_text_from_state(state), "{'raw_output': 'data'}")

    def test_validate_identifier_valid_names(self) -> None:
        self.assertEqual(validate_identifier("valid_name-123"), "valid_name-123")
        self.assertEqual(validate_identifier("  spaced_name  "), "spaced_name")

    def test_validate_identifier_invalid_names_raise_value_error(self) -> None:
        invalid_cases = [
            "",
            "   ",
            "name with spaces",
            "name/slash",
            "name.dot",
            "name@special",
            "con",
            "CON",
            "nul",
            "aux",
            "prn",
            "com1",
            "lpt9",
        ]
        for invalid_name in invalid_cases:
            with self.subTest(name=invalid_name):
                with self.assertRaises(ValueError):
                    validate_identifier(invalid_name)

    def test_allowed_reasoning_efforts_contains_defaults(self) -> None:
        self.assertIn(DEFAULT_REASONING_EFFORT, ALLOWED_REASONING_EFFORTS)
        self.assertIn("high", ALLOWED_REASONING_EFFORTS)
        self.assertIn("xhigh", ALLOWED_REASONING_EFFORTS)
        self.assertIn("low", ALLOWED_REASONING_EFFORTS)


if __name__ == "__main__":
    unittest.main()
