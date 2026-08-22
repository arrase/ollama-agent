from __future__ import annotations

import unittest
from types import SimpleNamespace

from ollama_agent.core.common import (
    ALLOWED_REASONING_EFFORTS,
    DEFAULT_REASONING_EFFORT,
    extract_text,
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
