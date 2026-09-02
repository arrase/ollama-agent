from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ollama_agent.core.common import (
    ALLOWED_REASONING_EFFORTS,
    DEFAULT_REASONING_EFFORT,
    atomic_write_text,
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

    def test_extract_text_from_tuple_of_strings(self) -> None:
        self.assertEqual(extract_text(("hello", "world")), "hello world")

    def test_extract_text_none_returns_empty(self) -> None:
        self.assertEqual(extract_text(None), "")

    def test_extract_text_unknown_types_raise_type_error(self) -> None:
        for unknown in (123, 4.5, object()):
            with self.subTest(value=unknown):
                with self.assertRaises(TypeError):
                    extract_text(unknown)

    def test_extract_text_dict_without_text_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            extract_text({"foo": "bar"})

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
            "com0",
            "com1",
            "lpt0",
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

    def test_atomic_write_text_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "test.txt"
            atomic_write_text(target, "hello atomic world")
            self.assertTrue(target.is_file())
            self.assertEqual(target.read_text(encoding="utf-8"), "hello atomic world")

    def test_atomic_write_text_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "deep" / "nested" / "dir" / "test.txt"
            atomic_write_text(target, "nested content")
            self.assertTrue(target.is_file())
            self.assertEqual(target.read_text(encoding="utf-8"), "nested content")

    def test_atomic_write_text_cleans_up_temp_file_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "target.txt"
            with patch("os.replace", side_effect=OSError("Disk write error")):
                with self.assertRaises(OSError):
                    atomic_write_text(target, "doomed content")
            # Target should not exist
            self.assertFalse(target.exists())
            # Parent directory should not have any leftover .tmp files
            tmp_files = list(Path(td).glob("*.tmp"))
            self.assertEqual(tmp_files, [])


if __name__ == "__main__":
    unittest.main()
