from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ollama_agent.core.resource_manager import (
    BaseFileStoreManager,
    require_text,
    resolve_unique_match,
)


class ItemNotFoundError(Exception):
    pass


class AmbiguousItemError(Exception):
    pass


class ItemValidationError(Exception):
    pass


class DummyFileManager(BaseFileStoreManager[str]):
    _ext = ".txt"
    items: dict[str, str]

    def __init__(self, base_dir: Path) -> None:
        super().__init__(base_dir)
        self.items = {}

    def get(self, item_id: str) -> str:
        path = self._path(item_id)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path.read_text()

    def find_matches(self, prefix: str) -> list[tuple[str, str]]:
        if prefix == "invalid!":
            raise ValueError("Invalid prefix")
        return [(k, v) for k, v in self.items.items() if k.startswith(prefix)]

    def list_all(self) -> list[tuple[str, str]]:
        return list(self.items.items())

    def delete(self, item_id: str) -> None:
        self._path(item_id).unlink()


class TestResourceManager(unittest.TestCase):
    """Unit tests for BaseFileStoreManager."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.mgr = DummyFileManager(self.base_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_base_dir_created_automatically(self) -> None:
        sub = self.base_path / "nested" / "dir"
        DummyFileManager(sub)
        self.assertTrue(sub.is_dir())

    def test_path_resolution_within_base_dir(self) -> None:
        resolved = self.mgr._path("test_item")
        self.assertEqual(resolved, (self.base_path / "test_item.txt").resolve())

    def test_path_traversal_detection_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.mgr._path("../outside_item")

    def test_require_text_valid(self) -> None:
        self.assertEqual(require_text("  hello world  ", "Field", ValueError), "hello world")

    def test_require_text_empty_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            require_text("   ", "Name", ValueError)
        self.assertIn("Name cannot be empty", str(cm.exception))

    def test_resolve_unique_match_success(self) -> None:
        match = resolve_unique_match(
            [("id1", "val1")],
            "id1",
            label="Item",
            not_found_error=ItemNotFoundError,
            ambiguous_error=AmbiguousItemError,
        )
        self.assertEqual(match, ("id1", "val1"))

    def test_resolve_unique_match_not_found_raises(self) -> None:
        with self.assertRaises(ItemNotFoundError):
            resolve_unique_match(
                [],
                "missing",
                label="Item",
                not_found_error=ItemNotFoundError,
                ambiguous_error=AmbiguousItemError,
            )

    def test_resolve_unique_match_ambiguous_raises(self) -> None:
        with self.assertRaises(AmbiguousItemError):
            resolve_unique_match(
                [("id1", "v1"), ("id2", "v2")],
                "id",
                label="Item",
                not_found_error=ItemNotFoundError,
                ambiguous_error=AmbiguousItemError,
            )

    def test_manager_resolve_success(self) -> None:
        self.mgr.items["tool-a"] = "alpha"
        res = self.mgr.resolve(
            "tool-a",
            label="Item",
            not_found_error=ItemNotFoundError,
            ambiguous_error=AmbiguousItemError,
            validation_error=ItemValidationError,
        )
        self.assertEqual(res, ("tool-a", "alpha"))

    def test_manager_resolve_not_found_raises(self) -> None:
        with self.assertRaises(ItemNotFoundError):
            self.mgr.resolve(
                "missing",
                label="Item",
                not_found_error=ItemNotFoundError,
                ambiguous_error=AmbiguousItemError,
                validation_error=ItemValidationError,
            )

    def test_manager_resolve_ambiguous_raises(self) -> None:
        self.mgr.items["tool-a"] = "alpha"
        self.mgr.items["tool-b"] = "beta"
        with self.assertRaises(AmbiguousItemError):
            self.mgr.resolve(
                "tool",
                label="Item",
                not_found_error=ItemNotFoundError,
                ambiguous_error=AmbiguousItemError,
                validation_error=ItemValidationError,
            )

    def test_manager_resolve_validation_error_raises(self) -> None:
        with self.assertRaises(ItemValidationError):
            self.mgr.resolve(
                "invalid!",
                label="Item",
                not_found_error=ItemNotFoundError,
                ambiguous_error=AmbiguousItemError,
                validation_error=ItemValidationError,
            )


if __name__ == "__main__":
    unittest.main()
