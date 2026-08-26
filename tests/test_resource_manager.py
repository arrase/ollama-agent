from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ollama_agent.core.resource_manager import BaseFileStoreManager


class DummyFileManager(BaseFileStoreManager[str]):
    _ext = ".txt"

    def get(self, item_id: str) -> str:
        path = self._path(item_id)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path.read_text()

    def find_matches(self, prefix: str) -> list[tuple[str, str]]:
        return []

    def list_all(self) -> list[tuple[str, str]]:
        return []

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


if __name__ == "__main__":
    unittest.main()
