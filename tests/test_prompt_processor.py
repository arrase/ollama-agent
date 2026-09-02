from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ollama_agent.core.prompt_processor import (
    PromptProcessingError,
    classify_multimodal_file,
    is_binary_file,
    process_prompt_mentions,
    read_binary_file_b64,
    read_file_content,
    resolve_context_files,
)


class TestPromptProcessor(unittest.TestCase):
    """Unit tests for @-mention parsing and file resolution logic."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_get_file_type_and_multimodal_classification(self) -> None:
        self.assertEqual(classify_multimodal_file(Path("photo.png")), "image")
        self.assertEqual(classify_multimodal_file(Path("video.mp4")), "video")
        self.assertEqual(classify_multimodal_file(Path("audio.wav")), "audio")
        self.assertEqual(classify_multimodal_file(Path("doc.pdf")), "file")
        self.assertIsNone(classify_multimodal_file(Path("script.py")))

    def test_is_binary_file_detection(self) -> None:
        text_file = self.base_path / "test.txt"
        text_file.write_text("plain text content", encoding="utf-8")
        self.assertFalse(is_binary_file(text_file))

        bin_file = self.base_path / "test.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03\x00")
        self.assertTrue(is_binary_file(bin_file))

    def test_read_file_content_success(self) -> None:
        file = self.base_path / "sample.py"
        file.write_text("print('hello')", encoding="utf-8")
        self.assertEqual(read_file_content(file), "print('hello')")

    def test_read_file_content_exceeds_max_size_raises(self) -> None:
        file = self.base_path / "large.txt"
        file.write_text("x" * 200, encoding="utf-8")
        with self.assertRaises(PromptProcessingError):
            read_file_content(file, max_file_size=100)

    def test_read_binary_file_b64(self) -> None:
        file = self.base_path / "image.png"
        file.write_bytes(b"\x89PNG\r\n\x1a\n")
        b64 = read_binary_file_b64(file)
        self.assertIsInstance(b64, str)
        self.assertTrue(len(b64) > 0)

    def test_resolve_context_files_text_file(self) -> None:
        file = self.base_path / "notes.md"
        file.write_text("# Title\nNotes here", encoding="utf-8")

        context = resolve_context_files(file)
        self.assertIn(file, context.text_contents)
        self.assertEqual(context.text_contents[file], "# Title\nNotes here")
        self.assertEqual(context.attachments, [])
        self.assertEqual(context.classifications, {})
        self.assertEqual(context.warnings, [])

    def test_resolve_context_files_multimodal_file_exposes_classification(self) -> None:
        image = self.base_path / "pic.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")

        context = resolve_context_files(image)
        self.assertEqual(context.classifications, {image: "image"})
        self.assertEqual(len(context.attachments), 1)

    def test_resolve_context_files_neither_file_nor_dir_raises(self) -> None:
        with self.assertRaises(PromptProcessingError):
            resolve_context_files(self.base_path / "does_not_exist")

    def test_resolve_context_files_directory_collects_warnings_for_skipped_files(self) -> None:
        project = self.base_path / "project"
        project.mkdir()
        (project / "good.txt").write_text("ok", encoding="utf-8")
        (project / "blob.bin").write_bytes(b"\x00\x01\x02")

        context = resolve_context_files(project)
        self.assertIn(project / "good.txt", context.text_contents)
        self.assertEqual(context.attachments, [])
        self.assertEqual(len(context.warnings), 1)
        self.assertIn("blob.bin", context.warnings[0])

    def test_process_prompt_mentions_with_existing_file(self) -> None:
        file = self.base_path / "app.py"
        file.write_text("def run(): pass", encoding="utf-8")

        prompt = f"Check @{file} for bugs"
        processed, attachments, warnings = process_prompt_mentions(prompt)

        self.assertIn("--- Attached Context ---", processed)
        self.assertIn("def run(): pass", processed)
        self.assertEqual(len(attachments), 0)
        self.assertEqual(warnings, [])

    def test_process_prompt_mentions_with_image_placeholder(self) -> None:
        image = self.base_path / "photo.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")

        prompt = f"Describe @{image} please"
        processed, attachments, warnings = process_prompt_mentions(prompt)

        self.assertNotIn("--- Attached Context ---", processed)
        self.assertIn(f"[image: {image}]", processed)
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0]["type"], "image")
        self.assertEqual(warnings, [])

    def test_process_prompt_mentions_surfaces_directory_warnings(self) -> None:
        project = self.base_path / "proj"
        project.mkdir()
        (project / "main.py").write_text("print('hi')", encoding="utf-8")
        (project / "data.bin").write_bytes(b"\x00\x01")

        prompt = f"Review @{project}"
        processed, attachments, warnings = process_prompt_mentions(prompt)

        self.assertIn("print('hi')", processed)
        self.assertEqual(attachments, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("data.bin", warnings[0])

    def test_process_prompt_mentions_with_quoted_nonexistent_file_raises(self) -> None:
        prompt = 'Inspect @"/nonexistent/path/file.py" please'
        with self.assertRaises(PromptProcessingError):
            process_prompt_mentions(prompt)

    def test_process_prompt_mentions_plain_prompt_without_mentions(self) -> None:
        prompt = "Hello agent, what is 2 + 2?"
        processed, attachments, warnings = process_prompt_mentions(prompt)
        self.assertEqual(processed, prompt)
        self.assertEqual(attachments, [])

    def test_process_prompt_mentions_ignores_decorators_as_literal_text(self) -> None:
        prompt = "def func():\n    @staticmethod\n    @classmethod\n    @property\n    @app.route('/api')\n    @pytest.mark.asyncio\n    def helper(): pass"
        processed, attachments, warnings = process_prompt_mentions(prompt)
        self.assertEqual(processed, prompt)
        self.assertEqual(attachments, [])

    def test_process_prompt_mentions_with_unquoted_missing_relative_file_raises(self) -> None:
        prompt = "Please look at @./nonexistent_file.py"
        with self.assertRaises(PromptProcessingError):
            process_prompt_mentions(prompt)

    def test_process_prompt_mentions_with_unquoted_missing_file_with_separator_raises(self) -> None:
        prompt = "Please look at @src/missing"
        with self.assertRaises(PromptProcessingError):
            process_prompt_mentions(prompt)

    def test_process_prompt_mentions_trailing_ellipsis_stripped(self) -> None:
        file = self.base_path / "hello.py"
        file.write_text("print('world')", encoding="utf-8")
        prompt = f"Please look at @{file}..."
        processed, _, _ = process_prompt_mentions(prompt)
        self.assertIn("print('world')", processed)

    def test_get_file_type_typescript_classified_as_text(self) -> None:
        self.assertEqual(classify_multimodal_file(Path("index.ts")), None)

    def test_process_prompt_mentions_with_file_uri(self) -> None:
        file = self.base_path / "service.py"
        file.write_text("class Service: pass", encoding="utf-8")

        file_uri = file.as_uri()
        prompt = f"Analyze @{file_uri}"
        processed, attachments, warnings = process_prompt_mentions(prompt)

        self.assertIn("--- Attached Context ---", processed)
        self.assertIn("class Service: pass", processed)
        self.assertEqual(len(attachments), 0)


if __name__ == "__main__":
    unittest.main()
