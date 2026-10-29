from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ollama_agent.core.prompt_processor import (
    ContextLimitExceededError,
    FileTooLargeError,
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
        assert classify_multimodal_file(Path("photo.png")) == "image"
        assert classify_multimodal_file(Path("video.mp4")) == "video"
        assert classify_multimodal_file(Path("audio.wav")) == "audio"
        assert classify_multimodal_file(Path("doc.pdf")) == "file"
        assert classify_multimodal_file(Path("script.py")) is None

    def test_is_binary_file_detection(self) -> None:
        text_file = self.base_path / "test.txt"
        text_file.write_text("plain text content", encoding="utf-8")
        assert not is_binary_file(text_file)

        bin_file = self.base_path / "test.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03\x00")
        assert is_binary_file(bin_file)

    def test_read_file_content_success(self) -> None:
        file = self.base_path / "sample.py"
        file.write_text("print('hello')", encoding="utf-8")
        assert read_file_content(file) == "print('hello')"

    def test_read_file_content_exceeds_max_size_raises(self) -> None:
        file = self.base_path / "large.txt"
        file.write_text("x" * 200, encoding="utf-8")
        with pytest.raises(PromptProcessingError):
            read_file_content(file, max_file_size=100)

    def test_read_binary_file_b64(self) -> None:
        file = self.base_path / "image.png"
        file.write_bytes(b"\x89PNG\r\n\x1a\n")
        b64 = read_binary_file_b64(file)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_resolve_context_files_text_file(self) -> None:
        file = self.base_path / "notes.md"
        file.write_text("# Title\nNotes here", encoding="utf-8")

        context = resolve_context_files(file)
        assert file in context.text_contents
        assert context.text_contents[file] == "# Title\nNotes here"
        assert context.attachments == []
        assert context.warnings == []

    def test_resolve_context_files_multimodal_file_exposes_classification(self) -> None:
        image = self.base_path / "pic.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")

        context = resolve_context_files(image)
        assert len(context.attachments) == 1
        assert context.attachments[0]["type"] == "image"

    def test_resolve_context_files_neither_file_nor_dir_raises(self) -> None:
        with pytest.raises(PromptProcessingError):
            resolve_context_files(self.base_path / "does_not_exist")

    def test_resolve_context_files_directory_collects_warnings_for_skipped_files(self) -> None:
        project = self.base_path / "project"
        project.mkdir()
        (project / "good.txt").write_text("ok", encoding="utf-8")
        (project / "blob.bin").write_bytes(b"\x00\x01\x02")

        context = resolve_context_files(project)
        assert project / "good.txt" in context.text_contents
        assert context.attachments == []
        assert len(context.warnings) == 1
        assert "blob.bin" in context.warnings[0]

    def test_process_prompt_mentions_with_existing_file(self) -> None:
        file = self.base_path / "app.py"
        file.write_text("def run(): pass", encoding="utf-8")

        prompt = f"Check @{file} for bugs"
        processed, attachments, warnings = process_prompt_mentions(prompt)

        assert "--- Attached Context ---" in processed
        assert "def run(): pass" in processed
        assert len(attachments) == 0
        assert warnings == []

    def test_process_prompt_mentions_with_image_placeholder(self) -> None:
        image = self.base_path / "photo.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")

        prompt = f"Describe @{image} please"
        processed, attachments, warnings = process_prompt_mentions(prompt)

        assert "--- Attached Context ---" not in processed
        assert f"[image: {image}]" in processed
        assert len(attachments) == 1
        assert attachments[0]["type"] == "image"
        assert warnings == []

    def test_process_prompt_mentions_surfaces_directory_warnings(self) -> None:
        project = self.base_path / "proj"
        project.mkdir()
        (project / "main.py").write_text("print('hi')", encoding="utf-8")
        (project / "data.bin").write_bytes(b"\x00\x01")

        prompt = f"Review @{project}"
        processed, attachments, warnings = process_prompt_mentions(prompt)

        assert "print('hi')" in processed
        assert attachments == []
        assert len(warnings) == 1
        assert "data.bin" in warnings[0]

    def test_process_prompt_mentions_with_quoted_nonexistent_file_raises(self) -> None:
        prompt = 'Inspect @"/nonexistent/path/file.py" please'
        with pytest.raises(PromptProcessingError):
            process_prompt_mentions(prompt)

    def test_process_prompt_mentions_plain_prompt_without_mentions(self) -> None:
        prompt = "Hello agent, what is 2 + 2?"
        processed, attachments, _warnings = process_prompt_mentions(prompt)
        assert processed == prompt
        assert attachments == []

    def test_process_prompt_mentions_ignores_decorators_as_literal_text(self) -> None:
        prompt = (
            "def func():\n    @staticmethod\n    @classmethod\n    @property\n"
            "    @app.route('/api')\n    @pytest.mark.asyncio\n    def helper(): pass"
        )
        processed, attachments, _warnings = process_prompt_mentions(prompt)
        assert processed == prompt
        assert attachments == []

    def test_process_prompt_mentions_with_unquoted_missing_relative_file_raises(self) -> None:
        prompt = "Please look at @./nonexistent_file.py"
        with pytest.raises(PromptProcessingError):
            process_prompt_mentions(prompt)

    def test_process_prompt_mentions_with_unquoted_missing_file_with_separator_raises(self) -> None:
        prompt = "Please look at @src/missing"
        with pytest.raises(PromptProcessingError):
            process_prompt_mentions(prompt)

    def test_process_prompt_mentions_trailing_ellipsis_stripped(self) -> None:
        file = self.base_path / "hello.py"
        file.write_text("print('world')", encoding="utf-8")
        prompt = f"Please look at @{file}..."
        processed, _, _ = process_prompt_mentions(prompt)
        assert "print('world')" in processed

    def test_get_file_type_typescript_classified_as_text(self) -> None:
        assert classify_multimodal_file(Path("index.ts")) is None

    def test_process_prompt_mentions_with_file_uri(self) -> None:
        file = self.base_path / "service.py"
        file.write_text("class Service: pass", encoding="utf-8")

        file_uri = file.as_uri()
        prompt = f"Analyze @{file_uri}"
        processed, attachments, _warnings = process_prompt_mentions(prompt)

        assert "--- Attached Context ---" in processed
        assert "class Service: pass" in processed
        assert len(attachments) == 0

    def test_resolve_context_files_total_size_tracking(self) -> None:
        folder = self.base_path / "mixed_folder"
        folder.mkdir()
        txt = folder / "doc.txt"
        txt.write_text("12345", encoding="utf-8")
        img = folder / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        context = resolve_context_files(folder)
        assert context.total_size == txt.stat().st_size + img.stat().st_size

    def test_process_prompt_mentions_tracks_multimodal_directory_cumulative_size(self) -> None:
        folder = self.base_path / "media_folder"
        folder.mkdir()
        img = folder / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        other_file = self.base_path / "extra.txt"
        other_file.write_text("extra", encoding="utf-8")

        prompt = f"Check @{folder} and @{other_file}"
        img_size = img.stat().st_size
        extra_size = other_file.stat().st_size

        with pytest.raises(ContextLimitExceededError):
            process_prompt_mentions(prompt, max_total_size=img_size + extra_size - 1)

    def test_process_prompt_mentions_allow_traversal_false_blocks_external_file(self) -> None:
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_file = Path(outside_dir) / "secret.txt"
            outside_file.write_text("confidential", encoding="utf-8")

            with pytest.raises(PromptProcessingError) as ctx:
                process_prompt_mentions(
                    f"Check @{outside_file}",
                    allow_traversal=False,
                    base_dir=self.base_path,
                )
            assert "Access to path outside working directory is not allowed" in str(ctx.value)

    def test_process_prompt_mentions_allow_traversal_false_allows_internal_file(self) -> None:
        inside_file = self.base_path / "allowed.txt"
        inside_file.write_text("allowed content", encoding="utf-8")

        processed, attachments, warnings = process_prompt_mentions(
            f"Check @{inside_file}",
            allow_traversal=False,
            base_dir=self.base_path,
        )
        assert "allowed content" in processed
        assert attachments == []
        assert warnings == []

    def test_process_prompt_mentions_allow_traversal_true_allows_external_file(self) -> None:
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_file = Path(outside_dir) / "external.txt"
            outside_file.write_text("external content", encoding="utf-8")

            processed, attachments, warnings = process_prompt_mentions(
                f"Check @{outside_file}",
                allow_traversal=True,
                base_dir=self.base_path,
            )
            assert "external content" in processed
            assert attachments == []
            assert warnings == []

    def test_process_prompt_mentions_allow_traversal_false_blocks_relative_traversal(self) -> None:
        with pytest.raises(PromptProcessingError) as ctx:
            process_prompt_mentions(
                'Check @"../outside_escape.txt"',
                allow_traversal=False,
                base_dir=self.base_path,
            )
        assert "Access to path outside working directory is not allowed" in str(ctx.value)

    def test_process_prompt_mentions_allow_traversal_false_default_base_dir_blocks_outside_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_file = Path(outside_dir) / "host_secret.txt"
            outside_file.write_text("sensitive", encoding="utf-8")

            with pytest.raises(PromptProcessingError) as ctx:
                process_prompt_mentions(
                    f"Check @{outside_file}",
                    allow_traversal=False,
                )
            assert "Access to path outside working directory is not allowed" in str(ctx.value)

    def test_process_prompt_mentions_allow_traversal_false_blocks_tilde_expansion(self) -> None:
        with pytest.raises(PromptProcessingError) as ctx:
            process_prompt_mentions(
                'Check @"~/secret_user_file.txt"',
                allow_traversal=False,
                base_dir=self.base_path,
            )
        assert "Access to path outside working directory is not allowed" in str(ctx.value)

        with pytest.raises(PromptProcessingError) as ctx:
            process_prompt_mentions(
                "Check @~/secret_user_file.txt",
                allow_traversal=False,
                base_dir=self.base_path,
            )
        assert "Access to path outside working directory is not allowed" in str(ctx.value)

    def test_process_prompt_mentions_allow_traversal_false_blocks_symlink_pointing_outside(self) -> None:
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_file = Path(outside_dir) / "outside_target.txt"
            outside_file.write_text("external secret", encoding="utf-8")
            symlink = self.base_path / "symlink_escape.txt"
            symlink.symlink_to(outside_file)

            with pytest.raises(PromptProcessingError) as ctx:
                process_prompt_mentions(
                    f"Check @{symlink}",
                    allow_traversal=False,
                    base_dir=self.base_path,
                )
            assert "Access to path outside working directory is not allowed" in str(ctx.value)

    def test_process_prompt_mentions_preserves_parent_and_current_dir_mentions(self) -> None:
        sub = self.base_path / "sub"
        sub.mkdir()
        (self.base_path / "root.txt").write_text("root file", encoding="utf-8")

        prompt = f"Check @{sub}/.. please"
        processed, _, warnings = process_prompt_mentions(prompt, base_dir=self.base_path)
        assert "root file" in processed
        assert warnings == []

        prompt_colon = f"Check @{sub}/..: please"
        processed_colon, _, _ = process_prompt_mentions(prompt_colon, base_dir=self.base_path)
        assert "root file" in processed_colon

        (sub / "subfile.txt").write_text("sub file", encoding="utf-8")
        prompt_dot = f"Check @{sub}/. please"
        processed_dot, _, _ = process_prompt_mentions(prompt_dot, base_dir=self.base_path)
        assert "sub file" in processed_dot

    def test_resolve_context_files_raises_file_too_large_before_total_size(self) -> None:
        file = self.base_path / "big.txt"
        file.write_text("x" * 500, encoding="utf-8")

        with pytest.raises(FileTooLargeError):
            resolve_context_files(file, max_file_size=100, max_total_size=200)

    def test_resolve_context_files_attachment_concrete_mime_types(self) -> None:
        pdf = self.base_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.5 test")
        png = self.base_path / "image.png"
        png.write_bytes(b"\x89PNG test")

        ctx_pdf = resolve_context_files(pdf)
        assert ctx_pdf.attachments[0]["mime_type"] == "application/pdf"

        ctx_png = resolve_context_files(png)
        assert ctx_png.attachments[0]["mime_type"] == "image/png"

    def test_resolve_context_files_attachment_mime_fallback(self) -> None:
        png = self.base_path / "image.png"
        png.write_bytes(b"\x89PNG test")
        with patch("ollama_agent.core.prompt_processor.get_file_type", return_value=None):
            ctx = resolve_context_files(png)
            assert ctx.attachments[0]["mime_type"] == "image/png"

    def test_resolve_context_files_directory_handles_unreadable_file(self) -> None:
        folder = self.base_path / "unreadable_folder"
        folder.mkdir()
        ok_file = folder / "ok.txt"
        ok_file.write_text("ok content", encoding="utf-8")
        bad_file = folder / "bad.txt"
        bad_file.write_text("bad content", encoding="utf-8")

        original_read_text = Path.read_text

        def mock_read_text(self_path: Path, *args: Any, **kwargs: Any) -> str:
            if self_path == bad_file:
                raise OSError("Simulated I/O error")
            return original_read_text(self_path, *args, **kwargs)

        with patch.object(Path, "read_text", side_effect=mock_read_text, autospec=True):
            context = resolve_context_files(folder)
            assert ok_file in context.text_contents
            assert bad_file not in context.text_contents
            assert any("Simulated I/O error" in w for w in context.warnings)

    def test_process_prompt_mentions_skips_symlink_outside_base_dir_inside_directory(self) -> None:
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_file = Path(outside_dir) / "outside_secret.txt"
            outside_file.write_text("secret data", encoding="utf-8")

            sub = self.base_path / "subdir"
            sub.mkdir()
            (sub / "local.txt").write_text("local data", encoding="utf-8")
            (sub / "escape_link.txt").symlink_to(outside_file)

            processed, attachments, warnings = process_prompt_mentions(
                f"Check @{sub}",
                allow_traversal=False,
                base_dir=self.base_path,
            )
            assert "local data" in processed
            assert "secret data" not in processed
            assert attachments == []
            assert len(warnings) == 1
            assert "Access to path outside working directory is not allowed" in warnings[0]

    def test_process_prompt_mentions_deduplicates_file_and_directory_mentions(self) -> None:
        folder = self.base_path / "dup_folder"
        folder.mkdir()
        img = folder / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        processed, attachments, warnings = process_prompt_mentions(
            f"Check @{img} and @{folder}",
            max_total_size=img.stat().st_size,
            base_dir=self.base_path,
        )
        assert f"[image: {img}]" in processed
        assert len(attachments) == 1
        assert warnings == []

        processed2, attachments2, warnings2 = process_prompt_mentions(
            f"Check @{folder} and @{img}",
            max_total_size=img.stat().st_size,
            base_dir=self.base_path,
        )
        assert f"[image: {img}]" in processed2
        assert len(attachments2) == 1
        assert warnings2 == []


if __name__ == "__main__":
    unittest.main()
