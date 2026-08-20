"""Core logic for parsing, resolving, and injecting file context from @-mentions."""

from __future__ import annotations

import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Any


class PromptProcessingError(Exception):
    """Exception raised when prompt processing fails, e.g., referenced file not found."""


_MIME_EXTENSIONS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/m4a",
    ".aac": "audio/aac",
    ".aiff": "audio/aiff",
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".flv": "video/x-flv",
    ".mpg": "video/mpeg",
    ".webm": "video/webm",
    ".wmv": "video/x-ms-wmv",
    ".3gpp": "video/3gpp",
    ".pdf": "application/pdf",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def get_file_type(file_path: Path) -> str:
    """Guess the MIME type of a file using mimetypes, with custom extension fallback."""
    mime, _ = mimetypes.guess_type(str(file_path))
    if mime:
        return mime
    return _MIME_EXTENSIONS.get(file_path.suffix.lower(), "")


def classify_multimodal_file(file_path: Path) -> str | None:
    """Classify a file into a LangChain multimodal type, or None if it should be treated as text."""
    mime = get_file_type(file_path)
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("video/"):
        return "video"
    if mime.startswith("audio/"):
        return "audio"
    if (
        mime == "application/pdf"
        or mime.startswith("application/vnd.ms-powerpoint")
        or mime.startswith("application/vnd.openxmlformats-officedocument.presentationml")
    ):
        return "file"
    return None


def is_binary_file(file_path: Path) -> bool:
    """Check if a file is binary by searching for null bytes in the first block."""
    with file_path.open("rb") as f:
        chunk = f.read(1024)
        return b"\x00" in chunk


def read_file_content(file_path: Path, max_file_size: int = 1024 * 1024) -> str:
    """Read file content as string, ensuring it is a text file and fits in size limit."""
    if not file_path.is_file():
        raise PromptProcessingError(f"Path is not a file: {file_path}")

    try:
        file_size = file_path.stat().st_size
    except OSError as e:
        raise PromptProcessingError(f"Failed to get file stats for {file_path}: {e}") from e

    if file_size > max_file_size:
        raise PromptProcessingError(
            f"File too large: {file_path} ({file_size} bytes, limit is {max_file_size} bytes)"
        )

    if is_binary_file(file_path):
        raise PromptProcessingError(f"Cannot read binary file as text: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as e:
        raise PromptProcessingError(f"Failed to read file {file_path}: {e}") from e


def read_binary_file_b64(file_path: Path, max_file_size: int = 1024 * 1024) -> str:
    """Read a binary file and return its content as a base64 encoded string."""
    try:
        file_size = file_path.stat().st_size
    except OSError as e:
        raise PromptProcessingError(f"Failed to get file stats for {file_path}: {e}") from e

    if file_size > max_file_size:
        raise PromptProcessingError(
            f"File too large: {file_path} ({file_size} bytes, limit is {max_file_size} bytes)"
        )

    try:
        with file_path.open("rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except OSError as e:
        raise PromptProcessingError(f"Failed to read binary file {file_path}: {e}") from e


def resolve_context_files(
    target_path: Path,
    max_file_size: int = 1024 * 1024,
    max_files: int = 100,
    max_total_size: int = 10 * 1024 * 1024,
) -> tuple[dict[Path, str], list[dict[str, Any]]]:
    """Resolve a target file or directory into context data."""
    text_contents: dict[Path, str] = {}
    binary_attachments: list[dict[str, Any]] = []
    total_size = 0

    def add_file(file_path: Path, ignore_errors: bool = False) -> None:
        nonlocal total_size

        try:
            size = file_path.stat().st_size
        except OSError as e:
            if ignore_errors:
                return
            raise PromptProcessingError(f"Failed to get file stats for {file_path}: {e}") from e

        attachment_type = classify_multimodal_file(file_path)
        if attachment_type is None and is_binary_file(file_path):
            if ignore_errors:
                return
            raise PromptProcessingError(f"Cannot read binary file as text: {file_path}")

        if len(text_contents) + len(binary_attachments) >= max_files:
            raise PromptProcessingError(
                f"Mentions limit exceeded: max {max_files} files."
            )

        if total_size + size > max_total_size:
            raise PromptProcessingError(
                f"Total context size limit of {max_total_size} bytes exceeded."
            )

        try:
            if attachment_type is not None:
                mime = get_file_type(file_path) or f"{attachment_type}/*"
                b64_data = read_binary_file_b64(file_path, max_file_size)
                binary_attachments.append({
                    "type": attachment_type,
                    "base64": b64_data,
                    "mime_type": mime,
                })
            else:
                content = read_file_content(file_path, max_file_size)
                text_contents[file_path] = content
            total_size += size
        except PromptProcessingError:
            if ignore_errors:
                return
            raise
        except OSError as e:
            if ignore_errors:
                return
            raise PromptProcessingError(f"Failed to add file {file_path}: {e}") from e

    if target_path.is_file():
        add_file(target_path, ignore_errors=False)
        return text_contents, binary_attachments

    if not target_path.is_dir():
        return text_contents, binary_attachments

    for root, _, files in os.walk(target_path):
        for file_name in files:
            add_file(Path(root) / file_name, ignore_errors=True)

    return text_contents, binary_attachments


def process_prompt_mentions(
    prompt: str,
    max_file_size: int = 1024 * 1024,
    max_files: int = 100,
    max_total_size: int = 10 * 1024 * 1024,
) -> tuple[str, list[dict[str, Any]]]:
    """Find all @<path> mentions, resolve their contents, and attach/append them.

    If a mention looks like a path but does not exist, raises PromptProcessingError.

    Returns:
        tuple containing:
        - The processed prompt string (with text context appended and binary placeholders replaced).
        - A list of binary attachment dicts (suitable for HumanMessage content list).
    """
    pattern = re.compile(
        r'(?:^|(?<=[\s\(\[\{<]))@(?:"([^"]*)"|\'([^\']*)\'|([^\s"\'\(\[\{<>,;]+))'
    )

    matches = list(pattern.finditer(prompt))
    resolved_paths: set[Path] = set()
    all_context_contents: dict[Path, str] = {}
    all_binary_attachments: list[dict[str, Any]] = []

    # Map of match range or original mention text -> replacement placeholder
    replacements: list[tuple[int, int, str]] = []

    for match in matches:
        start, end = match.span()
        if match.group(1) is not None:
            path_str = match.group(1)
            is_quoted = True
        elif match.group(2) is not None:
            path_str = match.group(2)
            is_quoted = True
        else:
            path_str = match.group(3)
            is_quoted = False

        if not path_str:
            continue

        if not is_quoted:
            while path_str and path_str[-1] in ".,?:;!":
                if path_str.endswith("..") or path_str == ".":
                    break
                path_str = path_str[:-1]

        if not path_str:
            continue

        candidate_path = Path(path_str).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = (Path.cwd() / candidate_path).resolve()
        else:
            candidate_path = candidate_path.resolve()

        if candidate_path.exists():
            if candidate_path not in resolved_paths:
                resolved_paths.add(candidate_path)
                text_content, bin_attachments = resolve_context_files(
                    candidate_path,
                    max_file_size=max_file_size,
                    max_files=max_files,
                    max_total_size=max_total_size,
                )
                all_context_contents.update(text_content)
                all_binary_attachments.extend(bin_attachments)

            attachment_type = classify_multimodal_file(candidate_path)
            if candidate_path.is_file() and attachment_type is not None:
                replacements.append((start, end, f"[{attachment_type}: {path_str}]"))
        else:
            has_separator = "/" in path_str or "\\" in path_str
            has_extension = bool(re.search(r"\.[a-zA-Z0-9]{1,5}$", path_str))

            if has_separator or has_extension or is_quoted:
                raise PromptProcessingError(
                    f"File or directory not found: '{path_str}'"
                )

    # Perform placeholder replacements in the original prompt (in reverse order to preserve offsets)
    processed_prompt = prompt
    for start, end, placeholder in sorted(replacements, key=lambda x: x[0], reverse=True):
        processed_prompt = processed_prompt[:start] + placeholder + processed_prompt[end:]

    if not all_context_contents:
        return processed_prompt, all_binary_attachments

    context_blocks = []
    cwd = Path.cwd()
    for file_path, content in sorted(all_context_contents.items()):
        try:
            rel_path = file_path.relative_to(cwd)
        except ValueError:
            rel_path = file_path

        context_blocks.append(
            f'<context_file path="{rel_path}">\n{content}\n</context_file>'
        )

    context_str = "\n\n".join(context_blocks)

    processed_prompt = (
        f"{processed_prompt}\n\n"
        f"--- Attached Context ---\n"
        f"{context_str}\n"
        f"--- End of Attached Context ---"
    )
    return processed_prompt, all_binary_attachments
