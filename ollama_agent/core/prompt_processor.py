"""Core logic for parsing, resolving, and injecting file context from @-mentions."""

from __future__ import annotations

import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from ..i18n import _


class PromptProcessingError(Exception):
    """Exception raised when prompt processing fails, e.g., referenced file not found."""


class ContextLimitExceededError(PromptProcessingError):
    """Raised when the total number of files or total context size limit is exceeded."""


class FileTooLargeError(PromptProcessingError):
    """Raised when a single referenced file exceeds the maximum allowed file size."""


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


def get_file_type(file_path: Path) -> str | None:
    """Guess the MIME type of a file, prioritizing explicit text overrides and custom extensions."""
    suffix = file_path.suffix.lower()
    if suffix == ".ts":
        return "text/plain"
    if suffix in _MIME_EXTENSIONS:
        return _MIME_EXTENSIONS[suffix]
    mime, _encoding = mimetypes.guess_type(str(file_path))
    return mime


class ResolvedContext(NamedTuple):
    """Result of resolving a @mention target into context data."""

    text_contents: dict[Path, str]
    attachments: list[dict[str, Any]]
    classifications: dict[Path, str]
    warnings: list[str]
    total_size: int


def _multimodal_kind(mime: str | None) -> str | None:
    """Map a MIME type to a LangChain multimodal type, or None for text."""
    if not mime:
        return None
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


def classify_multimodal_file(file_path: Path) -> str | None:
    """Classify a file into a LangChain multimodal type, or None if it should be treated as text."""
    return _multimodal_kind(get_file_type(file_path))


def is_binary_file(file_path: Path) -> bool:
    """Check if a file is binary by searching for null bytes in the first block."""
    with file_path.open("rb") as f:
        chunk = f.read(1024)
        return b"\x00" in chunk


def _check_file_size(file_path: Path, max_file_size: int) -> int:
    """Verify that file exists and its size does not exceed max_file_size."""
    try:
        file_size = file_path.stat().st_size
    except OSError as e:
        raise PromptProcessingError(_("Failed to read file {file_path}: {e}", file_path=file_path, e=e)) from e

    if file_size > max_file_size:
        raise FileTooLargeError(
            _(
                "File too large: {file_path} ({file_size} bytes, limit is {max_file_size} bytes)",
                file_path=file_path,
                file_size=file_size,
                max_file_size=max_file_size,
            )
        )
    return file_size


def read_file_content(file_path: Path, max_file_size: int = 1024 * 1024) -> str:
    """Read file content as string, ensuring it is a text file and fits in size limit."""
    if not file_path.is_file():
        raise PromptProcessingError(_("Path is not a file: {file_path}", file_path=file_path))

    _check_file_size(file_path, max_file_size)

    if is_binary_file(file_path):
        raise PromptProcessingError(_("Cannot read binary file as text: {file_path}", file_path=file_path))

    try:
        return file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise PromptProcessingError(_("Failed to read file {file_path}: {e}", file_path=file_path, e=e)) from e


def read_binary_file_b64(file_path: Path, max_file_size: int = 1024 * 1024) -> str:
    """Read a binary file and return its content as a base64 encoded string."""
    _check_file_size(file_path, max_file_size)

    try:
        return base64.b64encode(file_path.read_bytes()).decode("utf-8")
    except OSError as e:
        raise PromptProcessingError(_("Failed to read file {file_path}: {e}", file_path=file_path, e=e)) from e


def resolve_context_files(
    target_path: Path,
    max_file_size: int = 1024 * 1024,
    max_files: int = 100,
    max_total_size: int = 10 * 1024 * 1024,
    *,
    initial_count: int = 0,
    initial_size: int = 0,
) -> ResolvedContext:
    """Resolve a target file or directory into context data."""
    text_contents: dict[Path, str] = {}
    binary_attachments: list[dict[str, Any]] = []
    classifications: dict[Path, str] = {}
    warnings: list[str] = []
    total_size = initial_size
    resolved_size = 0
    file_count = initial_count

    def add_file(file_path: Path) -> None:
        nonlocal total_size, resolved_size, file_count

        if file_count >= max_files:
            raise ContextLimitExceededError(_("Mentions limit exceeded: max {max_files} files.", max_files=max_files))

        size = _check_file_size(file_path, max_file_size)

        if total_size + size > max_total_size:
            raise ContextLimitExceededError(
                _("Total context size limit of {max_total_size} bytes exceeded.", max_total_size=max_total_size)
            )

        mime = get_file_type(file_path)
        attachment_type = _multimodal_kind(mime)
        if attachment_type is not None:
            classifications[file_path] = attachment_type
            b64_data = read_binary_file_b64(file_path, max_file_size)
            binary_attachments.append(
                {
                    "type": attachment_type,
                    "base64": b64_data,
                    "mime_type": mime or f"{attachment_type}/*",
                }
            )
        else:
            text_contents[file_path] = read_file_content(file_path, max_file_size)
        file_count += 1
        total_size += size
        resolved_size += size

    if target_path.is_file():
        add_file(target_path)
        return ResolvedContext(text_contents, binary_attachments, classifications, warnings, resolved_size)

    if not target_path.is_dir():
        raise PromptProcessingError(_("Path is neither a file nor a directory: {file_path}", file_path=target_path))

    for root, _dirs, files in os.walk(target_path):
        stop = False
        for file_name in files:
            try:
                add_file(Path(root) / file_name)
            except ContextLimitExceededError as exc:
                warnings.append(str(exc))
                stop = True
                break
            except PromptProcessingError as exc:
                warnings.append(str(exc))
        if stop:
            break

    return ResolvedContext(
        text_contents, binary_attachments, classifications, list(dict.fromkeys(warnings)), resolved_size
    )


def process_prompt_mentions(
    prompt: str,
    max_file_size: int = 1024 * 1024,
    max_files: int = 100,
    max_total_size: int = 10 * 1024 * 1024,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Find all @<path> mentions, resolve their contents, and attach/append them.

    If a mention looks like a path but does not exist, raises PromptProcessingError.

    Returns:
        tuple containing:
        - The processed prompt string (with text context appended and binary placeholders replaced).
        - A list of binary attachment dicts (suitable for HumanMessage content list).
        - A list of formatted warnings for files skipped during directory resolution.
    """
    pattern = re.compile(r'(?:^|(?<=[\s\(\[\{<]))@(?:"([^"]*)"|\'([^\']*)\'|([^\s"\'\(\[\{<>,;]+))')

    matches = list(pattern.finditer(prompt))
    resolved_paths: set[Path] = set()
    all_context_contents: dict[Path, str] = {}
    all_binary_attachments: list[dict[str, Any]] = []
    all_classifications: dict[Path, str] = {}
    all_warnings: list[str] = []
    cumulative_files = 0
    cumulative_size = 0

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
                if (
                    path_str in (".", "..")
                    or path_str.endswith(("/..", "\\..", "/.", "\\."))
                    or bool(re.match(r"^[a-zA-Z]:$", path_str))
                ):
                    break
                path_str = path_str[:-1]
                end -= 1

        if not path_str:
            continue

        resolved_target = path_str
        if resolved_target.startswith(("file://", "file:")):
            parsed = urlparse(resolved_target)
            resolved_target = url2pathname(unquote(parsed.path))

        candidate_path = Path(resolved_target).expanduser().resolve()

        if candidate_path.exists():
            if candidate_path not in resolved_paths:
                resolved_paths.add(candidate_path)
                context = resolve_context_files(
                    candidate_path,
                    max_file_size=max_file_size,
                    max_files=max_files,
                    max_total_size=max_total_size,
                    initial_count=cumulative_files,
                    initial_size=cumulative_size,
                )
                all_context_contents.update(context.text_contents)
                all_binary_attachments.extend(context.attachments)
                all_classifications.update(context.classifications)
                all_warnings.extend(context.warnings)
                cumulative_files = len(all_context_contents) + len(all_binary_attachments)
                cumulative_size += context.total_size

            attachment_type = all_classifications.get(candidate_path)
            if candidate_path.is_file() and attachment_type is not None:
                replacements.append((start, end, f"[{attachment_type}: {path_str}]"))
        else:
            has_separator = "/" in path_str or "\\" in path_str
            has_relative_prefix = path_str.startswith(("./", "../", ".\\", "..\\"))
            is_intended_file = is_quoted or has_separator or has_relative_prefix

            if is_intended_file:
                raise PromptProcessingError(_("File or directory not found: '{path_str}'", path_str=path_str))

    # Perform placeholder replacements in the original prompt (in reverse order to preserve offsets)
    processed_prompt = prompt
    for start, end, placeholder in sorted(replacements, key=lambda x: x[0], reverse=True):
        processed_prompt = processed_prompt[:start] + placeholder + processed_prompt[end:]

    if not all_context_contents:
        return processed_prompt, all_binary_attachments, all_warnings

    context_blocks = []
    cwd = Path.cwd()
    for file_path, content in sorted(all_context_contents.items()):
        try:
            rel_path = file_path.relative_to(cwd).as_posix()
        except ValueError:
            rel_path = file_path.as_posix()

        context_blocks.append(f'<context_file path="{rel_path}">\n{content}\n</context_file>')

    context_str = "\n\n".join(context_blocks)

    processed_prompt = f"{processed_prompt}\n\n--- Attached Context ---\n{context_str}\n--- End of Attached Context ---"
    return processed_prompt, all_binary_attachments, all_warnings
