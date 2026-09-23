"""Core logic for parsing, resolving, and injecting file context from @-mentions."""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from ..i18n import _


class PromptProcessingError(Exception):
    """Exception raised when prompt processing fails."""


class ContextLimitExceededError(PromptProcessingError):
    """Raised when context limits are exceeded."""


class FileTooLargeError(PromptProcessingError):
    """Raised when a single referenced file exceeds maximum allowed size."""


class ResolvedContext(NamedTuple):
    """Context data resolved from files or directories."""

    text_contents: dict[Path, str]
    attachments: list[dict[str, Any]]
    classifications: dict[Path, str]
    warnings: list[str]
    total_size: int


_MULTIMODAL_KINDS: dict[str, set[str]] = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg", ".heic", ".heif"},
    "audio": {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".aiff"},
    "video": {".mp4", ".mpeg", ".mov", ".avi", ".flv", ".mpg", ".webm", ".wmv", ".3gpp"},
    "file": {".pdf", ".ppt", ".pptx"},
}

_MIME_FALLBACKS: dict[str, str] = {
    "image": "image/png",
    "video": "video/mp4",
    "audio": "audio/mpeg",
    "file": "application/pdf",
}


_FILE_TOO_LARGE_MSG = "File too large: {file_path} ({file_size} bytes, limit is {max_file_size} bytes)"


def classify_multimodal_file(file_path: Path) -> str | None:
    """Classify file into image, video, audio, or file, or None for text."""
    suffix = file_path.suffix.lower()
    for kind, extensions in _MULTIMODAL_KINDS.items():
        if suffix in extensions:
            return kind
    return None


def get_file_type(file_path: Path) -> str | None:
    """Guess file MIME type, with explicit plain text for typescript."""
    if file_path.suffix.lower() == ".ts":
        return "text/plain"
    return mimetypes.guess_type(str(file_path))[0]


def is_binary_file(file_path: Path) -> bool:
    """Return True if file contains null bytes in the first 1024 bytes."""
    with file_path.open("rb") as f:
        return b"\x00" in f.read(1024)


def read_file_content(file_path: Path, max_file_size: int = 1024 * 1024) -> str:
    """Read file content as text, enforcing file type and size limit."""
    if not file_path.is_file():
        raise PromptProcessingError(_("Path is not a file: {file_path}", file_path=file_path))
    size = file_path.stat().st_size
    if size > max_file_size:
        raise FileTooLargeError(
            _(_FILE_TOO_LARGE_MSG, file_path=file_path, file_size=size, max_file_size=max_file_size)
        )
    if is_binary_file(file_path):
        raise PromptProcessingError(_("Cannot read binary file as text: {file_path}", file_path=file_path))
    return file_path.read_text(encoding="utf-8", errors="replace")


def read_binary_file_b64(file_path: Path, max_file_size: int = 1024 * 1024) -> str:
    """Read binary file content and return as base64 string."""
    size = file_path.stat().st_size
    if size > max_file_size:
        raise FileTooLargeError(
            _(_FILE_TOO_LARGE_MSG, file_path=file_path, file_size=size, max_file_size=max_file_size)
        )
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def _resolve_single_file(
    target_path: Path,
    max_file_size: int,
    max_files: int,
    max_total_size: int,
    initial_count: int,
    initial_size: int,
) -> ResolvedContext:
    """Resolve a single file into context data."""
    if initial_count >= max_files:
        raise ContextLimitExceededError(_("Mentions limit exceeded: max {max_files} files.", max_files=max_files))
    size = target_path.stat().st_size
    if size > max_file_size:
        raise FileTooLargeError(
            _(
                _FILE_TOO_LARGE_MSG,
                file_path=target_path,
                file_size=size,
                max_file_size=max_file_size,
            )
        )
    if initial_size + size > max_total_size:
        raise ContextLimitExceededError(
            _("Total context size limit of {max_total_size} bytes exceeded.", max_total_size=max_total_size)
        )
    kind = classify_multimodal_file(target_path)
    if kind:
        b64_data = read_binary_file_b64(target_path, max_file_size)
        mime = get_file_type(target_path) or _MIME_FALLBACKS.get(kind, "application/octet-stream")
        return ResolvedContext({}, [{"type": kind, "base64": b64_data, "mime_type": mime}], {target_path: kind}, [], size)
    return ResolvedContext({target_path: read_file_content(target_path, max_file_size)}, [], {}, [], size)


def _append_dir_file_content(
    file_path: Path,
    max_file_size: int,
    text_contents: dict[Path, str],
    attachments: list[dict[str, Any]],
    classifications: dict[Path, str],
    warnings: list[str],
) -> bool:
    """Classify and append file content to text_contents or attachments. Returns True if appended."""
    kind = classify_multimodal_file(file_path)
    if kind:
        classifications[file_path] = kind
        b64_data = read_binary_file_b64(file_path, max_file_size)
        mime = get_file_type(file_path) or _MIME_FALLBACKS.get(kind, "application/octet-stream")
        attachments.append({"type": kind, "base64": b64_data, "mime_type": mime})
        return True
    if is_binary_file(file_path):
        warnings.append(_("Cannot read binary file as text: {file_path}", file_path=file_path))
        return False
    text_contents[file_path] = file_path.read_text(encoding="utf-8", errors="replace")
    return True


def _resolve_directory(
    target_path: Path,
    max_file_size: int,
    max_files: int,
    max_total_size: int,
    initial_count: int,
    initial_size: int,
) -> ResolvedContext:
    """Resolve files in a directory into context data."""
    text_contents: dict[Path, str] = {}
    attachments: list[dict[str, Any]] = []
    classifications: dict[Path, str] = {}
    warnings: list[str] = []
    resolved_size = 0

    for file_path in sorted(target_path.rglob("*")):
        try:
            if not file_path.is_file():
                continue

            if initial_count + len(text_contents) + len(attachments) >= max_files:
                warnings.append(_("Mentions limit exceeded: max {max_files} files.", max_files=max_files))
                break

            size = file_path.stat().st_size
            if size > max_file_size:
                warnings.append(
                    _(
                        _FILE_TOO_LARGE_MSG,
                        file_path=file_path,
                        file_size=size,
                        max_file_size=max_file_size,
                    )
                )
                continue

            if initial_size + resolved_size + size > max_total_size:
                warnings.append(_("Total context size limit of {max_total_size} bytes exceeded.", max_total_size=max_total_size))
                break

            if _append_dir_file_content(file_path, max_file_size, text_contents, attachments, classifications, warnings):
                resolved_size += size
        except (OSError, UnicodeDecodeError, PromptProcessingError) as exc:
            warnings.append(str(exc))

    return ResolvedContext(text_contents, attachments, classifications, list(dict.fromkeys(warnings)), resolved_size)


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
    if not target_path.exists():
        raise PromptProcessingError(_("Path is neither a file nor a directory: {file_path}", file_path=target_path))

    if target_path.is_file():
        return _resolve_single_file(
            target_path,
            max_file_size=max_file_size,
            max_files=max_files,
            max_total_size=max_total_size,
            initial_count=initial_count,
            initial_size=initial_size,
        )

    return _resolve_directory(
        target_path,
        max_file_size=max_file_size,
        max_files=max_files,
        max_total_size=max_total_size,
        initial_count=initial_count,
        initial_size=initial_size,
    )


def _extract_mention_path(match: re.Match[str]) -> tuple[str, int, bool]:
    """Extract path string, handling unquoted punctuation stripping, and end index."""
    q1, q2, unquoted = match.groups()
    if q1 is not None:
        return q1, match.end(), True
    if q2 is not None:
        return q2, match.end(), True

    path_str = unquoted
    end = match.end()
    while path_str and path_str[-1] in ".,?:;!":
        if (
            path_str in (".", "..")
            or path_str.endswith(("/..", "\\..", "/.", "\\."))
            or bool(re.match(r"^[a-zA-Z]:$", path_str))
        ):
            break
        path_str = path_str[:-1]
        end -= 1
    return path_str, end, False


def _resolve_mention_target(
    path_str: str,
    resolved_base: Path,
    allow_traversal: bool,
    is_quoted: bool,
) -> Path | None:
    """Resolve mention target path string to an existing Path or None."""
    if not path_str:
        return None

    target = url2pathname(unquote(urlparse(path_str).path)) if path_str.startswith(("file://", "file:")) else path_str
    path = (resolved_base / Path(target).expanduser()).resolve()

    if not allow_traversal and not path.is_relative_to(resolved_base):
        raise PromptProcessingError(_("Access to path outside working directory is not allowed: '{path_str}'", path_str=path_str))

    if not path.exists():
        if is_quoted or "/" in path_str or "\\" in path_str or path_str.startswith(("./", "../")):
            raise PromptProcessingError(_("File or directory not found: '{path_str}'", path_str=path_str))
        return None

    return path


def process_prompt_mentions(
    prompt: str,
    max_file_size: int = 1024 * 1024,
    max_files: int = 100,
    max_total_size: int = 10 * 1024 * 1024,
    allow_traversal: bool = True,
    base_dir: Path | None = None,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Resolve file @-mentions in prompt, returning updated prompt, attachments, and warnings."""
    if base_dir is None:
        base_dir = Path.cwd()
    resolved_base = base_dir.resolve()

    pattern = re.compile(r'(?:^|(?<=[\s\(\[\{<]))@(?:\"([^\"]*)\"|\'([^\']*)\'|([^\s\"\'\(\[\{<>,;]+))')

    resolved_paths: set[Path] = set()
    text_contents: dict[Path, str] = {}
    attachments: list[dict[str, Any]] = []
    warnings: list[str] = []
    replacements: list[tuple[int, int, str]] = []
    total_size = 0

    for match in pattern.finditer(prompt):
        path_str, end, is_quoted = _extract_mention_path(match)
        path = _resolve_mention_target(path_str, resolved_base, allow_traversal, is_quoted)
        if path is None:
            continue

        if path.is_file():
            kind = classify_multimodal_file(path)
            if kind:
                replacements.append((match.start(), end, f"[{kind}: {path_str}]"))

        if path in resolved_paths:
            continue
        resolved_paths.add(path)

        ctx = resolve_context_files(
            path,
            max_file_size=max_file_size,
            max_files=max_files,
            max_total_size=max_total_size,
            initial_count=len(text_contents) + len(attachments),
            initial_size=total_size,
        )
        text_contents.update(ctx.text_contents)
        attachments.extend(ctx.attachments)
        warnings.extend(ctx.warnings)
        total_size += ctx.total_size

    for start, end, rep in sorted(replacements, key=lambda x: x[0], reverse=True):
        prompt = prompt[:start] + rep + prompt[end:]

    if text_contents:
        blocks = [
            f'<context_file path="{p.relative_to(resolved_base).as_posix() if p.is_relative_to(resolved_base) else p.as_posix()}">\n{c}\n</context_file>'
            for p, c in sorted(text_contents.items())
        ]
        prompt = f"{prompt}\n\n--- Attached Context ---\n" + "\n\n".join(blocks) + "\n--- End of Attached Context ---"

    return prompt, attachments, warnings


__all__ = [
    "ContextLimitExceededError",
    "FileTooLargeError",
    "PromptProcessingError",
    "ResolvedContext",
    "classify_multimodal_file",
    "get_file_type",
    "is_binary_file",
    "process_prompt_mentions",
    "read_binary_file_b64",
    "read_file_content",
    "resolve_context_files",
]
