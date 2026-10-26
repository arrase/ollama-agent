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


def _check_file_size(file_path: Path, max_file_size: int) -> int:
    """Return file size in bytes, raising FileTooLargeError if it exceeds max_file_size."""
    size = file_path.stat().st_size
    if size > max_file_size:
        raise FileTooLargeError(
            _(
                "File too large: {file_path} ({file_size} bytes, limit is {max_file_size} bytes)",
                file_path=file_path,
                file_size=size,
                max_file_size=max_file_size,
            )
        )
    return size


def read_file_content(file_path: Path, max_file_size: int = 1024 * 1024) -> str:
    """Read file content as text, enforcing file type and size limit."""
    if not file_path.is_file():
        raise PromptProcessingError(_("Path is not a file: {file_path}", file_path=file_path))
    _check_file_size(file_path, max_file_size)
    if is_binary_file(file_path):
        raise PromptProcessingError(_("Cannot read binary file as text: {file_path}", file_path=file_path))
    return file_path.read_text(encoding="utf-8", errors="replace")


def read_binary_file_b64(file_path: Path, max_file_size: int = 1024 * 1024) -> str:
    """Read binary file content and return as base64 string."""
    _check_file_size(file_path, max_file_size)
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def _load_validated_file(
    file_path: Path,
    max_file_size: int,
    max_total_size: int,
    current_size: int,
    text_contents: dict[Path, str],
    attachments: list[dict[str, Any]],
) -> int:
    """Validate size and load file into text_contents or attachments, returning its byte size."""
    size = _check_file_size(file_path, max_file_size)
    if current_size + size > max_total_size:
        raise ContextLimitExceededError(
            _("Total context size limit of {max_total_size} bytes exceeded.", max_total_size=max_total_size)
        )
    kind = classify_multimodal_file(file_path)
    if kind:
        b64_data = read_binary_file_b64(file_path, max_file_size)
        mime = get_file_type(file_path) or _MIME_FALLBACKS[kind]
        attachments.append({"type": kind, "base64": b64_data, "mime_type": mime})
        return size
    text_contents[file_path] = read_file_content(file_path, max_file_size)
    return size


def _resolve_single_file(
    target_path: Path,
    max_file_size: int,
    max_files: int,
    max_total_size: int,
    initial_count: int,
    initial_size: int,
    seen_files: set[Path],
) -> ResolvedContext:
    """Resolve a single file into context data."""
    resolved_file = target_path.resolve()
    if resolved_file in seen_files:
        return ResolvedContext({}, [], [], 0)
    if initial_count >= max_files:
        raise ContextLimitExceededError(_("Mentions limit exceeded: max {max_files} files.", max_files=max_files))
    text_contents: dict[Path, str] = {}
    attachments: list[dict[str, Any]] = []
    size = _load_validated_file(target_path, max_file_size, max_total_size, initial_size, text_contents, attachments)
    seen_files.add(resolved_file)
    return ResolvedContext(text_contents, attachments, [], size)


def _resolve_directory(
    target_path: Path,
    max_file_size: int,
    max_files: int,
    max_total_size: int,
    initial_count: int,
    initial_size: int,
    seen_files: set[Path],
    resolved_base: Path,
    allow_traversal: bool,
) -> ResolvedContext:
    """Resolve files in a directory into context data."""
    text_contents: dict[Path, str] = {}
    attachments: list[dict[str, Any]] = []
    warnings: list[str] = []
    resolved_size = 0

    for file_path in sorted(target_path.rglob("*")):
        try:
            if not file_path.is_file():
                continue

            resolved_file = file_path.resolve()
            if not allow_traversal and not resolved_file.is_relative_to(resolved_base):
                warnings.append(
                    _("Access to path outside working directory is not allowed: '{path_str}'", path_str=file_path)
                )
                continue

            if resolved_file in seen_files:
                continue

            if initial_count + len(text_contents) + len(attachments) >= max_files:
                warnings.append(_("Mentions limit exceeded: max {max_files} files.", max_files=max_files))
                break

            size = _load_validated_file(
                file_path,
                max_file_size,
                max_total_size,
                initial_size + resolved_size,
                text_contents,
                attachments,
            )
            seen_files.add(resolved_file)
            resolved_size += size
        except ContextLimitExceededError as exc:
            warnings.append(str(exc))
            break
        except (OSError, UnicodeDecodeError, PromptProcessingError) as exc:
            warnings.append(str(exc))

    return ResolvedContext(text_contents, attachments, list(dict.fromkeys(warnings)), resolved_size)


def resolve_context_files(
    target_path: Path,
    max_file_size: int = 1024 * 1024,
    max_files: int = 100,
    max_total_size: int = 10 * 1024 * 1024,
    *,
    initial_count: int = 0,
    initial_size: int = 0,
    seen_files: set[Path] | None = None,
    resolved_base: Path | None = None,
    allow_traversal: bool = True,
) -> ResolvedContext:
    """Resolve a target file or directory into context data."""
    if not target_path.exists():
        raise PromptProcessingError(_("Path is neither a file nor a directory: {file_path}", file_path=target_path))

    seen = seen_files if seen_files is not None else set()
    if target_path.is_file():
        return _resolve_single_file(
            target_path,
            max_file_size=max_file_size,
            max_files=max_files,
            max_total_size=max_total_size,
            initial_count=initial_count,
            initial_size=initial_size,
            seen_files=seen,
        )

    return _resolve_directory(
        target_path,
        max_file_size=max_file_size,
        max_files=max_files,
        max_total_size=max_total_size,
        initial_count=initial_count,
        initial_size=initial_size,
        seen_files=seen,
        resolved_base=resolved_base if resolved_base is not None else target_path.resolve(),
        allow_traversal=allow_traversal,
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
            or re.match(r"^[a-zA-Z]:$", path_str)
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
    try:
        expanded = Path(target).expanduser()
    except RuntimeError as exc:
        # expanduser() raises RuntimeError when the home directory cannot be determined.
        raise PromptProcessingError(
            _("Cannot expand home directory in path '{path_str}': {exc}", path_str=path_str, exc=exc)
        ) from exc
    path = (resolved_base / expanded).resolve()

    if not allow_traversal and not path.is_relative_to(resolved_base):
        raise PromptProcessingError(
            _("Access to path outside working directory is not allowed: '{path_str}'", path_str=path_str)
        )

    if not path.exists():
        if is_quoted or "/" in path_str or "\\" in path_str or path_str.startswith(("./", "../")):
            raise PromptProcessingError(_("File or directory not found: '{path_str}'", path_str=path_str))
        return None

    return path


def _context_path(path: Path, base: Path) -> str:
    return (path.relative_to(base) if path.is_relative_to(base) else path).as_posix()


def process_prompt_mentions(
    prompt: str,
    max_file_size: int = 1024 * 1024,
    max_files: int = 100,
    max_total_size: int = 10 * 1024 * 1024,
    allow_traversal: bool = True,
    base_dir: Path = Path(),
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Resolve file @-mentions in prompt, returning updated prompt, attachments, and warnings."""
    resolved_base = base_dir.resolve()

    pattern = re.compile(r"(?:^|(?<=[\s\(\[\{<]))@(?:\"([^\"]*)\"|\'([^\']*)\'|([^\s\"\'\(\[\{<>,;]+))")

    resolved_targets: set[Path] = set()
    seen_files: set[Path] = set()
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

        if path in resolved_targets or path in seen_files:
            continue
        resolved_targets.add(path)

        ctx = resolve_context_files(
            path,
            max_file_size=max_file_size,
            max_files=max_files,
            max_total_size=max_total_size,
            initial_count=len(text_contents) + len(attachments),
            initial_size=total_size,
            seen_files=seen_files,
            resolved_base=resolved_base,
            allow_traversal=allow_traversal,
        )
        text_contents.update(ctx.text_contents)
        attachments.extend(ctx.attachments)
        warnings.extend(ctx.warnings)
        total_size += ctx.total_size

    for start, end, rep in sorted(replacements, key=lambda x: x[0], reverse=True):
        prompt = prompt[:start] + rep + prompt[end:]

    if text_contents:
        blocks = [
            f'<context_file path="{_context_path(p, resolved_base)}">\n{c}\n</context_file>'
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
