"""Core logic for parsing, resolving, and injecting file context from @-mentions."""

import base64
import mimetypes
import os
import re
from pathlib import Path

# Shared set of directory names to skip during traversal and autocompletion.
# Importable by other modules (e.g. the completer) to avoid duplication.
IGNORED_DIRECTORY_NAMES: frozenset[str] = frozenset({
    ".git",
    ".github",
    ".vscode",
    ".idea",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    "build",
    "dist",
})


class PromptProcessingError(Exception):
    """Exception raised when prompt processing fails, e.g., referenced file not found."""


def get_file_type(file_path: Path) -> str:
    """Guess the MIME type of a file using mimetypes, with custom extension fallback."""
    mime, _ = mimetypes.guess_type(str(file_path))
    if mime:
        return mime

    # Fallback extension matching
    ext = file_path.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"):
        return "image/" + (ext[1:] if ext != ".jpg" else "jpeg")
    if ext in (".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"):
        return "audio/" + (ext[1:] if ext != ".mp3" else "mpeg")
    return ""


def is_binary_file(file_path: Path) -> bool:
    """Check if a file is binary by searching for null bytes in the first block."""
    try:
        with file_path.open("rb") as f:
            chunk = f.read(1024)
            return b"\x00" in chunk
    except Exception:
        return True


def should_ignore_path(path: Path, root_path: Path) -> bool:
    """Determine if a file or directory should be ignored during traversal."""
    try:
        rel_path = path.relative_to(root_path)
    except ValueError:
        rel_path = path

    for part in rel_path.parts:
        if (
            part in IGNORED_DIRECTORY_NAMES
            or part.endswith(".egg-info")
            or (part.startswith(".") and part not in (".", ".."))
        ):
            return True
    return False


def read_file_content(file_path: Path, max_file_size: int = 1024 * 1024) -> str:
    """Read file content as string, ensuring it is a text file and fits in size limit."""
    if not file_path.is_file():
        raise PromptProcessingError(f"Path is not a file: {file_path}")

    try:
        file_size = file_path.stat().st_size
    except Exception as e:
        raise PromptProcessingError(f"Failed to get file stats for {file_path}: {e}")

    if file_size > max_file_size:
        raise PromptProcessingError(
            f"File too large: {file_path} ({file_size} bytes, limit is {max_file_size} bytes)"
        )

    if is_binary_file(file_path):
        raise PromptProcessingError(f"Cannot read binary file as text: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        raise PromptProcessingError(f"Failed to read file {file_path}: {e}")


def read_binary_file_b64(file_path: Path, max_file_size: int = 1024 * 1024) -> str:
    """Read a binary file and return its content as a base64 encoded string."""
    try:
        file_size = file_path.stat().st_size
    except Exception as e:
        raise PromptProcessingError(f"Failed to get file stats for {file_path}: {e}")

    if file_size > max_file_size:
        raise PromptProcessingError(
            f"File too large: {file_path} ({file_size} bytes, limit is {max_file_size} bytes)"
        )

    try:
        with file_path.open("rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        raise PromptProcessingError(f"Failed to read binary file {file_path}: {e}")


def resolve_context_files(
    target_path: Path,
    max_file_size: int = 1024 * 1024,
    max_files: int = 100,
    max_total_size: int = 10 * 1024 * 1024,
    allow_binary_traversal: bool = False,
    allowed_capabilities: set[str] | None = None,
) -> tuple[dict[Path, str], list[dict[str, Any]], list[str]]:
    """Resolve a target file or directory into context data.

    Returns:
        tuple containing:
        - dict mapping Path to text content.
        - list of attachment dictionaries for binary files.
        - list of warning/info strings.
    """
    text_contents: dict[Path, str] = {}
    binary_attachments: list[dict[str, Any]] = []
    warnings: list[str] = []
    total_size = 0
    capabilities = allowed_capabilities or set()

    cwd = Path.cwd()

    def get_display_path(p: Path) -> str:
        try:
            return str(p.relative_to(cwd))
        except ValueError:
            return str(p)

    if target_path.is_file():
        # For explicit file references, we check MIME type first.
        mime = get_file_type(target_path)
        is_image = mime.startswith("image/")
        is_audio = mime.startswith("audio/")

        if is_image or is_audio:
            cap_needed = "vision" if is_image else "audio"
            file_type_label = "image" if is_image else "audio"

            if cap_needed not in capabilities:
                raise PromptProcessingError(
                    f"Model does not support {cap_needed} capability required for {file_type_label} file: '{get_display_path(target_path)}'"
                )

            b64_data = read_binary_file_b64(target_path, max_file_size)
            binary_attachments.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{b64_data}"
                }
            })
            return text_contents, binary_attachments, warnings

        # Fallback to text reading
        content = read_file_content(target_path, max_file_size)
        text_contents[target_path] = content
        return text_contents, binary_attachments, warnings

    if not target_path.is_dir():
        return text_contents, binary_attachments, warnings

    for root, dirs, files in os.walk(target_path):
        dirs[:] = [
            d
            for d in dirs
            if not should_ignore_path(Path(root) / d, target_path)
        ]

        for file_name in files:
            file_path = Path(root) / file_name
            if should_ignore_path(file_path, target_path):
                continue

            try:
                size = file_path.stat().st_size
            except Exception:
                continue

            if size > max_file_size:
                continue

            # Check if it's a binary file (either via null byte check or MIME prefix)
            mime = get_file_type(file_path)
            is_image = mime.startswith("image/")
            is_audio = mime.startswith("audio/")
            is_binary = is_image or is_audio or is_binary_file(file_path)

            if is_binary:
                if not allow_binary_traversal:
                    display_p = get_display_path(file_path)
                    warnings.append(
                        f"Note: Skipped binary file '{display_p}' during directory walk. "
                        "Set 'allow_binary_traversal: true' in settings.yaml to include it."
                    )
                    continue

                if is_image or is_audio:
                    cap_needed = "vision" if is_image else "audio"
                    file_type_label = "image" if is_image else "audio"

                    if cap_needed not in capabilities:
                        display_p = get_display_path(file_path)
                        warnings.append(
                            f"Note: Skipped binary {file_type_label} file '{display_p}' "
                            f"because the active model lacks '{cap_needed}' capability."
                        )
                        continue

                    if len(text_contents) + len(binary_attachments) >= max_files:
                        raise PromptProcessingError(
                            f"Directory traversal exceeded limit of {max_files} files in: {get_display_path(target_path)}"
                        )

                    if total_size + size > max_total_size:
                        raise PromptProcessingError(
                            f"Directory traversal exceeded total size limit of {max_total_size} bytes in: {get_display_path(target_path)}"
                        )

                    try:
                        b64_data = read_binary_file_b64(file_path, max_file_size)
                        binary_attachments.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64_data}"
                            }
                        })
                        total_size += size
                    except PromptProcessingError:
                        continue
                continue

            # Text file resolution
            if len(text_contents) + len(binary_attachments) >= max_files:
                raise PromptProcessingError(
                    f"Directory traversal exceeded limit of {max_files} files in: {get_display_path(target_path)}"
                )

            if total_size + size > max_total_size:
                raise PromptProcessingError(
                    f"Directory traversal exceeded total size limit of {max_total_size} bytes in: {get_display_path(target_path)}"
                )

            try:
                content = read_file_content(file_path, max_file_size)
                text_contents[file_path] = content
                total_size += size
            except PromptProcessingError:
                continue

    return text_contents, binary_attachments, warnings


def process_prompt_mentions(
    prompt: str,
    max_file_size: int = 1024 * 1024,
    max_files: int = 100,
    max_total_size: int = 10 * 1024 * 1024,
    allow_binary_traversal: bool = False,
    allowed_capabilities: set[str] | None = None,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Find all @<path> mentions, resolve their contents, and attach/append them.

    If a mention looks like a path but does not exist, raises PromptProcessingError.

    Returns:
        tuple containing:
        - The processed prompt string (with text context appended and binary placeholders replaced).
        - A list of binary attachment dicts (suitable for HumanMessage content list).
        - A list of warnings/notices about skipped or modified files.
    """
    pattern = re.compile(
        r'(?:^|(?<=[\s\(\[\{<]))@(?:"([^"]*)"|\'([^\']*)\'|([^\s"\'\(\[\{<>,;]+))'
    )

    matches = list(pattern.finditer(prompt))
    resolved_paths: set[Path] = set()
    all_context_contents: dict[Path, str] = {}
    all_binary_attachments: list[dict[str, Any]] = []
    all_warnings: list[str] = []

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
                text_content, bin_attachments, warns = resolve_context_files(
                    candidate_path,
                    max_file_size=max_file_size,
                    max_files=max_files,
                    max_total_size=max_total_size,
                    allow_binary_traversal=allow_binary_traversal,
                    allowed_capabilities=allowed_capabilities,
                )
                all_context_contents.update(text_content)
                all_binary_attachments.extend(bin_attachments)
                all_warnings.extend(warns)

            # If it resolved to a binary file, replace mention in prompt with a placeholder.
            # (If it's a text file or directory, we keep the mention in the text as is,
            # and append the context block at the end).
            mime = get_file_type(candidate_path)
            is_image = mime.startswith("image/")
            is_audio = mime.startswith("audio/")
            if candidate_path.is_file() and (is_image or is_audio):
                file_type_label = "image" if is_image else "audio"
                replacements.append((start, end, f"[{file_type_label}: {path_str}]"))
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
        # Preserve the leading character (e.g. space, bracket) if matching boundary
        match_str = processed_prompt[start:end]
        leading_char = ""
        if match_str.startswith(" "):
            leading_char = " "
        processed_prompt = processed_prompt[:start] + leading_char + placeholder + processed_prompt[end:]

    if not all_context_contents:
        return processed_prompt, all_binary_attachments, all_warnings

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
    return processed_prompt, all_binary_attachments, all_warnings
