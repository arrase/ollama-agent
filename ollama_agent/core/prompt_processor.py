"""Core logic for parsing, resolving, and injecting file context from @-mentions."""

import os
import re
from pathlib import Path


class PromptProcessingError(Exception):
    """Exception raised when prompt processing fails, e.g., referenced file not found."""


def is_binary_file(file_path: Path) -> bool:
    """Check if a file is binary by searching for null bytes in the first block."""
    try:
        with open(file_path, "rb") as f:
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

    ignored_names = {
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
    }
    for part in rel_path.parts:
        if (
            part in ignored_names
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
        raise PromptProcessingError(f"Cannot read binary file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        raise PromptProcessingError(f"Failed to read file {file_path}: {e}")


def resolve_context_files(
    target_path: Path,
    max_file_size: int = 1024 * 1024,
    max_files: int = 100,
    max_total_size: int = 10 * 1024 * 1024,
) -> dict[Path, str]:
    """Resolve a target file or directory into a dict mapping Paths to contents."""
    files_content: dict[Path, str] = {}
    total_size = 0

    if target_path.is_file():
        content = read_file_content(target_path, max_file_size)
        files_content[target_path] = content
        return files_content

    if target_path.is_dir():
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

                if is_binary_file(file_path):
                    continue

                try:
                    size = file_path.stat().st_size
                except Exception:
                    continue

                if size > max_file_size:
                    continue

                if len(files_content) >= max_files:
                    raise PromptProcessingError(
                        f"Directory traversal exceeded limit of {max_files} files in: {target_path}"
                    )

                if total_size + size > max_total_size:
                    raise PromptProcessingError(
                        f"Directory traversal exceeded total size limit of {max_total_size} bytes in: {target_path}"
                    )

                try:
                    content = read_file_content(file_path, max_file_size)
                    files_content[file_path] = content
                    total_size += size
                except PromptProcessingError:
                    continue

        return files_content

    raise PromptProcessingError(f"Target path does not exist: {target_path}")


def process_prompt_mentions(
    prompt: str,
    max_file_size: int = 1024 * 1024,
    max_files: int = 100,
    max_total_size: int = 10 * 1024 * 1024,
) -> str:
    """Find all @<path> mentions, resolve their contents, and append to the prompt.

    If a mention looks like a path but does not exist, raises PromptProcessingError.
    """
    pattern = re.compile(
        r'(?:^|(?<=[\s\(\[\{\<]))@(?:"([^"]*)"|\'([^\']*)\'|([^\s"\'\(\[\{\<\>,;]+))'
    )

    matches = list(pattern.finditer(prompt))
    resolved_paths: set[Path] = set()
    all_context_contents: dict[Path, str] = {}

    for match in matches:
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
                contents = resolve_context_files(
                    candidate_path,
                    max_file_size=max_file_size,
                    max_files=max_files,
                    max_total_size=max_total_size,
                )
                all_context_contents.update(contents)
        else:
            has_separator = "/" in path_str or "\\" in path_str
            has_extension = bool(re.search(r"\.[a-zA-Z0-9]{1,5}$", path_str))

            if has_separator or has_extension or is_quoted:
                raise PromptProcessingError(
                    f"File or directory not found: '{path_str}'"
                )

    if not all_context_contents:
        return prompt

    context_blocks = []
    cwd = Path.cwd()
    for file_path, content in all_context_contents.items():
        try:
            rel_path = file_path.relative_to(cwd)
        except ValueError:
            rel_path = file_path

        context_blocks.append(
            f'<context_file path="{rel_path}">\n{content}\n</context_file>'
        )

    context_str = "\n\n".join(context_blocks)

    processed_prompt = (
        f"{prompt}\n\n"
        f"--- Attached Context ---\n"
        f"{context_str}\n"
        f"--- End of Attached Context ---"
    )
    return processed_prompt
