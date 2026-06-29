"""Autocomplete support for REPL slash commands and file mentions."""

import os
import re
from pathlib import Path
from typing import Callable, Iterable

from prompt_toolkit.completion import CompleteEvent, Completion, Completer
from prompt_toolkit.document import Document

from .dispatch import REPLCommand


class SlashCommandCompleter(Completer):
    """Tab-completer that suggests slash commands and file mentions.

    Slash commands: trigger when typing `/` as the first word.
    File mentions: trigger when typing `@` anywhere in the line.
    """

    def __init__(self, get_commands: Callable[[], dict[str, REPLCommand]]) -> None:
        self._get_commands = get_commands

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text_before = document.text_before_cursor

        # 1. Check if user is typing a file/directory reference (starts with @)
        # Match rightmost @-mention prefix before the cursor
        match = re.search(
            r'(?:^|[\s\(\[\{\<])@(?:"([^"]*)|\'([^\']*)|([^\s"\'\(\[\{\<\>,;]*))$',
            text_before,
        )
        if match:
            double_quote = match.group(1) is not None
            single_quote = match.group(2) is not None

            if double_quote:
                prefix = match.group(1)
                quote_char = '"'
            elif single_quote:
                prefix = match.group(2)
                quote_char = "'"
            else:
                prefix = match.group(3)
                quote_char = None

            yield from self._get_path_completions(prefix, quote_char)
            return

        # 2. Check if user is typing a slash command
        text = text_before.lstrip()
        if not text.startswith("/"):
            return

        # If we're past the first word, nothing to complete for slash commands.
        if " " in text:
            return

        for name, spec in self._get_commands().items():
            if name.startswith(text):
                yield Completion(
                    name,
                    start_position=-len(text),
                    display_meta=spec.summary,
                )

    def _get_path_completions(
        self, prefix: str, quote_char: str | None
    ) -> Iterable[Completion]:
        """Generate autocompletion suggestions for paths starting with the prefix."""
        try:
            dir_part, file_part = os.path.split(prefix)
        except Exception:
            return

        search_dir = Path.cwd()
        if dir_part:
            try:
                resolved_dir = Path(dir_part).expanduser()
                if not resolved_dir.is_absolute():
                    resolved_dir = Path.cwd() / resolved_dir
                if resolved_dir.is_dir():
                    search_dir = resolved_dir
                else:
                    return
            except Exception:
                return

        if not search_dir.exists():
            return

        try:
            for item in search_dir.iterdir():
                # Ignore hidden files unless explicitly typed
                if item.name.startswith(".") and not file_part.startswith("."):
                    continue

                # Filter out extremely common ignored directories to keep autocompletion clean
                if item.is_dir() and item.name in {
                    ".git",
                    ".venv",
                    "venv",
                    "__pycache__",
                    "node_modules",
                    "build",
                    "dist",
                }:
                    continue

                if item.name.startswith(file_part):
                    completed_path = (
                        os.path.join(dir_part, item.name) if dir_part else item.name
                    )

                    needs_quote = bool(
                        re.search(r'[\s"\'\(\)\[\]\{\},;]', completed_path)
                    )

                    is_dir = item.is_dir()
                    suffix = "/" if is_dir else ""
                    display_name = item.name + suffix
                    meta = "Directory" if is_dir else "File"

                    if not is_dir:
                        try:
                            size_kb = item.stat().st_size / 1024
                            meta = f"File ({size_kb:.1f} KB)"
                        except Exception:
                            pass

                    if quote_char is not None:
                        # User already opened quotes.
                        # Replacement text is completed_path + quote_char (closing quote)
                        # Replace starting from prefix.
                        text = completed_path + quote_char
                        start_pos = -len(prefix)
                    else:
                        # User did not open quotes.
                        if needs_quote:
                            # We wrap the whole thing in double quotes, replacing the @ as well.
                            text = f'"{completed_path}"'
                            start_pos = -len(prefix) - 1
                        else:
                            text = completed_path
                            start_pos = -len(prefix)

                    yield Completion(
                        text,
                        start_position=start_pos,
                        display=display_name,
                        display_meta=meta,
                    )
        except Exception:
            return
