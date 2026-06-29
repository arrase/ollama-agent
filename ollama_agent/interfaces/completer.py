"""Autocomplete support for REPL slash commands and file mentions."""

import logging
import os
import re
from pathlib import Path
from typing import Callable, Iterable

from prompt_toolkit.completion import CompleteEvent, Completion, Completer
from prompt_toolkit.document import Document

from ..core.prompt_processor import IGNORED_DIRECTORY_NAMES
from .dispatch import REPLCommand

_log = logging.getLogger(__name__)

# @-mention regex: matches @"quoted", @'quoted', or @bare at word boundaries.
_AT_MENTION_RE = re.compile(
    r"""(?:^|[\s\(\[\{<])@(?:"([^"]*)|'([^']*)|([^\s"'\(\[\{<>,;]*))$"""
)

# Characters whose presence in a completed path require quoting.
_NEEDS_QUOTE_RE = re.compile(r"""[\s"'\(\)\[\]\{\},;]""")


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
        match = _AT_MENTION_RE.search(text_before)
        if match:
            if match.group(1) is not None:
                prefix = match.group(1)
                quote_char = '"'
            elif match.group(2) is not None:
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
        except (TypeError, ValueError):
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
            except (OSError, ValueError) as exc:
                _log.debug("Path resolution failed for '%s': %s", dir_part, exc)
                return

        if not search_dir.exists():
            return

        try:
            entries = sorted(search_dir.iterdir(), key=lambda p: p.name)
        except OSError as exc:
            _log.debug("Cannot list directory '%s': %s", search_dir, exc)
            return

        for item in entries:
            # Ignore hidden files unless explicitly typed
            if item.name.startswith(".") and not file_part.startswith("."):
                continue

            # Filter out common ignored directories to keep autocompletion clean
            if item.is_dir() and item.name in IGNORED_DIRECTORY_NAMES:
                continue

            if not item.name.startswith(file_part):
                continue

            is_dir = item.is_dir()
            dir_suffix = "/" if is_dir else ""
            completed_path = (
                os.path.join(dir_part, item.name) if dir_part else item.name
            )

            needs_quote = bool(_NEEDS_QUOTE_RE.search(completed_path))

            display_name = item.name + dir_suffix
            meta = "Directory" if is_dir else "File"

            if not is_dir:
                try:
                    size_kb = item.stat().st_size / 1024
                    meta = f"File ({size_kb:.1f} KB)"
                except OSError:
                    pass

            # Append trailing slash to directory completions so the user
            # can continue typing a deeper path immediately.
            completion_text = completed_path + dir_suffix

            if quote_char is not None:
                # User already opened quotes — close them after the path.
                text = completion_text + quote_char
                start_pos = -len(prefix)
            elif needs_quote:
                # Wrap in double quotes, replacing the bare @ prefix.
                text = f'"{completion_text}"'
                start_pos = -len(prefix) - 1
            else:
                text = completion_text
                start_pos = -len(prefix)

            yield Completion(
                text,
                start_position=start_pos,
                display=display_name,
                display_meta=meta,
            )
