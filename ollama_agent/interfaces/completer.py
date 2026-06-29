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

# Maximum number of completion candidates to prevent UI slowdown.
_MAX_COMPLETIONS = 200


class SlashCommandCompleter(Completer):
    """Tab-completer that suggests slash commands and file mentions.

    Slash commands: trigger when typing ``/`` as the first word.
    File mentions: trigger when typing ``@`` anywhere in the line.
    File completions recursively walk the project tree and present every
    file and directory as a flat, filterable list of relative paths —
    similar to Codex or Claude Code.
    """

    def __init__(self, get_commands: Callable[[], dict[str, REPLCommand]]) -> None:
        self._get_commands = get_commands

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text_before = document.text_before_cursor

        # 1. File/directory @-mention completion
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

        # 2. Slash command completion
        text = text_before.lstrip()
        if not text.startswith("/"):
            return

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
        """Recursively walk the project tree and yield matching paths.

        Every file and directory under the working directory is presented as
        a relative path.  The typed *prefix* filters the flat list so the
        user can drill into deep paths by simply typing (e.g. ``@core/pr``
        immediately narrows to ``ollama_agent/core/prompt_processor.py``).
        """
        cwd = Path.cwd()
        show_hidden = prefix.startswith(".")
        count = 0

        try:
            tree = os.walk(cwd)
        except OSError as exc:
            _log.debug("Cannot walk '%s': %s", cwd, exc)
            return

        for root, dirs, files in tree:
            root_path = Path(root)

            # Build the filtered, sorted list of child directories with
            # their precomputed relative paths.  This list simultaneously
            # controls which directories os.walk descends into (via the
            # dirs[:] in-place mutation) and which ones are emitted as
            # completion candidates.
            candidate_dirs: list[tuple[str, str]] = []
            for dirname in sorted(dirs):
                if dirname in IGNORED_DIRECTORY_NAMES:
                    continue
                if not show_hidden and dirname.startswith("."):
                    continue
                rel = str((root_path / dirname).relative_to(cwd)) + "/"
                # Prune branches that cannot contain prefix matches:
                # keep only ancestors of the prefix or descendants of it.
                if not (prefix.startswith(rel) or rel.startswith(prefix)):
                    continue
                candidate_dirs.append((dirname, rel))

            # Control os.walk traversal — only descend into kept dirs.
            dirs[:] = [d for d, _ in candidate_dirs]

            # --- Emit directory completions ---
            for _, rel in candidate_dirs:
                if count >= _MAX_COMPLETIONS:
                    return
                # Skip the directory itself when the prefix matches exactly
                # — the user already typed it and wants its contents.
                if rel == prefix or not rel.startswith(prefix):
                    continue
                count += 1
                yield self._build_completion(
                    rel, prefix, quote_char, display_meta="Directory"
                )

            # --- Emit file completions ---
            for filename in sorted(files):
                if count >= _MAX_COMPLETIONS:
                    return
                if not show_hidden and filename.startswith("."):
                    continue
                rel = str((root_path / filename).relative_to(cwd))
                if not rel.startswith(prefix):
                    continue

                meta = "File"
                try:
                    size_kb = (root_path / filename).stat().st_size / 1024
                    meta = f"File ({size_kb:.1f} KB)"
                except OSError:
                    pass

                count += 1
                yield self._build_completion(
                    rel, prefix, quote_char, display_meta=meta
                )

    def _build_completion(
        self,
        rel_path: str,
        prefix: str,
        quote_char: str | None,
        display_meta: str,
    ) -> Completion:
        """Build a ``Completion`` for a relative path string."""
        needs_quote = bool(_NEEDS_QUOTE_RE.search(rel_path))

        if quote_char is not None:
            text = rel_path + quote_char
            start_pos = -len(prefix)
        elif needs_quote:
            text = f'"{rel_path}"'
            start_pos = -len(prefix) - 1
        else:
            text = rel_path
            start_pos = -len(prefix)

        return Completion(
            text,
            start_position=start_pos,
            display=rel_path,
            display_meta=display_meta,
        )
