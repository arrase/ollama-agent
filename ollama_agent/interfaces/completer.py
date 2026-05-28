"""Autocomplete support for REPL slash commands."""

from typing import Callable, Iterable

from prompt_toolkit.completion import CompleteEvent, Completion, Completer
from prompt_toolkit.document import Document

from .dispatch import REPLCommand


class SlashCommandCompleter(Completer):
    """Tab-completer that suggests slash commands from the REPL registry.

    Only activates when the input starts with ``/``.  Command summaries
    are shown as meta-text next to each suggestion.
    """

    def __init__(self, get_commands: Callable[[], dict[str, REPLCommand]]) -> None:
        self._get_commands = get_commands

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor.lstrip()

        # Only complete when the user is typing a slash command (first word).
        if not text.startswith("/"):
            return

        # If we're past the first word, nothing to complete.
        if " " in text:
            return

        for name, spec in self._get_commands().items():
            if name.startswith(text):
                yield Completion(
                    name,
                    start_position=-len(text),
                    display_meta=spec.summary,
                )
