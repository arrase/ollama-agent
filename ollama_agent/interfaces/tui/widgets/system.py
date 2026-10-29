"""System widgets for prompt queue and command outputs."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import islice
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual.widgets import Static

from ....core import shorten
from ....i18n import _

_MAX_VISIBLE_QUEUED = 3


def format_queue_preview(item: str) -> str:
    """Return a single-line, truncated, Rich-markup-escaped preview of a queued prompt."""
    return escape(shorten(item))


def format_queue_line(index: int, item: str) -> str:
    """Format a numbered queue entry line for console and widget display."""
    return f"  [dim]#{index}[/dim] {format_queue_preview(item)}"


class PromptQueueWidget(Static):
    """Widget displaying currently queued prompts and commands."""

    can_focus = False

    def update_queue(self, queue: Sequence[str]) -> None:
        if not queue:
            self.display = False
            return

        self.display = True
        count = len(queue)
        header = f"[bold #38bdf8]⏳ {_('Queued ({count})', count=count)}[/bold #38bdf8]"
        lines = [header]

        for i, item in enumerate(islice(queue, _MAX_VISIBLE_QUEUED), 1):
            lines.append(format_queue_line(i, item))

        if count > _MAX_VISIBLE_QUEUED:
            remaining = count - _MAX_VISIBLE_QUEUED
            lines.append(f"  [dim]... +{remaining} {_('more')}[/dim]")

        self.update("\n".join(lines))


class SystemOutputWidget(Static):
    """Dedicated TUI widget displaying system notifications and slash command outputs."""

    can_focus = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.display = False

    def show_output(self, content: str | Text, title: str | None = None) -> None:
        self.display = True
        display_title = title or _("System Output")
        header = f"[bold #38bdf8]⚙ {escape(display_title)}[/bold #38bdf8]  [dim]({_('esc to dismiss')})[/dim]"
        if isinstance(content, Text):
            header_text = Text.from_markup(header + "\n")
            self.update(Text.assemble(header_text, content))
        else:
            self.update(f"{header}\n{content}")

    def show_notice(self, notice: str | Text) -> None:
        self.display = True
        self.update(notice)

    def clear_output(self) -> None:
        self.display = False
        self.update("")
