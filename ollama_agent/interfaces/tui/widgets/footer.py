"""Footer widget displaying active modes, keyboard shortcuts, and queue status."""

from __future__ import annotations

from typing import Any

from textual.widgets import Static

from ....i18n import _


class AgentFooter(Static):
    """Dynamic TUI Footer displaying keyboard shortcuts and live status."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._is_generating = False
        self._is_approval = False
        self._queued_count = 0

    def on_mount(self) -> None:
        self.update_footer()

    def set_generating(self, is_generating: bool) -> None:
        self._is_generating = is_generating
        self.update_footer()

    def set_approval(self, is_approval: bool) -> None:
        self._is_approval = is_approval
        self.update_footer()

    def set_queued_count(self, count: int) -> None:
        self._queued_count = count
        self.update_footer()

    def update_footer(self) -> None:
        queue_info = (
            f"  [dim]│[/dim]  [bold #38bdf8]⏳ {_('{count} queued', count=self._queued_count)}[/bold #38bdf8]"
            if self._queued_count > 0
            else ""
        )
        if self._is_approval:
            self.update(
                f"[bold #fbbf24]⚠ {_('Approval required:')}[/bold #fbbf24]   "
                f"[bold #7d8590]y[/bold #7d8590] [#c9d1d9]{_('approve')}[/#c9d1d9]   "
                f"[bold #7d8590]n[/bold #7d8590] [#c9d1d9]{_('reject')}[/#c9d1d9]   "
                f"[bold #7d8590]a[/bold #7d8590] [#c9d1d9]{_('allow session')}[/#c9d1d9]   "
                f"[bold #7d8590]esc[/bold #7d8590] [#c9d1d9]{_('cancel')}[/#c9d1d9]   "
                f"[bold #7d8590]←→[/bold #7d8590] [#c9d1d9]{_('select')}[/#c9d1d9]"
                f"{queue_info}"
            )
        elif self._is_generating:
            self.update(
                f"[bold #38bdf8]⟡ {_('Generating response...')}[/bold #38bdf8]   "
                f"[dim]{_('press esc or ^C to interrupt')}[/dim]"
                f"{queue_info}"
            )
        else:
            self.update(
                f"[bold #7d8590]enter[/bold #7d8590] [#c9d1d9]{_('send')}[/#c9d1d9]   "
                f"[bold #7d8590]\\+enter[/bold #7d8590] [#c9d1d9]{_('newline')}[/#c9d1d9]   "
                f"[bold #7d8590]tab[/bold #7d8590] [#c9d1d9]{_('complete')}[/#c9d1d9]   "
                f"[bold #7d8590]↑↓[/bold #7d8590] [#c9d1d9]{_('history')}[/#c9d1d9]   "
                f"[bold #7d8590]esc[/bold #7d8590] [#c9d1d9]{_('interrupt')}[/#c9d1d9]   "
                f"[bold #7d8590]/[/bold #7d8590] [#c9d1d9]{_('commands')}[/#c9d1d9]"
                f"{queue_info}"
            )
