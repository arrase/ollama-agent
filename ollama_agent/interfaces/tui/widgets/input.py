"""Interactive multiline input widget with prompt history and autocomplete handling."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from textual import events
from textual.message import Message
from textual.widgets import OptionList, TextArea

from ....agent.episodic_memory import HistoryError, load_past_user_prompts
from ....i18n import _

_log = logging.getLogger(__name__)


class ReplInput(TextArea):
    """Interactive input field that captures Tab/arrow keys for autocomplete."""

    BINDINGS = [
        ("ctrl+v", "paste", _("Paste")),
        ("super+v", "paste", _("Paste")),
        ("shift+insert", "paste", _("Paste")),
    ]

    def action_paste(self) -> None:
        """Paste text from clipboard."""
        if self.app.clipboard:
            self.insert(self.app.clipboard)

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("highlight_cursor_line", False)
        kwargs.setdefault("show_line_numbers", False)
        kwargs.setdefault("placeholder", _("Ask anything, / for commands, @ for files..."))
        super().__init__(**kwargs)
        self._history: list[str] = []
        self._history_index: int = 0
        self._temp_input: str = ""

    def on_mount(self) -> None:
        super().on_mount()
        self._update_height()
        self.run_worker(self._load_history())

    def _update_height(self) -> None:
        lines = max(1, min(8, self.document.line_count))
        self.styles.height = lines

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._update_height()

    async def _load_history(self) -> None:
        try:
            db_entries = await asyncio.to_thread(load_past_user_prompts)
        except HistoryError as exc:
            _log.warning("Prompt history unavailable: %s", exc)
            self.app.show_system_notice(f"[yellow]⚠ {_('Prompt history unavailable: {exc}', exc=exc)}[/yellow]")
            db_entries = []
        self._history = list(db_entries)
        self._history_index = len(self._history)
        self._temp_input = ""

    def add_history_entry(self, entry: str) -> None:
        if not entry or entry.startswith("/"):
            return
        if self._history and self._history[-1] == entry:
            self._history_index = len(self._history)
            self._temp_input = ""
            return
        self._history.append(entry)
        self._history_index = len(self._history)
        self._temp_input = ""

    class Submitted(Message):
        """Emitted when the user submits the input."""

        def __init__(self, input_widget: ReplInput, value: str) -> None:
            super().__init__()
            self.input = input_widget
            self.value = value

    def _navigate_autocomplete(self, key: str, autolist: OptionList) -> None:
        if key == "down":
            if autolist.highlighted is None:
                autolist.highlighted = 0
            elif autolist.highlighted < autolist.option_count - 1:
                autolist.highlighted += 1
        elif key == "up" and autolist.highlighted is not None and autolist.highlighted > 0:
            autolist.highlighted -= 1

    def _handle_autocomplete_key(self, event: events.Key, app: Any, autolist: OptionList) -> bool:
        if not autolist.display or autolist.option_count == 0:
            return False

        if event.key in ("down", "up"):
            event.stop()
            event.prevent_default()
            self._navigate_autocomplete(event.key, autolist)
            return True
        if event.key == "tab":
            event.stop()
            event.prevent_default()
            if autolist.highlighted is not None:
                app.accept_completion(autolist.highlighted)
            return True
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            app.hide_autocomplete()
            return True
        if event.key == "enter" and autolist.highlighted is not None:
            event.stop()
            event.prevent_default()
            app.accept_completion(autolist.highlighted)
            return True
        return False

    def _handle_history_up(self, event: events.Key) -> bool:
        if self.document.line_count > 1:
            row, col = self.cursor_location
            if row > 0:
                return False
            if col > 0:
                self.action_cursor_line_start()
                event.stop()
                event.prevent_default()
                return True
        event.stop()
        event.prevent_default()
        if self._history:
            if self._history_index == len(self._history):
                self._temp_input = self.text
            if self._history_index > 0:
                self._history_index -= 1
                self.text = self._history[self._history_index]
                self.action_cursor_line_end()
                self._update_height()
        return True

    def _handle_history_down(self, event: events.Key) -> bool:
        if self.document.line_count > 1:
            row, col = self.cursor_location
            last_row = self.document.line_count - 1
            if row < last_row:
                return False
            last_line_len = len(self.document.get_line(last_row))
            if col < last_line_len:
                self.action_cursor_line_end()
                event.stop()
                event.prevent_default()
                return True
        event.stop()
        event.prevent_default()
        if self._history and self._history_index < len(self._history):
            self._history_index += 1
            if self._history_index == len(self._history):
                self.text = self._temp_input
            else:
                self.text = self._history[self._history_index]
            self.action_cursor_line_end()
            self._update_height()
        return True

    def _handle_history_key(self, event: events.Key) -> bool:
        if event.key == "up":
            return self._handle_history_up(event)
        if event.key == "down":
            return self._handle_history_down(event)
        return False

    def on_key(self, event: events.Key) -> None:
        app: Any = self.app
        autolist = app.query_one("#autocomplete-list", OptionList)
        if autolist.display:
            if self._handle_autocomplete_key(event, app, autolist):
                return
        else:
            if self._handle_history_key(event):
                return

        if event.key == "enter":
            row, col = self.cursor_location
            current_line = self.document.get_line(row)
            line_before_cursor = current_line[:col]
            if line_before_cursor.rstrip().endswith("\\"):
                event.stop()
                event.prevent_default()
                idx = line_before_cursor.rfind("\\")
                self.delete((row, idx), (row, col))
                self.insert("\n")
                self._update_height()
                return

            event.stop()
            event.prevent_default()
            val = self.text.strip()
            if val:
                self.post_message(self.Submitted(self, val))
