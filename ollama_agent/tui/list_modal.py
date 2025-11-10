"""Shared modal screen helpers for list-based dialogs."""

from __future__ import annotations

from typing import Iterable, Sequence

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Label


class ListModalScreen(ModalScreen):
    """Modal screen that renders a list of items with a close button."""

    CSS = """
    ListModalScreen {
        align: center middle;
    }

    #modal-dialog {
        width: 90;
        height: 30;
        border: thick $background 80%;
        background: $surface;
        padding: 1 2;
        overflow-x: hidden;
    }

    #modal-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        color: $accent;
    }

    #items-list {
        height: 18;
        border: solid $primary;
        margin: 1 0;
        overflow-x: hidden;
    }

    #button-container {
        height: 3;
        align: center middle;
    }

    .entry-row {
        width: 100%;
        margin-bottom: 1;
    }

    .entry-info {
        width: 3fr;
    }

    .entry-btn {
        width: 1fr;
        min-width: 10;
        margin-left: 1;
    }
    """

    BINDINGS = [Binding("escape", "dismiss", "Cancel")]

    def __init__(self, title: str, empty_message: str) -> None:
        super().__init__()
        self._title = title
        self._empty_message = empty_message

    def compose(self) -> ComposeResult:
        items = list(self.load_items())
        with Container(id="modal-dialog"):
            yield Label(self._title, id="modal-title")
            if items:
                with VerticalScroll(id="items-list"):
                    yield from self.render_rows(items)
            else:
                yield Label(self._empty_message, id="modal-empty")
            with Container(id="button-container"):
                yield Button("Close", variant="default", id="cancel-button")

    def render_rows(self, items: Sequence[object]) -> Iterable[Widget]:  # pragma: no cover - abstract
        raise NotImplementedError

    def load_items(self) -> Iterable[object]:  # pragma: no cover - abstract
        raise NotImplementedError
