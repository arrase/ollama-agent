"""Base modal screen for list-based dialogs."""

from typing import Iterable, Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Label, Static

ButtonVariant = Literal["default", "primary", "success", "warning", "error"]

# Common CSS for all modal screens
MODAL_CSS = """
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
        height: auto;
    }

    .entry-info {
        width: 1fr;
        height: auto;
        padding-right: 1;
    }

    .entry-btn {
        width: 12;
        min-width: 12;
        height: 3;
        min-height: 3;
        margin-left: 1;
        content-align: center middle;
        text-style: bold;
        text-opacity: 100%;
    }
"""


class ListModalScreen(ModalScreen):
    """Base modal screen that displays a list of items."""

    CSS = MODAL_CSS
    BINDINGS = [Binding("escape", "dismiss", "Cancel")]

    def __init__(self, title: str, empty_message: str = "No items found"):
        super().__init__()
        self._title = title
        self._empty_message = empty_message

    def compose(self) -> ComposeResult:
        items = list(self.get_items())
        with Container(id="modal-dialog"):
            yield Label(self._title, id="modal-title")
            if items:
                with VerticalScroll(id="items-list"):
                    yield from self.render_items(items)
            else:
                yield Label(self._empty_message, id="modal-empty")
            with Container(id="button-container"):
                yield Button("Close", variant="default", id="cancel-button")

    def get_items(self) -> Iterable[object]:
        """Override to provide items to display."""
        return []

    def render_items(self, items: list[object]) -> Iterable[Widget]:
        """Override to render item rows."""
        return []

    def handle_action(self, action: str, item_id: str) -> bool:
        """Handle an action. Return True to dismiss, False to refresh."""
        return True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "cancel-button":
            self.dismiss(None)
            return

        # Parse "action-id" pattern
        if "-" in button_id:
            action, item_id = button_id.split("-", 1)
            should_dismiss = self.handle_action(action, item_id)
            if should_dismiss:
                self.dismiss(f"{action}:{item_id}")
            else:
                self.refresh(recompose=True)


def make_row(
    text: str,
    item_id: str,
    actions: list[tuple[str, str, ButtonVariant]],
) -> Widget:
    """Create a standard row with info label and action buttons.
    
    Args:
        text: The markup text to display
        item_id: ID to use in button IDs
        actions: List of (action_name, label, variant) tuples
    """
    buttons = [
        Button(label, variant=variant, id=f"{action}-{item_id}", classes="entry-btn")
        for action, label, variant in actions
    ]
    return Horizontal(
        Static(text, classes="entry-info", markup=True),
        *buttons,
        classes="entry-row",
    )
