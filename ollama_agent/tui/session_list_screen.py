from datetime import datetime
from typing import Iterable, Sequence, cast

from textual.containers import Horizontal
from textual.widgets import Button, Label

from ..agent import OllamaAgent
from .list_modal import ListModalScreen


class SessionListScreen(ListModalScreen):
    """Modal screen to list and select sessions."""

    def __init__(self, agent: OllamaAgent):
        super().__init__("Select a Session to Load or Delete", "No sessions found")
        self.agent = agent

    def load_items(self) -> Iterable[object]:
        return self.agent.list_sessions()

    def render_rows(self, items: Sequence[object]):
        for session in items:
            data = cast(dict[str, object], session)
            session_id = str(data.get("session_id", ""))
            raw_count = data.get("message_count", 0)
            try:
                count = int(raw_count)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                count = 0
            preview = str(data.get("preview", ""))
            timestamp = self._format_timestamp(str(data.get("last_message", "Unknown")))
            text = f"[bold]{session_id[:8]}...[/bold] ({count} msgs)\n{timestamp}\n{preview[:40]}..."
            yield Horizontal(
                Label(text, classes="entry-info"),
                Button("Load", variant="primary", id=f"load-{session_id}", classes="entry-btn"),
                Button("Delete", variant="error", id=f"delete-{session_id}", classes="entry-btn"),
                classes="entry-row",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "cancel-button":
            self.dismiss(None)
        elif button_id.startswith("load-"):
            self.dismiss(f"load:{button_id.removeprefix('load-')}")
        elif button_id.startswith("delete-"):
            session_id = button_id.removeprefix("delete-")
            if self.agent.delete_session(session_id):
                self.refresh(recompose=True)

    @staticmethod
    def _format_timestamp(value: str) -> str:
        if value == "Unknown":
            return value
        try:
            return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return value
