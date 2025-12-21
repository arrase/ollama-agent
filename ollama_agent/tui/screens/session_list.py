"""Session list modal screen."""

from datetime import datetime
from typing import Iterable, cast

from textual.widget import Widget

from ...agent import OllamaAgent
from .base import ListModalScreen, make_row


class SessionListScreen(ListModalScreen):
    """Modal screen to list and manage sessions."""

    def __init__(self, agent: OllamaAgent):
        super().__init__("Select a Session", "No sessions found")
        self.agent = agent

    def get_items(self) -> Iterable[object]:
        return self.agent.session_manager.list_sessions()

    def render_items(self, items: list[object]) -> Iterable[Widget]:
        for session in items:
            data = cast(dict[str, object], session)
            session_id = str(data.get("session_id", ""))
            count = int(data.get("message_count", 0) or 0)
            preview = str(data.get("preview", "")).strip()
            if not preview:
                preview = "(no messages)"
            preview = preview.replace("\n", " ")
            preview = preview[:80]
            timestamp = self._format_time(str(data.get("last_message", "")))

            text = (
                f"[bold]{session_id[:8]}...[/bold] ({count} msgs)\n"
                f"Last activity: {timestamp}\n"
                f"{preview}"
            )
            yield make_row(text, session_id, [
                ("load", "Load", "primary"),
                ("delete", "Delete", "error"),
            ])

    def handle_action(self, action: str, item_id: str) -> bool:
        if action == "delete":
            self.agent.session_manager.delete_session(item_id)
            return False  # Refresh, don't dismiss
        return True  # Dismiss for "load"

    @staticmethod
    def _format_time(value: str) -> str:
        if not value or value == "Unknown":
            return "Unknown"
        try:
            return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return value
