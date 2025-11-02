from datetime import datetime

from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label

from ..agent import OllamaAgent


class SessionListScreen(ModalScreen):
    """Modal screen to list and select sessions."""

    CSS = """
    SessionListScreen {
        align: center middle;
    }

    #session-dialog {
        width: 90;
        height: 30;
        border: thick $background 80%;
        background: $surface;
        padding: 1 2;
        overflow-x: hidden;
    }

    #session-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        color: $accent;
    }

    #session-list {
        height: 18;
        border: solid $primary;
        margin: 1 0;
        overflow-x: hidden;
    }

    #button-container {
        height: 3;
        align: center middle;
    }

    .session-row {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    .session-info {
        width: 3fr;
        height: auto;
    }

    .session-load-btn {
        width: 1fr;
        min-width: 10;
        margin-left: 1;
    }

    .session-delete-btn {
        width: 1fr;
        min-width: 10;
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
    ]

    def __init__(self, sessions: list[dict], agent: OllamaAgent):
        """
        Initialize the session list screen.

        Args:
            sessions: List of session dictionaries.
            agent: The agent instance for deleting sessions.
        """
        super().__init__()
        self.sessions = sessions
        self.agent = agent

    def compose(self) -> "ComposeResult":
        """Create the session list dialog."""
        with Container(id="session-dialog"):
            yield Label("Select a Session to Load or Delete", id="session-title")

            if not self.sessions:
                yield Label("No sessions found", id="no-sessions")
            else:
                with VerticalScroll(id="session-list"):
                    for session in self.sessions:
                        session_id = session['session_id']
                        count = session['message_count']
                        preview = session['preview']
                        last_time = session.get('last_message', 'Unknown')

                        # Format the time
                        time_str = last_time
                        if last_time != 'Unknown':
                            try:
                                dt = datetime.fromisoformat(last_time)
                                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                            except:
                                pass

                        # Create a row with session info and buttons
                        with Horizontal(classes="session-row"):
                            session_text = f"[bold]{session_id[:8]}...[/bold] ({count} msgs)\n{time_str}\n{preview[:40]}..."
                            yield Label(session_text, classes="session-info")
                            yield Button("Load", variant="primary", id=f"load-{session_id}",
                                         classes="session-load-btn")
                            yield Button("Delete", variant="error", id=f"delete-{session_id}",
                                         classes="session-delete-btn")

            with Container(id="button-container"):
                yield Button("Close", variant="default", id="cancel-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        button_id = event.button.id
        if not button_id:
            return

        if button_id == "cancel-button":
            self.dismiss(None)
        elif button_id.startswith("load-"):
            session_id = button_id.removeprefix("load-")
            self.dismiss(f"load:{session_id}")
        elif button_id.startswith("delete-"):
            session_id = button_id.removeprefix("delete-")
            if self.agent.delete_session(session_id):
                self.sessions = self.agent.list_sessions()
                self.refresh(recompose=True)
