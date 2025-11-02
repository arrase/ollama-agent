"""Terminal user interface (TUI) using Textual."""

from datetime import datetime
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, RichLog, Static
from rich.text import Text

from .agent import OllamaAgent


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
        self.action_result = None  # 'load:session_id' or 'delete:session_id' or None
    
    def compose(self) -> ComposeResult:
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
                            yield Button("Load", variant="primary", id=f"load-{session_id}", classes="session-load-btn")
                            yield Button("Delete", variant="error", id=f"delete-{session_id}", classes="session-delete-btn")
            
            with Container(id="button-container"):
                yield Button("Close", variant="default", id="cancel-button")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        button_id = event.button.id
        
        if button_id == "cancel-button":
            self.dismiss(None)
        elif button_id and button_id.startswith("load-"):
            session_id = button_id.replace("load-", "")
            self.dismiss(f"load:{session_id}")
        elif button_id and button_id.startswith("delete-"):
            session_id = button_id.replace("delete-", "")
            # Delete the session
            success = self.agent.delete_session(session_id)
            if success:
                # Refresh the session list
                self.sessions = self.agent.list_sessions()
                self.refresh(recompose=True)
            else:
                # Could show an error message here
                pass


class ChatInterface(App):
    """Chat interface to interact with the AI agent."""
    
    CSS = """
    #chat-container {
        height: 1fr;
        border: solid $primary;
        padding: 1;
        margin-bottom: 1;
    }
    
    #chat-log {
        height: 1fr;
        scrollbar-gutter: stable;
    }
    
    #input-container {
        height: 3;
        border: solid $accent;
        padding: 0 1;
    }
    
    #user-input {
        width: 100%;
        height: 100%;
        border: none;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+r", "reset_session", "New Session"),
        Binding("ctrl+s", "load_session", "Load Session"),
    ]
    
    def __init__(self, agent: OllamaAgent):
        """
        Initialize the interface.
        
        Args:
            agent: The AI agent to use.
        """
        super().__init__()
        self.agent = agent
        
    def compose(self) -> ComposeResult:
        """Create the interface widgets."""
        yield Header()
        
        with Vertical(id="chat-container"):
            yield RichLog(id="chat-log", highlight=True, markup=True)
        
        with Container(id="input-container"):
            yield Input(placeholder="Type your message here...", id="user-input")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Execute when the application is mounted."""
        session_id = self.agent.get_session_id()
        self.title = "Ollama Agent - Chat"
        self.sub_title = f"Model: {self.agent.model} | Session: {session_id[:8]}..." if session_id else f"Model: {self.agent.model}"
        
        chat_log = self.query_one("#chat-log", RichLog)
        self._write_system_message(chat_log, "Welcome to Ollama Agent!")
        self._write_system_message(chat_log, f"Session ID: {session_id}")
        self._write_system_message(chat_log, "Type your message and press Enter to send.")
        self._write_system_message(chat_log, "Press Ctrl+C to exit, Ctrl+L to clear, Ctrl+R for new session, or Ctrl+S to load a session.")
        chat_log.write("")
        
        # Focus the input
        self.query_one("#user-input", Input).focus()
    
    def _write_system_message(self, chat_log: RichLog, message: str) -> None:
        """
        Write a system message to the chat log.
        
        Args:
            chat_log: The RichLog widget.
            message: The message to write.
        """
        chat_log.write(Text(message, style="italic cyan"))
    
    def _write_user_message(self, chat_log: RichLog, message: str) -> None:
        """
        Write a user message to the chat log.
        
        Args:
            chat_log: The RichLog widget.
            message: The message to write.
        """
        chat_log.write(Text(f"User: {message}", style="bold blue"))
    
    def _write_agent_message(self, chat_log: RichLog, message: str) -> None:
        """
        Write an agent message to the chat log.
        
        Args:
            chat_log: The RichLog widget.
            message: The message to write.
        """
        chat_log.write(Text(f"Agent: {message}", style="bold green"))
    
    def _write_error_message(self, chat_log: RichLog, message: str) -> None:
        """
        Write an error message to the chat log.
        
        Args:
            chat_log: The RichLog widget.
            message: The message to write.
        """
        chat_log.write(Text(f"Error: {message}", style="bold red"))
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user message submission."""
        message = event.value.strip()
        if not message:
            return
        
        # Clear the input
        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""
        
        chat_log = self.query_one("#chat-log", RichLog)
        self._write_user_message(chat_log, message)
        chat_log.write(Text("Agent: thinking...", style="italic yellow"))
        
        # Get response from agent
        try:
            response = await self.agent.run_async(message)
            self._write_agent_message(chat_log, response)
        except Exception as e:
            self._write_error_message(chat_log, str(e))
        
        chat_log.write("")
        chat_log.scroll_end(animate=False)
    
    def action_clear(self) -> None:
        """Clear the chat."""
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.clear()
        self._write_system_message(chat_log, "Chat cleared.")
        chat_log.write("")
    
    def action_reset_session(self) -> None:
        """Reset the session and start a new conversation."""
        session_id = self.agent.reset_session()
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.clear()
        self._write_system_message(chat_log, "New session started!")
        self._write_system_message(chat_log, f"Session ID: {session_id}")
        self._write_system_message(chat_log, "Previous conversation history has been cleared.")
        chat_log.write("")
        
        # Update subtitle with new session ID
        self.sub_title = f"Model: {self.agent.model} | Session: {session_id[:8]}..."
    
    def action_load_session(self) -> None:
        """Show the session list dialog."""
        sessions = self.agent.list_sessions()
        
        async def handle_session_action(action: str | None) -> None:
            """Handle the selected action."""
            if action and action.startswith("load:"):
                session_id = action.replace("load:", "")
                await self._load_selected_session(session_id)
        
        self.push_screen(SessionListScreen(sessions, self.agent), handle_session_action)
    
    async def _load_selected_session(self, session_id: str) -> None:
        """Load the selected session and display its history."""
        self.agent.load_session(session_id)
        
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.clear()
        
        self._write_system_message(chat_log, f"Loaded session: {session_id}")
        chat_log.write("")
        
        # Load and display session history
        history = await self.agent.get_session_history(session_id)
        
        for item in history:
            if isinstance(item, dict):
                role = item.get('role', 'unknown')
                content = item.get('content', '')
                
                # Extract text from content (handles both string and array formats)
                text = ''
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    # For assistant messages with array content
                    text_parts = []
                    for c in content:
                        if isinstance(c, dict) and 'text' in c:
                            text_parts.append(c['text'])
                    text = ' '.join(text_parts)
                
                if role == 'user' and text:
                    self._write_user_message(chat_log, text)
                elif role == 'assistant' and text:
                    self._write_agent_message(chat_log, text)
        
        chat_log.write("")
        chat_log.scroll_end(animate=False)
        
        # Update subtitle
        self.sub_title = f"Model: {self.agent.model} | Session: {session_id[:8]}..."
