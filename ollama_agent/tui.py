"""Terminal user interface (TUI) using Textual."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Input, RichLog
from rich.text import Text

from .agent import OllamaAgent


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
        self._write_system_message(chat_log, "Press Ctrl+C to exit, Ctrl+L to clear the chat, or Ctrl+R to start a new session.")
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
