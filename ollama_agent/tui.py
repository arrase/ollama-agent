"""Terminal user interface (TUI) using Textual."""

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Header, Footer, Input, RichLog, Static
from textual.binding import Binding
from rich.text import Text
from .agent import OllamaAgent


class ChatInterface(App):
    """Chat interface to interact with the AI agent."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #chat-container {
        height: 1fr;
        border: solid $primary;
        background: $panel;
        padding: 1;
        margin-bottom: 1;
    }
    
    #chat-log {
        height: 1fr;
        background: $panel;
        scrollbar-gutter: stable;
    }
    
    #input-container {
        height: 3;
        border: solid $accent;
        background: $surface;
        padding: 0 1;
    }
    
    #user-input {
        width: 100%;
        height: 100%;
        border: none;
    }
    
    .user-message {
        color: $accent;
        text-style: bold;
    }
    
    .agent-message {
        color: $success;
    }
    
    .system-message {
        color: $warning;
        text-style: italic;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
    ]
    
    def __init__(self, agent: OllamaAgent):
        """
        Initializes the interface.
        
        Args:
            agent: The AI agent to use.
        """
        super().__init__()
        self.agent = agent
        
    def compose(self) -> ComposeResult:
        """Creates the interface widgets."""
        yield Header()
        
        with Vertical(id="chat-container"):
            yield RichLog(id="chat-log", highlight=True, markup=True)
        
        with Container(id="input-container"):
            yield Input(placeholder="Escribe tu mensaje aquí...", id="user-input")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Executes when the application is mounted."""
        self.title = "Ollama Agent - Chat"
        self.sub_title = f"Model: {self.agent.model}"
        
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(Text("Welcome to Ollama Agent!", style="bold cyan"))
        chat_log.write(Text("Type your message and press Enter to send.", style="italic"))
        chat_log.write(Text("Press Ctrl+C to exit or Ctrl+L to clear the chat.", style="italic"))
        chat_log.write("")
        
        # Focus the input
        self.query_one("#user-input", Input).focus()
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Handles user message submission.
        
        Args:
            event: The submission event.
        """
        message = event.value.strip()
        if not message:
            return
        
        # Clear the input
        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""
        
        # Show user message
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(Text(f"User: {message}", style="bold blue"))
        
        # Show "thinking..." message
        chat_log.write(Text("Agent: thinking...", style="italic yellow"))
        
        # Get response from agent
        try:
            response = await self.agent.run_async(message)
            
            # Clear the last line (the "thinking...")
            # Instead, show the real response
            chat_log.write(Text(f"Agent: {response}", style="bold green"))
            chat_log.write("")
        except Exception as e:
            chat_log.write(Text(f"Error: {str(e)}", style="bold red"))
            chat_log.write("")
        
        # Scroll to the end
        chat_log.scroll_end(animate=False)
    
    def action_clear(self) -> None:
        """Clears the chat."""
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.clear()
        chat_log.write(Text("Chat cleared.", style="italic yellow"))
        chat_log.write("")
