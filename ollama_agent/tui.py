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
        """Ejecuta cuando la aplicación se monta."""
        self.title = "Ollama Agent - Chat"
        self.sub_title = f"Modelo: {self.agent.model}"
        
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(Text("¡Bienvenido a Ollama Agent!", style="bold cyan"))
        chat_log.write(Text("Escribe tu mensaje y presiona Enter para enviar.", style="italic"))
        chat_log.write(Text("Presiona Ctrl+C para salir o Ctrl+L para limpiar el chat.", style="italic"))
        chat_log.write("")
        
        # Enfocar el input
        self.query_one("#user-input", Input).focus()
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Maneja el envío de mensajes del usuario.
        
        Args:
            event: El evento de envío.
        """
        message = event.value.strip()
        if not message:
            return
        
        # Limpiar el input
        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""
        
        # Mostrar mensaje del usuario
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(Text(f"Usuario: {message}", style="bold blue"))
        
        # Mostrar mensaje de "pensando..."
        chat_log.write(Text("Agente: pensando...", style="italic yellow"))
        
        # Obtener respuesta del agente
        try:
            response = await self.agent.run_async(message)
            
            # Limpiar la última línea (el "pensando...")
            # En su lugar, mostramos la respuesta real
            chat_log.write(Text(f"Agente: {response}", style="bold green"))
            chat_log.write("")
        except Exception as e:
            chat_log.write(Text(f"Error: {str(e)}", style="bold red"))
            chat_log.write("")
        
        # Hacer scroll al final
        chat_log.scroll_end(animate=False)
    
    def action_clear(self) -> None:
        """Limpia el chat."""
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.clear()
        chat_log.write(Text("Chat limpiado.", style="italic yellow"))
        chat_log.write("")
