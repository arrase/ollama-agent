"""REPL interface for Ollama Agent."""

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

from rich.console import Console
from rich.panel import Panel

from ..agent import AgentRuntime
from ..agent.builtin_tools import set_rag_manager, set_tool_timeout
from ..rag import RAGContext, RAGManager, load_rag_database
from ..skills import SkillsContext
from ..streaming import ConsoleStreamingRenderer, stream_agent_events
from ..tasks.commands import CLIContext
from .completer import SlashCommandCompleter
from .dispatch import build_repl_handlers, render_repl_help
from .model_commands import set_model
from .repl_wizards import safe_call, skill_create_wizard, task_create_wizard
from .session_commands import new_session


class OllamaREPL:
    """Read-Eval-Print Loop for interacting with the Ollama Agent."""

    def __init__(
        self,
        runtime: AgentRuntime,
        rag_database: str | None = None,
    ):
        self.runtime = runtime
        self.console = Console()
        self.session: PromptSession = PromptSession(
            style=Style.from_dict({
                "prompt": "#81a1c1 bold",  # Nord blue
                "arrow": "#88c0d0 bold",   # Frost cyan
            }),
            completer=SlashCommandCompleter(self._get_commands),
        )
        self._task_ctx = CLIContext(console=self.console)
        self._skills_ctx = SkillsContext(console=self.console)
        self._initial_rag_database = rag_database
        self._rag_ctx: RAGContext | None = None
        self._commands: dict | None = None

    def _get_rag_ctx(self) -> RAGContext:
        if self._rag_ctx is None:
            mgr = RAGManager(self.runtime.settings.rag)
            self._rag_ctx = RAGContext(console=self.console, rag_manager=mgr)
            set_rag_manager(mgr)
        return self._rag_ctx

    def _get_commands(self) -> dict:
        """Lazily build and cache REPL command handlers."""
        if self._commands is None:
            self._commands = build_repl_handlers(
                task_ctx=self._task_ctx,
                skills_ctx=self._skills_ctx,
                get_rag_ctx=self._get_rag_ctx,
                console=self.console,
                current_model=lambda: self.runtime.settings.model.name,
                base_url=lambda: self.runtime.settings.model.base_url,
                switch_model=self._switch_model,
                handle_exit=self._handle_exit,
                handle_clear=self._handle_clear,
                handle_new=self._handle_new_session,
                handle_task_create=lambda args: task_create_wizard(
                    self.session, self.console, self._task_ctx,
                    self.runtime.settings.model.name,
                    self.runtime.settings.model.reasoning_effort,
                    args,
                ),
                handle_skill_create=lambda args: skill_create_wizard(
                    self.session, self.console, self._skills_ctx, args,
                ),
            )
        return self._commands


    async def cleanup(self) -> None:
        if self._rag_ctx:
            self._rag_ctx.rag_manager.unload()
        await self.runtime.aclose()

    def _print_header(self, new_session: bool = False) -> None:
        """Print the REPL header inside a beautiful box Panel."""
        ms = self.runtime.settings.model
        
        rag_info = ""
        if self._rag_ctx and self._rag_ctx.rag_manager.current_database:
            rag_info = f"  |  [bold]RAG:[/bold] [cyan]{self._rag_ctx.rag_manager.current_database}[/cyan]"

        title = "[bold green]🤖 Ollama Agent[/bold green]"
        if new_session:
            title += " [dim](New Session)[/dim]"

        info_lines = [
            f"[bold]Model:[/bold] [cyan]{ms.name}[/cyan]  |  [bold]Effort:[/bold] [cyan]{ms.reasoning_effort}[/cyan]{rag_info}",
            "[dim]─────────────────────────────────────────────────────[/dim]",
            "Type [bold green]/help[/bold green] for commands or just start typing to chat."
        ]

        panel = Panel(
            "\n".join(info_lines),
            title=title,
            title_align="left",
            border_style="green",
            expand=True,
            padding=(1, 1)
        )
        self.console.print()
        self.console.print(panel)
        self.console.print()

    async def run(self) -> None:
        # Load initial RAG database if specified
        rag_ctx = self._get_rag_ctx()
        if self._initial_rag_database:
            try:
                load_rag_database(rag_ctx, self._initial_rag_database)
            except SystemExit:
                pass

        self._print_header(new_session=False)

        # Initialize the runtime
        set_tool_timeout(self.runtime.settings.runtime.builtin_tool_timeout)
        await self.runtime.reload()

        try:
            while True:
                try:
                    if user_input := (
                        await self.session.prompt_async(HTML("<arrow>❯</arrow> "))
                    ).strip():
                        await (
                            self.handle_command(user_input)
                            if user_input.startswith("/")
                            else self.handle_chat(user_input)
                        )
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break
                except Exception as e:
                    self.console.print(f"[red]Error:[/red] {e}")
        finally:
            await self.cleanup()
            self.console.print("[bold yellow]Goodbye![/bold yellow]")

    async def handle_command(self, command: str) -> None:
        """Handle slash commands."""
        parts = command.split()
        cmd, args = parts[0].lower(), parts[1:]
        commands = self._get_commands()

        if cmd == "/help":
            render_repl_help(self.console, commands)
            return

        if cmd not in commands:
            self.console.print(f"[red]Unknown command:[/red] {cmd}")
            return

        spec = commands[cmd]
        if spec.usage and not args:
            self.console.print(f"[red]Usage: {spec.usage}[/red]")
            return

        await safe_call(spec.handler, args)

    async def _handle_new_session(self, args: list[str]) -> None:
        self.runtime.thread_id = new_session(self.console)
        self.console.clear()
        self._print_header(new_session=True)

    def _handle_exit(self, args: list[str]) -> None:
        raise EOFError

    def _handle_clear(self, args: list[str]) -> None:
        self.console.clear()

    async def _switch_model(self, model_name: str) -> None:
        await set_model(
            self.console,
            model_name,
            runtime=self.runtime,
        )
        # Model is already updated in runtime by set_model

    async def handle_chat(self, prompt: str) -> None:
        renderer = ConsoleStreamingRenderer(self.console)
        await stream_agent_events(self.runtime, prompt, renderer, auto_close=True)

