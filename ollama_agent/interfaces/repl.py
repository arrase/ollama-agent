"""REPL interface for Ollama Agent."""

import inspect

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel

from ..agent import AgentRuntime
from ..agent.builtin_tools import set_rag_manager, set_tool_timeout
from ..rag import RAGContext, RAGManager, load_rag_database
from ..skills import SkillsContext, create_skill
from ..streaming import ConsoleStreamingRenderer, stream_agent_events
from ..tasks.commands import CLIContext, create_task
from .dispatch import build_repl_handlers, render_repl_help
from .model_commands import set_model
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
            })
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
                switch_model=self._switch_model,
            )
        return self._commands

    @staticmethod
    async def _safe(fn, *args, **kwargs):
        """Call fn(*args, **kwargs), silencing SystemExit (already printed)."""
        try:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                await result
        except SystemExit:
            pass

    async def cleanup(self) -> None:
        if self._rag_ctx:
            self._rag_ctx.rag_manager.unload()
        await self.runtime.aclose()

    async def run(self) -> None:
        # Load initial RAG database if specified
        rag_ctx = self._get_rag_ctx()
        if self._initial_rag_database:
            try:
                load_rag_database(rag_ctx, self._initial_rag_database)
            except SystemExit:
                pass

        rag_info = ""
        if rag_ctx.rag_manager.current_database:
            rag_info = f" | RAG: [cyan]{rag_ctx.rag_manager.current_database}[/cyan]"

        ms = self.runtime.settings.model
        self.console.print()
        self.console.print("  [bold green]🤖 Ollama Agent[/bold green]")
        self.console.print("  [dim]──────────────────────────────────────────────────────────────────[/dim]")
        self.console.print(f"  [bold]Model:[/bold] [cyan]{ms.name}[/cyan]  |  [bold]Effort:[/bold] [cyan]{ms.reasoning_effort}[/cyan]{rag_info}")
        self.console.print("  [dim]──────────────────────────────────────────────────────────────────[/dim]")
        self.console.print("  Type [bold green]/help[/bold green] for commands or just start typing to chat.\n")

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

        # Dispatch to registered handlers (excludes inline-handled commands)
        _INLINE = {"/exit", "/quit", "/clear", "/new", "/task-create", "/skill-create"}
        if cmd in commands and cmd not in _INLINE:
            if (
                cmd
                in {
                    "/task-run",
                    "/task-delete",
                    "/model-set",
                    "/rag-create",
                    "/rag-delete",
                    "/rag-load",
                    "/skill-show",
                    "/skill-delete",
                }
                and not args
            ):
                self.console.print(f"[red]Usage: {cmd} <value>[/red]")
                return
            if cmd == "/rag-add" and not args:
                self.console.print("[red]Usage: /rag-add <file_or_dir> [--dir][/red]")
                return
            await self._safe(commands[cmd].handler, args)
            return

        match cmd:
            case "/exit" | "/quit":
                raise EOFError
            case "/clear":
                self.console.clear()
            case "/new":
                await self._handle_new_session()
            case "/task-create":
                await self._handle_task_create(args)
            case "/skill-create":
                await self._handle_skill_create(args)
            case _:
                self.console.print(f"[red]Unknown command:[/red] {cmd}")

    async def _prompt_multiline(self, prompt_html: str, hint: str) -> str:
        self.console.print(f"[dim]{hint}[/dim]")
        buf = self.session.default_buffer
        old_multiline = buf.multiline
        try:
            return await self.session.prompt_async(HTML(prompt_html), multiline=True)
        finally:
            buf.multiline = old_multiline

    async def _prompt_line(self, label: str, default: str = "") -> str:
        prompt_html = f"<b>{label}</b>"
        if default:
            prompt_html += f" (default: {default})"
        prompt_html += "> "
        val = (await self.session.prompt_async(HTML(prompt_html))).strip()
        return val or default

    async def _handle_task_create(self, args: list[str]) -> None:
        if not args:
            self.console.print("[red]Usage: /task-create <task_id> [--force][/red]")
            return
        task_id, force = args[0], "--force" in args[1:]
        ms = self.runtime.settings.model

        title = await self._prompt_line("title")
        model = await self._prompt_line("model", ms.name)
        effort = await self._prompt_line("effort", ms.reasoning_effort)
        task_prompt = await self._prompt_multiline(
            "<b>prompt> </b>",
            "Enter the task prompt (multiline). Finish with Esc+Enter.",
        )
        await self._safe(
            create_task,
            self._task_ctx,
            task_id,
            title=title,
            prompt=task_prompt,
            model=model,
            reasoning_effort=effort,
            force=force,
        )

    async def _handle_skill_create(self, args: list[str]) -> None:
        if not args:
            self.console.print("[red]Usage: /skill-create <skill_id> [--force][/red]")
            return
        skill_id, force = args[0], "--force" in args[1:]

        name = await self._prompt_line("name")
        description = await self._prompt_line("description")
        instructions = await self._prompt_multiline(
            "<b>instructions> </b>",
            "Enter skill instructions (multiline markdown). Finish with Esc+Enter.",
        )
        await self._safe(
            create_skill,
            self._skills_ctx,
            skill_id,
            name=name,
            description=description,
            instructions=instructions,
            force=force,
        )

    async def _handle_new_session(self) -> None:
        self.runtime.thread_id = new_session(self.console)
        self.console.clear()
        ms = self.runtime.settings.model
        self.console.print()
        self.console.print("  [bold green]🤖 Ollama Agent[/bold green] [dim](New Session)[/dim]")
        self.console.print("  [dim]──────────────────────────────────────────────────────────────────[/dim]")
        self.console.print(f"  [bold]Model:[/bold] [cyan]{ms.name}[/cyan]  |  [bold]Effort:[/bold] [cyan]{ms.reasoning_effort}[/cyan]")
        self.console.print("  [dim]──────────────────────────────────────────────────────────────────[/dim]")
        self.console.print("  Type [bold green]/help[/bold green] for commands or just start typing to chat.\n")

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
