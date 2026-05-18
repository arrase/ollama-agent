"""REPL interface for Ollama Agent."""

import uuid

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel

from ..agent import AgentRuntime
from ..agent.builtin_tools import set_rag_manager, set_tool_timeout
from ..core import validate_reasoning_effort
from ..rag import RAGContext, RAGManager, load_rag_database
from ..settings import RAGSettings, load_settings
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
            style=Style.from_dict({"prompt": "#ansiwhite bold"})
        )
        self._task_ctx = CLIContext(console=self.console)
        self._skills_ctx = SkillsContext(console=self.console)
        self._initial_rag_database = rag_database
        self._rag_ctx: RAGContext | None = None

    def _get_rag_ctx(self) -> RAGContext:
        if self._rag_ctx is None:
            rag_settings = RAGSettings(
                rag_dir=self.runtime.settings.rag.rag_dir,
                embedder_model=self.runtime.settings.rag.embedder_model,
                embedder_base_url=self.runtime.settings.rag.embedder_base_url,
                embedding_dims=self.runtime.settings.rag.embedding_dims,
                default_top_k=self.runtime.settings.rag.default_top_k,
                chunk_size=self.runtime.settings.rag.chunk_size,
                chunk_overlap=self.runtime.settings.rag.chunk_overlap,
            )
            mgr = RAGManager(rag_settings)  # type: ignore[arg-type]
            self._rag_ctx = RAGContext(console=self.console, rag_manager=mgr)
            set_rag_manager(mgr)
        return self._rag_ctx

    @staticmethod
    async def _safe(fn, *args, **kwargs):
        """Call fn(*args, **kwargs), silencing SystemExit (already printed)."""
        try:
            result = fn(*args, **kwargs)
            if hasattr(result, "__await__"):
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
        self.console.print(
            Panel(
                f"[bold green]Ollama Agent[/bold green]\n"
                f"Model: [cyan]{ms.name}[/cyan] | Effort: [cyan]{ms.reasoning_effort}[/cyan]{rag_info}\n"
                "Type [bold]/help[/bold] for commands or just start typing to chat.",
                title="Welcome",
                border_style="green",
            )
        )

        # Initialize the runtime
        set_tool_timeout(self.runtime.settings.runtime.builtin_tool_timeout)
        await self.runtime.reload()

        try:
            while True:
                try:
                    if user_input := (
                        await self.session.prompt_async(HTML("<b>❯ </b>"))
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
        inline_commands = {
            "/exit",
            "/quit",
            "/clear",
            "/new",
            "/task-create",
            "/skill-create",
        }

        async def _require_arg(usage: str) -> str | None:
            if not args:
                self.console.print(f"[red]Usage: {usage}[/red]")
            return args[0] if args else None

        commands = build_repl_handlers(
            task_ctx=self._task_ctx,
            skills_ctx=self._skills_ctx,
            get_rag_ctx=self._get_rag_ctx,
            console=self.console,
            current_model=lambda: self.runtime.settings.model.name,
            switch_model=self._switch_model,
        )

        if cmd == "/help":
            render_repl_help(self.console, commands)
            return

        if cmd in inline_commands:
            pass
        elif cmd in commands:
            if cmd in {
                "/task-run",
                "/task-delete",
                "/model-set",
                "/rag-create",
                "/rag-delete",
                "/rag-load",
                "/skill-show",
                "/skill-delete",
            }:
                if await _require_arg(f"{cmd} <value>") is None:
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
            case "/task-create":
                await self._handle_task_create(args)
            case "/new":
                await self._handle_new_session()
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

    async def _handle_task_create(self, args: list[str]) -> None:
        if not args:
            self.console.print("[red]Usage: /task-create <task_id> [--force][/red]")
            return
        task_id, force = args[0], "--force" in args[1:]
        ms = self.runtime.settings.model
        title = (await self.session.prompt_async(HTML("<b>title> </b>"))).strip()
        model = (
            await self.session.prompt_async(
                HTML(f"<b>model</b> (default: {ms.name})> ")
            )
        ).strip() or ms.name
        effort = (
            await self.session.prompt_async(
                HTML(f"<b>effort</b> (default: {ms.reasoning_effort})> ")
            )
        ).strip() or ms.reasoning_effort
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
        name = (await self.session.prompt_async(HTML("<b>name> </b>"))).strip()
        description = (
            await self.session.prompt_async(HTML("<b>description> </b>"))
        ).strip()
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
        self.runtime.thread_id = new_session(self.console, self.runtime.thread_id)
        self.console.clear()
        ms = self.runtime.settings.model
        self.console.print(
            Panel(
                f"[bold green]Ollama Agent[/bold green]\n"
                f"Model: [cyan]{ms.name}[/cyan] | Effort: [cyan]{ms.reasoning_effort}[/cyan]\n"
                "Type [bold]/help[/bold] for commands or just start typing to chat.",
                title="New Session",
                border_style="green",
            )
        )

    async def _switch_model(self, model_name: str) -> None:
        new_model = await set_model(
            self.console,
            model_name,
            runtime=self.runtime,
        )
        # Model is already updated in runtime by set_model

    async def handle_chat(self, prompt: str) -> None:
        renderer = ConsoleStreamingRenderer(self.console)
        await stream_agent_events(self.runtime, prompt, renderer, auto_close=True)
