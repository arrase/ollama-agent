"""REPL interface for Ollama Agent."""

from typing import Callable
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel

from ..agent import OllamaAgent
from ..streaming import ConsoleStreamingRenderer, stream_agent_events
from ..rag import RAGContext, load_rag_database
from ..skills import SkillsContext, create_skill
from ..tasks.commands import CLIContext, create_task
from .dispatch import build_repl_handlers, render_repl_help
from .model_commands import set_model


class OllamaREPL:
    """Read-Eval-Print Loop for interacting with the Ollama Agent."""

    def __init__(
        self,
        agent_factory: Callable[..., OllamaAgent],
        model: str,
        effort: str,
        rag_database: str | None = None,
    ):
        self.agent_factory, self.model, self.effort = agent_factory, model, effort
        self.console = Console()
        self.session = PromptSession(
            style=Style.from_dict({"prompt": "#ansiwhite bold"})
        )
        self.ctx = CLIContext(agent_factory, console=self.console)
        self._skills_ctx = SkillsContext(console=self.console)
        self.active_agent: OllamaAgent | None = None
        self._initial_rag_database = rag_database
        # RAGContext will be created with agent's RAGManager once agent is initialized
        self._rag_ctx: RAGContext | None = None

    def _get_rag_ctx(self) -> RAGContext:
        """Get or create RAGContext using agent's RAGManager."""
        if self._rag_ctx is None:
            agent = self._ensure_agent()
            self._rag_ctx = RAGContext(
                console=self.console, rag_manager=agent.rag_manager
            )
        return self._rag_ctx

    def _ensure_agent(self) -> OllamaAgent:
        if not self.active_agent:
            self.active_agent = self.agent_factory(
                model=self.model, reasoning_effort=self.effort
            )
        return self.active_agent

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
        if self.active_agent:
            await self.active_agent.cleanup()
            self.active_agent = None
        if self._rag_ctx:
            self._rag_ctx.rag_manager.unload()

    async def run(self) -> None:
        # Load initial RAG database if specified
        rag_ctx = self._get_rag_ctx()
        if self._initial_rag_database:
            try:
                load_rag_database(rag_ctx, self._initial_rag_database)
            except SystemExit:
                pass  # Error already printed

        rag_info = ""
        if rag_ctx.rag_manager.current_database:
            rag_info = f" | RAG: [cyan]{rag_ctx.rag_manager.current_database}[/cyan]"

        self.console.print(
            Panel(
                f"[bold green]Ollama Agent REPL[/bold green]\n"
                f"Model: [cyan]{self.model}[/cyan] | Effort: [cyan]{self.effort}[/cyan]{rag_info}\n"
                "Type [bold]/help[/bold] for commands or just start typing to chat.",
                title="Welcome",
                border_style="green",
            )
        )
        try:
            while True:
                try:
                    if user_input := (
                        await self.session.prompt_async(HTML("<b>>>> </b>"))
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
            task_ctx=self.ctx,
            skills_ctx=self._skills_ctx,
            get_rag_ctx=self._get_rag_ctx,
            ensure_agent=self._ensure_agent,
            console=self.console,
            current_model=lambda: self.model,
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
                "/session-load",
                "/session-delete",
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
        title = (await self.session.prompt_async(HTML("<b>title> </b>"))).strip()
        model = (
            await self.session.prompt_async(
                HTML(f"<b>model</b> (default: {self.model})> ")
            )
        ).strip() or self.model
        effort = (
            await self.session.prompt_async(
                HTML(f"<b>effort</b> (default: {self.effort})> ")
            )
        ).strip() or self.effort
        task_prompt = await self._prompt_multiline(
            "<b>prompt> </b>",
            "Enter the task prompt (multiline). Finish with Esc+Enter.",
        )
        await self._safe(
            create_task,
            self.ctx,
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
        if self.active_agent:
            self.active_agent.session_manager.reset_session()
        self.console.clear()
        self.console.print(
            Panel(
                f"[bold green]Ollama Agent REPL[/bold green]\n"
                f"Model: [cyan]{self.model}[/cyan] | Effort: [cyan]{self.effort}[/cyan]\n"
                "Type [bold]/help[/bold] for commands or just start typing to chat.",
                title="New Session",
                border_style="green",
            )
        )

    async def _switch_model(self, model_name: str) -> None:
        self.model, self.active_agent = await set_model(
            self.console,
            model_name,
            current_model=self.model,
            current_effort=self.effort,
            active_agent=self.active_agent,
            agent_factory=self.agent_factory,
        )

    async def handle_chat(self, prompt: str) -> None:
        agent = self._ensure_agent()
        renderer = ConsoleStreamingRenderer(self.console)
        await stream_agent_events(agent, prompt, renderer, auto_close=True)
