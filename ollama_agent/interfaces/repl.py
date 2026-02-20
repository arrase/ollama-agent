"""REPL interface for Ollama Agent."""

from typing import Callable
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from ..agent import OllamaAgent
from ..streaming import ConsoleStreamingRenderer, stream_agent_events
from ..rag import (
    RAGContext,
    add_rag_directory,
    add_rag_file,
    create_rag_database,
    delete_rag_database,
    list_rag_databases,
    load_rag_database,
    show_rag_status,
    unload_rag_database,
)
from ..skills import SkillsContext, create_skill, delete_skill, list_skills, show_skill
from ..tasks.commands import CLIContext, create_task, delete_task, list_tasks, run_task
from .agent_commands import (
    delete_session,
    find_session,
    list_models,
    list_sessions,
    load_session,
    set_model,
)


class OllamaREPL:
    """Read-Eval-Print Loop for interacting with the Ollama Agent."""

    def __init__(self, agent_factory: Callable[..., OllamaAgent], model: str, effort: str,
                 rag_database: str | None = None):
        self.agent_factory, self.model, self.effort = agent_factory, model, effort
        self.console = Console()
        self.session = PromptSession(
            style=Style.from_dict({"prompt": "#ansiwhite bold"}))
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
            self._rag_ctx = RAGContext(console=self.console, rag_manager=agent.rag_manager)
        return self._rag_ctx

    def _ensure_agent(self) -> OllamaAgent:
        if not self.active_agent:
            self.active_agent = self.agent_factory(
                model=self.model, reasoning_effort=self.effort)
        return self.active_agent

    @staticmethod
    async def _safe(fn, *args, **kwargs):
        """Call fn(*args, **kwargs), silencing SystemExit (already printed)."""
        try:
            result = fn(*args, **kwargs)
            if hasattr(result, '__await__'):
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

        self.console.print(Panel(
            f"[bold green]Ollama Agent REPL[/bold green]\n"
            f"Model: [cyan]{self.model}[/cyan] | Effort: [cyan]{self.effort}[/cyan]{rag_info}\n"
            "Type [bold]/help[/bold] for commands or just start typing to chat.",
            title="Welcome", border_style="green"))
        try:
            while True:
                try:
                    if user_input := (await self.session.prompt_async(HTML("<b>>>> </b>"))).strip():
                        await (self.handle_command(user_input) if user_input.startswith("/") else self.handle_chat(user_input))
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
        parts, cmd, args = command.split(), command.split()[
            0].lower(), command.split()[1:]

        async def _require_arg(usage: str) -> str | None:
            if not args:
                self.console.print(f"[red]Usage: {usage}[/red]")
            return args[0] if args else None

        match cmd:
            case "/exit" | "/quit": raise EOFError
            case "/help": self.show_help()
            case "/clear": self.console.clear()
            case "/tasks": list_tasks(self.ctx)
            case "/task-run":
                if tid := await _require_arg("/task-run <task_id>"):
                    await self._safe(run_task, self.ctx, tid)
            case "/task-delete":
                if tid := await _require_arg("/task-delete <task_id>"):
                    await self._safe(delete_task, self.ctx, tid)
            case "/task-create":
                if not args:
                    self.console.print(
                        "[red]Usage: /task-create <task_id> [--force][/red]")
                    return
                task_id, force = args[0], "--force" in args[1:]
                title = (await self.session.prompt_async(HTML("<b>title> </b>"))).strip()
                model = (await self.session.prompt_async(HTML(f"<b>model</b> (default: {self.model})> "))).strip() or self.model
                effort = (await self.session.prompt_async(HTML(f"<b>effort</b> (default: {self.effort})> "))).strip() or self.effort
                self.console.print(
                    "[dim]Enter the task prompt (multiline). Finish with Esc+Enter.[/dim]")
                buf, old_multiline = self.session.default_buffer, self.session.default_buffer.multiline
                try:
                    task_prompt = await self.session.prompt_async(HTML("<b>prompt> </b>"), multiline=True)
                finally:
                    buf.multiline = old_multiline
                await self._safe(create_task, self.ctx, task_id, title=title, prompt=task_prompt,
                                 model=model, reasoning_effort=effort, force=force)
            case "/new":
                if self.active_agent:
                    self.active_agent.session_manager.reset_session()
                self.console.clear()
                self.console.print(Panel(
                    f"[bold green]Ollama Agent REPL[/bold green]\n"
                    f"Model: [cyan]{self.model}[/cyan] | Effort: [cyan]{self.effort}[/cyan]\n"
                    "Type [bold]/help[/bold] for commands or just start typing to chat.",
                    title="New Session", border_style="green"))
            case "/sessions": await self._list_sessions(page=int(args[0]) if args and args[0].isdigit() else 1)
            case "/session-load":
                if sid := await _require_arg("/session-load <session_id>"):
                    await self._load_session(sid)
            case "/session-delete":
                if sid := await _require_arg("/session-delete <session_id>"):
                    await self._delete_session(sid)
            case "/models": self._list_models()
            case "/model-set":
                if mid := await _require_arg("/model-set <model_name>"):
                    await self._set_model(mid)
            # RAG commands
            case "/rag": show_rag_status(self._get_rag_ctx())
            case "/rag-list": list_rag_databases(self._get_rag_ctx())
            case "/rag-create":
                if name := await _require_arg("/rag-create <name>"):
                    await self._safe(create_rag_database, self._get_rag_ctx(), name)
            case "/rag-delete":
                if name := await _require_arg("/rag-delete <name>"):
                    await self._safe(delete_rag_database, self._get_rag_ctx(), name)
            case "/rag-load":
                if name := await _require_arg("/rag-load <name>"):
                    await self._safe(load_rag_database, self._get_rag_ctx(), name)
            case "/rag-unload":
                unload_rag_database(self._get_rag_ctx())
            case "/rag-add":
                if not args:
                    self.console.print("[red]Usage: /rag-add <file_or_dir> [--dir][/red]")
                    return
                path, is_dir = args[0], "--dir" in args[1:]
                await self._safe(add_rag_directory if is_dir else add_rag_file, self._get_rag_ctx(), path)
            # Skill commands
            case "/skills": list_skills(self._skills_ctx)
            case "/skill-show":
                if sid := await _require_arg("/skill-show <skill_id>"):
                    await self._safe(show_skill, self._skills_ctx, sid)
            case "/skill-delete":
                if sid := await _require_arg("/skill-delete <skill_id>"):
                    await self._safe(delete_skill, self._skills_ctx, sid)
            case "/skill-create":
                if not args:
                    self.console.print("[red]Usage: /skill-create <skill_id> [--force][/red]")
                    return
                skill_id, force = args[0], "--force" in args[1:]
                name = (await self.session.prompt_async(HTML("<b>name> </b>"))).strip()
                description = (await self.session.prompt_async(HTML("<b>description> </b>"))).strip()
                self.console.print("[dim]Enter skill instructions (multiline markdown). Finish with Esc+Enter.[/dim]")
                buf, old_multiline = self.session.default_buffer, self.session.default_buffer.multiline
                try:
                    instructions = await self.session.prompt_async(HTML("<b>instructions> </b>"), multiline=True)
                finally:
                    buf.multiline = old_multiline
                await self._safe(create_skill, self._skills_ctx, skill_id, name=name,
                                 description=description, instructions=instructions, force=force)
            case _: self.console.print(f"[red]Unknown command:[/red] {cmd}")

    async def _list_sessions(self, page: int = 1, per_page: int = 10) -> None:
        list_sessions(self._ensure_agent(), self.console, page=page, per_page=per_page)

    def _find_session(self, prefix: str, sessions: list) -> dict | None:
        return find_session(prefix, sessions, self.console)

    async def _load_session(self, session_id_prefix: str) -> None:
        load_session(self._ensure_agent(), self.console, session_id_prefix)

    async def _delete_session(self, session_id_prefix: str) -> None:
        delete_session(self._ensure_agent(), self.console, session_id_prefix)

    def _list_models(self) -> None:
        list_models(self.console, self.model)

    async def _set_model(self, model_name: str) -> None:
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

    def show_help(self) -> None:
        self.console.print(Panel("""[bold]Available Commands:[/bold]
        [green]/help[/green]            Show this help message
        [green]/exit[/green], [green]/quit[/green]    Exit the REPL
        [green]/clear[/green]           Clear the screen

        [bold]Model Management:[/bold]
        [green]/models[/green]          List available Ollama models
        [green]/model-set[/green]       Switch to a different model (Usage: /model-set <model>)

        [bold]Session Management:[/bold]
        [green]/new[/green]             Start a new chat session (clears context)
        [green]/sessions[/green]        List saved sessions (Usage: /sessions [page])
        [green]/session-load[/green]    Load a saved session (Usage: /session-load <id>)
        [green]/session-delete[/green]  Delete a saved session (Usage: /session-delete <id>)

        [bold]Task Management:[/bold]
        [green]/tasks[/green]           List saved tasks
        [green]/task-create[/green]     Create a task (Usage: /task-create <id> [--force])
        [green]/task-run[/green]        Run a saved task (Usage: /task-run <id>)
        [green]/task-delete[/green]     Delete a saved task (Usage: /task-delete <id>)

        [bold]RAG (Document Retrieval):[/bold]
        [green]/rag[/green]             Show current RAG status
        [green]/rag-list[/green]        List all RAG databases
        [green]/rag-create[/green]      Create a new RAG database (Usage: /rag-create <name>)
        [green]/rag-delete[/green]      Delete a RAG database (Usage: /rag-delete <name>)
        [green]/rag-load[/green]        Load a RAG database (Usage: /rag-load <name>)
        [green]/rag-unload[/green]      Unload the current RAG database
        [green]/rag-add[/green]         Add file(s) to RAG (Usage: /rag-add <path> [--dir])

        [bold]Skills Management:[/bold]
        [green]/skills[/green]          List all skills
        [green]/skill-show[/green]      Show skill details (Usage: /skill-show <id>)
        [green]/skill-create[/green]    Create a skill (Usage: /skill-create <id> [--force])
        [green]/skill-delete[/green]    Delete a skill (Usage: /skill-delete <id>)""", title="Help"))
