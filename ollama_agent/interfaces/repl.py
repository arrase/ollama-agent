"""REPL interface for Ollama Agent."""

from typing import Callable
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from ..agent import OllamaAgent
from ..core import ModelCapabilityError, model_supports_tools
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
from ..tasks.commands import CLIContext, create_task, delete_task, list_tasks, run_task


class OllamaREPL:
    """Read-Eval-Print Loop for interacting with the Ollama Agent."""

    def __init__(self, agent_factory: Callable[..., OllamaAgent], model: str, effort: str,
                 rag_database: str | None = None):
        self.agent_factory, self.model, self.effort = agent_factory, model, effort
        self.console = Console()
        self.session = PromptSession(
            style=Style.from_dict({"prompt": "#ansiwhite bold"}))
        self.ctx = CLIContext(agent_factory, console=self.console)
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
                    try:
                        await run_task(self.ctx, tid)
                    except SystemExit:
                        pass
            case "/task-delete":
                if tid := await _require_arg("/task-delete <task_id>"):
                    try:
                        delete_task(self.ctx, tid)
                    except SystemExit:
                        pass
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
                try:
                    create_task(self.ctx, task_id, title=title, prompt=task_prompt,
                                model=model, reasoning_effort=effort, force=force)
                except SystemExit:
                    pass
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
                    try:
                        create_rag_database(self._get_rag_ctx(), name)
                    except SystemExit:
                        pass
            case "/rag-delete":
                if name := await _require_arg("/rag-delete <name>"):
                    try:
                        delete_rag_database(self._get_rag_ctx(), name)
                    except SystemExit:
                        pass
            case "/rag-load":
                if name := await _require_arg("/rag-load <name>"):
                    try:
                        load_rag_database(self._get_rag_ctx(), name)
                    except SystemExit:
                        pass
            case "/rag-unload":
                unload_rag_database(self._get_rag_ctx())
            case "/rag-add":
                if not args:
                    self.console.print("[red]Usage: /rag-add <file_or_dir> [--dir][/red]")
                    return
                path, is_dir = args[0], "--dir" in args[1:]
                try:
                    if is_dir:
                        add_rag_directory(self._get_rag_ctx(), path)
                    else:
                        add_rag_file(self._get_rag_ctx(), path)
                except SystemExit:
                    pass
            case _: self.console.print(f"[red]Unknown command:[/red] {cmd}")

    async def _list_sessions(self, page: int = 1, per_page: int = 10) -> None:
        agent = self._ensure_agent()
        sessions = agent.session_manager.list_sessions(
            limit=per_page, offset=(page - 1) * per_page)
        if not sessions:
            self.console.print(
                f"[yellow]{'No saved sessions found.' if page == 1 else f'No more sessions (page {page} is empty).'}[/yellow]")
            return
        current_id = agent.session_manager.get_session_id()
        self.console.print(
            f"[bold]Sessions (page {page}):[/bold]\n[dim]─" + "─" * 59 + "[/dim]")
        for s in sessions:
            marker = " [green]◀ current[/green]" if s["session_id"] == current_id else ""
            preview = s["preview"][:40] + \
                "..." if len(s["preview"]) > 40 else s["preview"]
            self.console.print(
                f"[cyan]{s['session_id'][:8]}[/cyan] │ {s['message_count']:>3} msgs │ {s['last_message'][:16]} │ [dim]{preview}[/dim]{marker}")
        self.console.print("[dim]─" * 60 + "[/dim]")
        hint = f"Next page: /sessions {page + 1} | " if len(
            sessions) == per_page else ""
        self.console.print(f"[dim]{hint}Load: /session-load <id>[/dim]")

    def _find_session(self, prefix: str, sessions: list) -> dict | None:
        matches = [s for s in sessions if s["session_id"].startswith(prefix)]
        if not matches:
            self.console.print(
                f"[red]No session found matching '{prefix}'[/red]")
            return None
        if len(matches) > 1:
            self.console.print(
                f"[yellow]Multiple sessions match '{prefix}':[/yellow]")
            for m in matches[:5]:
                self.console.print(
                    f"  [cyan]{m['session_id'][:8]}[/cyan] - {m['preview'][:40]}")
            return None
        return matches[0]

    async def _load_session(self, session_id_prefix: str) -> None:
        agent = self._ensure_agent()
        if not (target := self._find_session(session_id_prefix, agent.session_manager.list_sessions(limit=100))):
            return
        history = agent.session_manager.get_readable_history(
            target["session_id"])
        agent.session_manager.load_session(target["session_id"])
        self.console.print(
            f"\n[bold green]━━━ Session Loaded: {target['session_id'][:8]}... ━━━[/bold green]")
        self.console.print(
            f"[dim]Messages: {target['message_count']} | Last active: {target['last_message']}[/dim]\n")
        if history:
            self.console.print(
                "[bold]Conversation History:[/bold]\n[dim]─" + "─" * 49 + "[/dim]")
            for msg in history[-10:]:
                limit = 200 if msg["role"] == "user" else 300
                content = msg["content"][:limit] + \
                    "..." if len(msg["content"]) > limit else msg["content"]
                prefix = "[bold blue]>>>[/bold blue] " if msg["role"] == "user" else ""
                self.console.print(f"{prefix}{content}\n")
            if len(history) > 10:
                self.console.print(
                    f"[dim]... and {len(history) - 10} earlier messages[/dim]\n")
            self.console.print("[dim]─" * 50 + "[/dim]")
        self.console.print(
            "[green]✓ Session loaded. Continue typing to resume the conversation.[/green]\n")

    async def _delete_session(self, session_id_prefix: str) -> None:
        agent = self._ensure_agent()
        if not (target := self._find_session(session_id_prefix, agent.session_manager.list_sessions(limit=100))):
            return
        msg = f"[green]✓ Deleted session:[/green] [cyan]{target['session_id'][:8]}[/cyan]" \
            if agent.session_manager.delete_session(target["session_id"]) else "[red]Failed to delete session[/red]"
        self.console.print(msg)

    def _list_models(self) -> None:
        try:
            models = getattr(ollama.list(), "models", [])
            if not models:
                self.console.print(
                    "[yellow]No models found in Ollama.[/yellow]")
                return
            self.console.print(
                "[bold]Available Models:[/bold]\n[dim]─" + "─" * 59 + "[/dim]")
            for item in models:
                if not (name := getattr(item, "model", None)):
                    continue
                marker = " [green]◀ current[/green]" if name == self.model else ""
                size_gb = getattr(item, "size", 0) / (1024 ** 3)
                size_str = f"{size_gb:.1f}GB" if size_gb else ""
                try:
                    tool_icon = "[green]✓[/green]" if model_supports_tools(
                        name) else "[red]✗[/red]"
                except ModelCapabilityError:
                    tool_icon = "[yellow]?[/yellow]"
                self.console.print(
                    f"  {tool_icon} [cyan]{name}[/cyan] {size_str}{marker}")
            self.console.print(
                "[dim]─" * 60 + "[/dim]\n[dim]✓ = supports tools | Use /model-set <model> to switch[/dim]")
        except Exception as e:
            self.console.print(f"[red]Error listing models: {e}[/red]")

    async def _set_model(self, model_name: str) -> None:
        try:
            available = {getattr(m, "model", "")
                         for m in getattr(ollama.list(), "models", [])}
            if model_name not in available:
                self.console.print(
                    f"[red]Model '{model_name}' not found.[/red]\n[dim]Use /models to see available models.[/dim]")
                return
        except Exception as e:
            self.console.print(f"[red]Error checking model: {e}[/red]")
            return
        if model_name == self.model:
            self.console.print(
                f"[yellow]Already using model '{model_name}'.[/yellow]")
            return
        try:
            if not model_supports_tools(model_name):
                self.console.print(
                    f"[red]Model '{model_name}' does not support tools.[/red]\n[dim]The agent requires tool support.[/dim]")
                return
        except ModelCapabilityError as e:
            self.console.print(
                f"[red]Cannot verify model capabilities: {e}[/red]")
            return

        old_model = self.model
        old_session_id = self.active_agent.session_manager.get_session_id(
        ) if self.active_agent else None

        if self.active_agent:
            await self.active_agent.cleanup()
            self.active_agent = None

        self.model = model_name
        try:
            self.active_agent = self.agent_factory(
                model=self.model, reasoning_effort=self.effort)
            await self.active_agent.initialize()
        except (ModelCapabilityError, SystemExit) as e:
            self.console.print(
                f"[red]Failed to create agent with model '{model_name}': {e}[/red]")
            self.model = old_model
            self.active_agent = self.agent_factory(
                model=self.model, reasoning_effort=self.effort)
            if old_session_id:
                self.active_agent.session_manager.load_session(old_session_id)
            return

        if old_session_id:
            self.active_agent.session_manager.load_session(old_session_id)
        self.console.print(
            f"[green]✓ Switched from [cyan]{old_model}[/cyan] to [cyan]{model_name}[/cyan][/green]\n[dim]Conversation preserved. Continue chatting.[/dim]")

    async def handle_chat(self, prompt: str) -> None:
        agent = self._ensure_agent()
        full_response, reasoning_active, response_banner_shown = "", False, False
        live = Live(console=self.console, refresh_per_second=12,
                    vertical_overflow="visible")
        try:
            async for payload in agent.run_async_streamed(prompt):
                msg_type = payload.get("type")
                if msg_type == "text_delta":
                    if reasoning_active:
                        reasoning_active = False
                        self.console.print()
                    if not response_banner_shown:
                        self.console.print()
                        response_banner_shown = True
                        live.start()
                    full_response += payload["content"]
                    live.update(Markdown(full_response))
                elif msg_type == "reasoning_delta":
                    if not reasoning_active:
                        self.console.print(
                            "\n[bold magenta]🧠 Thinking:[/bold magenta] ", end="")
                        reasoning_active = True
                    self.console.print(payload.get(
                        "content", ""), end="", style="dim italic magenta")
                elif msg_type == "tool_call":
                    if reasoning_active:
                        reasoning_active = False
                        self.console.print()
                    live.stop()
                    self.console.print(
                        f"\n[bold magenta]tool -> {payload.get('name')}...[/bold magenta]")
                elif msg_type == "tool_output":
                    self.console.print(
                        f"[dim]<- {payload.get('output')}[/dim]")
                elif msg_type == "error":
                    live.stop()
                    self.console.print(f"[red]{payload['content']}[/red]")
            if reasoning_active:
                self.console.print()
            if not response_banner_shown:
                self.console.print()
                live.start()
            live.update(Markdown(full_response))
        except Exception as e:
            live.stop()
            self.console.print(f"[red]Error running agent: {e}[/red]")
        finally:
            live.stop()
        self.console.print()

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
        [green]/rag-add[/green]         Add file(s) to RAG (Usage: /rag-add <path> [--dir])""", title="Help"))
