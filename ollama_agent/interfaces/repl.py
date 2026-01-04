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
from ..tasks.commands import CLIContext, create_task, delete_task, list_tasks, run_task


class OllamaREPL:
    """Read-Eval-Print Loop for interacting with the Ollama Agent."""

    def __init__(self, agent_factory: Callable[..., OllamaAgent], model: str, effort: str):
        self.agent_factory = agent_factory
        self.model = model
        self.effort = effort
        self.console = Console()
        self.session = PromptSession(
            style=Style.from_dict({
                "prompt": "#ansiwhite bold",
            })
        )
        # We need a context for reusing task commands
        self.ctx = CLIContext(agent_factory, console=self.console)
        self.active_agent: OllamaAgent | None = None

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.active_agent:
            await self.active_agent.cleanup()
            self.active_agent = None

    async def run(self) -> None:
        """Start the REPL loop."""
        self.console.print(
            Panel(
                f"[bold green]Ollama Agent REPL[/bold green]\n"
                f"Model: [cyan]{self.model}[/cyan] | Effort: [cyan]{self.effort}[/cyan]\n"
                "Type [bold]/help[/bold] for commands or just start typing to chat.",
                title="Welcome",
                border_style="green",
            )
        )

        try:
            while True:
                try:
                    user_input = await self.session.prompt_async(
                        HTML("<b>>>> </b>"),
                        multiline=False,
                    )
                    user_input = user_input.strip()

                    if not user_input:
                        continue

                    if user_input.startswith("/"):
                        await self.handle_command(user_input)
                    else:
                        await self.handle_chat(user_input)

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
        cmd = parts[0].lower()
        args = parts[1:]

        match cmd:
            case "/exit" | "/quit":
                raise EOFError
            case "/help":
                self.show_help()
            case "/clear":
                self.console.clear()
            case "/tasks":
                list_tasks(self.ctx)
            case "/task-run":
                if not args:
                    self.console.print("[red]Usage: /task-run <task_id>[/red]")
                    return
                try:
                    await run_task(self.ctx, args[0])
                except SystemExit:
                    return
            case "/task-delete":
                if not args:
                    self.console.print("[red]Usage: /task-delete <task_id>[/red]")
                    return
                try:
                    delete_task(self.ctx, args[0])
                except SystemExit:
                    return
            case "/task-create":
                if not args:
                    self.console.print(
                        "[red]Usage: /task-create <task_id> [--force][/red]"
                    )
                    return

                task_id = args[0]
                force = "--force" in args[1:]

                title = (await self.session.prompt_async(HTML("<b>title> </b>"))).strip()

                model = (
                    await self.session.prompt_async(
                        HTML(f"<b>model</b> (default: {self.model})> ")
                    )
                ).strip()
                if not model:
                    model = self.model

                effort = (
                    await self.session.prompt_async(
                        HTML(f"<b>effort</b> (default: {self.effort})> ")
                    )
                ).strip()
                if not effort:
                    effort = self.effort

                self.console.print(
                    "[dim]Enter the task prompt (multiline). Finish with Esc+Enter.[/dim]"
                )

                buf = self.session.default_buffer
                old_multiline = buf.multiline
                try:
                    task_prompt = await self.session.prompt_async(
                        HTML("<b>prompt> </b>"),
                        multiline=True,
                    )
                finally:
                    buf.multiline = old_multiline

                try:
                    create_task(
                        self.ctx,
                        task_id,
                        title=title,
                        prompt=task_prompt,
                        model=model,
                        reasoning_effort=effort,
                        force=force,
                    )
                except SystemExit:
                    return
            case "/new":
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
            case "/sessions":
                page = int(args[0]) if args and args[0].isdigit() else 1
                await self._list_sessions(page=page)
            case "/session-load":
                if not args:
                    self.console.print("[red]Usage: /session-load <session_id>[/red]")
                    return
                await self._load_session(args[0])
            case "/session-delete":
                if not args:
                    self.console.print("[red]Usage: /session-delete <session_id>[/red]")
                    return
                await self._delete_session(args[0])
            case _:
                self.console.print(f"[red]Unknown command:[/red] {cmd}")

    async def _list_sessions(self, page: int = 1, per_page: int = 10) -> None:
        """List saved sessions with pagination."""
        # Ensure we have an agent to access the session manager
        if not self.active_agent:
            self.active_agent = self.agent_factory(
                model=self.model,
                reasoning_effort=self.effort,
            )

        offset = (page - 1) * per_page
        sessions = self.active_agent.session_manager.list_sessions(limit=per_page, offset=offset)
        
        if not sessions:
            if page == 1:
                self.console.print("[yellow]No saved sessions found.[/yellow]")
            else:
                self.console.print(f"[yellow]No more sessions (page {page} is empty).[/yellow]")
            return

        current_id = self.active_agent.session_manager.get_session_id()

        self.console.print(f"[bold]Sessions (page {page}):[/bold]")
        self.console.print("[dim]─" * 60 + "[/dim]")
        for i, s in enumerate(sessions, 1):
            is_current = s["session_id"] == current_id
            marker = " [green]◀ current[/green]" if is_current else ""
            short_id = s["session_id"][:8]
            preview = s["preview"][:40] + "..." if len(s["preview"]) > 40 else s["preview"]
            self.console.print(
                f"[cyan]{short_id}[/cyan] │ {s['message_count']:>3} msgs │ {s['last_message'][:16]} │ [dim]{preview}[/dim]{marker}"
            )
        self.console.print("[dim]─" * 60 + "[/dim]")
        
        nav_hint = f"/sessions {page + 1}" if len(sessions) == per_page else ""
        if nav_hint:
            self.console.print(f"[dim]Next page: {nav_hint} | Load: /session-load <id>[/dim]")
        else:
            self.console.print("[dim]Use /session-load <id> to continue a conversation[/dim]")

    async def _load_session(self, session_id_prefix: str) -> None:
        """Load a session and display its conversation history."""
        # Ensure we have an agent to access the session manager
        if not self.active_agent:
            self.active_agent = self.agent_factory(
                model=self.model,
                reasoning_effort=self.effort,
            )

        sessions = self.active_agent.session_manager.list_sessions(limit=100)
        matches = [s for s in sessions if s["session_id"].startswith(session_id_prefix)]

        if not matches:
            self.console.print(f"[red]No session found matching '{session_id_prefix}'[/red]")
            return

        if len(matches) > 1:
            self.console.print(f"[yellow]Multiple sessions match '{session_id_prefix}':[/yellow]")
            for m in matches[:5]:
                self.console.print(f"  [cyan]{m['session_id'][:8]}[/cyan] - {m['preview'][:40]}")
            return

        target = matches[0]
        
        # Get conversation history BEFORE loading (to display it)
        history = self.active_agent.session_manager.get_readable_history(target["session_id"])
        
        # Now load the session
        self.active_agent.session_manager.load_session(target["session_id"])
        
        # Display header
        self.console.print(f"\n[bold green]━━━ Session Loaded: {target['session_id'][:8]}... ━━━[/bold green]")
        self.console.print(f"[dim]Messages: {target['message_count']} | Last active: {target['last_message']}[/dim]\n")
        
        # Display conversation history
        if history:
            self.console.print("[bold]Conversation History:[/bold]")
            self.console.print("[dim]─" * 50 + "[/dim]")
            for msg in history[-10:]:  # Show last 10 messages
                if msg["role"] == "user":
                    content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                    self.console.print(f"[bold blue]>>>[/bold blue] {content}")
                else:
                    content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
                    self.console.print(f"{content}")
                self.console.print()
            
            if len(history) > 10:
                self.console.print(f"[dim]... and {len(history) - 10} earlier messages[/dim]\n")
            self.console.print("[dim]─" * 50 + "[/dim]")
        
        self.console.print("[green]✓ Session loaded. Continue typing to resume the conversation.[/green]\n")

    async def _delete_session(self, session_id_prefix: str) -> None:
        """Delete a session by ID or prefix."""
        # Ensure we have an agent to access the session manager
        if not self.active_agent:
            self.active_agent = self.agent_factory(
                model=self.model,
                reasoning_effort=self.effort,
            )

        sessions = self.active_agent.session_manager.list_sessions(limit=100)
        matches = [s for s in sessions if s["session_id"].startswith(session_id_prefix)]

        if not matches:
            self.console.print(f"[red]No session found matching '{session_id_prefix}'[/red]")
            return

        if len(matches) > 1:
            self.console.print(f"[yellow]Multiple sessions match '{session_id_prefix}':[/yellow]")
            for m in matches[:5]:
                self.console.print(f"  [cyan]{m['session_id'][:8]}[/cyan] - {m['preview'][:40]}")
            return

        target = matches[0]
        if self.active_agent.session_manager.delete_session(target["session_id"]):
            self.console.print(f"[green]✓ Deleted session:[/green] [cyan]{target['session_id'][:8]}[/cyan]")
        else:
            self.console.print("[red]Failed to delete session[/red]")

    async def handle_chat(self, prompt: str) -> None:
        """Send prompt to the agent and stream response."""
        if not self.active_agent:
            self.active_agent = self.agent_factory(
                model=self.model,
                reasoning_effort=self.effort,
            )

        full_response = ""
        reasoning_active = False
        response_banner_shown = False
        live = Live(
            console=self.console,
            refresh_per_second=12,
            vertical_overflow="visible",
        )
        try:
            async for payload in self.active_agent.run_async_streamed(prompt):
                msg_type = payload.get("type")
                if msg_type == "text_delta":
                    # End reasoning block if active
                    if reasoning_active:
                        reasoning_active = False
                        self.console.print()
                    # Show banner before first text (like ConsoleStreamingRenderer)
                    if not response_banner_shown:
                        self.console.print()
                        response_banner_shown = True
                        live.start()
                    content = payload["content"]
                    full_response += content
                    live.update(Markdown(full_response))
                elif msg_type == "reasoning_delta":
                    if not reasoning_active:
                        self.console.print(
                            "\n[bold magenta]🧠 Thinking:[/bold magenta] ", end=""
                        )
                        reasoning_active = True
                    self.console.print(
                        payload.get("content", ""), end="", style="dim italic magenta"
                    )
                elif msg_type == "tool_call":
                    if reasoning_active:
                        reasoning_active = False
                        self.console.print()
                    live.stop()
                    self.console.print(
                        f"\n[bold magenta]tool -> {payload.get('name')}...[/bold magenta]"
                    )
                elif msg_type == "tool_output":
                    self.console.print(f"[dim]<- {payload.get('output')}[/dim]")
                elif msg_type == "error":
                    live.stop()
                    self.console.print(f"[red]{payload['content']}[/red]")

            # End reasoning block if still active at the end
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
        """Show available commands."""
        help_text = """
        [bold]Available Commands:[/bold]
        [green]/help[/green]            Show this help message
        [green]/exit[/green], [green]/quit[/green]    Exit the REPL
        [green]/clear[/green]           Clear the screen

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
        """
        self.console.print(Panel(help_text.strip(), title="Help"))
