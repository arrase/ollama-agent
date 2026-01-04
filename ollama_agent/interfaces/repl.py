"""REPL interface for Ollama Agent."""

import asyncio
from typing import Callable, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from ..agent import OllamaAgent
from ..execution.runner import run_non_interactive
from ..tasks.commands import CLIContext, delete_task, list_tasks, run_task


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
        self.ctx = CLIContext(agent_factory)
        self.active_agent: Optional[OllamaAgent] = None

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
                    user_input = await self.session.prompt_async(HTML("<b>>>> </b>"))
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

        if cmd in ("/exit", "/quit"):
            raise EOFError
        if cmd == "/help":
            self.show_help()
            return
        if cmd == "/clear":
            self.console.clear()
            return
        if cmd == "/tasks":
            list_tasks(self.ctx)
            return
        if cmd == "/task-run":
            if not args:
                self.console.print("[red]Usage: /task-run <task_id>[/red]")
                return
            await run_task(self.ctx, args[0])
            return
        if cmd == "/task-delete":
            if not args:
                self.console.print("[red]Usage: /task-delete <task_id>[/red]")
                return
            delete_task(self.ctx, args[0])
            return
        if cmd == "/new":
            if self.active_agent:
                await self.active_agent.cleanup()
            self.active_agent = None
            self.console.print("[green]Started new session.[/green]")
            return

        self.console.print(f"[red]Unknown command:[/red] {cmd}")

    async def handle_chat(self, prompt: str) -> None:
        """Send prompt to the agent and stream response."""
        if not self.active_agent:
            self.active_agent = self.agent_factory(
                model=self.model,
                reasoning_effort=self.effort,
            )

        self.console.print("[bold green]Agent:[/bold green]")

        full_response = ""
        live = Live(
            console=self.console,
            refresh_per_second=12,
            vertical_overflow="visible",
        )
        live.start()

        try:
            async for payload in self.active_agent.run_async_streamed(prompt):
                msg_type = payload.get("type")
                if msg_type == "text_delta":
                    content = payload["content"]
                    full_response += content
                    live.update(Markdown(full_response))
                elif msg_type == "reasoning_delta":
                    pass
                elif msg_type == "tool_call":
                    live.stop()
                    self.console.print(f"[bold magenta]tool -> {payload.get('name')}...[/bold magenta]")
                    live.start()
                elif msg_type == "tool_output":
                    live.stop()
                    self.console.print(f"[dim]<- {payload.get('output')}[/dim]")
                    live.start()
                elif msg_type == "error":
                    live.stop()
                    self.console.print(f"[red]{payload['content']}[/red]")
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
        [green]/help[/green]          Show this help message
        [green]/exit[/green], [green]/quit[/green]  Exit the REPL
        [green]/clear[/green]         Clear the screen
        [green]/tasks[/green]         List saved tasks
        [green]/task-run[/green]      Run a saved task (Usage: /task-run <id>)
        [green]/task-delete[/green]   Delete a saved task (Usage: /task-delete <id>)
        [green]/new[/green]           Start a new chat session (clears context)
        """
        self.console.print(Panel(help_text.strip(), title="Help"))
