"""Read-Eval-Print Loop controller for the interactive TUI."""

from __future__ import annotations

from collections import deque

from rich.console import Console
from rich.markup import escape

from ...agent import AgentRuntime
from ...agent.builtin_tools import set_rag_manager, set_tool_timeout
from ...i18n import _
from ...rag import RAGContext, RAGManager, load_rag_database
from ...skills import SkillsContext
from ...tasks import TasksContext
from ..commands.dispatch import REPLHandler, build_repl_handlers
from ..commands.models import set_model
from ..commands.params import set_context_window, set_effort
from ..commands.sessions import new_session
from .app import OllamaAgentApp


class OllamaREPL:
    """Read-Eval-Print Loop for interacting with the Ollama Agent."""

    def __init__(
        self,
        runtime: AgentRuntime,
        rag_database: str | None = None,
    ):
        self.runtime = runtime
        self.console = Console(force_terminal=True, color_system="truecolor")
        self._task_ctx = TasksContext(console=self.console, settings=self.runtime.settings)
        self._skills_ctx = SkillsContext(console=self.console)
        self._initial_rag_database = rag_database
        self._rag_ctx: RAGContext | None = None
        self._commands: dict[str, REPLHandler] | None = None
        self.app: OllamaAgentApp | None = None

    def _get_rag_ctx(self) -> RAGContext:
        if self._rag_ctx is None:
            mgr = RAGManager(self.runtime.settings.rag)
            self._rag_ctx = RAGContext(console=self.console, rag_manager=mgr)
            set_rag_manager(mgr)
        return self._rag_ctx

    def _get_commands(self) -> dict[str, REPLHandler]:
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
                handle_yolo=self._handle_yolo_cmd,
                handle_stealth=self._handle_stealth_cmd,
                handle_queue=self._handle_queue_cmd,
                get_runtime=lambda: self.runtime,
                current_thread_id=lambda: self.runtime.thread_id,
                switch_effort=self._switch_effort,
                switch_context_window=self._switch_context_window,
            )
        return self._commands

    async def cleanup(self) -> None:
        if self._rag_ctx:
            self._rag_ctx.rag_manager.unload()
        await self.runtime.aclose()

    async def run(self) -> None:
        if self._initial_rag_database:
            rag_ctx = self._get_rag_ctx()
            load_rag_database(rag_ctx, self._initial_rag_database)

        set_tool_timeout(self.runtime.settings.runtime.builtin_tool_timeout)

        app = OllamaAgentApp(self)
        try:
            await app.run_async()
        except KeyboardInterrupt:
            pass
        finally:
            await self.cleanup()

    async def _switch_model(self, model_name: str) -> None:
        await set_model(self.console, model_name, runtime=self.runtime)

    async def _switch_effort(self, effort: str) -> None:
        await set_effort(self.console, effort, runtime=self.runtime)

    async def _switch_context_window(self, context_window: str) -> None:
        await set_context_window(self.console, context_window, runtime=self.runtime)

    def _handle_new_session(self) -> None:
        self.runtime.thread_id = new_session(self.console)
        self.runtime.last_context_tokens = 0

    def _handle_yolo_cmd(self, args: list[str]) -> None:
        if args:
            val = args[0].lower()
            if val in ("on", "true", "yes", "1"):
                self.runtime.yolo_mode = True
            elif val in ("off", "false", "no", "0"):
                self.runtime.yolo_mode = False
            else:
                self.console.print(f"[red]{escape(_('Usage: /yolo [on|off]'))}[/red]")
                return
        else:
            self.runtime.yolo_mode = not self.runtime.yolo_mode

        status = _("on") if self.runtime.yolo_mode else _("off")
        color = "red" if self.runtime.yolo_mode else "green"
        self.console.print(f"[bold {color}]{_('YOLO mode is now {status}', status=status)}[/bold {color}]")

    async def _handle_stealth_cmd(self, args: list[str]) -> None:
        if args:
            val = args[0].lower()
            if val in ("on", "true", "yes", "1"):
                self.runtime.stealth_mode = True
            elif val in ("off", "false", "no", "0"):
                self.runtime.stealth_mode = False
            else:
                self.console.print(f"[red]{escape(_('Usage: /stealth [on|off]'))}[/red]")
                return
        else:
            self.runtime.stealth_mode = not self.runtime.stealth_mode

        await self.runtime.reload()

        status = _("on") if self.runtime.stealth_mode else _("off")
        color = "#c084fc" if self.runtime.stealth_mode else "green"
        desc = (
            _("chat history will not be saved to SQLite")
            if self.runtime.stealth_mode
            else _("chat history will be saved to SQLite")
        )
        self.console.print(
            f"[bold {color}]{_('Stealth mode is now {status} ({desc})', status=status, desc=desc)}[/bold {color}]"
        )

    def _handle_queue_rm(self, queue: deque, pos_str: str) -> None:
        if not queue:
            self.console.print(f"[dim]{_('Prompt queue is empty.')}[/dim]")
            return
        raw_pos = pos_str.lstrip("#")
        if not raw_pos.isdigit():
            msg = _("Invalid queue position '{pos}'. Usage: /queue rm <position>", pos=raw_pos)
            self.console.print(f"[red]{msg}[/red]")
            return
        pos = int(raw_pos)
        if pos < 1 or pos > len(queue):
            err = _(
                "Queue position {pos} out of range (queue has {count} items).",
                pos=pos,
                count=len(queue),
            )
            self.console.print(f"[red]{err}[/red]")
            return
        item = queue[pos - 1]
        del queue[pos - 1]
        if self.app is not None:
            self.app._update_queue_ui()
        truncated_text = item.replace("\n", " ")
        if len(truncated_text) > 60:
            truncated_text = truncated_text[:57] + "..."
        msg = _("Removed #{pos} from prompt queue: {text}", pos=pos, text=truncated_text)
        self.console.print(f"[bold #34d399]✓ {msg}[/bold #34d399]")

    def _handle_queue_cmd(self, args: list[str]) -> None:
        queue = self.app._prompt_queue if self.app is not None else deque()
        if not args or args[0] == "list":
            if not queue:
                self.console.print(f"[dim]{_('Prompt queue is empty.')}[/dim]")
                return
            self.console.print(f"[bold #38bdf8]{_('Queued prompts ({count}):', count=len(queue))}[/bold #38bdf8]")
            for i, item in enumerate(queue, 1):
                self.console.print(f"  [dim]#{i}[/dim] {item}")
            return
        if args[0] == "clear":
            count = len(queue)
            queue.clear()
            if self.app is not None:
                self.app._update_queue_ui()
            msg = _("Prompt queue cleared ({count} removed).", count=count)
            self.console.print(f"[bold #34d399]✓ {msg}[/bold #34d399]")
            return
        if args[0] in ("rm", "remove", "delete"):
            if len(args) < 2:
                self.console.print(f"[red]{_('Usage: /queue rm <position>')}[/red]")
                return
            self._handle_queue_rm(queue, args[1])
            return
        err_msg = _("Unknown queue subcommand '{sub}'. Usage: /queue [clear | rm <position>]", sub=args[0])
        self.console.print(f"[red]{err_msg}[/red]")
