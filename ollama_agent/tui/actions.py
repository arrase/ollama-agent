"""User actions for the TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..core import extract_text
from ..tasks import Task
from .create_task_screen import CreateTaskScreen
from .session_list_screen import SessionListScreen
from .task_list_screen import TaskListScreen

if TYPE_CHECKING:
    from .app import ChatInterface


class UIActions:
    """Handles user-triggered actions in the chat interface."""

    def __init__(self, app: "ChatInterface") -> None:
        self._app = app

    def reset_session(self) -> None:
        """Reset the session and start a new conversation."""
        session_id = self._app.agent.session_manager.reset_session()
        self._app.chat_log.clear()
        self._write_session_banner(session_id, is_new=True)
        self._app._set_subtitle(session_id)

    def load_session(self) -> None:
        """Show the session list dialog."""
        async def on_select(action: str | None) -> None:
            if action and action.startswith("load:"):
                await self._load_selected_session(action.removeprefix("load:"))

        self._app.push_screen(
            SessionListScreen(self._app.agent), on_select
        )

    def create_task(self) -> None:
        """Show the create task dialog."""
        def on_save(task: Optional[Task]) -> None:
            if task:
                task_id = self._app.task_manager.save_task(task)
                self._app.chat_logger.write_message(
                    f"Task saved: {task.title} ({task_id})",
                    style="italic cyan",
                )
                self._app.chat_logger.blank_line()

        self._app.push_screen(CreateTaskScreen(self._app.agent), on_save)

    def list_tasks(self) -> None:
        """Show the task list dialog."""
        def on_select(action: Optional[str]) -> None:
            if action and action.startswith("run:"):
                self._app.run_worker(
                    self._run_selected_task(action.removeprefix("run:"))
                )

        self._app.push_screen(TaskListScreen(self._app.task_manager), on_select)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _write_session_banner(self, session_id: str, *, is_new: bool = False) -> None:
        """Write session info to the chat log."""
        logger = self._app.chat_logger
        if is_new:
            logger.write_message("New session started!", style="italic cyan")
            logger.write_message(
                "Previous conversation history has been cleared.",
                style="italic cyan",
            )
        else:
            logger.write_message(f"Loaded session: {session_id}", style="italic cyan")
        logger.write_message(f"Session ID: {session_id}", style="italic cyan")
        logger.blank_line()

    async def _load_selected_session(self, session_id: str) -> None:
        """Load the selected session and display its history."""
        sm = self._app.agent.session_manager
        sm.load_session(session_id)

        log = self._app.chat_log
        log.clear()
        self._write_session_banner(session_id, is_new=False)

        for item in await sm.get_session_history(session_id):
            if not isinstance(item, dict):
                continue
            role = item.get("role", "")
            text = extract_text(item.get("content", ""))
            if role == "user" and text:
                self._app.chat_logger.write_message(
                    text, style="bold blue", prefix="User"
                )
            elif role == "assistant" and text:
                self._app.chat_logger.write_message(
                    text, style="bold green", prefix="Agent", markdown=True
                )

        self._app.chat_logger.blank_line()
        log.scroll_end(animate=False)
        self._app._set_subtitle(session_id)

    async def _run_selected_task(self, task_id: str) -> None:
        """Execute the selected task."""
        task = self._app.task_manager.load_task(task_id)
        if not task:
            self._app.chat_logger.write_message(
                f"Task not found: {task_id}",
                style="bold red",
                prefix="Error",
            )
            return

        self._app.chat_logger.write_message(
            f"Executing task: {task.title} ({task_id})",
            style="italic cyan",
        )
        self._app.chat_logger.write_message(
            task.prompt, style="bold blue", prefix="User"
        )

        await self._app._stream_agent_response(
            task.prompt,
            model=task.model,
            reasoning_effort=task.reasoning_effort,
        )
