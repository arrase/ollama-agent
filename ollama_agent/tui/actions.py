"""User actions for the TUI."""

from typing import Optional

from ..agent import OllamaAgent
from ..tasks import Task, TaskManager
from .chat_logger import ChatLogger
from .create_task_screen import CreateTaskScreen
from .session_list_screen import SessionListScreen
from .task_list_screen import TaskListScreen
from ..utils import extract_text


class UIActions:
    def __init__(self, app):
        self._app = app

    @property
    def agent(self) -> OllamaAgent:
        return self._app.agent

    @property
    def task_manager(self) -> TaskManager:
        return self._app.task_manager

    @property
    def chat_logger(self) -> ChatLogger:
        return self._app.chat_logger

    def reset_session(self) -> None:
        """Reset the session and start a new conversation."""
        session_id = self.agent.reset_session()
        self._app.chat_log.clear()
        self.chat_logger.write_message("New session started!", style="italic cyan")
        self.chat_logger.write_message(
            f"Session ID: {session_id}", style="italic cyan")
        self.chat_logger.write_message(
            "Previous conversation history has been cleared.",
            style="italic cyan",
        )
        self.chat_logger.blank_line()

        # Update subtitle with new session ID
        self._app._set_subtitle(session_id)

    def load_session(self) -> None:
        """Show the session list dialog."""
        async def handle_session_action(action: str | None) -> None:
            """Handle the selected action."""
            if action and action.startswith("load:"):
                session_id = action.replace("load:", "")
                await self._load_selected_session(session_id)

        self._app.push_screen(SessionListScreen(self.agent), handle_session_action)

    async def _load_selected_session(self, session_id: str) -> None:
        """Load the selected session and display its history."""
        self.agent.load_session(session_id)

        log = self._app.chat_log
        log.clear()
        self.chat_logger.write_message(
            f"Loaded session: {session_id}", style="italic cyan")
        self.chat_logger.blank_line()

        history = await self.agent.get_session_history(session_id)

        for item in history:
            if isinstance(item, dict):
                role = item.get('role', 'unknown')
                content = item.get('content', '')
                text = extract_text(content)

                if role == 'user' and text:
                    self.chat_logger.write_message(text, style="bold blue", prefix="User")
                elif role == 'assistant' and text:
                    self.chat_logger.write_message(
                        text,
                        style="bold green",
                        prefix="Agent",
                        markdown=True,
                    )

        self.chat_logger.blank_line()
        log.scroll_end(animate=False)

        # Update subtitle
        self._app._set_subtitle(session_id)

    def create_task(self) -> None:
        """Show the create task dialog."""
        def handle_task_creation(task: Optional[Task]) -> None:
            """Handle the created task."""
            if task:
                task_id = self.task_manager.save_task(task)
                self.chat_logger.write_message(
                    f"Task saved: {task.title} ({task_id})",
                    style="italic cyan",
                )
                self.chat_logger.blank_line()

        self._app.push_screen(CreateTaskScreen(self.agent), handle_task_creation)

    def list_tasks(self) -> None:
        """Show the task list dialog."""
        def handle_task_action(action: Optional[str]) -> None:
            """Handle the selected action."""
            if action and action.startswith("run:"):
                task_id = action.replace("run:", "")
                self._app.run_worker(self._run_selected_task(task_id))

        self._app.push_screen(TaskListScreen(self.task_manager), handle_task_action)

    async def _run_selected_task(self, task_id: str) -> None:
        """Execute the selected task."""
        task = self.task_manager.load_task(task_id)

        if not task:
            self.chat_logger.write_message(
                f"Task not found: {task_id}", style="bold red", prefix="Error")
            return

        self.chat_logger.write_message(
            f"Executing task: {task.title} ({task_id})",
            style="italic cyan",
        )
        self.chat_logger.write_message(task.prompt, style="bold blue", prefix="User")

        await self._app._stream_agent_response(
            task.prompt,
            model=task.model,
            reasoning_effort=task.reasoning_effort,
        )
