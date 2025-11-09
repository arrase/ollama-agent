from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select

from ..agent import OllamaAgent
from ..tasks import Task
from ..utils import (
    ALLOWED_REASONING_EFFORTS,
    ModelCapabilityError,
    get_tool_compatible_models,
    validate_reasoning_effort,
)


class CreateTaskScreen(ModalScreen):
    """Modal screen to create a new task."""

    CSS = """
    CreateTaskScreen {
        align: center middle;
    }

    #task-dialog {
        width: 80;
        height: 30;
        border: thick $background 80%;
        background: $surface;
        padding: 1 2;
    }

    #task-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        color: $accent;
    }

    .field-label {
        margin-top: 1;
        text-style: bold;
    }

    .field-input {
        margin-bottom: 1;
    }

    #button-container {
        height: 3;
        align: center middle;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
    ]

    def __init__(self, agent: OllamaAgent):
        """
        Initialize the create task screen.

        Args:
            agent: The agent instance for getting current settings.
        """
        super().__init__()
        self.agent = agent
        try:
            self._models = get_tool_compatible_models(self.agent.model)
        except ModelCapabilityError:
            # Preferred model is not tool-compatible; try to get all tool-compatible models
            try:
                self._models = get_tool_compatible_models()
            except ModelCapabilityError:
                self._models = []
        if not self._models:
            # No tool-compatible models available; leave list empty
            self._models = []

    def compose(self) -> ComposeResult:
        """Create the task creation dialog."""
        with Container(id="task-dialog"):
            yield Label("Create New Task", id="task-title")

            yield Label("Title:", classes="field-label")
            yield Input(placeholder="Enter task title...", id="task-title-input", classes="field-input")

            yield Label("Prompt:", classes="field-label")
            yield Input(placeholder="Enter task prompt...", id="task-prompt-input", classes="field-input")

            yield Label("Model:", classes="field-label")
            default_model = self.agent.model if self.agent.model in self._models else (self._models[0] if self._models else "")
            yield Select(
                [(model, model) for model in self._models],
                value=default_model,
                id="task-model-select",
                classes="field-input"
            )

            yield Label("Reasoning Effort:", classes="field-label")
            yield Select(
                [(effort, effort) for effort in ALLOWED_REASONING_EFFORTS],
                value=self.agent.reasoning_effort,
                id="task-effort-select",
                classes="field-input"
            )

            with Horizontal(id="button-container"):
                yield Button("Save", variant="primary", id="save-button")
                yield Button("Cancel", variant="default", id="cancel-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "cancel-button":
            self.dismiss(None)
        elif event.button.id == "save-button":
            task = self._create_task_from_inputs()
            if task:
                self.dismiss(task)

    def _create_task_from_inputs(self) -> Optional[Task]:
        """
        Create a task from user inputs.

        Returns:
            Task instance if all inputs are valid, None otherwise.
        """
        title = self.query_one("#task-title-input", Input).value.strip()
        prompt = self.query_one("#task-prompt-input", Input).value.strip()
        model_select = self.query_one("#task-model-select", Select)
        model = str(model_select.value) if model_select.value else ""

        if not all([title, prompt, model]):
            return None

        effort_select = self.query_one("#task-effort-select", Select)
        effort = str(
            effort_select.value) if effort_select.value else self.agent.reasoning_effort

        return Task(
            title=title,
            prompt=prompt,
            model=model,
            reasoning_effort=validate_reasoning_effort(effort)
        )
