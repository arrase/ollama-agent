"""Create task modal screen."""

from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from ...agent import OllamaAgent
from ...core import (
    ALLOWED_REASONING_EFFORTS,
    ModelCapabilityError,
    get_tool_compatible_models,
    validate_reasoning_effort,
)
from ...tasks import Task


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

    #task-title { text-align: center; text-style: bold; padding: 1; color: $accent; }
    .field-label { margin-top: 1; text-style: bold; }
    .field-input { margin-bottom: 1; }
    #button-container { height: 3; align: center middle; margin-top: 1; }
    """

    BINDINGS = [Binding("escape", "dismiss", "Cancel")]

    def __init__(self, agent: OllamaAgent):
        super().__init__()
        self.agent = agent
        self._models: list[str] = []
        self._error: Optional[str] = None
        self._load_models()

    def _load_models(self) -> None:
        """Load available models with tool support."""
        try:
            self._models = get_tool_compatible_models(self.agent.model)
        except ModelCapabilityError as e:
            self._error = str(e)

        if not self._models:
            try:
                self._models = get_tool_compatible_models()
            except ModelCapabilityError as e:
                self._error = self._error or str(e)

        if not self._models:
            self._error = self._error or "No models with tool support available."

    def compose(self) -> ComposeResult:
        with Container(id="task-dialog"):
            yield Label("Create New Task", id="task-title")

            yield Label("Title:", classes="field-label")
            yield Input(placeholder="Task title...", id="title-input", classes="field-input")

            yield Label("Prompt:", classes="field-label")
            yield Input(placeholder="Task prompt...", id="prompt-input", classes="field-input")

            if self._error:
                yield Static(self._error, classes="field-label")
            else:
                yield Label("Model:", classes="field-label")
                default = self.agent.model if self.agent.model in self._models else self._models[0]
                yield Select(
                    [(m, m) for m in self._models],
                    value=default,
                    id="model-select",
                    classes="field-input",
                )

            yield Label("Reasoning Effort:", classes="field-label")
            yield Select(
                [(e, e) for e in ALLOWED_REASONING_EFFORTS],
                value=self.agent.reasoning_effort,
                id="effort-select",
                classes="field-input",
            )

            with Horizontal(id="button-container"):
                yield Button("Save", variant="primary", id="save-btn", disabled=bool(self._error))
                yield Button("Cancel", variant="default", id="cancel-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.dismiss(None)
        elif event.button.id == "save-btn":
            task = self._build_task()
            if task:
                self.dismiss(task)

    def _build_task(self) -> Optional[Task]:
        """Build task from form inputs."""
        if self._error:
            return None

        title = self.query_one("#title-input", Input).value.strip()
        prompt = self.query_one("#prompt-input", Input).value.strip()
        model_sel = self.query_one("#model-select", Select)
        model = str(model_sel.value) if model_sel.value else ""

        if not all([title, prompt, model]):
            return None

        effort_sel = self.query_one("#effort-select", Select)
        effort = str(effort_sel.value) if effort_sel.value else self.agent.reasoning_effort

        return Task(
            title=title,
            prompt=prompt,
            model=model,
            reasoning_effort=validate_reasoning_effort(effort),
        )
