"""Interactive tool execution approval prompt widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.markup import escape
from textual import events
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Static

from ....i18n import _
from ....streaming import build_approval_decisions
from .footer import AgentFooter
from .input import ReplInput
from .messages import AgentResponse

if TYPE_CHECKING:
    from ..app import OllamaAgentApp


class ToolApprovalWidget(Container):
    """Inline widget prompting the user for approval of sensitive tool calls."""

    BUTTON_IDS = ["approve-btn", "reject-btn", "allow-btn", "cancel-btn"]
    DECISION_KEYS = {
        "y": "approve-btn",
        "n": "reject-btn",
        "a": "allow-btn",
        "c": "cancel-btn",
        "escape": "cancel-btn",
    }

    def __init__(
        self,
        action_requests: list[dict[str, Any]],
        app_ref: OllamaAgentApp,
        scroll: Any,
        agent_msg: AgentResponse,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.action_requests = action_requests
        self.app_ref = app_ref
        self.scroll = scroll
        self.agent_msg = agent_msg
        self.buttons_container: Horizontal | None = None

    def compose(self) -> ComposeResult:
        yield Static(f"[bold #fbbf24]⚠ {_('Action Approval Required')}[/bold #fbbf24]", classes="approval-title")
        for req in self.action_requests:
            name = escape(str(req["name"]))
            args = escape(str(req["args"]))
            details = f"{_('Tool:')} [bold #38bdf8]{name}[/bold #38bdf8]\n{_('Arguments:')} [dim]{args}[/dim]"
            yield Static(details, classes="approval-details")

        with Horizontal(classes="approval-buttons") as buttons:
            self.buttons_container = buttons
            yield Button(_("Approve (y)"), id="approve-btn", variant="success", classes="approval-btn")
            yield Button(_("Reject (n)"), id="reject-btn", variant="error", classes="approval-btn")
            yield Button(_("Allow Session (a)"), id="allow-btn", variant="primary", classes="approval-btn")
            yield Button(_("Cancel (c)"), id="cancel-btn", classes="approval-btn")

    def on_mount(self) -> None:
        buttons = self.query("#approve-btn")
        if buttons:
            buttons.first().focus()
        else:
            self.call_after_refresh(lambda: self.query_one("#approve-btn", Button).focus())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._handle_decision(event.button.id)

    def _cycle_button_focus(self, forward: bool) -> None:
        focused = self.app.focused
        current_id = focused.id if focused and focused.id in self.BUTTON_IDS else None
        if forward:
            idx = self.BUTTON_IDS.index(current_id) if current_id else -1
            target_id = self.BUTTON_IDS[(idx + 1) % len(self.BUTTON_IDS)]
        else:
            idx = self.BUTTON_IDS.index(current_id) if current_id else 0
            target_id = self.BUTTON_IDS[(idx - 1) % len(self.BUTTON_IDS)]
        self.query_one(f"#{target_id}", Button).focus()

    def on_key(self, event: events.Key) -> None:
        key = event.key.lower()
        if key in self.DECISION_KEYS:
            event.stop()
            self._handle_decision(self.DECISION_KEYS[key])
            return

        if key in ("left", "up", "shift+tab"):
            event.stop()
            event.prevent_default()
            self._cycle_button_focus(forward=False)
            return

        if key in ("right", "down", "tab"):
            event.stop()
            event.prevent_default()
            self._cycle_button_focus(forward=True)
            return

        if key in ("enter", "space"):
            focused = self.app.focused
            decision = focused.id if focused and focused.id in self.BUTTON_IDS else "approve-btn"
            event.stop()
            event.prevent_default()
            self._handle_decision(decision)

    def _handle_decision(self, decision_type: str | None) -> None:
        if not self.buttons_container:
            return

        for child in self.buttons_container.query(Button):
            child.disabled = True

        if decision_type == "approve-btn":
            decisions = build_approval_decisions(self.action_requests, "approve")
        elif decision_type == "reject-btn":
            decisions = build_approval_decisions(self.action_requests, "reject")
        elif decision_type == "allow-btn":
            decisions = build_approval_decisions(self.action_requests, "allow", runtime=self.app_ref.repl.runtime)
        elif decision_type == "cancel-btn":
            decisions = build_approval_decisions(
                self.action_requests, "reject", reject_message=_("User cancelled the execution.")
            )
        else:
            decisions = []

        self.buttons_container.remove()
        self.buttons_container = None

        status_map = {
            "approve-btn": f"[bold #34d399]✓ {_('Approved')}[/bold #34d399]",
            "reject-btn": f"[bold #f87171]✗ {_('Rejected')}[/bold #f87171]",
            "allow-btn": f"[bold #38bdf8]✓ {_('Allowed for session & approved')}[/bold #38bdf8]",
            "cancel-btn": f"[bold #f87171]✗ {_('Cancelled')}[/bold #f87171]",
        }
        status_text = status_map.get(decision_type or "", "")
        self.mount(Static(f"  {status_text}", classes="approval-status"))

        inp = self.app_ref.query_one(ReplInput)
        inp.disabled = False

        footer = self.app_ref.query_one(AgentFooter)
        footer.set_approval(False)
        self.app_ref._is_approval_pending = False

        self.app_ref._current_worker = self.app_ref.run_worker(
            self.app_ref._handle_approval_decision(decisions, self.scroll, self.agent_msg)
        )
