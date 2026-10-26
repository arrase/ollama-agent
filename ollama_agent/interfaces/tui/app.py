"""Main Textual application representing the agent's interactive TUI."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langgraph.types import Command
from rich.markup import escape
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option
from textual.worker import Worker

from ...core import AudioPrompt, AudioRecorder
from ...i18n import _
from ...streaming import extract_action_requests, stream_agent_events
from ..clipboard import ClipboardError, copy_to_system_clipboard, get_system_clipboard
from .commands import _is_immediate_command, run_slash_command
from .completion import (
    NO_ARG_COMMANDS,
    _AT_MENTION_RE,
    _AUTOCOMPLETE_LIST_ID,
    _CHAT_SCROLL_ID,
    file_completions,
    slash_completions,
)
from .renderer import _TUIStreamingRenderer
from .screens import MainScreen
from .widgets import (
    AgentFooter,
    AgentHeader,
    AgentResponse,
    AudioRecordButton,
    PromptQueueWidget,
    ReplInput,
    SystemOutputWidget,
    ToolApprovalWidget,
    UserMessage,
)

_log = logging.getLogger(__name__)


class OllamaAgentApp(App):
    """Main Textual Application representing the Agent's interactive TUI."""

    BINDINGS = [
        ("escape", "cancel_generation", _("Interrupt")),
        ("ctrl+c", "cancel_or_quit", _("Interrupt/Quit")),
        ("super+c", "copy_selection", _("Copy")),
        ("ctrl+shift+c", "copy_selection", _("Copy")),
        ("ctrl+insert", "copy_selection", _("Copy")),
    ]

    CSS_PATH = Path(__file__).parent / "repl.tcss"

    _is_generating: bool
    _is_approval_pending: bool
    _is_recording: bool

    def action_cancel_generation(self) -> None:
        if self._is_recording:
            self.cancel_audio_recording()
            return

        sys_out = self.query_one(SystemOutputWidget)
        if sys_out.display:
            self.clear_system_output()
            return

        if self._prompt_queue:
            self._prompt_queue.clear()
            self._update_queue_ui()
            self.show_system_notice(f"[bold #f87171]🛑 {_('Prompt queue cleared.')}[/bold #f87171]")
        if self._is_generating and self._current_worker is not None:
            self._current_worker.cancel()
        elif self._is_approval_pending:
            for w in self.query(ToolApprovalWidget):
                if not w._decided:
                    w._handle_decision("cancel-btn")
                    return
            self._begin_cancel_approval()
            self.run_worker(self._finish_cancel_approval())

    def action_cancel_or_quit(self) -> None:
        if self._is_recording or self._is_generating or self._is_approval_pending or self._prompt_queue:
            self.action_cancel_generation()
        else:
            self.exit()

    def action_copy_selection(self) -> None:
        selected_text = self.screen.get_selected_text()
        if selected_text:
            self.copy_to_clipboard(selected_text)

    def copy_to_clipboard(self, text: str) -> None:
        super().copy_to_clipboard(text)
        try:
            copy_to_system_clipboard(text)
        except ClipboardError as exc:
            self.notify(_("Failed to copy to system clipboard: {exc}", exc=exc), severity="warning")

    @property
    def clipboard(self) -> str:
        try:
            sys_clip = get_system_clipboard()
        except ClipboardError:
            return super().clipboard
        if sys_clip:
            return sys_clip
        return super().clipboard

    def on_text_selected(self, event: events.TextSelected) -> None:
        self.action_copy_selection()

    def get_default_screen(self) -> Screen:
        return MainScreen(id="_default")

    def __init__(self, repl: Any) -> None:
        super().__init__()
        self.repl = repl
        self.repl.app = self
        self._is_generating = False
        self._is_approval_pending = False
        self._is_recording = False
        self._recorder: AudioRecorder | None = None
        self._recording_timer: Timer | None = None
        self._recording_started_at = 0.0
        self._recording_max_duration = 0.0
        self._prompt_queue: deque[str | AudioPrompt] = deque()
        self._current_worker: Worker | None = None
        self._post_stream_cleanup: Callable[[], Awaitable[None]] | None = None

    def compose(self) -> ComposeResult:
        yield AgentHeader(self.repl)
        yield ScrollableContainer(id=_CHAT_SCROLL_ID.lstrip("#"))
        yield OptionList(id=_AUTOCOMPLETE_LIST_ID.lstrip("#"))
        yield PromptQueueWidget(id="prompt-queue")
        yield SystemOutputWidget(id="system-output")
        with Container(id="input-container"), Horizontal(id="input-bar"):
            yield Static("❯ ", id="prompt-char")
            yield ReplInput(id="repl-input")
            yield AudioRecordButton(id="audio-record-btn")
        yield AgentFooter()

    def show_system_output(self, content: str | Text, title: str | None = None) -> None:
        self.query_one(SystemOutputWidget).show_output(content, title=title)

    def show_system_notice(self, notice: str | Text) -> None:
        self.query_one(SystemOutputWidget).show_notice(notice)

    def clear_system_output(self) -> None:
        self.query_one(SystemOutputWidget).clear_output()

    def on_mount(self) -> None:
        self.query_one(ReplInput).focus()
        self.update_mode_ui()
        # exit_on_error=False: a startup failure must be reported in the UI, not crash the app.
        self.run_worker(self._warmup_agent(), group="warmup", exit_on_error=False)

    def on_unmount(self) -> None:
        if self._is_recording:
            self._stop_recorder()

    async def _warmup_agent(self) -> None:
        try:
            await self.repl.runtime._ensure_graph()
        except asyncio.CancelledError:
            raise
        # Startup failures are reported in the TUI instead of killing it.
        except Exception as exc:  # noqa: BLE001
            _log.exception("Agent warmup failed")
            notice = escape(_("Agent startup failed: {exc}", exc=exc))
            self.show_system_notice(f"[bold #f87171]✕ {notice}[/bold #f87171]")
            return
        self.update_mode_ui()

    def update_mode_ui(self) -> None:
        prompt_char = self.query_one("#prompt-char")
        input_container = self.query_one("#input-container")
        yolo = self.repl.runtime.yolo_mode
        stealth = self.repl.runtime.stealth_mode

        input_container.set_class(yolo, "yolo-mode")
        input_container.set_class(stealth, "stealth-mode")

        if yolo and stealth:
            prompt_char.styles.color = "#fbbf24"  # Amber / Dual mode
        elif yolo:
            prompt_char.styles.color = "#f87171"  # Red / Coral
        elif stealth:
            prompt_char.styles.color = "#c084fc"  # Purple / Violet
        else:
            prompt_char.styles.color = "#38bdf8"  # Sky Blue

        audio_btn = self.query_one(AudioRecordButton)
        supports_audio = bool(self.repl.runtime.supports_audio)
        audio_btn.display = supports_audio
        if supports_audio:
            max_s = self.repl.runtime.settings.audio.max_duration
            audio_btn.tooltip = _("Record audio prompt (max {seconds}s)", seconds=max_s)
        else:
            audio_btn.tooltip = None
            if self._is_recording:
                self.cancel_audio_recording()

        header = self.query_one(AgentHeader)
        header.update_header()

    def _update_queue_ui(self) -> None:
        footer = self.query_one(AgentFooter)
        footer.set_queued_count(len(self._prompt_queue))
        self.query_one(PromptQueueWidget).update_queue(self._prompt_queue)

    # ── Audio recording ───────────────────────────────────────────────────

    def on_audio_record_button_toggle_requested(self, event: AudioRecordButton.ToggleRequested) -> None:
        if self._is_recording:
            self.stop_and_submit_audio()
        else:
            self.start_audio_recording()

    def start_audio_recording(self) -> None:
        if self._is_recording or not self.repl.runtime.supports_audio:
            return
        self.clear_system_output()
        recorder = AudioRecorder()
        try:
            recorder.start()
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to start audio recording: %s", exc)
            notice = escape(_("Could not start audio recording: {exc}", exc=exc))
            self.show_system_notice(f"[bold #f87171]✕ {notice}[/bold #f87171]")
            return

        self._recorder = recorder
        self._is_recording = True
        self._recording_max_duration = float(self.repl.runtime.settings.audio.max_duration)
        self._recording_started_at = time.monotonic()

        self.query_one("#input-container").add_class("recording-mode")
        self.query_one(AudioRecordButton).set_recording(True, self._recording_max_duration)
        self.query_one(AgentFooter).set_recording(True)
        self._recording_timer = self.set_interval(0.1, self._on_recording_tick)
        self.query_one(ReplInput).focus()

    def _stop_recorder(self) -> AudioRecorder | None:
        if self._recording_timer is not None:
            self._recording_timer.stop()
            self._recording_timer = None
        self._is_recording = False
        self.query_one("#input-container").remove_class("recording-mode")
        self.query_one(AudioRecordButton).set_recording(False)
        self.query_one(AgentFooter).set_recording(False)
        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            try:
                recorder.stop()
            except Exception as exc:  # noqa: BLE001
                _log.warning("Error stopping audio recorder: %s", exc)
        return recorder

    def _on_recording_tick(self) -> None:
        if not self._is_recording:
            return
        elapsed = time.monotonic() - self._recording_started_at
        remaining = max(0.0, self._recording_max_duration - elapsed)
        self.query_one(AudioRecordButton).set_recording(True, remaining)
        if elapsed >= self._recording_max_duration:
            self.stop_and_submit_audio()

    def cancel_audio_recording(self) -> None:
        if not self._is_recording:
            return
        self._stop_recorder()
        self.show_system_notice(f"[bold #f87171]🛑 {escape(_('Audio recording cancelled.'))}[/bold #f87171]")

    def stop_and_submit_audio(self) -> None:
        if not self._is_recording:
            return
        recorder = self._stop_recorder()
        if recorder is None:
            return
        if recorder.is_silent:
            self.show_system_notice(
                f"[bold #fbbf24]⚠ {escape(_('No speech detected in audio recording.'))}[/bold #fbbf24]"
            )
            return

        inp = self.query_one(ReplInput)
        text = inp.text.strip()
        inp.text = ""
        if text:
            inp.add_history_entry(text)
        audio_prompt = recorder.build_prompt(text)

        if self._is_generating or self._is_approval_pending:
            self._prompt_queue.append(audio_prompt)
            self._update_queue_ui()
            return

        self._start_stream(audio_prompt)

    # ── Input events ──────────────────────────────────────────────────────

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "repl-input":
            self.update_autocomplete(event.text_area.text)

    def on_repl_input_submitted(self, event: ReplInput.Submitted) -> None:
        val = event.value.strip()
        if not val:
            return
        self.clear_system_output()
        event.input.text = ""
        event.input.add_history_entry(val)

        if _is_immediate_command(val):
            worker = self.run_worker(run_slash_command(self, val))
            if not self._is_generating:
                self._current_worker = worker
            return

        if self._is_generating or self._is_approval_pending:
            self._prompt_queue.append(val)
            self._update_queue_ui()
            return

        if val.startswith("/"):
            self._start_slash(val)
        else:
            self._start_stream(val)

    def _start_slash(self, cmd_line: str) -> None:
        """Schedule a slash-command worker, claiming the generation slot first.

        ``_is_generating`` is flipped here rather than inside the coroutine because
        ``run_worker`` only creates the task: until it is first scheduled the flag
        would still read ``False`` and a second Enter could start a parallel run.
        """
        if not _is_immediate_command(cmd_line):
            self._is_generating = True
            self.query_one(AgentFooter).set_generating(True)
        self._current_worker = self.run_worker(run_slash_command(self, cmd_line))

    def mount_turn_widgets(
        self, user_text: str | AudioPrompt, scroll: Any = None
    ) -> tuple[Any, AgentResponse]:
        if scroll is None:
            scroll = self.query_one(_CHAT_SCROLL_ID)
        scroll.mount(UserMessage(user_text))
        agent_msg = AgentResponse()
        scroll.mount(agent_msg)
        self._deferred_scroll()
        return scroll, agent_msg

    def _start_stream(self, prompt: str | AudioPrompt) -> None:
        """Mount the prompt widgets and schedule a streaming worker (see _start_slash)."""
        scroll, agent_msg = self.mount_turn_widgets(prompt)
        self._is_generating = True
        self.query_one(AgentFooter).set_generating(True)
        self._current_worker = self.run_worker(self._run_stream(prompt, scroll, agent_msg))

    def _process_next_in_queue(self) -> None:
        if not self._prompt_queue or self._is_generating or self._is_approval_pending:
            return
        item = self._prompt_queue.popleft()
        self._update_queue_ui()
        if isinstance(item, str) and item.startswith("/"):
            self._start_slash(item)
        else:
            self._start_stream(item)

    # ── Autocomplete ──────────────────────────────────────────────────────

    def hide_autocomplete(self) -> None:
        autolist = self.query_one(_AUTOCOMPLETE_LIST_ID, OptionList)
        autolist.clear_options()
        autolist.display = False
        autolist.highlighted = None

    def update_autocomplete(self, text: str) -> None:
        autolist = self.query_one(_AUTOCOMPLETE_LIST_ID, OptionList)

        # 1. File completions via @-mention
        match = _AT_MENTION_RE.search(text)
        if match:
            token = match.group(1) or match.group(2) or match.group(3) or ""
            items = list(
                file_completions(
                    token,
                    max_completions=self.repl.runtime.settings.mentions.max_completions,
                )
            )
            if items:
                autolist.clear_options()
                for insert_val, display_markup in items:
                    autolist.add_option(Option(prompt=display_markup, id=insert_val))
                autolist.display = True
                autolist.highlighted = 0
                return

        # 2. Slash command completions
        if text.startswith("/") and text not in NO_ARG_COMMANDS:
            items = slash_completions(text, self.repl, self._prompt_queue)
            if items:
                autolist.clear_options()
                for insert_val, display_text in items:
                    autolist.add_option(Option(prompt=display_text, id=insert_val))
                autolist.display = True
                autolist.highlighted = 0
                return

        self.hide_autocomplete()

    def accept_completion(self, option_index: int) -> None:
        autolist = self.query_one(_AUTOCOMPLETE_LIST_ID, OptionList)
        opt = autolist.get_option_at_index(option_index)
        if not opt or not opt.id:
            return

        inp = self.query_one(ReplInput)
        text = inp.text

        match = _AT_MENTION_RE.search(text)
        if match:
            # File mention completion
            start = match.start()
            # Preserve character before @ if it was part of boundary match
            prefix_char = text[start] if text[start] != "@" else ""
            before = text[:start] + prefix_char + "@"
            quote = '"' if " " in opt.id else ""
            inp.text = before + quote + opt.id + quote + " "
        else:
            # Slash command completion
            trailing = "" if opt.id in NO_ARG_COMMANDS else " "
            inp.text = opt.id + trailing

        inp.action_cursor_line_end()
        self.hide_autocomplete()

    # ── Deferred scroll helper ────────────────────────────────────────────

    def _deferred_scroll(self) -> None:
        """Schedule a scroll-to-end after the next layout refresh."""
        scroll = self.query_one(_CHAT_SCROLL_ID)
        self.call_after_refresh(scroll.scroll_end, animate=False)

    # ── Streaming chat ────────────────────────────────────────────────────

    async def _run_post_stream_cleanup(self) -> None:
        if self._post_stream_cleanup is not None:
            cleanup = self._post_stream_cleanup
            self._post_stream_cleanup = None
            await cleanup()

    def _begin_cancel_approval(self) -> None:
        self._is_approval_pending = False
        self.query_one(AgentFooter).set_approval(False)
        self.show_system_notice(f"[bold #f87171]🛑 {_('Approval cancelled.')}[/bold #f87171]")
        self.query_one(ReplInput).focus()

    async def _finish_cancel_approval(self) -> None:
        await self._run_post_stream_cleanup()
        self.query_one(AgentHeader).update_header()
        self._process_next_in_queue()

    async def cancel_approval(self) -> None:
        self._begin_cancel_approval()
        await self._finish_cancel_approval()

    async def _handle_approval_decision(
        self, decisions: list[dict[str, Any]], scroll: Any, agent_msg: AgentResponse
    ) -> None:
        self._is_approval_pending = False
        command: Command[Any] = Command(resume={"decisions": decisions})
        await self._run_stream(command, scroll, agent_msg)

    async def _run_stream(
        self, prompt: str | AudioPrompt | Command[Any], scroll: Any, agent_msg: AgentResponse
    ) -> None:
        """Stream one agent turn, reporting failures in the message widget.

        A run that raises must not propagate: ``run_worker`` defaults to
        ``exit_on_error=True``, which turns any exception into a fatal app crash.
        """
        self._is_generating = True
        self.query_one(AgentFooter).set_generating(True)
        renderer = _TUIStreamingRenderer(self, scroll, agent_msg)
        try:
            try:
                await stream_agent_events(
                    self.repl.runtime,
                    prompt,
                    renderer,
                )
            except asyncio.CancelledError:
                self.show_system_notice(f"[bold #f87171]🛑 {_('Execution interrupted by user.')}[/bold #f87171]")
                if self._prompt_queue:
                    self._prompt_queue.clear()
                    self._update_queue_ui()
                raise
            # Surface the failure in the transcript and keep the app alive.
            except Exception as exc:  # noqa: BLE001
                _log.exception("Agent run failed")
                agent_msg.add_error(_("Run failed: {exc}", exc=exc))
                self.query_one(ReplInput).focus()
                return

            if renderer.last_interrupt is not None:
                action_requests = extract_action_requests(renderer.last_interrupt)
                self._is_approval_pending = True
                footer = self.query_one(AgentFooter)
                footer.set_approval(True)
                approval_widget = ToolApprovalWidget(
                    action_requests=action_requests,
                    app_ref=self,
                    scroll=scroll,
                    agent_msg=agent_msg,
                )
                await agent_msg.mount(approval_widget)
                self._deferred_scroll()
            else:
                self.query_one(ReplInput).focus()
        finally:
            try:
                if not self._is_approval_pending:
                    await self._run_post_stream_cleanup()
            finally:
                self._is_generating = False
                self.query_one(AgentFooter).set_generating(False)
                self.query_one(AgentHeader).update_header()
                if not self._is_approval_pending:
                    self._process_next_in_queue()
