"""TUI renderers for streaming markdown and reasoning."""

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual.widgets import RichLog


def _clear_log_lines(log: RichLog, start_line: int) -> None:
    """Remove rendered lines after ``start_line`` and clear RichLog caches."""

    while len(log.lines) > start_line:
        log.lines.pop()

    # RichLog lacks a public API to drop cached rendered lines; clear manually.
    if hasattr(log, "_line_cache"):
        log._line_cache.clear()


class StreamingMarkdownRenderer:
    """Render markdown content progressively inside a RichLog widget."""

    def __init__(self, chat_log: RichLog, update_frequency: int = 5):
        self.chat_log = chat_log
        self.update_frequency = max(1, update_frequency)
        self.buffer = ""
        self.token_count = 0
        self._start_line_count = 0
        self._rendering_started = False

    def start_rendering(self) -> None:
        """Prepare the log for streaming output (only once)."""
        if not self._rendering_started:
            self.chat_log.write(Text("Agent:", style="bold green"))
            self._start_line_count = len(self.chat_log.lines)
            self._rendering_started = True

    def append_token(self, token: str) -> None:
        """Append a token and refresh the view at the configured cadence."""
        if not self._rendering_started:
            self.start_rendering()

        self.buffer += token
        self.token_count += 1
        if self.token_count % self.update_frequency == 0:
            self._update_display()

    def finalize(self) -> None:
        """Render the buffered markdown one last time."""
        self._update_display()

    def _update_display(self) -> None:
        if not self.buffer:
            return

        _clear_log_lines(self.chat_log, self._start_line_count)
        self.chat_log.write(RichMarkdown(self.buffer))
        self.chat_log.scroll_end(animate=False)
        self.chat_log.refresh()


class ReasoningRenderer:
    """Render reasoning/thinking process in real-time."""

    def __init__(self, chat_log: RichLog):
        self.chat_log = chat_log
        self.reasoning_buffer = ""
        self.is_active = False
        self._start_line_count = 0
        self._update_counter = 0
        self._update_frequency = 5  # Update every N tokens for performance

    def start_reasoning(self) -> None:
        """Start capturing reasoning tokens."""
        if not self.is_active:
            self.chat_log.write("")  # Blank line before reasoning
            self.is_active = True
            self.reasoning_buffer = ""
            self._start_line_count = len(self.chat_log.lines)
            self._update_counter = 0

    def append_reasoning_token(self, token: str) -> None:
        """Append a reasoning token and update display in real-time."""
        self.reasoning_buffer += token
        self._update_counter += 1

        # Update display every N tokens for performance
        if self._update_counter % self._update_frequency == 0:
            self._update_display()

    def finalize_reasoning(self) -> None:
        """Finish the reasoning display with final update."""
        if self.is_active and self.reasoning_buffer:
            # Final update to ensure all tokens are shown
            self._update_display()
            self.chat_log.write("")  # Add a blank line after reasoning
        self.is_active = False
        self.reasoning_buffer = ""
        self._start_line_count = 0

    def _update_display(self) -> None:
        """Update the reasoning line in the chat log."""
        # Remove any lines we added before (if updating)
        _clear_log_lines(self.chat_log, self._start_line_count)

        # Write the updated reasoning line
        reasoning_line = Text()
        reasoning_line.append("🧠 Thinking: ", style="bold magenta")
        reasoning_line.append(self.reasoning_buffer,
                              style="dim italic magenta")
        self.chat_log.write(reasoning_line)

        # Scroll to end to show the update
        self.chat_log.scroll_end(animate=False)
        self.chat_log.refresh()
