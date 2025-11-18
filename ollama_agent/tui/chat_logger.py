"""Chat logger for writing messages to the TUI."""

from typing import Optional

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual.widgets import RichLog


class ChatLogger:
    def __init__(self, chat_log: RichLog):
        self._chat_log = chat_log

    def write_message(
        self,
        message: str,
        *,
        style: str,
        prefix: Optional[str] = None,
        markdown: bool = False,
    ) -> None:
        if markdown:
            if prefix:
                self._chat_log.write(Text(f"{prefix}:", style=style))
            self._chat_log.write(RichMarkdown(message))
            return

        text = f"{prefix}: {message}" if prefix else message
        self._chat_log.write(Text(text, style=style))

    def blank_line(self) -> None:
        self._chat_log.write("")
