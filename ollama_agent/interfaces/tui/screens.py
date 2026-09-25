"""Textual screens for the TUI application."""

from __future__ import annotations

import logging

from textual import events
from textual.screen import Screen


class MainScreen(Screen):
    """Default screen with a guard against Textual crashes when text selection
    ends over a widget detached mid-drag (``parent`` is ``None``)."""

    def _forward_event(self, event: events.Event) -> None:
        try:
            super()._forward_event(event)
        except AttributeError as err:
            if "'NoneType' object has no attribute 'region'" in str(err):
                # Known upstream Textual crash: text selection released over a
                # widget detached mid-drag. Keep observable via logging.
                logging.warning("Worked around Textual detached-widget selection crash", exc_info=err)
                self._select_state = None
                return
            raise
