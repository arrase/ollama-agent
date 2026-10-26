"""Header widget displaying agent status and context window usage."""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from textual.widgets import Static

from ....i18n import _


def _format_context_info(tokens: int, eff_ctx: int, ms_context_window: int | str) -> str:
    num_ctx = 0
    if eff_ctx > 0:
        num_ctx = eff_ctx
    elif isinstance(ms_context_window, int):
        num_ctx = ms_context_window
    elif ms_context_window.isdigit():
        num_ctx = int(ms_context_window)

    if num_ctx <= 0:
        return ""

    pct = int((tokens / num_ctx) * 100)
    if pct > 90:
        color = "#f87171"
    elif pct > 75:
        color = "#fbbf24"
    else:
        color = "#38bdf8"

    tok_str = f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)
    ctx_str = f"{num_ctx / 1000:.1f}k" if num_ctx >= 1000 else str(num_ctx)
    ctx_label = _("Context:")
    return (
        f"  [#30363d]│[/]  [bold #8b949e]{ctx_label}[/bold #8b949e] "
        f"[bold {color}]{tok_str}/{ctx_str} ({pct}%)[/bold {color}]"
    )


class AgentHeader(Static):
    """Dynamic TUI Header displaying agent status information."""

    def __init__(self, repl: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.repl = repl

    def on_mount(self) -> None:
        self.update_header()

    def update_header(self) -> None:
        ms = self.repl.runtime.settings.model
        tokens = self.repl.runtime.last_context_tokens
        eff_ctx = self.repl.runtime.effective_context_window
        ctx_info = _format_context_info(tokens, eff_ctx, ms.context_window)

        rag_ctx = self.repl._rag_ctx
        rag_db = rag_ctx.rag_manager.current_database if rag_ctx else None
        rag_label = _("RAG:")
        rag_info = (
            f"  [#30363d]│[/]  [bold #8b949e]{rag_label}[/bold #8b949e] [bold #a78bfa]{escape(rag_db)}[/bold #a78bfa]"
            if rag_db
            else ""
        )
        yolo = self.repl.runtime.yolo_mode
        stealth = self.repl.runtime.stealth_mode

        yolo_status = (
            f"[bold #0d1117 on #f87171] {_('YOLO: ON')} [/bold #0d1117 on #f87171]"
            if yolo
            else f"[#7d8590]{_('YOLO: OFF')}[/#7d8590]"
        )
        stealth_status = (
            f"[bold #0d1117 on #c084fc] {_('STEALTH: ON')} [/bold #0d1117 on #c084fc]"
            if stealth
            else f"[#7d8590]{_('STEALTH: OFF')}[/#7d8590]"
        )
        self.update(
            f"[bold #38bdf8]● ollama-agent[/bold #38bdf8]  [#30363d]│[/]  "
            f"[bold #8b949e]{_('Model:')}[/bold #8b949e] "
            f"[bold #e6edf3]{escape(ms.name)}[/bold #e6edf3]{ctx_info}  [#30363d]│[/]  "
            f"[bold #8b949e]{_('Effort:')}[/bold #8b949e] "
            f"[#e6edf3]{escape(ms.reasoning_effort)}[/#e6edf3]{rag_info}  [#30363d]│[/]  "
            f"{yolo_status}  [#30363d]│[/]  {stealth_status}"
        )
