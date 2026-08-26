"""Shared prompt-environment helpers for the main agent and subagents."""

from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path

#: Virtual skill mount points exposed by the CompositeBackend.
SKILL_ROOTS: list[str | tuple[str, str]] = [("/system_skills/", "Built-in"), ("/skills/", "User")]


def environment_block(*, include_cwd: bool) -> str:
    """Build the '# ENVIRONMENT' section appended to system prompts."""
    lines = [
        "# ENVIRONMENT",
        f"Operating System: {platform.system()} ({platform.release()})",
    ]
    if include_cwd:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        cwd = Path.cwd().resolve()
        lines.append(f"Current Date & Time: {now_str}")
        lines.append(
            f'Working Directory: {cwd} (directory where shell commands start in; this is what execute(command="pwd") reports)'
        )
    return "\n\n" + "\n".join(lines) + "\n"
