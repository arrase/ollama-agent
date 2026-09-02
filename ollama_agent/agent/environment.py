"""Shared prompt-environment helpers for the main agent and subagents."""

from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path

#: Virtual skill mount points exposed by the CompositeBackend.
SKILL_ROOTS: tuple[tuple[str, str], ...] = (("/system_skills/", "Built-in"), ("/skills/", "User"))


def environment_block(*, include_cwd: bool) -> str:
    """Build the '# ENVIRONMENT' section appended to system prompts."""
    now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# ENVIRONMENT",
        f"Operating System: {platform.system()} ({platform.release()}, {platform.machine()})",
        f"Current Date & Time: {now_str}",
    ]
    if include_cwd:
        cwd = Path.cwd().resolve()
        lines.append(
            f"Working Directory: {cwd} "
            '(directory where shell commands start in; this is what execute(command="pwd") reports)'
        )
    return "\n\n" + "\n".join(lines) + "\n"
