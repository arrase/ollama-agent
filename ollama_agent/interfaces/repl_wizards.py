from __future__ import annotations

import inspect
from typing import Any, Callable

from ..skills.commands import SkillError

async def safe_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Call *fn*(*args, **kwargs), awaiting if necessary and silencing SystemExit/SkillError."""
    try:
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            await result
    except (SystemExit, SkillError):
        pass
