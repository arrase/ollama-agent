"""Interactive wizard helpers for the REPL.

Provides prompt utilities and safe calling helpers.
"""

import inspect

async def safe_call(fn, *args, **kwargs):
    """Call *fn*(*args, **kwargs), awaiting if necessary and silencing SystemExit."""
    try:
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            await result
    except SystemExit:
        pass
