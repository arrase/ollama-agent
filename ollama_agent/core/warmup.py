"""Model warm-up utilities for reducing cold start latency.

Ollama loads model weights into VRAM only on the first inference request.
By sending a no-op generate call (empty prompt) during application startup,
we can overlap the weight loading with the time the user spends composing
their first prompt.
"""

import logging
import time

import ollama

_log = logging.getLogger(__name__)


async def preload_model(model: str, base_url: str) -> None:
    """Preload a model into VRAM by issuing an empty generate request.

    This triggers Ollama to load the model weights without producing any
    tokens.  The function is designed to be launched as a fire-and-forget
    ``asyncio.Task`` — it never raises and logs outcomes at debug/warning
    level.
    """
    host = base_url.rstrip("/")
    t0 = time.monotonic()
    try:
        client = ollama.AsyncClient(host=host)
        await client.generate(model=model, prompt="")
        elapsed = time.monotonic() - t0
        _log.info("Model '%s' preloaded in %.1fs", model, elapsed)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        _log.warning(
            "Model preload for '%s' failed after %.1fs: %s",
            model, elapsed, exc,
        )
