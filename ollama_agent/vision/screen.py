from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import mss
from PIL import Image

_DISPLAY_TOKEN_RE = re.compile(r"@dp(\d+)")


class ScreenCaptureError(RuntimeError):
    """Error during screen capture operations."""


@dataclass(frozen=True, slots=True)
class CapturedImage:
    mime_type: str
    base64_data: str

    @property
    def data_url(self) -> str:
        return f"data:{self.mime_type};base64,{self.base64_data}"


def extract_display_tokens(prompt: str) -> tuple[str, list[int]]:
    """Extract @dpN tokens from prompt. Returns (cleaned_prompt, unique_indexes)."""
    indexes = [int(m.group(1)) for m in _DISPLAY_TOKEN_RE.finditer(prompt)]
    cleaned = " ".join(_DISPLAY_TOKEN_RE.sub("", prompt).split())
    # preserva orden, elimina duplicados
    return cleaned, list(dict.fromkeys(indexes))


def _require_display_session() -> None:
    """Raise if no graphical session is available (Linux only)."""
    if os.name == "posix" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise ScreenCaptureError(
            "No graphical session detected. Screen capture unavailable in headless mode."
        )


def capture_display_as_base64(
    display_index: int,
    *,
    image_format: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 85,
) -> CapturedImage:
    """Capture monitor screenshot as base64. display_index is 0-based."""
    _require_display_session()

    with mss.mss() as sct:
        # mss.monitors[0] = all monitors combined; [1..N] = individual monitors
        if display_index < 0 or display_index >= len(sct.monitors) - 1:
            max_idx = max(0, len(sct.monitors) - 2)
            raise ScreenCaptureError(
                f"Invalid monitor dp{display_index}. Available: 0..{max_idx}"
            )

        shot = sct.grab(sct.monitors[display_index + 1])
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

        buf = io.BytesIO()
        if image_format == "jpeg":
            img.save(buf, format="JPEG", quality=min(95, max(1, jpeg_quality)))
            mime = "image/jpeg"
        else:
            img.save(buf, format="PNG")
            mime = "image/png"

        return CapturedImage(
            mime_type=mime,
            base64_data=base64.b64encode(buf.getvalue()).decode("ascii"),
        )


def build_multimodal_responses_input(
    text: str,
    images: Iterable[CapturedImage],
) -> list[dict[str, Any]]:
    """Build a standard multimodal user message for LangChain/Ollama.

    Returns a DeepAgents/LangChain-compatible messages list:
    [{"role": "user", "content": [ ...content blocks... ]}]
    where blocks are standard content blocks ("image_url" + "text").
    """

    blocks: list[dict[str, Any]] = []
    for img in images:
        blocks.append({"type": "image_url", "image_url": img.data_url})
    if text:
        blocks.append({"type": "text", "text": text})
    return [{"role": "user", "content": blocks}]


def build_multimodal_user_message(text: str, images: Iterable[CapturedImage]) -> dict[str, Any]:
    """Convenience wrapper returning a single user message dict."""
    return build_multimodal_responses_input(text, images)[0]
