from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal


_DISPLAY_TOKEN_RE = re.compile(r"@dp(?P<index>\d+)")


class ScreenCaptureError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CapturedImage:
    mime_type: str
    base64_data: str


def extract_display_tokens(prompt: str) -> tuple[str, list[int]]:
    """Extract @dpN tokens from a prompt.

    Returns (clean_prompt, display_indexes).

    Example: "describe @dp0" -> ("describe", [0])
    """
    indexes: list[int] = []

    def _repl(match: re.Match[str]) -> str:
        idx = int(match.group("index"))
        indexes.append(idx)
        return ""

    cleaned = _DISPLAY_TOKEN_RE.sub(_repl, prompt)
    cleaned = " ".join(cleaned.split())

    # keep order, remove duplicates
    seen: set[int] = set()
    unique: list[int] = []
    for idx in indexes:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)

    return cleaned, unique


def _require_display_session() -> None:
    """Best-effort guard for headless sessions (Linux)."""
    if os.name != "posix":
        return

    # For X11 and Wayland, these env vars are commonly present.
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise ScreenCaptureError(
            "No graphical session detected (missing DISPLAY/WAYLAND_DISPLAY). "
            "Screen capture is not available in headless mode."
        )


def capture_display_as_base64(
    display_index: int,
    *,
    image_format: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 85,
) -> CapturedImage:
    """Capture a screenshot of the requested monitor and return as base64 data URL payload.

    display_index is 0-based (dp0 = monitor 0).
    """
    _require_display_session()

    try:
        import mss  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ScreenCaptureError(
            "Missing dependency 'mss'. Install it to use @dpN."
        ) from exc

    # Prefer JPEG if Pillow is available; otherwise fall back to PNG.
    pillow_available = True
    try:
        from PIL import Image  # type: ignore
    except Exception:
        pillow_available = False

    with mss.mss() as sct:
        monitors = getattr(sct, "monitors", None)
        if not monitors or not isinstance(monitors, list):
            raise ScreenCaptureError("Unable to enumerate monitors for capture.")

        # mss uses monitors[1] as the first real monitor; monitors[0] is all monitors.
        mss_index = display_index + 1
        if mss_index < 1 or mss_index >= len(monitors):
            raise ScreenCaptureError(
                f"Invalid monitor dp{display_index}. Available: 0..{max(0, len(monitors) - 2)}"
            )

        monitor = monitors[mss_index]
        shot = sct.grab(monitor)

        if image_format == "jpeg" and pillow_available:
            from PIL import Image  # type: ignore

            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=max(1, min(95, int(jpeg_quality))))
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return CapturedImage(mime_type="image/jpeg", base64_data=b64)

        # Fallback to PNG (works without Pillow)
        import mss.tools  # type: ignore

        png_bytes = mss.tools.to_png(shot.rgb, shot.size)
        if png_bytes is None:
            raise ScreenCaptureError("Failed to convert screenshot to PNG.")
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return CapturedImage(mime_type="image/png", base64_data=b64)


def build_multimodal_responses_input(
    text: str,
    images: Iterable[CapturedImage],
) -> list[dict[str, Any]]:
    """Build OpenAI Responses-style multimodal input.

    Matches the documented structure:
    [{"role":"user","content":[{"type":"input_text",...},{"type":"input_image",...}]}]
    """
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "input_text", "text": text})

    for img in images:
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{img.mime_type};base64,{img.base64_data}",
            }
        )

    return [{"role": "user", "content": content}]
