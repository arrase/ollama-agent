"""Internationalization (i18n) support for ollama-agent."""

from __future__ import annotations

import json
import locale
import os
from importlib import resources
from typing import Any

SUPPORTED_LOCALES: tuple[str, ...] = (
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "zh",
    "ja",
    "ru",
    "hi",
    "ko",
    "ar",
    "tr",
    "pl",
    "nl",
    "uk",
)
DEFAULT_LOCALE = "en"

_current_locale: str = DEFAULT_LOCALE
_translations: dict[str, str] = {}


def _normalize_lang(raw: str) -> str:
    clean = raw.strip().split(".")[0].split("@")[0]
    return clean.split("_")[0].split("-")[0].lower()


def _system_locale_candidates() -> list[str]:
    candidates = []
    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        if var in os.environ:
            candidates.extend(os.environ[var].split(":"))
    loc = locale.getlocale()[0]
    if loc:
        candidates.append(loc)
    return candidates


def detect_system_language() -> str:
    for item in _system_locale_candidates():
        code = _normalize_lang(item)
        if code in SUPPORTED_LOCALES:
            return code
    return DEFAULT_LOCALE


def set_locale(lang: str | None = None) -> str:
    global _current_locale, _translations
    target = _normalize_lang(lang) if lang else detect_system_language()
    if target not in SUPPORTED_LOCALES:
        raise ValueError(f"Unsupported language: {lang}")

    _current_locale = target
    _translations = {}
    if target != DEFAULT_LOCALE:
        data = resources.files(__name__).joinpath("locales", f"{target}.json").read_text(encoding="utf-8")
        _translations = json.loads(data)
    return _current_locale


def get_locale() -> str:
    return _current_locale


def get_text(message: str, **kwargs: Any) -> str:
    text = _translations[message] if message in _translations else message
    if kwargs:
        return text.format(**kwargs)
    return text


_ = get_text

__all__ = [
    "DEFAULT_LOCALE",
    "SUPPORTED_LOCALES",
    "_",
    "detect_system_language",
    "get_locale",
    "get_text",
    "set_locale",
]
