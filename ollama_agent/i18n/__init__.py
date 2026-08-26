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
    """Normalize language string to lowercase language code (e.g. 'es_ES.UTF-8' -> 'es')."""
    clean = raw.strip().split(".")[0].split("@")[0]
    return clean.split("_")[0].split("-")[0].lower()


def detect_system_language() -> str:
    """Detect the system language code, falling back to DEFAULT_LOCALE."""
    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        val = os.environ.get(var)
        if val is not None and val.strip():
            for part in val.split(":"):
                if part.strip():
                    norm = _normalize_lang(part)
                    if norm in SUPPORTED_LOCALES:
                        return norm
            return DEFAULT_LOCALE

    loc = locale.getlocale()[0]
    if loc:
        norm = _normalize_lang(loc)
        if norm in SUPPORTED_LOCALES:
            return norm

    return DEFAULT_LOCALE


def _load_translations(lang: str) -> dict[str, str]:
    """Load translation mapping for a given language code."""
    if lang == DEFAULT_LOCALE:
        return {}
    data = (
        resources.files(__package__)
        .joinpath(f"locales/{lang}.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(data)


def set_locale(lang: str | None = None) -> str:
    """Set the active application locale and load its translations."""
    global _current_locale, _translations
    if not lang:
        target = detect_system_language()
    else:
        norm = _normalize_lang(lang)
        if norm not in SUPPORTED_LOCALES:
            raise ValueError(f"Unsupported language: {lang}")
        target = norm

    _current_locale = target
    _translations = _load_translations(target)
    return _current_locale


def get_text(message: str, **kwargs: Any) -> str:
    """Translate a message string with optional format arguments."""
    if _current_locale == DEFAULT_LOCALE:
        template = message
    else:
        if message not in _translations:
            raise KeyError(
                f"Missing translation for {message!r} in locale {_current_locale}"
            )
        template = _translations[message]
    if kwargs:
        return template.format(**kwargs)
    return template


_ = get_text
