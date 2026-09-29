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
    if lang is not None and not lang.strip():
        raise ValueError("Language must be a non-empty locale code")
    target = _normalize_lang(lang) if lang else detect_system_language()
    if target not in SUPPORTED_LOCALES:
        raise ValueError(f"Unsupported language: {lang}. Expected one of: {', '.join(SUPPORTED_LOCALES)}")

    # Load and validate before mutating global state: a corrupt locale file must not
    # leave the process on the new locale with an empty translation table.
    translations: dict[str, str] = {}
    if target != DEFAULT_LOCALE:
        path = resources.files(__name__).joinpath("locales", f"{target}.json")
        try:
            data: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Failed to load locale '{target}': {exc}") from exc
        if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
            raise ValueError(f"Locale file for '{target}' must be an object of string -> string")
        translations = data

    _translations = translations
    _current_locale = target
    return _current_locale


def get_locale() -> str:
    return _current_locale


def get_text(message: str, **kwargs: Any) -> str:
    text = _translations[message] if message in _translations else message
    if not kwargs:
        return text
    try:
        return text.format(**kwargs)
    except (KeyError, IndexError):
        # A literal brace in a (possibly translated) message must not crash the caller.
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
