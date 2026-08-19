"""Optional Tavily-backed web search and page extraction tools."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Literal
from urllib.parse import urlparse

from langchain.tools import tool
from tavily import AsyncTavilyClient
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)
from tavily.errors import TimeoutError as TavilyTimeoutError

from ..settings import TavilySettings

SearchTopic = Literal["general", "news", "finance"]
TimeRange = Literal["day", "week", "month", "year"]

_TAVILY_ERRORS = (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    TavilyTimeoutError,
    UsageLimitExceededError,
)


def _api_key(settings: TavilySettings) -> str:
    """Resolve the YAML key first, then fall back to the standard environment variable."""
    return settings.api_key.strip() or os.getenv("TAVILY_API_KEY", "").strip()


def _configured_limit(settings: TavilySettings, requested: int | None) -> int:
    """Clamp result counts to Tavily's limits and the configured maximum."""
    configured = max(1, min(int(settings.max_results), 20))
    if requested is None:
        return configured
    return max(1, min(int(requested), configured))


def _valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _search_results(response: dict[str, Any]) -> list[dict[str, Any]]:
    fields = ("title", "url", "content", "score", "published_date")
    return [
        {key: result[key] for key in fields if result.get(key) is not None}
        for result in response.get("results", [])
    ]


async def _run_search(
    client: AsyncTavilyClient,
    settings: TavilySettings,
    query: str,
    max_results: int | None = None,
    topic: SearchTopic = "general",
    time_range: TimeRange | None = None,
    include_domains: Sequence[str] | None = None,
    exclude_domains: Sequence[str] | None = None,
) -> dict[str, Any]:
    response = await client.search(
        query=query,
        search_depth=settings.search_depth,
        topic=topic,
        time_range=time_range,
        max_results=_configured_limit(settings, max_results),
        chunks_per_source=max(1, min(int(settings.chunks_per_source), 3)),
        include_domains=list(include_domains or []),
        exclude_domains=list(exclude_domains or []),
        include_answer=False,
        include_raw_content=False,
        include_images=False,
    )
    return {
        "success": True,
        "query": response.get("query", query),
        "results": _search_results(response),
    }


async def _run_fetch(
    client: AsyncTavilyClient,
    settings: TavilySettings,
    url: str,
    query: str | None = None,
) -> dict[str, Any]:
    if not _valid_url(url):
        return {
            "success": False,
            "error": "URL must use http or https and include a host.",
        }

    extract_kwargs: dict[str, Any] = {
        "urls": url,
        "extract_depth": settings.extract_depth,
        "format": "markdown",
        "include_images": False,
    }
    if query:
        extract_kwargs["query"] = query
        extract_kwargs["chunks_per_source"] = max(
            1, min(int(settings.chunks_per_source), 5)
        )
    response = await client.extract(**extract_kwargs)
    results = response.get("results", [])
    if not results:
        failures = response.get("failed_results", [])
        error = failures[0].get("error") if failures else "No content returned."
        return {"success": False, "url": url, "error": error}

    content = str(results[0].get("raw_content") or "")
    limit = max(1, int(settings.max_content_chars))
    return {
        "success": True,
        "url": results[0].get("url", url),
        "content": content[:limit],
        "truncated": len(content) > limit,
    }


def build_tavily_tools(settings: TavilySettings) -> list[Any]:
    """Build Tavily tools when an API key is configured."""
    api_key = _api_key(settings)
    if not api_key:
        return []

    @tool
    async def web_search(
        query: str,
        max_results: int | None = None,
        topic: SearchTopic = "general",
        time_range: TimeRange | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search the web for current information and return relevant source snippets."""
        try:
            async with AsyncTavilyClient(api_key=api_key) as client:
                return await _run_search(
                    client,
                    settings,
                    query,
                    max_results,
                    topic,
                    time_range,
                    include_domains,
                    exclude_domains,
                )
        except _TAVILY_ERRORS as exc:
            return {"success": False, "error": str(exc)}

    @tool
    async def web_fetch(url: str, query: str | None = None) -> dict[str, Any]:
        """Fetch readable Markdown from a web page, optionally focused on a query."""
        try:
            async with AsyncTavilyClient(api_key=api_key) as client:
                return await _run_fetch(client, settings, url, query)
        except _TAVILY_ERRORS as exc:
            return {"success": False, "error": str(exc)}

    return [web_search, web_fetch]
