"""Shared search text, URL source and HTTP timeout helpers."""
from __future__ import annotations

import html
import re
from urllib.parse import urlparse

import httpx


def _clean_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return _clean_text(text)


def _clean_text(value: str) -> str:
    text = html.unescape(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _source_from_url(url: str) -> str:
    try:
        host = urlparse(url).netloc
    except ValueError:
        return ""
    return host.removeprefix("www.")


def _httpx_timeout(timeout_seconds: float) -> httpx.Timeout:
    total = max(0.1, float(timeout_seconds))
    connect = min(1.5, max(0.4, total * 0.3))
    return httpx.Timeout(timeout=total, connect=connect)


def _wikipedia_host(host: str) -> bool:
    host = _normalized_host(host)
    return any(host == suffix or host.endswith("." + suffix) for suffix in _WIKIPEDIA_HOST_SUFFIXES)


_WIKIPEDIA_HOST_SUFFIXES = (
    "wikipedia.org",
    "wikimedia.org",
    "m.wikipedia.org",
)


def _normalized_host(host: str) -> str:
    host = (host or "").lower().split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host
