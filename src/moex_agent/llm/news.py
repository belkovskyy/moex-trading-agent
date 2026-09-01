"""MOEX news fetcher for the LLM brief.

Pulls recent headlines from MOEX iss (`https://iss.moex.com/iss/news.json`),
caches them with a short TTL (default 20 min — more often than brief TTL of
60 min so a fresh brief sees recent news).

Fails soft: on any network/parse error returns the last cached snapshot
(or an empty list), never raises. The trading loop must keep running even
if MOEX news is unreachable.

The fetcher is intentionally independent of moexalgo — `iss.moex.com` is
public, doesn't need MOEX_ALGO_TOKEN, and has its own schema.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


# Public MOEX iss news endpoints. No auth needed. We try in order:
#   1) sitenews — MOEX-wide news feed with titles + publish dates
#   2) news     — alternative iss endpoint (fallback for older schemas)
# We scan ALL top-level blocks in the response and use the first one that
# has a title-like column. ISS schemas drift; this is more robust than
# hardcoding a single block name.
_NEWS_URLS = [
    "https://iss.moex.com/iss/sitenews.json",
    "https://iss.moex.com/iss/news.json",
]
_USER_AGENT = "moex-agent/1.0"
_CACHE_STATE_KEY = "moex_news"

# Columns that signal "this block is news". Case-insensitive prefix match.
_TITLE_KEYS = ("title", "headline", "name", "subject")
_DATE_KEYS = ("publish_date", "modified_at", "created_at", "publish_time", "tradedate")


def _scan_iss_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Scan ALL blocks in an ISS response and return the first newsy one
    as a list of {title, publish_date}."""
    if not isinstance(data, dict):
        return []
    for block_name, block in data.items():
        if not isinstance(block, dict):
            continue
        columns = block.get("columns")
        rows = block.get("data")
        if not isinstance(columns, list) or not isinstance(rows, list):
            continue
        if not rows:
            continue
        # find title column (case-insensitive)
        lower_columns = [str(c).lower() for c in columns]
        title_idx = next(
            (i for i, c in enumerate(lower_columns) if any(c.startswith(k) for k in _TITLE_KEYS)),
            None,
        )
        if title_idx is None:
            continue
        date_idx = next(
            (i for i, c in enumerate(lower_columns) if any(c.startswith(k) for k in _DATE_KEYS)),
            None,
        )
        out: list[dict[str, Any]] = []
        for raw in rows:
            if not isinstance(raw, list) or len(raw) != len(columns):
                continue
            title = str(raw[title_idx] or "").strip()
            if not title:
                continue
            publish_date = str(raw[date_idx]).strip() if date_idx is not None and raw[date_idx] else None
            out.append({"title": title, "publish_date": publish_date})
        if out:
            logger.debug("iss block matched", extra={"extra": {"block": block_name, "items": len(out)}})
            return out
    return []


def _is_fresh(payload: dict[str, Any], ttl_seconds: int, now: datetime) -> bool:
    fetched_at = payload.get("fetched_at")
    if not fetched_at:
        return False
    try:
        ts = datetime.fromisoformat(str(fetched_at))
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts) < timedelta(seconds=max(60, int(ttl_seconds)))


class NewsService:
    """Cached fetcher for MOEX iss news.

    Stores snapshot in `data/state/moex_news.json` so it survives bot
    restarts. TTL ~20 minutes. Returns a compact list of headlines suitable
    for handing to an LLM brief.
    """

    def __init__(self, store, *, ttl_seconds: int = 1200, timeout: float = 8.0, max_items: int = 12):
        self.store = store
        self.ttl_seconds = ttl_seconds
        self.timeout = timeout
        self.max_items = max_items

    def get_or_refresh(self, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or datetime.now(timezone.utc)
        cached = self.store.read_state(_CACHE_STATE_KEY) or {}
        if _is_fresh(cached, self.ttl_seconds, now):
            items = cached.get("items") or []
            return items if isinstance(items, list) else []

        items: list[dict[str, Any]] = []
        last_error: str | None = None
        for url in _NEWS_URLS:
            try:
                response = requests.get(
                    url,
                    headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                last_error = f"{type(exc).__name__} on {url}: {exc}"
                continue
            parsed = _scan_iss_blocks(data)
            if parsed:
                items = parsed
                break
            # If we got 200 but no newsy block, log the keys so we know what to fix.
            keys = list(data.keys()) if isinstance(data, dict) else []
            logger.info(
                "moex news endpoint returned no newsy blocks",
                extra={"extra": {"url": url, "top_keys": keys[:8]}},
            )

        if not items:
            logger.info(
                "moex news fetch returned no items; using cached/empty",
                extra={"extra": {"error": last_error}},
            )
            stale = dict(cached) if isinstance(cached, dict) else {}
            stale["fetched_at"] = now.isoformat()
            stale.setdefault("items", [])
            self.store.write_state(_CACHE_STATE_KEY, stale)
            return list(stale.get("items") or [])

        # Keep only most recent N (ISS already sorted desc, but we slice
        # defensively in case the order changes).
        items = items[: self.max_items]
        payload = {
            "fetched_at": now.isoformat(),
            "items": items,
        }
        self.store.write_state(_CACHE_STATE_KEY, payload)
        logger.info(
            "moex news refreshed",
            extra={"extra": {"count": len(items), "sample": items[0]["title"][:100] if items else None}},
        )
        return items


def compact_headlines(items: list[dict[str, Any]], *, max_chars: int = 1500) -> list[str]:
    """Return a list of short headlines, truncating to a max char budget."""
    out: list[str] = []
    total = 0
    for item in items:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        ts = item.get("publish_date") or ""
        line = f"[{ts}] {title}" if ts else title
        if total + len(line) > max_chars:
            break
        out.append(line)
        total += len(line) + 1
    return out
