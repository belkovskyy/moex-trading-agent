"""Polza model catalog auditor.

Hits `GET /v1/models` and ranks the catalog by relevance to our agent:
- cheap workhorses for post-trade explanations (low cost, decent JSON)
- premium models for daily retrospectives
- shows context length and pricing in one screen

Run with:
    python -m moex_agent.llm_models_audit
or to save full catalog as JSON:
    python -m moex_agent.llm_models_audit --save data/llm_catalog.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests

from moex_agent.config import settings


def fetch_models() -> list[dict[str, Any]]:
    if not settings.polza_api_key:
        raise SystemExit("POLZA_API_KEY is empty in .env")
    url = f"{settings.polza_base_url.rstrip('/')}/v1/models"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {settings.polza_api_key}"},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    # OpenAI-compatible format: {"data": [...]}
    if isinstance(data, dict) and "data" in data:
        return list(data["data"])
    if isinstance(data, list):
        return data
    raise SystemExit(f"Unexpected /v1/models response shape: {type(data).__name__}")


def _price(model: dict[str, Any], key: str) -> float:
    """Extract per-1M-token price. Polza/OpenRouter style nests it under 'pricing'."""
    pricing = model.get("pricing") or {}
    raw = pricing.get(key)
    if raw is None:
        # Some catalogs put it at top level
        raw = model.get(key)
    if raw is None:
        return 0.0
    try:
        # Often returned as string per-token, multiply to per-1M
        v = float(raw)
        # Heuristic: per-token values are <1.0, per-1M are big.
        if v < 1.0 and v > 0:
            v *= 1_000_000
        return v
    except (TypeError, ValueError):
        return 0.0


def _context(model: dict[str, Any]) -> int:
    raw = (
        model.get("context_length")
        or model.get("context_window")
        or (model.get("top_provider") or {}).get("context_length")
    )
    try:
        return int(raw) if raw else 0
    except (TypeError, ValueError):
        return 0


def _id(model: dict[str, Any]) -> str:
    return str(model.get("id") or model.get("name") or "?")


def _name(model: dict[str, Any]) -> str:
    return str(model.get("name") or model.get("id") or "?")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=Path, default=None, help="Save full catalog as JSON")
    parser.add_argument("--top", type=int, default=15, help="How many top entries per ranking to show")
    args = parser.parse_args()

    print(f"Fetching {settings.polza_base_url}/v1/models ...")
    models = fetch_models()
    print(f"Got {len(models)} models\n")

    enriched = []
    for m in models:
        p_in = _price(m, "prompt")
        p_out = _price(m, "completion")
        enriched.append({
            "id": _id(m),
            "name": _name(m),
            "context": _context(m),
            "price_in_per_1M": p_in,
            "price_out_per_1M": p_out,
            "blended": p_in * 0.7 + p_out * 0.3,  # roughly our input-heavy workload
            "raw": m,
        })

    # Filter: chat-capable only (drop image/audio/embedding by heuristic)
    def is_chat(m: dict[str, Any]) -> bool:
        mid = m["id"].lower()
        if any(skip in mid for skip in ["embed", "whisper", "tts", "image", "stable-diffusion", "dalle", "yandex-art", "midjourney"]):
            return False
        return True

    chat = [m for m in enriched if is_chat(m)]
    print(f"Chat-capable: {len(chat)}\n")

    # Top-cheap
    print("=" * 70)
    print(f"TOP-{args.top} CHEAPEST (by blended cost — workhorse candidates)")
    print("=" * 70)
    print(f"{'ID':<55} {'IN/1M':>9} {'OUT/1M':>9} {'CTX':>8}")
    for m in sorted(chat, key=lambda x: x["blended"])[:args.top]:
        print(f"{m['id']:<55} {m['price_in_per_1M']:>9.2f} {m['price_out_per_1M']:>9.2f} {m['context']:>8}")

    # Top by context (might matter for daily retrospective)
    print()
    print("=" * 70)
    print(f"TOP-{args.top} BY CONTEXT (premium / retrospective candidates)")
    print("=" * 70)
    print(f"{'ID':<55} {'IN/1M':>9} {'OUT/1M':>9} {'CTX':>8}")
    for m in sorted(chat, key=lambda x: -x["context"])[:args.top]:
        print(f"{m['id']:<55} {m['price_in_per_1M']:>9.2f} {m['price_out_per_1M']:>9.2f} {m['context']:>8}")

    # Look for our current picks
    print()
    print("=" * 70)
    print("OUR CURRENT PICKS — verify they exist in catalog")
    print("=" * 70)
    targets = [settings.llm_model_fast, settings.llm_model_premium]
    for target in targets:
        match = next((m for m in chat if m["id"].lower() == target.lower()), None)
        if match:
            print(f"OK   {target}: in={match['price_in_per_1M']:.2f} out={match['price_out_per_1M']:.2f} ctx={match['context']}")
        else:
            similar = [m['id'] for m in chat if target.split('/')[-1].lower()[:6] in m['id'].lower()][:5]
            print(f"MISS {target}: not found. Similar IDs: {similar}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nFull catalog saved: {args.save}")


if __name__ == "__main__":
    main()
