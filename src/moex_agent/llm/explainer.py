"""Post-trade explanation service.

After the bot places an order, this module asks an LLM to write a short
human-readable explanation of WHY the trade was made: which signal fired,
what the regime was, what the ML filter said, and what the main risks are
going forward.

Critical contract: this is fire-and-forget and never blocks the trading
loop. A ThreadPoolExecutor (size 2) holds in-flight requests; if it's full
the explanation is silently dropped. Failures are logged but never raised.

Output lands in `data/logs/llm_explanations.jsonl`, one line per trade.
Experts evaluating the project see these explanations directly in the logs.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from moex_agent.llm.polza_client import PolzaClient
from moex_agent.storage import JsonStore

logger = logging.getLogger(__name__)


def _build_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You write concise post-trade explanations for an autonomous trading "
        "agent on the Moscow Exchange (MOEX). The agent already executed the "
        "trade — your job is to explain it, not approve it. "
        "Return a STRICT JSON object:\n"
        "{\n"
        '  "summary": string (1-2 sentences, Russian, plain text),\n'
        '  "primary_signal": string (e.g. "oversold near lower Bollinger band"),\n'
        '  "key_risks": [string]  // 1-3 short bullets\n'
        "}\n"
        "Be specific about numbers. Do not invent facts not present in the input."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
    ]


class ExplainerService:
    def __init__(
        self,
        store: JsonStore,
        client: PolzaClient,
        *,
        model: str,
        max_tokens: int = 350,
        max_workers: int = 2,
    ):
        self.store = store
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="llm-explain")

    def explain_async(self, context: dict[str, Any]) -> None:
        if not self.client.enabled():
            return
        # Defensive: don't queue if pool already saturated. ThreadPoolExecutor's
        # _work_queue is a private attribute but reading it is safe (read-only).
        queue_len = getattr(self._pool, "_work_queue", None)
        if queue_len is not None and queue_len.qsize() > 4:
            logger.warning("llm explainer queue saturated, dropping explanation")
            return
        try:
            self._pool.submit(self._run, context)
        except RuntimeError as exc:
            logger.warning("llm explainer pool rejected job", extra={"extra": {"error": str(exc)}})

    def _run(self, context: dict[str, Any]) -> None:
        try:
            resp = self.client.chat(
                model=self.model,
                messages=_build_messages(context),
                max_tokens=self.max_tokens,
                temperature=0.2,
                json_mode=True,
            )
            payload: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": context.get("symbol"),
                "order": context.get("order"),
            }
            if resp is None:
                payload["status"] = "no_response"
            else:
                parsed = resp.as_json() or {}
                summary = parsed.get("summary") or ""
                # If JSON-mode produced empty fields, fall back to raw content
                # so we never silently drop useful text. Trim to keep logs sane.
                raw_snippet = (resp.content or "").strip()[:1500]
                payload.update({
                    "status": "ok",
                    "model": resp.model,
                    "tokens_total": resp.total_tokens,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "summary": summary if summary else raw_snippet,
                    "primary_signal": parsed.get("primary_signal"),
                    "key_risks": parsed.get("key_risks") or [],
                    "raw_content": raw_snippet if not summary else None,
                })
            self.store.append_jsonl("llm_explanations", payload)
        except Exception:
            logger.exception("llm explainer failed")

    def shutdown(self) -> None:
        try:
            self._pool.shutdown(wait=False)
        except Exception:
            pass
