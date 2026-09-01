"""Pre-cycle market brief.

Once per `LLM_BRIEF_TTL_SECONDS` (default 1h), we ask a fast LLM to look at
a compact market snapshot and return a structured JSON brief:

    {
      "market_regime": "normal" | "risk_off" | "news_shock" | "macro_uncertain",
      "cash_multiplier": 1.0,        # 0.3 in risk_off, ~1.0 in normal
      "blocked_symbols": ["SBER"],   # symbols to skip this hour
      "rationale": "...",            # 1-2 sentence explanation
      "risks": ["..."]               # 1-3 short risk bullets
    }

The brief is cached in `data/state/llm_brief.json`. If a refresh fails,
the cached value is returned even if stale — the bot never blocks on LLM.

The brief does NOT directly authorize/block trades. It's logged and made
available to the risk layer as a soft modifier (cash_multiplier). If LLM
is disabled or all attempts fail, the bot trades as if regime=normal.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from moex_agent.llm.polza_client import PolzaClient
from moex_agent.storage import JsonStore

logger = logging.getLogger(__name__)


BRIEF_STATE_KEY = "llm_brief"
BRIEF_HISTORY_STATE_KEY = "llm_brief_history"
BRIEF_HISTORY_MAX = 4  # last 4 briefs (≈ last 2 hours at 30min cadence)

_REGIME_TO_DEFAULT_MULTIPLIER = {
    "normal": 1.0,
    "macro_uncertain": 0.7,
    "news_shock": 0.5,
    "risk_off": 0.3,
}


@dataclass
class MarketBrief:
    ts: datetime
    market_regime: str
    cash_multiplier: float
    blocked_symbols: list[str]
    rationale: str
    risks: list[str] = field(default_factory=list)
    model: str = ""
    tokens_total: int = 0
    failed_snippet: str = ""

    @property
    def is_default(self) -> bool:
        return self.market_regime == "normal" and self.cash_multiplier == 1.0 and not self.blocked_symbols

    def to_payload(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "market_regime": self.market_regime,
            "cash_multiplier": self.cash_multiplier,
            "blocked_symbols": self.blocked_symbols,
            "rationale": self.rationale,
            "risks": self.risks,
            "model": self.model,
            "tokens_total": self.tokens_total,
            "failed_snippet": self.failed_snippet,
        }


def default_brief() -> MarketBrief:
    return MarketBrief(
        ts=datetime.now(timezone.utc),
        market_regime="normal",
        cash_multiplier=1.0,
        blocked_symbols=[],
        rationale="default brief (LLM disabled or never refreshed)",
        risks=[],
    )


def degraded_brief(reason: str, snippet: str = "") -> MarketBrief:
    """Fail-SAFE brief used when the LLM refresh keeps failing.

    A blind risk overlay must REDUCE risk, not maximise it: resetting to a
    normal / full-size brief on failure would let the bot buy full-size into a
    falling tape with no defensive throttle. So on repeated failure we drop to a
    cautious stance: macro_uncertain regime (raises the BUY-confidence bar in
    risk.py) + 0.6x size. This is deliberately mild (not risk_off) and, with a
    short TTL, self-corrects within one refresh cycle the moment the LLM answers
    again — avoiding a "stuck cautious" state.
    """
    return MarketBrief(
        ts=datetime.now(timezone.utc),
        market_regime="macro_uncertain",
        cash_multiplier=0.6,
        blocked_symbols=[],
        rationale=f"refresh failed twice ({reason}); fail-safe to cautious until LLM recovers",
        risks=["LLM brief unavailable — trading at reduced size with a higher BUY bar"],
        failed_snippet=snippet[:500],
    )


def _build_messages(snapshot: dict[str, Any], watchlist: list[str]) -> list[dict[str, str]]:
    system = (
        "You are a risk-overlay analyst for an autonomous MOEX trading bot. "
        "You return ONLY a strict JSON object. No prose outside JSON.\n\n"
        "Input snapshot contains:\n"
        "- portfolio: equity, cash, exposure, open_symbols\n"
        "- regime_distribution: counts of detected regimes across watchlist\n"
        "- per_symbol[ticker]: {rsi, regime, bb_position (-1=lower band, +1=upper band), "
        "atr_pct (%), trade_imbalance (-1..+1, buyer/seller pressure), book_imbalance, "
        "has_position, pnl_pct (% if has_position)}\n"
        "- market_breadth: {above_mid, below_mid, oversold_rsi_lt_30, overbought_rsi_gt_70, "
        "in_trend, in_range, avg_atr_pct, median_rsi} — one-glance read of the tape.\n"
        "- macro_context: {imoex_close, imoex_return_60m_pct, imoex_return_4h_pct} — "
        "broad index movement. Negative 4h return + most tickers below_mid → risk_off bias.\n"
        "- futoi (optional): {Si: {phys_long, phys_short, jur_long, jur_short, phys_net, jur_net}, "
        "RTSI: {...}, BR: {...}} — open positions in key futures. Heavy retail (phys) net short on Si "
        "while juridical net long usually precedes rouble strength; opposite alignment = rouble weakness "
        "= risk-off for Russian equities. Use as macro tilt, not for blocking specific tickers.\n"
        "- macro_quotes (optional): {usd_rub_spot: {secid, close, ts}, brent_fut: {secid, close, ts}, ...} "
        "— current spot/active-futures price for USDRUB and Brent. Use as macro-frame anchor: high USDRUB "
        "(>95) + falling Brent (<70) often coincides with risk-off in Russian equities. Don't block specific "
        "tickers from this — feed into market_regime classification.\n"
        "- anomalies (optional): list of {symbol, flags} for tickers with volume_spike, "
        "stretched_lower_band, stretched_upper_band, or wide_spread. Cross-reference with news.\n"
        "- news_headlines: list of recent MOEX iss headlines (may be empty). "
        "Use these to detect news_shock or risk_off regimes, and to populate "
        "blocked_symbols when a ticker is named in concerning news (sanctions, "
        "earnings miss, suspension, dividend cut). Do NOT block based on neutral "
        "news (M&A, executive changes without scandal, etc.).\n"
        "- brief_history (optional): up to 4 most recent prior briefs in compact "
        "form {ts, regime, cash_multiplier, blocked_symbols, median_rsi, "
        "imoex_return_4h_pct}. Use it to detect TREND of market mood — e.g. "
        "regime drift (normal→macro_uncertain→risk_off means deteriorating), "
        "median_rsi climbing means accelerating overbought, repeated blocks on "
        "same symbol means persistent risk. Use historical drift to confirm "
        "or temper your current call, NOT to copy past decisions verbatim.\n\n"
        "Your job: classify the market regime, optionally block tickers that look "
        "particularly risky to BUY this hour, set a cash multiplier scaling order size.\n\n"
        "Output schema (all fields required):\n"
        "{\n"
        '  "market_regime": "normal" | "risk_off" | "news_shock" | "macro_uncertain",\n'
        '  "cash_multiplier": float in [0.2, 1.0],\n'
        '  "blocked_symbols": [string],   // tickers from watchlist; empty if no concerns\n'
        '  "rationale": string (1-3 short sentences in Russian referencing concrete tickers/numbers),\n'
        '  "risks": [string]              // 1-3 short bullets in Russian\n'
        "}\n\n"
        "Calibration rules:\n"
        "- normal: most tickers in normal/range, no extreme RSI clusters, multiplier 0.9-1.0, no blocks.\n"
        "- macro_uncertain: mixed regimes, RSI bimodal or atr_pct elevated, multiplier 0.6-0.8.\n"
        "- news_shock: one or two tickers with extreme moves (|bb_position|>0.9, atr>2x peers), "
        "block them, multiplier 0.4-0.6 for the rest.\n"
        "- risk_off: most tickers oversold AND trade_imbalance negative (broad selling), "
        "multiplier 0.2-0.4, may block multiple.\n"
        "Don't block tickers just because they're oversold (RSI<30) — those are buy candidates "
        "the bot might still want. Block when oversold+strong_selling+no_book_support."
    )
    user_payload = {
        "watchlist": watchlist,
        "snapshot": snapshot,
        "instruction": "Return only the JSON object.",
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def _parse_brief(payload: dict[str, Any], *, model: str, tokens_total: int) -> MarketBrief | None:
    regime = str(payload.get("market_regime") or "normal").lower()
    if regime not in _REGIME_TO_DEFAULT_MULTIPLIER:
        regime = "normal"
    mult_raw = payload.get("cash_multiplier")
    try:
        mult = float(mult_raw) if mult_raw is not None else _REGIME_TO_DEFAULT_MULTIPLIER[regime]
    except (TypeError, ValueError):
        mult = _REGIME_TO_DEFAULT_MULTIPLIER[regime]
    mult = max(0.2, min(1.0, mult))
    blocked = payload.get("blocked_symbols") or []
    if not isinstance(blocked, list):
        blocked = []
    blocked = [str(s).upper() for s in blocked if str(s).strip()]
    risks = payload.get("risks") or []
    if not isinstance(risks, list):
        risks = []
    risks = [str(r) for r in risks if str(r).strip()][:5]
    rationale = str(payload.get("rationale") or "").strip()[:500]
    return MarketBrief(
        ts=datetime.now(timezone.utc),
        market_regime=regime,
        cash_multiplier=mult,
        blocked_symbols=blocked,
        rationale=rationale,
        risks=risks,
        model=model,
        tokens_total=tokens_total,
    )


class BriefService:
    """Manages the cached brief lifecycle."""

    def __init__(
        self,
        store: JsonStore,
        client: PolzaClient,
        *,
        model: str,
        ttl_seconds: int = 3600,
        max_tokens: int = 500,
        fallback_model: str = "",
    ):
        self.store = store
        self.client = client
        self.model = model
        # Tried in the same refresh if the primary returns empty/invalid JSON.
        self.fallback_model = fallback_model
        self.ttl = timedelta(seconds=max(60, int(ttl_seconds)))
        self.max_tokens = max_tokens

    def get_or_refresh(
        self,
        *,
        snapshot: dict[str, Any],
        watchlist: list[str],
        now: datetime | None = None,
    ) -> MarketBrief:
        now = now or datetime.now(timezone.utc)
        cached = self._load_cached()
        if cached is not None and (now - cached.ts) < self.ttl:
            return cached

        if not self.client.enabled():
            return cached or default_brief()

        # Enrich snapshot with last 4 brief summaries so the LLM can see the
        # trend, not just the current moment. Each entry is compact: regime,
        # multiplier, blocked symbols, plus key macro markers if available.
        try:
            history = self.store.read_state(BRIEF_HISTORY_STATE_KEY) or []
            if isinstance(history, list) and history:
                snapshot = dict(snapshot)  # don't mutate caller's dict
                snapshot["brief_history"] = history[-BRIEF_HISTORY_MAX:]
        except Exception:
            pass

        messages = _build_messages(snapshot=snapshot, watchlist=watchlist)

        def _persist_failure(reason: str, snippet: str = "") -> MarketBrief:
            # If the cached brief was already a failure (consecutive failures),
            # fail safe to a cautious stance instead of holding a possibly stale
            # state forever.
            cached_was_failure = (
                cached is not None
                and "refresh failed" in (cached.rationale or "")
            )
            if cached_was_failure:
                # Circuit broken twice in a row — fail SAFE to a cautious stance
                # (reduced size + higher BUY bar) instead of fail-OPEN to normal.
                # See degraded_brief() for the rationale.
                reset = degraded_brief(reason, snippet=snippet)
                reset = MarketBrief(
                    ts=now,
                    market_regime=reset.market_regime,
                    cash_multiplier=reset.cash_multiplier,
                    blocked_symbols=reset.blocked_symbols,
                    rationale=reset.rationale,
                    risks=reset.risks,
                    model="",
                    tokens_total=0,
                    failed_snippet=snippet[:500],
                )
                self._save(reset)
                logger.warning(
                    "llm brief refresh failed twice — fail-safe to cautious (macro_uncertain, 0.6x)",
                    extra={"extra": {"reason": reason}},
                )
                return reset
            # First failure: keep last known state (existing behavior).
            fallback = cached if cached is not None else default_brief()
            persisted = MarketBrief(
                ts=now,
                market_regime=fallback.market_regime,
                cash_multiplier=fallback.cash_multiplier,
                blocked_symbols=fallback.blocked_symbols,
                rationale=f"refresh failed ({reason}); reusing previous/default until next TTL",
                risks=fallback.risks,
                model=fallback.model,
                tokens_total=fallback.tokens_total,
                failed_snippet=snippet[:500],
            )
            self._save(persisted)
            logger.info("llm brief refresh failed — circuit-broken until next TTL", extra={"extra": {"reason": reason}})
            return persisted

        def _try_model(model: str) -> tuple[MarketBrief | None, str, str]:
            """Returns (brief|None, failure_reason, snippet) for one model."""
            resp = self.client.chat(
                model=model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.1,
                json_mode=True,
            )
            if resp is None:
                return None, "no_response", ""
            parsed = resp.as_json()
            if not parsed:
                return None, "invalid_json", resp.content
            parsed_brief = _parse_brief(parsed, model=resp.model, tokens_total=resp.total_tokens)
            if parsed_brief is None:
                return None, "parse_failed", resp.content
            return parsed_brief, "", ""

        # Primary model, then fallback in the SAME refresh (different model
        # families rarely fail on the same call), then the cautious fail-safe.
        brief, reason, snippet = _try_model(self.model)
        if brief is None and self.fallback_model and self.fallback_model != self.model:
            logger.info("brief primary failed; trying fallback", extra={"extra": {"primary": self.model, "reason": reason, "fallback": self.fallback_model}})
            fb_brief, fb_reason, fb_snippet = _try_model(self.fallback_model)
            if fb_brief is not None:
                brief = fb_brief
            else:
                reason, snippet = f"{reason}+fallback_{fb_reason}", (fb_snippet or snippet)
        if brief is None:
            return _persist_failure(reason, snippet=snippet)
        self._save(brief)
        logger.info(
            "llm brief refreshed",
            extra={"extra": {
                "regime": brief.market_regime,
                "cash_multiplier": brief.cash_multiplier,
                "blocked": brief.blocked_symbols,
                "tokens": brief.tokens_total,
                "model": brief.model,
            }},
        )
        return brief

    def _load_cached(self) -> MarketBrief | None:
        raw = self.store.read_state(BRIEF_STATE_KEY)
        if not raw or not isinstance(raw, dict):
            return None
        try:
            return MarketBrief(
                ts=datetime.fromisoformat(str(raw.get("ts"))),
                market_regime=str(raw.get("market_regime") or "normal"),
                cash_multiplier=float(raw.get("cash_multiplier") or 1.0),
                blocked_symbols=list(raw.get("blocked_symbols") or []),
                rationale=str(raw.get("rationale") or ""),
                risks=list(raw.get("risks") or []),
                model=str(raw.get("model") or ""),
                tokens_total=int(raw.get("tokens_total") or 0),
                failed_snippet=str(raw.get("failed_snippet") or ""),
            )
        except Exception as exc:
            logger.warning("failed to parse cached brief", extra={"extra": {"error": str(exc)}})
            return None

    def _save(self, brief: MarketBrief) -> None:
        self.store.write_state(BRIEF_STATE_KEY, brief.to_payload())
        # Append to history so the next refresh sees the trend.
        try:
            history = self.store.read_state(BRIEF_HISTORY_STATE_KEY) or []
            if not isinstance(history, list):
                history = []
            entry = {
                "ts": brief.ts.isoformat(),
                "regime": brief.market_regime,
                "cash_multiplier": brief.cash_multiplier,
                "blocked_symbols": brief.blocked_symbols,
            }
            history.append(entry)
            # Keep the tail bounded so the state file doesn't grow.
            history = history[-(BRIEF_HISTORY_MAX * 3):]
            self.store.write_state(BRIEF_HISTORY_STATE_KEY, history)
        except Exception:
            pass
