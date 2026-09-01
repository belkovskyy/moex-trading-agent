"""Mid-session LLM review.

Runs every ~3 hours during the trading day. Reads recent closed trades,
the current LLM brief, and asks the LLM to choose a session mode that
risk.py will apply for the next hours:

    - normal     : default behavior, no overrides
    - cautious   : tighter ML confidence bar (+0.05), size x0.7
    - pause_buys : block new BUYs for 1 hour (SELLs still pass)
    - accelerate : cooldown -50%, size x1.2  (ONLY allowed when win_rate>70%)

The mode lives in `data/state/mid_review.json` and is read by risk.py at
every decision. Mode expires after `expires_at` (default 3 hours).

Safety:
- Mode whitelisted to the four above. Anything else falls back to "normal".
- `accelerate` requires recent win_rate >= 70% (LLM can't force aggressive
  mode on a losing streak).
- Default expiry 3h, max expiry 6h.
- Logs every decision with rationale and inputs.

Usage:
    python -m moex_agent.mid_review               # apply for next 3h
    python -m moex_agent.mid_review --dry-run     # only log, no apply
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from moex_agent.config import settings
from moex_agent.llm import PolzaClient


VALID_MODES = {"normal", "cautious", "pause_buys", "accelerate"}
STATE_FILE_NAME = "mid_review.json"
LOG_FILE_NAME = "mid_review.jsonl"


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _read_jsonl(path: Path, n: int = 5000) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows[-n:]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _compute_recent_stats(orders: list[dict], window_hours: int = 6) -> dict:
    """FIFO-match buys with sells in the last N hours to compute fresh stats."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)

    def _extract(o: dict):
        # orders.jsonl rows are {"ts", "result": {...}} — trade fields are
        # NESTED: result.request.{secid,direction} (live submitted),
        # result.order.{symbol,side} (paper/rejected), result.actual_shares /
        # result.response.{price,quantity,success}. Earlier code read top-level
        # o["symbol"]/o["side"]/result["qty"] which never exist -> always 0 trades.
        result = o.get("result") or {}
        status = result.get("status") or ""
        response = result.get("response") or {}
        filled = status in ("filled", "ok") or (status == "submitted" and bool(response.get("success")))
        if not filled:
            return None
        req = result.get("request") or {}
        order = result.get("order") or {}
        sym = req.get("secid") or order.get("symbol")
        direction = str(req.get("direction") or order.get("side") or "").upper()
        side = "buy" if direction in ("B", "BUY") else "sell" if direction in ("S", "SELL") else ""
        qty = float(result.get("actual_shares") or response.get("quantity") or order.get("qty") or 0.0)
        price = float(response.get("price") or order.get("limit_price") or 0.0)
        if not sym or side not in ("buy", "sell") or qty <= 0 or price <= 0:
            return None
        return sym, side, qty, price

    by_symbol: dict[str, list[tuple]] = {}
    for o in orders:
        ts = _parse_ts(o.get("ts"))
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < cutoff:
            continue
        parsed = _extract(o)
        if parsed is not None:
            by_symbol.setdefault(parsed[0], []).append((ts, parsed))

    # Signed FIFO so SHORT round-trips (SELL open -> BUY cover) are counted too,
    # not just BUY->SELL longs (see models.match_round_trips).
    from moex_agent.models import match_round_trips
    closed = []
    for sym, rows in by_symbol.items():
        rows.sort(key=lambda x: x[0])
        seq = [(side, qty, price) for _ts, (_sym, side, qty, price) in rows]
        for rt in match_round_trips(seq):
            closed.append({"symbol": sym, "pnl_pct": rt["pnl_pct"], "pnl_abs": rt["pnl_abs"]})

    # Turnover in the window (so mid_review can weigh the venue's turnover requirement).
    turnover_rub = sum(
        qty * price
        for rows in by_symbol.values()
        for _ts, (_s, _side, qty, price) in rows
    )

    if not closed:
        return {"closed_trades": 0, "win_rate_pct": 0.0, "total_pnl_abs": 0.0,
                "avg_pnl_pct": 0.0, "turnover_rub": round(turnover_rub, 0)}

    wins = [t for t in closed if t["pnl_abs"] > 0]
    return {
        "closed_trades": len(closed),
        "win_rate_pct": (len(wins) / len(closed)) * 100,
        "total_pnl_abs": sum(t["pnl_abs"] for t in closed),
        "avg_pnl_pct": sum(t["pnl_pct"] for t in closed) / len(closed),
        "worst_symbol": min(closed, key=lambda t: t["pnl_abs"])["symbol"] if closed else None,
        "turnover_rub": round(turnover_rub, 0),
    }


def _recent_auto_tuner_change() -> dict | None:
    """Return the most recent non-dry-run auto_tuner result, compact form.
    Only the latest one — supervisor cares about what's affecting bot NOW,
    not a tuning history."""
    from moex_agent.config import settings as _s
    log_path = _s.runtime_data_dir / "logs" / "auto_tuner.jsonl"
    if not log_path.exists():
        return None
    try:
        entries: list[dict] = []
        with log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
        for e in reversed(entries):
            if e.get("dry_run"):
                continue
            changes = e.get("applied_changes") or []
            if not changes:
                continue
            return {
                "ts": e.get("ts"),
                "changes": [
                    {"key": c.get("key"), "from": c.get("from"), "to": c.get("to")}
                    for c in changes
                ],
            }
    except Exception:
        pass
    return None


def _build_messages(stats: dict, brief: dict, recent_blocks: dict, prior_modes: list | None = None, hours_in_current_mode: float = 0.0, current_ml_threshold: float = 0.70, fill_context: dict | None = None) -> list[dict]:
    system = (
        "You are an autonomous trading agent's mid-session supervisor. "
        "Every 90 minutes you choose the trading mode for the next window.\n\n"
        "Output a STRICT JSON object only:\n"
        '{ "mode": "normal" | "cautious" | "pause_buys" | "accelerate",\n'
        '  "expires_hours": float in [1, 6],\n'
        '  "ml_min_up_probability": float in [0.58, 0.74],\n'
        '  "rationale": "short Russian explanation" }\n\n'
        "ML THRESHOLD (`ml_min_up_probability`) — you are the FAST adapter of this\n"
        "buy-filter gate (auto_tuner only revisits it every 12h; you run every 90m).\n"
        "It is given to you as `current_ml_min_up_probability` plus `fill_context`\n"
        "(orders/blocks this window). Lower it (toward 0.60) when the market is\n"
        "FAVOURABLE but we are barely trading — e.g. broad OVERSOLD (median RSI<45),\n"
        "regime normal, near-zero fills, and the strategy is producing buy\n"
        "candidates that the filter rejects by a small margin. Raise it (toward\n"
        "0.72) when overbought / proven losses. Otherwise keep it ~current. You may\n"
        "move it at most 0.05 per cycle; values are hard-clamped to [0.58, 0.74].\n"
        "This is how the bot adapts to the regime instead of sitting idle.\n\n"
        "CRITICAL RULE: mode is a function of MARKET CONDITIONS and PROVEN "
        "problems, NOT of how many trades we made. cautious mode reduces "
        "trades by design — if you pick cautious because 'few trades', you "
        "create a deadlock: cautious -> fewer trades -> 'still few trades' "
        "-> stay cautious forever. DON'T DO THIS.\n\n"
        "Mode selection rules (in priority order):\n"
        "1. brief.regime == 'news_shock' AND a ticker is in active loss → pause_buys\n"
        "2. brief.regime == 'risk_off' AND market_breadth shows broad selling → cautious\n"
        "3. ≥3 consecutive losses in last 6h closed trades → cautious (proven problem)\n"
        "4. win_rate < 50% AND closed_trades >= 5 → cautious (statistically bad)\n"
        "5. win_rate >= 70% AND total_pnl > 0 AND regime == 'normal' AND closed_trades >= 5 → accelerate\n"
        "6. ELSE → normal (this is the default, including 'no data yet')\n\n"
        "DO NOT pick cautious because:\n"
        "- 'no closed trades' (could just mean market is quiet — normal mode handles this)\n"
        "- 'many blocked decisions' (filters are doing their job — that's normal)\n"
        "- 'we've been in cautious before' (each decision stands on its own)\n\n"
        "TURNOVER: the venue requires >10M RUB / 14 days (~700k/day). `turnover_rub` is\n"
        "this window's turnover. If it is LOW and win_rate is acceptable (>55%) and\n"
        "there's no proven problem, lean toward normal/accelerate — do NOT choose\n"
        "cautious, which would shrink turnover further and risk the turnover penalty.\n\n"
        "Mode effects (for context):\n"
        "- normal: ML threshold per regime, full position size\n"
        "- cautious: ML threshold +0.03, size 0.85x — modest defensive shift\n"
        "- pause_buys: block ALL new BUYs for 1h (SELLs still pass)\n"
        "- accelerate: cooldown -50%, size 1.2x"
    )
    user_payload = {
        "current_brief": {
            "regime": brief.get("market_regime"),
            "cash_multiplier": brief.get("cash_multiplier"),
            "blocked_symbols": brief.get("blocked_symbols"),
            "age_seconds": (
                int((datetime.now(timezone.utc) - _parse_ts(brief.get("ts"))).total_seconds())
                if brief.get("ts") and _parse_ts(brief.get("ts"))
                else None
            ),
        },
        "recent_window_stats": stats,
        "current_ml_min_up_probability": round(current_ml_threshold, 3),
        "fill_context": fill_context or {},
        "recent_block_reasons_top5": recent_blocks,
        # Last 5 mid_review decisions so the supervisor sees its own trend
        # and doesn't flip-flop between modes every cycle.
        "prior_mid_review_decisions": prior_modes or [],
        # How long we've been in the current mode. If hours_in_current_mode > 3
        # and it's cautious/pause_buys with no improvement — strong signal to
        # try normal as a probe. Anti-deadlock awareness.
        "hours_in_current_mode": round(hours_in_current_mode, 2),
        # Latest parameter changes from auto_tuner — gives the supervisor
        # context like "ML threshold was lowered yesterday, so increased
        # trade count is the EXPECTED effect, not a fluke".
        "recent_auto_tuner_change": _recent_auto_tuner_change(),
        "instruction": (
            "Return only the JSON object. Look at prior_mid_review_decisions "
            "before deciding: if you've been in cautious for 3 cycles in a row "
            "with no improvement, consider whether a stronger move (pause_buys) "
            "is justified. If accelerate worked last time, you can keep it. "
            "If recent_auto_tuner_change shows parameters were just tightened "
            "or loosened, attribute observed changes in trade frequency to "
            "that, not to market behavior — give it a cycle to settle before "
            "compounding via mode change."
        ),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def _validate_proposal(parsed: dict, stats: dict, regime: str, prior_modes: list | None = None, current_threshold: float = 0.70) -> dict:
    """Apply guardrails. Returns a sanitized proposal."""
    mode = str(parsed.get("mode") or "").strip().lower()
    if mode not in VALID_MODES:
        mode = "normal"

    # Guardrail: 'accelerate' requires healthy stats
    if mode == "accelerate":
        ok = (
            stats.get("closed_trades", 0) >= 5
            and stats.get("win_rate_pct", 0.0) >= 70.0
            and stats.get("total_pnl_abs", 0.0) > 0
            and regime == "normal"
        )
        if not ok:
            mode = "normal"

    # Anti-stuck guardrail: if cautious for 3+ cycles in a row with zero
    # closed trades, override to normal for ONE cycle to probe whether the
    # market actually justifies caution or we're stuck in a self-reinforcing
    # loop.
    if mode == "cautious" and stats.get("closed_trades", 0) == 0 and prior_modes:
        recent_cautious_streak = 0
        for entry in reversed(prior_modes):
            if str(entry.get("mode", "")).lower() == "cautious":
                recent_cautious_streak += 1
            else:
                break
        if recent_cautious_streak >= 3:
            mode = "normal"  # break the loop with one probe cycle

    # Guardrail: clamp expires_hours. Capped at 1h (was 6h) so no session mode —
    # especially pause_buys, a portfolio-wide BUY halt — can latch for hours like
    # the old daily breaker did. mid_review re-runs every 45min (and on events),
    # so a 1h ceiling means the mode is always re-evaluated before it expires.
    try:
        expires_h = float(parsed.get("expires_hours") or 0.75)
    except (TypeError, ValueError):
        expires_h = 0.75
    expires_h = max(0.5, min(1.0, expires_h))

    rationale = str(parsed.get("rationale") or "")[:400]

    # ML threshold fast-adapter: clamp Sonnet's proposal to a tight step
    # (<=0.05/cycle) and hard bounds [0.58, 0.74]. This is the LIVE buy-filter
    # gate (hot-reloaded), so guard it tightly — Sonnet tunes within a sandbox.
    try:
        thr = float(parsed.get("ml_min_up_probability"))
    except (TypeError, ValueError):
        thr = current_threshold
    thr = max(current_threshold - 0.05, min(current_threshold + 0.05, thr))
    thr = round(max(0.58, min(0.74, thr)), 3)

    return {
        "mode": mode,
        "expires_hours": expires_h,
        "rationale": rationale,
        "ml_min_up_probability": thr,
        "original_mode": str(parsed.get("mode") or "").strip().lower(),
    }


def run_mid_review(*, dry_run: bool = False) -> dict:
    data_dir = settings.runtime_data_dir
    logs = data_dir / "logs"
    state = data_dir / "state"
    state.mkdir(parents=True, exist_ok=True)
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)

    orders = _read_jsonl(logs / "orders.jsonl", n=2000)
    decisions = _read_jsonl(logs / "decisions.jsonl", n=2000)
    brief = _read_json(state / "llm_brief.json")
    stats = _compute_recent_stats(orders, window_hours=6)

    # Top blocked reasons in the last 6 hours
    cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
    block_counter: Counter = Counter()
    for d in decisions:
        ts = _parse_ts(d.get("ts"))
        if ts and (ts.tzinfo is None or ts.tzinfo is not None) and ts.replace(tzinfo=timezone.utc) >= cutoff:
            meta = d.get("meta") or {}
            for r in (meta.get("risk_reasons") or []):
                block_counter[r] += 1
    recent_blocks = dict(block_counter.most_common(5))

    # Fill context + current ML threshold for the fast-adapter. If the market
    # is favourable (regime normal) yet we're barely trading and the filter is
    # the main blocker, Sonnet should lower the threshold a notch.
    current_threshold = settings.ml_min_up_probability
    orders_window = sum(
        1 for o in orders
        if (_ts := _parse_ts(o.get("ts"))) and _ts.replace(tzinfo=timezone.utc) >= cutoff
    )
    ml_blocked_window = 0
    decisions_window = 0
    for d in decisions:
        ts = _parse_ts(d.get("ts"))
        if not ts or ts.replace(tzinfo=timezone.utc) < cutoff:
            continue
        decisions_window += 1
        if "ml buy filter blocked" in ((d.get("signal") or {}).get("reason") or ""):
            ml_blocked_window += 1
    fill_context = {
        "orders_window_6h": orders_window,
        "ml_blocked_window_6h": ml_blocked_window,
        "decisions_window_6h": decisions_window,
    }

    polza = PolzaClient(
        api_key=settings.polza_api_key,
        base_url=settings.polza_base_url,
        timeout=settings.llm_timeout_seconds,
    )
    if not polza.enabled():
        print("Polza disabled, skipping mid-review")
        return {"skipped": "polza_disabled"}

    # Read last 5 mid_review decisions to give Sonnet trend awareness +
    # compute hours-in-current-mode so it knows if we've been stuck.
    prior_modes: list = []
    hours_in_current_mode = 0.0
    try:
        log_entries = _read_jsonl(logs / LOG_FILE_NAME, n=15)
        for entry in log_entries[-5:]:
            prior_modes.append({
                "ts": entry.get("ts"),
                "mode": entry.get("mode"),
                "regime_at_decision": entry.get("regime_at_decision"),
                "rationale": (entry.get("rationale") or "")[:100],
            })
        # Walk back from latest to find the first mode change, count hours.
        if log_entries:
            latest_mode = str(log_entries[-1].get("mode", "")).lower()
            now_utc = datetime.now(timezone.utc)
            for entry in reversed(log_entries):
                if str(entry.get("mode", "")).lower() != latest_mode:
                    break
                entry_ts = _parse_ts(entry.get("ts"))
                if entry_ts is not None:
                    if entry_ts.tzinfo is None:
                        entry_ts = entry_ts.replace(tzinfo=timezone.utc)
                    hours_in_current_mode = max(
                        hours_in_current_mode,
                        (now_utc - entry_ts).total_seconds() / 3600,
                    )
    except Exception:
        pass

    messages = _build_messages(stats, brief, recent_blocks, prior_modes, hours_in_current_mode, current_threshold, fill_context)
    # mid_review picks one of 4 modes from an allowlist — a classification task,
    # not deep reasoning — so a cheaper model handles it well. The premium model
    # is reserved for auto_tuner and daily_retro where reasoning quality matters.
    mid_review_model = settings.llm_model_mid_review or "anthropic/claude-sonnet-4.6"
    resp = polza.chat(
        model=mid_review_model,
        messages=messages,
        max_tokens=400,
        temperature=0.1,
        json_mode=True,
    )
    if resp is None:
        print("LLM call failed, no proposal")
        return {"skipped": "llm_failed"}

    parsed = resp.as_json() or {}
    sanitized = _validate_proposal(parsed, stats, brief.get("market_regime") or "normal", prior_modes, current_threshold)

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=sanitized["expires_hours"])

    state_payload = {
        "ts": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "mode": sanitized["mode"],
        "rationale": sanitized["rationale"],
        "ml_min_up_probability": sanitized["ml_min_up_probability"],
        "ml_threshold_prev": round(current_threshold, 3),
        "stats_at_decision": stats,
        "regime_at_decision": brief.get("market_regime"),
        "model": resp.model,
        "tokens_total": resp.total_tokens,
    }

    new_thr = sanitized["ml_min_up_probability"]
    if not dry_run:
        (state / STATE_FILE_NAME).write_text(
            json.dumps(state_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Fast-adapt the live buy-filter threshold via the same atomic .env
        # writer auto_tuner uses; the bot's hot-reload picks it up next cycle.
        if abs(new_thr - current_threshold) >= 0.001:
            try:
                from moex_agent.auto_tuner import _write_env_atomic
                _write_env_atomic(Path(".env"), {"ML_MIN_UP_PROBABILITY": str(new_thr)})
            except Exception as exc:
                print(f"  (failed to write ML threshold to .env: {exc})")

    # Append to log regardless of dry-run
    log_path = logs / LOG_FILE_NAME
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({**state_payload, "dry_run": dry_run, "original_mode": sanitized["original_mode"]}, ensure_ascii=False) + "\n")

    print(f"Mode = {sanitized['mode']}  (expires {sanitized['expires_hours']:.1f}h)")
    if sanitized["mode"] != sanitized["original_mode"]:
        print(f"  (guardrail downgraded from '{sanitized['original_mode']}')")
    print(f"Rationale: {sanitized['rationale']}")
    if abs(new_thr - current_threshold) >= 0.001:
        print(f"ML threshold: {current_threshold:.3f} -> {new_thr:.3f}  (fast-adapt, hot-reloaded)")
    return state_payload


def read_active_mode(data_dir: Path) -> str:
    """Read currently active mid_review mode. Returns 'normal' if no active mode."""
    path = data_dir / "state" / STATE_FILE_NAME
    if not path.exists():
        return "normal"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "normal"
    expires_at = _parse_ts(payload.get("expires_at"))
    if expires_at is None:
        return "normal"
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) >= expires_at:
        return "normal"
    mode = str(payload.get("mode") or "normal").lower()
    return mode if mode in VALID_MODES else "normal"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and log but do NOT write state file.")
    args = parser.parse_args()
    run_mid_review(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
