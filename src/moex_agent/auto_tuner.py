"""Daily parameter auto-tuner.

End-of-day pipeline: reads recent trading performance, asks an LLM to
suggest small adjustments to a tightly-bounded set of risk/strategy
parameters, validates the response, and applies it to `.env`.

This closes the loop between `daily_retro` (which already analyzes the day)
and actual bot behavior. Without it, retro recommendations sit unused.

Safety design — multiple guards because LLM can hallucinate:
- Only a fixed allowlist of keys can be changed.
- Each key has hard min/max bounds outside which suggestions are clipped.
- Each adjustment must be within ±35% of the previous value (no jumps).
- Maximum 4 changes per day.
- All changes are logged with rationale to `data/logs/auto_tuner.jsonl`.
- Atomic .env write (temp file + rename) so partial failures can't corrupt config.
- Dry-run mode (AUTO_TUNER_DRY_RUN=true) for testing.

Usage:
    python -m moex_agent.auto_tuner             # apply changes (default)
    python -m moex_agent.auto_tuner --dry-run   # only log what would change
    python -m moex_agent.auto_tuner --date 2026-05-20
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from moex_agent.config import settings
from moex_agent.llm import PolzaClient


# Allowed parameters and their absolute bounds. LLM cannot ever push values
# outside these — even if it suggests something extreme, we clip.
# Format: KEY -> (hard_min, hard_max, default_step_max_pct)
TUNABLE_PARAMS: dict[str, tuple[float, float, float]] = {
    "MIN_TRADE_IMBALANCE_BUY": (-0.30, 0.0, 0.35),       # currently -0.15
    "MIN_BOOK_IMBALANCE_BUY": (-0.30, 0.0, 0.35),        # currently -0.15
    "MAX_BOOK_SPREAD_PCT": (0.005, 0.025, 0.35),         # currently 0.01
    "MIN_VOLUME_RATIO": (0.10, 0.60, 0.35),              # currently 0.30
    "MIN_SUPER_VOLUME": (200.0, 5000.0, 0.50),           # currently 1000
    # Bounds must STRADDLE the live config (stop 0.007, take 0.012) so the tuner
    # can hold/fine-tune the current reward:risk asymmetry instead of being
    # forced to widen the stop on first touch (which would re-grow losses).
    "STOP_LOSS_PCT": (0.005, 0.025, 0.30),               # currently 0.011
    "TAKE_PROFIT_PCT": (0.010, 0.040, 0.30),             # currently 0.028
    # Trailing stop: how much a winner may give back from its peak before we
    # lock the gain. Lower = lock sooner (smaller avg win), higher = let it run
    # (bigger avg win, more give-back). Floor 0.006 because tighter trails churn
    # (heavy commission drag; 0 gives no profit exits at all).
    "TRAIL_STOP_PCT": (0.006, 0.020, 0.35),              # currently 0.012
    # Position concentration cap. Wider lets winners accumulate, narrower
    # diversifies.
    "MAX_POSITION_PCT": (0.10, 0.25, 0.20),              # currently 0.20
    # Total portfolio exposure cap — tightens portfolio risk in tough markets,
    # loosens it when winning. Enforced in risk.py.
    "MAX_PORTFOLIO_EXPOSURE_PCT": (0.40, 0.85, 0.15),    # currently 0.75
    # NOTE: MAX_DAILY_LOSS_PCT is intentionally excluded — the field exists
    # in RiskConfig but no rule in risk.py reads it. Adding it to the
    # allowlist would let the LLM tune a dead parameter. Wire it up in risk.py
    # first (e.g. block new BUYs when realised PnL today < -MAX_DAILY_LOSS_PCT).
    # These are "gas pedal" parameters that shift the bot's risk profile, so the
    # tuner may only nudge them in small steps per run.
    "ML_MIN_UP_PROBABILITY": (0.55, 0.80, 0.07),         # currently 0.58
    "ORDER_CASH_PCT": (0.02, 0.12, 0.15),                # currently 0.10
    "ADD_POSITION_MIN_CONFIDENCE": (0.55, 0.80, 0.20),   # currently 0.64
    "ADD_POSITION_MIN_TRADE_IMBALANCE": (0.10, 0.45, 0.30),  # currently 0.20
    # Floor is 0.0 (not negative): a negative floor would let the tuner enable
    # averaging-down into losers, a dangerous feedback loop. Never allow adds to
    # an underwater position.
    "ADD_POSITION_MIN_PNL_PCT": (0.0, 0.010, 1.0),       # currently 0.004
    "ADD_POSITION_MAX_POSITION_PCT": (0.05, 0.20, 0.30), # currently 0.10
    "SYMBOL_COOLDOWN_MINUTES": (5, 60, 0.50),            # currently 30
    "MAX_DAILY_TRADES": (30, 200, 0.50),                 # currently 80
    # Conviction sizing knobs (active only when CONVICTION_SIZING_ENABLED=true).
    # GAIN: how aggressively order size scales with ML edge above threshold.
    # MAX_MULT: hard cap on the size multiplier. The final order size is still
    # clamped by MAX_POSITION_PCT in risk.py, so no combination here can breach
    # the position ceiling — the LLM tunes aggressiveness inside a sandbox.
    "CONVICTION_GAIN": (0.1, 2.0, 0.50),                 # currently 0.5
    "CONVICTION_MAX_MULT": (1.0, 2.5, 0.20),             # currently 1.6
    # ── Short side ──────────────────────────────────────────────────────────
    # Previously absent: the tuner adjusted the long book daily while shorts
    # ran on hardcoded defaults. The backtest (2023-2026, GMKN/VTBR/MTSS) shows
    # the short side is the profitable one (PF 1.34 with its ML filter) and was
    # throttled by a 6% exposure cap, so these need to be tunable too.
    "SHORT_STOP_LOSS_PCT": (0.005, 0.025, 0.30),         # currently 0.010
    "SHORT_TAKE_PROFIT_PCT": (0.010, 0.040, 0.30),       # currently 0.018
    "SHORT_ORDER_CASH_PCT": (0.02, 0.12, 0.30),          # currently 0.08
    "SHORT_MAX_TOTAL_EXPOSURE_PCT": (0.05, 0.50, 0.25),  # currently 0.35
    "ML_MIN_DOWN_PROBABILITY": (0.55, 0.80, 0.07),       # currently 0.60
    # Trailing / time exit for shorts measured HARMFUL in backtest (shorts need
    # ~30h to work; early exits killed the edge: PF 1.08 → 1.03 / 0.68). Bounds
    # start at 0 = off, and the tuner may only creep up if data ever disagrees.
    "SHORT_TRAIL_STOP_PCT": (0.0, 0.020, 0.30),          # currently 0 (off)
    "SHORT_TIME_EXIT_HOURS": (0.0, 48.0, 0.40),          # currently 0 (off)
}

# Maximum changes allowed per single run. Hard cap to prevent LLM from
# rewriting all params at once.
MAX_CHANGES_PER_RUN = 4

# When applying integer-valued params (cooldown minutes, daily trades),
# round to int.
INTEGER_PARAMS = {"SYMBOL_COOLDOWN_MINUTES", "MAX_DAILY_TRADES"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=lambda s: date.fromisoformat(s), default=None,
                        help="Date to base tuning on (default: today UTC; falls back to yesterday if today has no data).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and log changes but do NOT modify .env.")
    parser.add_argument("--window-hours", type=int, default=24,
                        help="Look back N hours of trades for the digest (default 24).")
    parser.add_argument("--min-closed-trades", type=int, default=5,
                        help="Refuse to tune if fewer closed trades in window (default 5).")
    parser.add_argument("--cooldown-hours", type=float, default=6.0,
                        help="Minimum hours between tuner runs (default 6). Set 0 to disable.")
    parser.add_argument("--trades-since-last", type=int, default=15,
                        help="Bypass cooldown if this many closed trades since last run (default 15).")
    parser.add_argument("--force", action="store_true",
                        help="Skip cooldown / min-trades checks. Use for manual analysis.")
    parser.add_argument("--data-dir", type=Path, default=settings.runtime_data_dir)
    parser.add_argument("--env-path", type=Path, default=Path(".env"),
                        help="Path to .env file relative to project root.")
    parser.add_argument("--model", type=str, default=None,
                        help="Override LLM model. Default settings.llm_model_premium.")
    return parser.parse_args()


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace(" ", "T"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _within_day(ts: Any, target: date) -> bool:
    dt = _parse_ts(ts)
    if dt is None:
        return False
    return dt.astimezone(timezone.utc).date() == target


def _within_window(ts: Any, window_start: datetime) -> bool:
    """True if ts >= window_start. Used for rolling N-hour windows."""
    dt = _parse_ts(ts)
    if dt is None:
        return False
    return dt >= window_start


def _last_run_info(data_dir: Path) -> dict[str, Any] | None:
    """Read the latest entry from auto_tuner.jsonl, if any. Used for cooldown
    + event-driven gating."""
    path = data_dir / "logs" / "auto_tuner.jsonl"
    if not path.exists():
        return None
    last: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def _current_brief_snapshot(data_dir: Path) -> dict:
    """Latest brief with full context: regime + multiplier + blocked symbols
    + rationale + key risks. Rationale trimmed to 200 chars to bound token
    cost. Lets auto_tuner understand the market context behind the digest
    numbers — e.g. low trade count in risk_off is expected, not a bug."""
    raw = _load_json(data_dir / "state" / "llm_brief.json")
    if not raw:
        return {}
    return {
        "regime": raw.get("market_regime"),
        "cash_multiplier": raw.get("cash_multiplier"),
        "blocked_symbols": raw.get("blocked_symbols", []),
        "rationale": (raw.get("rationale") or "")[:200],
        "risks": (raw.get("risks") or [])[:3],
    }


def _recent_mid_review_modes(data_dir: Path, n: int = 3) -> list[dict]:
    """Last N mode decisions with rationale (trimmed). Tells auto_tuner
    whether the bot was throttled by the supervisor (cautious / pause_buys),
    so low trade count isn't misread as filters being too strict."""
    log_path = data_dir / "logs" / "mid_review.jsonl"
    if not log_path.exists():
        return []
    entries = _load_jsonl(log_path)
    if not entries:
        return []
    out = []
    for e in entries[-n:]:
        if e.get("dry_run"):
            continue
        out.append({
            "ts": e.get("ts"),
            "mode": e.get("mode"),
            "rationale": (e.get("rationale") or "")[:150],
        })
    return out


def _recent_tuning_history(data_dir: Path, n: int = 3) -> list[dict]:
    """Return the last N auto_tuner runs (compact) so the LLM can see what it
    has been doing and avoid flip-flopping yesterday's decisions."""
    log_path = data_dir / "logs" / "auto_tuner.jsonl"
    if not log_path.exists():
        return []
    entries = _load_jsonl(log_path)
    if not entries:
        return []
    out = []
    for e in entries[-n:]:
        if e.get("dry_run"):
            continue
        changes = e.get("applied_changes") or []
        out.append({
            "ts": e.get("ts"),
            "performance_at_run": {
                "closed_trades": (e.get("performance_summary") or {}).get("closed_trades"),
                "win_rate_pct": (e.get("performance_summary") or {}).get("win_rate_pct"),
                "realized_pnl_total": (e.get("performance_summary") or {}).get("realized_pnl_total"),
            },
            "changes": [
                {"key": c.get("key"), "from": c.get("from"), "to": c.get("to")}
                for c in changes
            ],
        })
    return out


def _trades_since(data_dir: Path, since: datetime) -> int:
    """Count successful SELL orders since the given timestamp.

    Uses orders.jsonl (works in live mode); paper_ledger.json is not maintained
    when DRY_RUN/PAPER_TRADING are false, so the old path returned 0 in live.
    """
    n = 0
    orders = _load_jsonl(data_dir / "logs" / "orders.jsonl")
    for o in orders:
        if not _within_window(o.get("ts"), since):
            continue
        result = o.get("result") or {}
        status = result.get("status") or ""
        ok = status in ("filled", "ok") or (
            status == "submitted" and bool((result.get("response") or {}).get("success"))
        )
        if not ok:
            continue
        direction = str((result.get("request") or {}).get("direction") or "").upper()
        side = (result.get("order") or {}).get("side", "")
        if direction == "S" or str(side).lower() == "sell":
            n += 1
    return n


def _should_run(
    args: argparse.Namespace,
    data_dir: Path,
    now: datetime,
) -> tuple[bool, str]:
    """Hybrid event+time gating. Returns (should_run, reason)."""
    if args.force:
        return True, "force flag"
    last = _last_run_info(data_dir)
    if last is None:
        return True, "first ever run"
    last_ts = _parse_ts(last.get("ts"))
    if last_ts is None:
        return True, "previous run timestamp unreadable"
    hours_since = (now - last_ts).total_seconds() / 3600.0
    trades_since_last = _trades_since(data_dir, last_ts)
    cooldown = float(args.cooldown_hours)
    bypass_count = int(args.trades_since_last)
    # Event override: enough trades since last run → run even if cold
    if trades_since_last >= bypass_count:
        return True, f"event override: {trades_since_last} closed trades since last run"
    if hours_since < cooldown:
        return False, (
            f"cooldown: only {hours_since:.1f}h since last run "
            f"({trades_since_last} closed trades, need >= {bypass_count} to override)"
        )
    return True, f"time gate passed: {hours_since:.1f}h since last run"


def _read_env(env_path: Path) -> dict[str, str]:
    """Parse .env into dict of key→value strings. Preserves order & comments
    when writing back via _write_env."""
    if not env_path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def _write_env_atomic(env_path: Path, updates: dict[str, str]) -> None:
    """Replace KEY=value lines for each updated key, preserving comments and
    ordering. New keys are appended at end. Writes to .env.tmp then renames."""
    src = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    seen: set[str] = set()
    out_lines: list[str] = []
    for raw in src:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out_lines.append(raw)
            continue
        k = stripped.split("=", 1)[0].strip()
        if k in updates:
            out_lines.append(f"{k}={updates[k]}")
            seen.add(k)
        else:
            out_lines.append(raw)
    # Append any keys not present in original .env
    extras = [k for k in updates if k not in seen]
    if extras:
        out_lines.append("")
        out_lines.append("# Appended by auto_tuner")
        for k in extras:
            out_lines.append(f"{k}={updates[k]}")
    tmp = env_path.with_suffix(env_path.suffix + ".tmp")
    tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    # On Windows, replace is atomic across the same volume.
    tmp.replace(env_path)


def _build_digest(data_dir: Path, *, window_start: datetime, window_label: str) -> dict[str, Any]:
    """Compact summary of trading performance in [window_start, now) + current params.

    Uses a rolling time window instead of a calendar date so the same code
    works for daily, 6-hourly, or event-triggered runs.
    """
    # Use orders.jsonl as source of truth (same as monitor.py). paper_ledger
    # accumulates trades across all bot lifetimes which gives stale stats.
    paper = _load_json(data_dir / "state" / "paper_ledger.json")
    orders_path = data_dir / "logs" / "orders.jsonl"
    cycles_path = data_dir / "logs" / "cycle_summary.jsonl"
    orders_all = _load_jsonl(orders_path)
    cycles_all = _load_jsonl(cycles_path)

    def _is_filled(o: dict) -> bool:
        res = o.get("result") or {}
        status = res.get("status") or ""
        if status in ("filled", "ok"):
            return True
        if status == "submitted":
            return bool((res.get("response") or {}).get("success"))
        return False

    def _extract(o: dict) -> dict | None:
        if not _is_filled(o):
            return None
        result = o.get("result") or {}
        response = result.get("response") or {}
        sym = (result.get("request") or {}).get("secid") or (result.get("order") or {}).get("symbol")
        side_raw = (result.get("request") or {}).get("direction") or (result.get("order") or {}).get("side") or ""
        side = "buy" if str(side_raw).lower() in ("b", "buy") else ("sell" if str(side_raw).lower() in ("s", "sell") else "?")
        price = float(response.get("price") or result.get("price") or 0)
        qty = float(result.get("actual_shares") or response.get("quantity") or (result.get("order") or {}).get("qty") or 0)
        return {"ts": o.get("ts"), "symbol": sym, "side": side, "price": price, "qty": qty}

    # FIFO match buys+sells per symbol, restricted to the window.
    day_extracted = []
    for o in orders_all:
        e = _extract(o)
        if e is None or not e["symbol"] or e["price"] <= 0 or e["qty"] <= 0:
            continue
        if not _within_window(e["ts"], window_start):
            continue
        day_extracted.append(e)

    # avg_price lookup from cycle_summary for synthetic matching of SELLs whose
    # BUYs are outside the window (so PnL still gets computed).
    def _avg_before(sym: str, ts_str):
        if not ts_str:
            return 0.0
        target = _parse_ts(ts_str)
        if target is None:
            return 0.0
        best = 0.0
        for c in cycles_all:
            cts = _parse_ts(c.get("ts"))
            if cts is None or cts >= target:
                if cts is not None and cts >= target:
                    break
                continue
            pos = (c.get("portfolio") or {}).get("positions") or {}
            if sym in pos:
                best = float(pos[sym].get("avg_price") or 0)
        return best

    # Signed FIFO so SHORT round-trips (SELL open -> BUY cover) count too, not
    # just BUY->SELL longs (see models.match_round_trips).
    from moex_agent.models import match_round_trips
    pnls: list[float] = []
    pnls_by_side: dict[str, list[float]] = {}
    by_symbol: dict[str, list[dict]] = {}
    for e in day_extracted:
        by_symbol.setdefault(e["symbol"], []).append(e)
    for sym, rows in by_symbol.items():
        rows.sort(key=lambda r: _parse_ts(r.get("ts")) or datetime.min.replace(tzinfo=timezone.utc))
        seq = [(r["side"], r["qty"], r["price"]) for r in rows]
        # If the window opens with a SELL of a pre-existing long (no prior buy),
        # seed the entry from the avg_price snapshot so it isn't mis-read as a
        # short open.
        if seq and seq[0][0] == "sell":
            avg = _avg_before(sym, rows[0]["ts"])
            if avg > 0:
                seq.insert(0, ("buy", seq[0][1], avg))
        for rt in match_round_trips(seq):
            pnls.append(rt["pnl_abs"])
            # Side of the round-trip: a short opens with a SELL. Kept separate
            # so the tuner can see WHICH book is making money — a healthy short
            # side can otherwise hide behind a bleeding long one (and vice
            # versa), which is exactly how the short knobs went untouched.
            side = "short" if str(rt.get("side") or rt.get("direction") or "").lower() == "short" else "long"
            pnls_by_side.setdefault(side, []).append(rt["pnl_abs"])
    closed = [{"pnl": p} for p in pnls]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    def _side_stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"trades": 0}
        side_wins = [v for v in values if v > 0]
        side_losses = [v for v in values if v < 0]
        avg_win = sum(side_wins) / len(side_wins) if side_wins else 0.0
        avg_loss = sum(side_losses) / len(side_losses) if side_losses else 0.0
        return {
            "trades": len(values),
            "pnl": round(sum(values), 2),
            "win_rate_pct": round(len(side_wins) / len(values) * 100, 1),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "loss_win_ratio": round(abs(avg_loss) / avg_win, 2) if avg_win > 0 else None,
        }

    by_side = {side: _side_stats(vals) for side, vals in pnls_by_side.items()}
    # For backward compat
    day_trades = day_extracted

    decisions = _load_jsonl(data_dir / "logs" / "decisions.jsonl")
    day_decisions = [d for d in decisions if _within_window(d.get("ts"), window_start)]
    blocked: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    for d in day_decisions:
        sig = d.get("signal") or {}
        actions[str(sig.get("action", "?"))] += 1
        if not d.get("approved"):
            for r in (d.get("meta", {}).get("risk_reasons") or []):
                blocked[str(r)] += 1

    summaries = _load_jsonl(data_dir / "logs" / "cycle_summary.jsonl")
    # Use the same window_start filter as decisions/trades for consistency.
    day_summaries = [s for s in summaries if _within_window(s.get("ts"), window_start)]
    last_portfolio = day_summaries[-1].get("portfolio") if day_summaries else None
    if last_portfolio is None and isinstance(paper, dict):
        last_portfolio = paper.get("portfolio")

    # ML filter stats — how often it blocked
    ml_blocked = 0
    ml_passed = 0
    for d in day_decisions:
        reason = str((d.get("signal") or {}).get("reason", "") or "")
        if "ml buy filter blocked" in reason:
            ml_blocked += 1
        elif "ml P(up)=" in reason:
            ml_passed += 1

    # Current parameter values from .env (or defaults from config)
    current_params: dict[str, float] = {}
    env_dict = _read_env(Path(".env"))
    defaults = {
        "MIN_TRADE_IMBALANCE_BUY": settings.min_trade_imbalance_buy,
        "MIN_BOOK_IMBALANCE_BUY": settings.min_book_imbalance_buy,
        "MAX_BOOK_SPREAD_PCT": settings.max_book_spread_pct,
        "MIN_VOLUME_RATIO": settings.min_volume_ratio,
        "MIN_SUPER_VOLUME": settings.min_super_volume,
        "STOP_LOSS_PCT": settings.stop_loss_pct,
        "TAKE_PROFIT_PCT": settings.take_profit_pct,
        "TRAIL_STOP_PCT": settings.trail_stop_pct,
        "ML_MIN_UP_PROBABILITY": settings.ml_min_up_probability,
        "ORDER_CASH_PCT": settings.order_cash_pct,
        "ADD_POSITION_MIN_CONFIDENCE": settings.add_position_min_confidence,
        "ADD_POSITION_MIN_TRADE_IMBALANCE": settings.add_position_min_trade_imbalance,
        "ADD_POSITION_MIN_PNL_PCT": settings.add_position_min_pnl_pct,
        "ADD_POSITION_MAX_POSITION_PCT": settings.add_position_max_position_pct,
        "SYMBOL_COOLDOWN_MINUTES": float(settings.symbol_cooldown_minutes),
        "MAX_DAILY_TRADES": float(settings.max_daily_trades),
        "CONVICTION_GAIN": settings.conviction_gain,
        "CONVICTION_MAX_MULT": settings.conviction_max_mult,
        "SHORT_STOP_LOSS_PCT": settings.short_stop_loss_pct,
        "SHORT_TAKE_PROFIT_PCT": settings.short_take_profit_pct,
        "SHORT_ORDER_CASH_PCT": settings.short_order_cash_pct,
        "SHORT_MAX_TOTAL_EXPOSURE_PCT": settings.short_max_total_exposure_pct,
        "SHORT_TRAIL_STOP_PCT": settings.short_trail_stop_pct,
        "SHORT_TIME_EXIT_HOURS": settings.short_time_exit_hours,
        "ML_MIN_DOWN_PROBABILITY": settings.ml_min_down_probability,
    }
    for key, default in defaults.items():
        raw = env_dict.get(key)
        try:
            current_params[key] = float(raw) if raw is not None else float(default)
        except (TypeError, ValueError):
            current_params[key] = float(default)

    return {
        "window": window_label,
        "performance": {
            "decisions": len(day_decisions),
            "orders": len(day_trades),
            "closed_trades": len(closed),
            "realized_pnl_total": round(sum(pnls), 2),
            "win_rate_pct": round(len(wins) / len(pnls) * 100, 1) if pnls else None,
            "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
            "by_side": by_side,
            "ml_filter_blocked": ml_blocked,
            "ml_filter_passed": ml_passed,
            "actions": dict(actions),
            "top_blocked_reasons": dict(blocked.most_common(8)),
            "turnover_rub_window": round(sum(e["price"] * e["qty"] for e in day_extracted), 0),
        },
        # The venue requires >10M RUB turnover over 14 days. If the window is
        # ~24h, turnover_rub_window * 14 approximates the 14-day pace vs target.
        "stage2_turnover_target_14d": 10_000_000,
        "latest_portfolio": last_portfolio,
        "current_params": current_params,
        "tunable_bounds": {
            k: {"min": v[0], "max": v[1], "max_step_pct": v[2]}
            for k, v in TUNABLE_PARAMS.items()
        },
        # Last 3 tuning rounds — gives LLM trend awareness so it doesn't undo
        # yesterday's change without good reason (flip-flop prevention).
        "prior_tuning_history": _recent_tuning_history(data_dir, n=3),
        # Context from other LLM layers: what was the market situation during
        # this window and what session modes did the supervisor choose? Helps
        # auto_tuner attribute low performance to causes vs noise.
        "current_brief": _current_brief_snapshot(data_dir),
        "recent_mid_review_modes": _recent_mid_review_modes(data_dir, n=3),
    }


def _build_messages(digest: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are a quantitative risk officer for an autonomous MOEX trading bot. "
        "Given yesterday's performance and the current parameter values, "
        "propose SMALL, JUSTIFIED adjustments to a fixed allowlist of "
        "parameters. Your output is parsed by a safety validator that will "
        "clip out-of-range values, so stay within bounds.\n\n"
        "Output STRICT JSON. Schema:\n"
        "{\n"
        '  "rationale": string (2-4 sentences in Russian, references concrete numbers),\n'
        '  "changes": [\n'
        '    {"key": string, "from": float, "to": float, "why": string (1 sentence Russian)}\n'
        "  ],\n"
        '  "skip_reason": string | null  // set non-null to abort with no changes\n'
        "}\n\n"
        "Rules:\n"
        "- At most 4 changes per run.\n"
        "- Only keys in `tunable_bounds` are allowed; others ignored.\n"
        "- `to` must be within `tunable_bounds[key].min` ... `tunable_bounds[key].max`.\n"
        "- |to - from| / |from| must be <= `max_step_pct` (gradual changes only).\n"
        "- Prefer NO changes if performance is stable.\n"
        "- If yesterday had <5 closed trades or no clear signal, set skip_reason "
        "  and return empty changes.\n"
        "- Connect each change to a specific data point. Bad: 'increase stop_loss'. "
        "  Good: 'win_rate=43%, average_loss=-47, stop is too tight'.\n"
        "- The bot trades MOEX equities, 10-min candles, oversold + MACD setups. "
        "  Stop/take values are fractions (0.012 = 1.2%). PnL is in RUB.\n"
        "- `prior_tuning_history` shows your last 3 decisions. DO NOT reverse "
        "  a recent change unless data clearly shows it didn't work. If you "
        "  raised threshold yesterday and win_rate is still down, the issue "
        "  is probably elsewhere — don't undo, look further.\n"
        "- `current_brief` and `recent_mid_review_modes` show market mood and "
        "  if the bot was throttled. If recent modes were cautious/pause_buys, "
        "  low trade count is EXPECTED — don't loosen filters in response.\n\n"
        "WHICH PARAM FIXES WHICH BLOCKER (top_blocked_reasons -> knob):\n"
        "- 'book spread above threshold'        -> MAX_BOOK_SPREAD_PCT (raise to allow wider books)\n"
        "- 'trade imbalance blocks buy'         -> MIN_TRADE_IMBALANCE_BUY (lower/more negative)\n"
        "- 'negative book imbalance blocks buy' -> MIN_BOOK_IMBALANCE_BUY (lower)\n"
        "- 'volume ratio below threshold'       -> MIN_VOLUME_RATIO (lower)\n"
        "- 'super volume below threshold'       -> MIN_SUPER_VOLUME (lower)\n"
        "- 'add blocked in weak position'       -> ADD_POSITION_MIN_PNL_PCT (lower)\n"
        "- 'add requires stronger ...'          -> ADD_POSITION_MIN_CONFIDENCE / _MIN_TRADE_IMBALANCE\n"
        "- 'symbol cooldown active'             -> SYMBOL_COOLDOWN_MINUTES (lower)\n"
        "- 'daily trade limit reached'          -> MAX_DAILY_TRADES (raise)\n\n"
        "SCENARIO PLAYBOOK (act on these, don't just sit):\n"
        "1) TURNOVER: the venue requires >10M RUB / 14 days. If turnover_rub_window*14 "
        "is below stage2_turnover_target_14d AND the dominant blocker is a single "
        "filter (e.g. book spread) AND win_rate is acceptable (>55%), LOOSEN that "
        "filter a notch toward more turnover — missing the turnover target is "
        "penalized, worse than slightly worse fills.\n"
        "2) THIN/EVENING LIQUIDITY: a high 'book spread above threshold' count is "
        "normal in the low-liquidity evening session. A small MAX_BOOK_SPREAD_PCT "
        "raise captures evening turnover; the trade-off is wider fills — acceptable "
        "when turnover is short, NOT when win_rate is already poor.\n"
        "3) SMALL avg_win vs avg_loss: this usually means WINNERS ARE CUT TOO EARLY, "
        "not that losses are too big. The fix is to LET WINNERS RUN (raise "
        "TRAIL_STOP_PCT so trades give back more before locking), NOT to tighten "
        "STOP_LOSS_PCT — tightening cuts winners further and makes it worse.\n"
        "3b) RUB SIZE ASYMMETRY: if avg_win and avg_loss are similar in PERCENT but "
        "avg_loss is much bigger in RUBLES than avg_win (e.g. -950 vs +110), the "
        "problem is that LOSING positions are bigger than WINNING ones — usually "
        "pyramiding into losers. Exits are already full-position, so fix the SIZE: "
        "lower MAX_POSITION_PCT and/or tighten ADD_POSITION_* (fewer/smaller adds) "
        "so a single loser can't dwarf many wins. Per-trade %-expectancy can be "
        "POSITIVE while RUB PnL is negative purely from this size gap — watch for it.\n"
        "4) Balance: a justified loosening when a clear blocker throttles turnover "
        "is GOOD. 'Prefer no change' applies to NOISE, not to a standing structural "
        "blocker with a clear cause."
    )
    user = json.dumps(digest, ensure_ascii=False, default=str, indent=2)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _clip_change(key: str, from_val: float, to_val: float) -> tuple[float, str | None]:
    """Apply hard bounds + step-size clipping. Returns (clipped_value, note)."""
    if key not in TUNABLE_PARAMS:
        return to_val, "key not in allowlist"
    lo, hi, step_pct = TUNABLE_PARAMS[key]
    note: str | None = None
    if to_val < lo:
        to_val = lo
        note = f"clipped to min {lo}"
    if to_val > hi:
        to_val = hi
        note = f"clipped to max {hi}"
    # Step-size limit relative to current value. For values near zero, fall
    # back to absolute step = 0.5 * range.
    abs_from = abs(from_val) if abs(from_val) > 1e-6 else (hi - lo) * 0.5
    max_step = abs_from * step_pct
    if abs(to_val - from_val) > max_step:
        # Clip in the direction of the suggestion.
        direction = 1 if to_val > from_val else -1
        to_val = from_val + direction * max_step
        # Re-bound after step clip.
        to_val = max(lo, min(hi, to_val))
        note = (note + "; " if note else "") + f"step clipped to max ±{step_pct * 100:.0f}%"
    if key in INTEGER_PARAMS:
        to_val = float(int(round(to_val)))
    return to_val, note


def _format_value(key: str, value: float) -> str:
    if key in INTEGER_PARAMS:
        return str(int(round(value)))
    # Compact float — strip trailing zeros, but keep precision for small values.
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def run_tuner(args: argparse.Namespace) -> None:
    now = datetime.now(timezone.utc)
    window_hours = max(1, int(args.window_hours))
    window_start = now - timedelta(hours=window_hours)
    window_label = f"last {window_hours}h: {window_start.isoformat()} → {now.isoformat()}"
    print(f"Auto-tuner window: {window_label}. Dry-run: {args.dry_run}")

    if not settings.polza_api_key:
        raise SystemExit("POLZA_API_KEY empty — cannot reach LLM.")

    # Hybrid time + event-driven gate: don't tune too often, but skip the
    # cooldown if enough closed trades have accumulated.
    should_run, gate_reason = _should_run(args, args.data_dir, now)
    print(f"  gate: {gate_reason}")
    if not should_run:
        print("  → skip this run. Use --force to override.")
        return

    digest = _build_digest(args.data_dir, window_start=window_start, window_label=window_label)
    perf = digest["performance"]
    print(
        f"  closed={perf['closed_trades']}  pnl={perf['realized_pnl_total']}  "
        f"win_rate={perf['win_rate_pct']}  ml_blocked={perf['ml_filter_blocked']}"
    )

    # Refuse if too few closed trades — statistically meaningless tuning.
    if perf["closed_trades"] < int(args.min_closed_trades) and not args.force:
        print(
            f"  → skip: only {perf['closed_trades']} closed trades in window "
            f"(need >= {args.min_closed_trades}). Use --force to override."
        )
        _log_run(args.data_dir, window_label, digest, {"skip_reason": "insufficient_trades"},
                 applied=[], dry_run=args.dry_run)
        return

    # Generous timeout: this runs offline once a day, no rush. The LLM
    # typically responds in 5-15s but we leave headroom for retries.
    client = PolzaClient(
        api_key=settings.polza_api_key,
        base_url=settings.polza_base_url,
        timeout=90.0,
    )
    model = args.model or settings.llm_model_premium
    print(f"  model: {model}")

    resp = client.chat(
        model=model,
        messages=_build_messages(digest),
        max_tokens=1500,
        temperature=0.1,
        json_mode=True,
    )
    if resp is None:
        print("  LLM call failed; no changes applied.")
        return

    parsed = resp.as_json()
    if not parsed or not isinstance(parsed, dict):
        print("  LLM returned non-JSON or empty; no changes applied.")
        return

    skip_reason = parsed.get("skip_reason")
    if skip_reason:
        print(f"  LLM declined to tune: {skip_reason}")
        _log_run(args.data_dir, window_label, digest, parsed, applied=[], dry_run=args.dry_run)
        return

    raw_changes = parsed.get("changes") or []
    if not isinstance(raw_changes, list):
        print("  LLM `changes` is not a list — aborting.")
        return

    # Validate, clip, and prepare for .env update.
    validated: list[dict[str, Any]] = []
    env_dict = _read_env(args.env_path)
    for ch in raw_changes[:MAX_CHANGES_PER_RUN]:
        if not isinstance(ch, dict):
            continue
        key = str(ch.get("key", "")).strip()
        if key not in TUNABLE_PARAMS:
            print(f"  skipped {key}: not in allowlist")
            continue
        try:
            from_val = float(ch.get("from"))
            to_val = float(ch.get("to"))
        except (TypeError, ValueError):
            print(f"  skipped {key}: from/to not numeric")
            continue
        if from_val == to_val:
            continue
        # Use the actual current .env value if available, not the LLM-claimed `from`
        env_current = env_dict.get(key)
        if env_current is not None:
            try:
                actual_from = float(env_current)
                if abs(actual_from - from_val) > 1e-6:
                    print(f"  note {key}: LLM claimed from={from_val} but .env has {actual_from}; using .env value")
                from_val = actual_from
            except ValueError:
                pass
        clipped, note = _clip_change(key, from_val, to_val)
        if clipped == from_val:
            print(f"  skipped {key}: clipped value equals current")
            continue
        validated.append({
            "key": key,
            "from": from_val,
            "to": clipped,
            "llm_suggested_to": to_val,
            "clip_note": note,
            "why": str(ch.get("why", "")),
        })

    print(f"\n  rationale: {parsed.get('rationale', '(none)')}")
    if not validated:
        print("  No valid changes after validation. Nothing to apply.")
        _log_run(args.data_dir, window_label, digest, parsed, applied=[], dry_run=args.dry_run)
        return

    print(f"\n  {len(validated)} change(s) to apply:")
    updates: dict[str, str] = {}
    for c in validated:
        s = f"    {c['key']}: {c['from']:g} → {c['to']:g}"
        if c["clip_note"]:
            s += f"  [{c['clip_note']}]"
        print(s)
        print(f"      because: {c['why']}")
        updates[c["key"]] = _format_value(c["key"], c["to"])

    if args.dry_run:
        print("\n  (dry-run) .env NOT modified.")
    else:
        _write_env_atomic(args.env_path, updates)
        print(f"\n  Applied to {args.env_path}. Restart the bot to pick up changes.")

    _log_run(args.data_dir, window_label, digest, parsed, applied=validated, dry_run=args.dry_run)


def _log_run(
    data_dir: Path,
    window_label: str,
    digest: dict[str, Any],
    llm_response: dict[str, Any],
    *,
    applied: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    log_path = data_dir / "logs" / "auto_tuner.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "window": window_label,
        "dry_run": dry_run,
        "performance_summary": digest["performance"],
        "llm_rationale": llm_response.get("rationale"),
        "llm_skip_reason": llm_response.get("skip_reason"),
        "applied_changes": applied,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    args = _parse_args()
    run_tuner(args)


if __name__ == "__main__":
    main()
