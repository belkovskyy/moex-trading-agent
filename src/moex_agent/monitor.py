"""One-page bot health snapshot.

Usage:
    python -m moex_agent.monitor

Sections:
    A. LIVENESS       — is the bot alive, last cycle age, cycles seen
    B. PORTFOLIO      — equity / cash / open positions / unrealized PnL
    C. TODAY'S TAPE   — decisions, orders, fills, turnover, top blocked reasons
    D. CLOSED TRADES  — win rate, avg win/loss, profit factor (vs backtest)
    E. ML FILTER      — pass rate, avg P(up), regime impact
    F. LLM BRIEF      — current regime, cash_multiplier, blocked symbols
    G. TURNOVER READY — pace vs 10M requirement
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from moex_agent.config import settings


# === backtest baseline ===
BACKTEST_BASELINE = {
    "win_rate_pct": 90.83,
    "avg_win_pct": 0.93,
    "avg_loss_pct": -1.48,
    "profit_factor": 1.276,
    "trades_per_day": 7.2,
    "total_return_pct_per_year": 0.84,
}
STAGE2_DAYS = 14
STAGE2_TURNOVER_REQUIRED = 10_000_000.0


# ---------- helpers ----------
def _read_jsonl(path: Path, n: int = 10000) -> list[dict]:
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


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _fmt_age(dt: datetime | None) -> str:
    if dt is None:
        return "?"
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    s = int((now - dt).total_seconds())
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m ago"
    return f"{s // 86400}d ago"


def _color(value: float, good: float, bad: float, *, higher_is_better: bool = True) -> str:
    """Return a status marker for a metric vs thresholds."""
    if higher_is_better:
        if value >= good:
            return "OK"
        if value <= bad:
            return "BAD"
        return "WARN"
    if value <= good:
        return "OK"
    if value >= bad:
        return "BAD"
    return "WARN"


# ---------- analyzers ----------
def _liveness(cycles: list[dict]) -> dict:
    last = cycles[-1] if cycles else None
    last_ts = _parse_ts(last.get("ts") if last else None)
    age_sec = None
    if last_ts:
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        age_sec = (datetime.now(timezone.utc) - last_ts).total_seconds()
    # MOEX trading hours: 04:00–20:50 UTC weekdays. Outside of that the bot
    # is in idle mode by design — cycles are intentionally 5 min apart, so
    # "STALE" alert is a false alarm. Detect this case.
    now_utc = datetime.now(timezone.utc)
    is_weekday = now_utc.weekday() < 5
    mod = now_utc.hour * 60 + now_utc.minute
    market_open = is_weekday and 240 <= mod <= 1250
    return {
        "last_cycle_ts": last_ts,
        "age_sec": age_sec,
        "alive": age_sec is not None and age_sec < 300,
        "cycles_seen": len(cycles),
        "market_open": market_open,
    }


def _portfolio_state(cycles: list[dict]) -> dict:
    """Use the latest cycle_summary as truth (live ledger may lag)."""
    if not cycles:
        return {}
    latest = cycles[-1].get("portfolio") or {}
    first = cycles[0].get("portfolio") or {}
    starting_equity = float(first.get("equity") or 0.0)
    current_equity = float(latest.get("equity") or 0.0)
    pnl_session = current_equity - starting_equity
    pnl_session_pct = (pnl_session / starting_equity * 100) if starting_equity > 0 else 0.0
    return {
        "cash": float(latest.get("cash") or 0.0),
        "equity": current_equity,
        "exposure": float(latest.get("exposure") or 0.0),
        "exposure_pct": (float(latest.get("exposure") or 0.0) / max(current_equity, 1.0)) * 100,
        "positions": latest.get("positions") or {},
        "pnl_session": pnl_session,
        "pnl_session_pct": pnl_session_pct,
        "starting_equity": starting_equity,
    }


def _today_tape(orders: list[dict], decisions: list[dict]) -> dict:
    today = datetime.now(timezone.utc).date()

    def is_today(row):
        ts = _parse_ts(row.get("ts"))
        return ts is not None and ts.date() == today

    orders_today = [r for r in orders if is_today(r)]
    decisions_today = [r for r in decisions if is_today(r)]

    # ArenaGo returns status="submitted" with response.success=true for
    # accepted orders — those ARE filled trades, balance is debited.
    # status="below_lot_size" etc. are real rejections.
    def _is_filled(o):
        res = o.get("result") or {}
        status = res.get("status") or ""
        if status in ("filled", "ok"):
            return True
        if status == "submitted":
            response = res.get("response") or {}
            return bool(response.get("success"))
        return False

    filled = [o for o in orders_today if _is_filled(o)]
    rejected = [o for o in orders_today if (o.get("result") or {}).get("status") in ("below_lot_size", "rejected", "error")]
    pending = [o for o in orders_today if (o.get("result") or {}).get("status") == "submitted" and not _is_filled(o)]

    turnover = 0.0
    for o in filled:
        result = o.get("result") or {}
        response = result.get("response") or {}
        order_value = float(response.get("order_value") or 0.0)
        if order_value > 0:
            turnover += abs(order_value)
        else:
            price = float(response.get("price") or result.get("price") or 0.0)
            qty = float(result.get("actual_shares") or response.get("quantity") or 0.0)
            turnover += abs(price * qty)

    # decisions.jsonl stores risk reasons under decision["meta"]["risk_reasons"],
    # not at the top level. Old code read d["blocked_reasons"] which is empty.
    blocked_counter: Counter = Counter()
    for d in decisions_today:
        meta = d.get("meta") or {}
        for r in (meta.get("risk_reasons") or []):
            blocked_counter[r] += 1

    # Why no orders? Distinguish "ML vetoed buy candidates" (market regime
    # gives no high-probability setups — bot correctly idle) from "strategy
    # produced no candidate at all" vs "risk layer blocked". This stops the
    # health section from crying "filters too strict" when the bot is simply
    # right to sit out an unfavourable regime.
    import re as _re
    ml_blocked = 0
    strategy_hold = 0
    max_pup = 0.0
    for d in decisions_today:
        sig = d.get("signal") or {}
        reason = sig.get("reason") or ""
        if "ml buy filter blocked" in reason:
            ml_blocked += 1
            m = _re.search(r"P\(up\)=([0-9.]+)", reason)
            if m:
                max_pup = max(max_pup, float(m.group(1)))
        elif (sig.get("action") or "") == "hold":
            strategy_hold += 1

    return {
        "decisions": len(decisions_today),
        "orders_attempted": len(orders_today),
        "orders_filled": len(filled),
        "orders_submitted_unknown": len(pending),
        "orders_rejected": len(rejected),
        "turnover": turnover,
        "top_blocked": blocked_counter.most_common(5),
        "ml_blocked": ml_blocked,
        "strategy_hold": strategy_hold,
        "max_pup_today": max_pup,
        "risk_blocked_any": sum(blocked_counter.values()) > 0,
    }


def _avg_price_before(symbol: str, ts, cycles_snapshot: list[dict]) -> float:
    """Find the most recent avg_price for symbol in cycle_summary BEFORE ts."""
    if ts is None:
        return 0.0
    target = _parse_ts(ts) if isinstance(ts, str) else ts
    if target is None:
        return 0.0
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    best = 0.0
    for c in cycles_snapshot:
        c_ts = _parse_ts(c.get("ts"))
        if c_ts is None:
            continue
        if c_ts.tzinfo is None:
            c_ts = c_ts.replace(tzinfo=timezone.utc)
        if c_ts >= target:
            break
        pos = (c.get("portfolio") or {}).get("positions") or {}
        if symbol in pos:
            best = float(pos[symbol].get("avg_price") or 0)
    return best


def _closed_trades(orders: list[dict], cycles_snapshot: list[dict] | None = None) -> dict:
    """Match buys and sells by FIFO per symbol to compute realized PnL per trade.

    Handles two shapes of orders.jsonl rows:
    - Filled (ArenaGo accepted): result.request.{secid, direction}, result.response.{price, quantity}, result.actual_shares
    - Rejected (below_lot_size etc): result.order.{symbol, side, qty}
    Rejected are skipped — they didn't move money.
    """
    def _extract(row: dict) -> dict | None:
        result = row.get("result") or {}
        status = result.get("status") or ""
        # Accept only orders that actually moved money.
        if status in ("filled", "ok"):
            sym = (result.get("request") or {}).get("secid") or (result.get("order") or {}).get("symbol")
            side_raw = (result.get("request") or {}).get("direction") or (result.get("order") or {}).get("side") or ""
            price = float((result.get("response") or {}).get("price") or result.get("price") or 0.0)
            qty = float(result.get("actual_shares") or (result.get("response") or {}).get("quantity") or 0.0)
        elif status == "submitted":
            response = result.get("response") or {}
            if not response.get("success"):
                return None
            sym = (result.get("request") or {}).get("secid")
            side_raw = (result.get("request") or {}).get("direction") or ""
            price = float(response.get("price") or 0.0)
            qty = float(result.get("actual_shares") or response.get("quantity") or 0.0)
        else:
            return None
        if not sym or price <= 0 or qty <= 0:
            return None
        side_lower = side_raw.lower()
        if side_lower in ("b", "buy"):
            side = "buy"
        elif side_lower in ("s", "sell"):
            side = "sell"
        else:
            return None
        return {"symbol": sym, "side": side, "qty": qty, "price": price, "ts": row.get("ts")}

    extracted: list[dict] = []
    for o in orders:
        e = _extract(o)
        if e is not None:
            extracted.append(e)

    by_symbol: dict[str, list[dict]] = {}
    for e in extracted:
        by_symbol.setdefault(e["symbol"], []).append(e)

    # Signed FIFO so SHORT round-trips (SELL open -> BUY cover) are counted too,
    # not just BUY->SELL longs (see models.match_round_trips).
    from moex_agent.models import match_round_trips
    closed_trades: list[dict] = []
    for sym, rows in by_symbol.items():
        rows.sort(key=lambda r: _parse_ts(r.get("ts")) or datetime.min.replace(tzinfo=timezone.utc))
        seq = [(r["side"], r["qty"], r["price"]) for r in rows]
        # Seed a synthetic entry if the window opens with a SELL of a pre-log
        # long (avg_price snapshot), so it isn't mis-read as a short open.
        if seq and seq[0][0] == "sell":
            synth_avg = _avg_price_before(sym, rows[0].get("ts"), cycles_snapshot)
            if synth_avg > 0:
                seq.insert(0, ("buy", seq[0][1], synth_avg))
        for rt in match_round_trips(seq):
            closed_trades.append({
                "symbol": sym,
                "pnl_abs": rt["pnl_abs"],
                "pnl_pct": rt["pnl_pct"],
            })

    if not closed_trades:
        return {"count": 0}

    wins = [t for t in closed_trades if t["pnl_abs"] > 0]
    losses = [t for t in closed_trades if t["pnl_abs"] <= 0]
    sum_wins = sum(t["pnl_abs"] for t in wins)
    sum_losses = abs(sum(t["pnl_abs"] for t in losses))
    return {
        "count": len(closed_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0,
        "avg_win_pct": (sum(t["pnl_pct"] for t in wins) / len(wins)) if wins else 0.0,
        "avg_loss_pct": (sum(t["pnl_pct"] for t in losses) / len(losses)) if losses else 0.0,
        "profit_factor": (sum_wins / sum_losses) if sum_losses > 0 else float("inf"),
        "total_pnl_abs": sum(t["pnl_abs"] for t in closed_trades),
    }


def _stage2_pace(orders_today: int, turnover_today: float, cycles_age_days: float) -> dict:
    if cycles_age_days <= 0:
        return {"turnover_pace": 0.0, "trades_pace": 0.0, "extrapolated_turnover_14d": 0.0, "ratio_to_required": 0.0}
    turnover_per_day = turnover_today / max(cycles_age_days, 1e-6)
    trades_per_day = orders_today / max(cycles_age_days, 1e-6)
    extrapolated = turnover_per_day * STAGE2_DAYS
    return {
        "turnover_per_day": turnover_per_day,
        "trades_per_day": trades_per_day,
        "extrapolated_turnover_14d": extrapolated,
        "ratio_to_required": extrapolated / STAGE2_TURNOVER_REQUIRED,
    }


# ---------- render ----------
def main() -> None:
    data_dir = settings.runtime_data_dir
    logs = data_dir / "logs"
    state = data_dir / "state"

    cycles = _read_jsonl(logs / "cycle_summary.jsonl", n=2000)
    orders = _read_jsonl(logs / "orders.jsonl", n=20000)
    decisions = _read_jsonl(logs / "decisions.jsonl", n=20000)
    brief = _read_json(state / "llm_brief.json")

    live = _liveness(cycles)
    port = _portfolio_state(cycles)
    tape = _today_tape(orders, decisions)
    trades = _closed_trades(orders, cycles_snapshot=cycles)

    # Pace: cycles span from first to last
    pace_days = 0.0
    if len(cycles) >= 2:
        t0 = _parse_ts(cycles[0].get("ts"))
        t1 = _parse_ts(cycles[-1].get("ts"))
        if t0 and t1:
            pace_days = (t1 - t0).total_seconds() / 86400
    pace = _stage2_pace(tape["orders_filled"], tape["turnover"], pace_days)

    bar = "=" * 64
    print(bar)
    print(f"  BOT HEALTH SNAPSHOT  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} local")
    print(bar)

    # A. Liveness
    print("\n[A] LIVENESS")
    if not live.get("market_open", True) and not live["alive"]:
        status = "IDLE (market closed)"
    elif live["alive"]:
        status = "ALIVE"
    else:
        status = "STALE / DOWN"
    print(f"  Status         : {status}")
    print(f"  Last cycle     : {_fmt_age(live['last_cycle_ts'])}")
    print(f"  Cycles in tail : {live['cycles_seen']}")
    print(f"  Market open    : {'yes' if live.get('market_open') else 'no'}")

    # B. Portfolio
    print("\n[B] PORTFOLIO")
    if port:
        print(f"  Equity         : {port['equity']:>14,.2f} RUB")
        print(f"  Cash           : {port['cash']:>14,.2f} RUB")
        print(f"  Exposure       : {port['exposure']:>14,.2f} RUB  ({port['exposure_pct']:.1f}% of equity)")
        marker = "+" if port["pnl_session"] >= 0 else ""
        print(f"  Session PnL    : {marker}{port['pnl_session']:>13,.2f} RUB  ({port['pnl_session_pct']:+.3f}%)")
        print(f"  Open positions : {len(port['positions'])}")
        latest_prices_payload = _read_json(state / "latest_prices.json")
        latest_prices: dict[str, float] = latest_prices_payload.get("prices") or {}
        latest_prices_ts = _parse_ts(latest_prices_payload.get("ts"))
        total_unrealized = 0.0
        for sym, p in list(port["positions"].items())[:12]:
            avg = float(p.get("avg_price", 0) or 0)
            qty = float(p.get("qty", 0) or 0)
            fresh_mkt = float(latest_prices.get(sym) or 0)
            stale_mkt = float(p.get("market_price", 0) or 0)
            mkt = fresh_mkt or stale_mkt
            if avg > 0 and mkt > 0 and qty > 0:
                pnl_abs = (mkt - avg) * qty
                pnl_pct = (mkt / avg - 1) * 100
                total_unrealized += pnl_abs
                source = "live" if fresh_mkt else "stale"
                print(f"    {sym:>6} qty={qty:<5g} avg={avg:<8.2f} mkt={mkt:<8.2f} pnl={pnl_abs:+8.2f} ({pnl_pct:+.2f}%) [{source}]")
            else:
                print(f"    {sym:>6} qty={qty:<5g} avg={avg:<8.2f} mkt=?")
        if latest_prices_ts is not None:
            print(f"  Unrealized PnL : {total_unrealized:+.2f} RUB  (prices {_fmt_age(latest_prices_ts)})")
        elif port["positions"]:
            print(f"  Unrealized PnL : N/A (run bot at least one cycle to record latest_prices.json)")
    else:
        print("  (no cycle data yet)")

    # C. Today's tape
    print("\n[C] TODAY")
    print(f"  Decisions      : {tape['decisions']}")
    print(f"  Orders attempted: {tape['orders_attempted']}")
    print(f"    filled       : {tape['orders_filled']}")
    print(f"    rejected     : {tape['orders_rejected']}  (below_lot_size, etc.)")
    print(f"    submitted/?  : {tape['orders_submitted_unknown']}")
    print(f"  Turnover today : {tape['turnover']:>14,.2f} RUB")
    if tape["top_blocked"]:
        print(f"  Top blocked reasons:")
        for reason, n in tape["top_blocked"]:
            print(f"    - {reason}: {n}")

    # D. Closed trades vs backtest
    print("\n[D] CLOSED TRADES (since logs start)")
    if trades.get("count", 0) == 0:
        print("  No closed trades yet.")
    else:
        wr = trades["win_rate_pct"]
        pf = trades["profit_factor"]
        aw = trades["avg_win_pct"]
        al = trades["avg_loss_pct"]
        wr_marker = _color(wr, 80, 50)
        pf_marker = _color(pf, 1.2, 1.0)
        print(f"  Count          : {trades['count']}  (wins {trades['wins']} / losses {trades['losses']})")
        print(f"  Win rate       : {wr:.1f}%       [{wr_marker}, backtest: {BACKTEST_BASELINE['win_rate_pct']}%]")
        print(f"  Avg win        : {aw:+.2f}%      [backtest: {BACKTEST_BASELINE['avg_win_pct']:+.2f}%]")
        print(f"  Avg loss       : {al:+.2f}%      [backtest: {BACKTEST_BASELINE['avg_loss_pct']:+.2f}%]")
        print(f"  Profit factor  : {pf:.2f}        [{pf_marker}, backtest: {BACKTEST_BASELINE['profit_factor']}]")
        print(f"  Total PnL      : {trades['total_pnl_abs']:+,.2f} RUB")

    # E. LLM Brief
    print("\n[E] LLM BRIEF (DeepSeek refresh every 30 min)")
    if brief:
        print(f"  Regime         : {brief.get('market_regime')}")
        print(f"  Cash multiplier: {brief.get('cash_multiplier')}  (clamped to [0.5, 1.0] in risk.py)")
        print(f"  Blocked symbols: {brief.get('blocked_symbols')}")
        print(f"  Brief age      : {_fmt_age(_parse_ts(brief.get('ts')))}")
        rationale = (brief.get("rationale") or "")[:180]
        if rationale:
            print(f"  Rationale      : {rationale}")
    else:
        print("  (no brief cached yet)")

    # F. Turnover readiness
    print("\n[F] TURNOVER READINESS  (target: turnover >10M over 14 days)")
    if pace_days > 0.1:
        extrap = pace["extrapolated_turnover_14d"]
        ratio = pace["ratio_to_required"]
        ratio_marker = "OK" if ratio >= 1.2 else ("WARN" if ratio >= 0.9 else "BAD")
        print(f"  Days of data   : {pace_days:.2f}")
        print(f"  Trades/day pace: {pace['trades_per_day']:.1f}  (backtest expected: {BACKTEST_BASELINE['trades_per_day']:.1f})")
        print(f"  Turnover/day   : {pace['turnover_per_day']:>14,.2f} RUB")
        print(f"  14-day extrap  : {extrap:>14,.2f} RUB")
        print(f"  Vs requirement : {ratio:.2f}x  [{ratio_marker}]")
        if ratio < 0.9:
            print(f"   WARNING: turnover below requirement — penalty risk unless it picks up")
    else:
        print(f"  Not enough data yet (need >2.5h of live cycles).")

    # G. Supervisors
    print("\n[G] LLM SUPERVISORS")
    mid = _read_json(state / "mid_review.json")
    if mid:
        mode = mid.get("mode", "?")
        ts = _parse_ts(mid.get("ts"))
        exp = _parse_ts(mid.get("expires_at"))
        now_utc = datetime.now(timezone.utc)
        if exp and exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        active = exp is not None and now_utc < exp
        print(f"  Mid-review     : mode={mode}  ({'ACTIVE' if active else 'EXPIRED, falls back to normal'})")
        print(f"  Set at         : {_fmt_age(ts)}")
        rationale = (mid.get("rationale") or "")[:160]
        if rationale:
            print(f"  Rationale      : {rationale}")
    else:
        print(f"  Mid-review     : not run yet (first fire ~3h after bot start)")

    tuner_log = _read_jsonl(logs / "auto_tuner.jsonl", n=20)
    last_tune = tuner_log[-1] if tuner_log else None
    if last_tune:
        ts = _parse_ts(last_tune.get("ts"))
        changes = last_tune.get("applied_changes") or last_tune.get("changes") or []
        dry = last_tune.get("dry_run", False)
        suffix = " [DRY-RUN]" if dry else ""
        print(f"  Auto-tuner     : last run {_fmt_age(ts)}, changes={len(changes)}{suffix}")
        rationale = (last_tune.get("llm_rationale") or "")[:300]
        if rationale:
            print(f"  Rationale      : {rationale}")
        if changes:
            print(f"  Applied changes:")
            for ch in changes[:6]:
                key = ch.get("key", "?")
                f = ch.get("from", "?")
                t = ch.get("to", "?")
                why = (ch.get("why") or "")[:150]
                print(f"    - {key}: {f} → {t}")
                if why:
                    print(f"      ({why})")
    else:
        print(f"  Auto-tuner     : not run yet (first fire 24h after bot start)")

    # G2. Recent activity — last 10 orders with side/qty/price/status/PnL.
    # For each SELL we extract PnL% — first from the strategy's reason text
    # ("stop-loss exit at -1.23%"), and if that's missing we fall back to
    # the avg_price recorded in the latest cycle_summary BEFORE this SELL.
    print("\n[G2] RECENT ORDERS (last 10)")

    import re as _re_pnl

    def _avg_price_just_before(symbol: str, ts: datetime | None) -> float:
        if ts is None:
            return 0.0
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        best_avg = 0.0
        for c in cycles:
            c_ts = _parse_ts(c.get("ts"))
            if c_ts is None:
                continue
            if c_ts.tzinfo is None:
                c_ts = c_ts.replace(tzinfo=timezone.utc)
            if c_ts >= ts:
                break
            pos = (c.get("portfolio") or {}).get("positions") or {}
            if symbol in pos:
                best_avg = float(pos[symbol].get("avg_price") or 0)
        return best_avg

    if orders:
        recent = orders[-10:]
        for o in recent:
            ts = _parse_ts(o.get("ts"))
            result = o.get("result") or {}
            status = result.get("status") or "?"
            request = result.get("request") or {}
            order_field = result.get("order") or {}
            response = result.get("response") or {}
            sym = request.get("secid") or order_field.get("symbol") or "?"
            side_raw = request.get("direction") or order_field.get("side") or "?"
            side = "BUY" if str(side_raw).lower() in ("b", "buy") else "SELL"
            qty = result.get("actual_shares") or response.get("quantity") or order_field.get("qty") or 0
            price = float(response.get("price") or order_field.get("limit_price") or 0)
            value = response.get("order_value") or (price * float(qty) if price and qty else 0)
            reason = order_field.get("reason") or ""
            ok = status in ("filled", "ok") or (status == "submitted" and response.get("success"))
            mark = "OK " if ok else "REJ"
            pnl_str = ""
            if ok and side == "SELL" and price > 0 and qty:
                m = _re_pnl.search(r"(-?\d+\.\d+)%", reason)
                if m:
                    pnl_pct = float(m.group(1))
                    pnl_str = f"  PnL={pnl_pct:+.2f}%"
                else:
                    avg = _avg_price_just_before(sym, ts)
                    if avg > 0:
                        pnl_pct = (price / avg - 1) * 100
                        pnl_abs = (price - avg) * float(qty)
                        pnl_str = f"  PnL={pnl_pct:+.2f}% ({pnl_abs:+.0f} RUB)"
            print(f"  {_fmt_age(ts):>10} {mark} {side:>4} {sym:>6} qty={qty:<5} @ {price:<8} ({value:>9,.0f} RUB)  [{status}]{pnl_str}")
    else:
        print(f"  no orders yet")

    # G3. Last LLM explanation for context on the most recent trade
    print("\n[G3] LAST LLM EXPLANATION")
    explanations = _read_jsonl(logs / "llm_explanations.jsonl", n=3)
    if explanations:
        last = explanations[-1]
        ts = _parse_ts(last.get("ts"))
        sym = last.get("symbol") or "?"
        order_info = last.get("order") or {}
        side = (order_info.get("side") or "?").upper()
        qty = order_info.get("qty") or "?"
        price = order_info.get("limit_price") or "?"
        summary = (last.get("summary") or "")[:300]
        primary = last.get("primary_signal") or ""
        risks = last.get("key_risks") or []
        print(f"  {_fmt_age(ts):>10}  {side} {sym} qty={qty} @ {price}")
        if primary:
            print(f"  Signal       : {primary}")
        if summary:
            print(f"  Summary      : {summary}")
        if risks:
            print(f"  Risks        :")
            for r in risks[:3]:
                print(f"    - {str(r)[:140]}")
    else:
        print(f"  no LLM explanations yet")

    # H. Health issues (alerts only when something is off)
    print("\n[H] HEALTH ISSUES")
    issues = []
    if not live["alive"] and live.get("market_open", True):
        issues.append("bot stale — no cycle in 5+ minutes (market is OPEN)")
    if brief:
        b_age = _parse_ts(brief.get("ts"))
        if b_age:
            if b_age.tzinfo is None:
                b_age = b_age.replace(tzinfo=timezone.utc)
            age_min = (datetime.now(timezone.utc) - b_age).total_seconds() / 60
            if age_min > 90:
                issues.append(f"brief is {int(age_min)}m stale — DeepSeek refresh may be failing")
    if pace_days > 0.5 and tape["orders_filled"] == 0:
        thr = float(getattr(settings, "ml_min_up_probability", 0.70) or 0.70)
        max_pup = tape.get("max_pup_today", 0.0)
        ml_blocked = tape.get("ml_blocked", 0)
        if tape.get("risk_blocked_any"):
            issues.append(f"no fills in {pace_days*24:.1f}h — risk layer blocking (see [C] blocked reasons)")
        elif ml_blocked > 0 and max_pup < thr:
            # Strategy IS finding candidates; ML vetoes them all. This is the
            # regime, not a stuck filter — say so plainly.
            issues.append(
                f"no fills in {pace_days*24:.1f}h — NOT a bug: strategy found {ml_blocked} buy "
                f"candidates, ML vetoed all (today max P(up)={max_pup:.3f} < threshold {thr:.2f}). "
                f"Regime offers no high-probability setups; bot correctly idle."
            )
        elif ml_blocked == 0:
            issues.append(f"no fills in {pace_days*24:.1f}h — strategy produced no buy candidates (quiet/neutral market)")
        else:
            issues.append(f"no fills in {pace_days*24:.1f}h — investigate (max P(up)={max_pup:.3f} ≥ threshold {thr:.2f} but no orders)")
    if pace_days > 0.5 and tape["orders_attempted"] > 0 and tape["orders_filled"] == 0:
        rejected_pct = (tape["orders_rejected"] / max(tape["orders_attempted"], 1)) * 100
        if rejected_pct >= 50:
            issues.append(f"{rejected_pct:.0f}% of orders rejected (below_lot_size etc.) — lot size mismatch?")
    if trades.get("count", 0) >= 10:
        wr = trades["win_rate_pct"]
        if wr < 60:
            issues.append(f"closed-trade win_rate {wr:.0f}% well below backtest 90% — strategy mismatch")
        if trades["profit_factor"] < 1.0:
            issues.append(f"profit factor {trades['profit_factor']:.2f} below 1.0 — net negative")
    if pace_days > 0.5 and pace["ratio_to_required"] < 0.9:
        issues.append(f"turnover pace at {pace['ratio_to_required']:.2f}x of required — penalty risk")

    if issues:
        for i in issues:
            print(f"  ! {i}")
    else:
        print(f"  No issues detected.")

    print("\n" + bar)


if __name__ == "__main__":
    main()
