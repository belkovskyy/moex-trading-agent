from __future__ import annotations

import argparse
import logging
import os
import time
from collections import Counter
from dataclasses import asdict
from datetime import timedelta

from moex_agent.arena_client import ArenaClient
from moex_agent.config import settings
from moex_agent.indicators import build_features
from moex_agent.llm import BriefService, ExplainerService, NewsService, PolzaClient, compact_headlines, default_brief
from moex_agent.logging_config import configure_logging
from moex_agent.macro_data import LiveMacroCache, candles_to_pairs
from moex_agent.market_data import (
    AlgopackMarketDataClient,
    IssMarketDataClient,
    MarketDataClient,
    SampleMarketDataClient,
)
from moex_agent.memory import DecisionMemory, SimilarSetupsIndex
from moex_agent.ml_data import MLDataCollector
from moex_agent.ml_filter import MLBuyFilter
from moex_agent.models import Action, Decision, OrderIntent, OrderSide, Signal, utc_now
from moex_agent.paper import PaperLedger
from moex_agent.portfolio import portfolio_snapshot
from moex_agent.regime import detect_regime
from moex_agent.risk import RiskConfig, RiskManager
from moex_agent.storage import JsonStore
from moex_agent.strategy import (
    WATCHLIST,
    generate_cover_signal,
    generate_exit_signal,
    generate_short_signal,
    generate_signal,
)

logger = logging.getLogger(__name__)


_RECENT_STATS_CACHE: dict[str, dict] = {"by_symbol": {}, "computed_at": 0.0}
_LIVE_TRACK_CACHE: dict[str, dict] = {"last_ts": {}, "today_count": 0, "computed_at": 0.0}


def _live_order_tracking(now_dt) -> tuple[dict, int]:
    """Fallback for symbol cooldown + daily-trade-count when paper_ledger is
    None (live mode, PAPER_TRADING=false). Reads orders.jsonl (cached 30s) and
    returns ({symbol: last_order_dt}, today_order_count). Without this, in live
    mode last_trade_at=None and daily_trade_count=0 → the cooldown and daily
    limit are DEAD and the bot re-trades the same name every cycle (churn).
    """
    import json
    import time as _t
    from datetime import timezone, date as _date
    from datetime import datetime as _dt2
    from moex_agent.config import settings as _s

    if (_t.time() - _LIVE_TRACK_CACHE["computed_at"]) < 30.0:
        return _LIVE_TRACK_CACHE["last_ts"], _LIVE_TRACK_CACHE["today_count"]
    path = _s.runtime_data_dir / "logs" / "orders.jsonl"
    last_ts: dict = {}
    today_count = 0
    today = (now_dt.date() if now_dt else _date.today())
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                res = row.get("result") or {}
                # Count ONLY real executed live orders. Skip paper/dry-run,
                # below-lot, kill-switch-blocked, and rejected rows — otherwise
                # cooldown/daily-count get inflated by non-trades (and by any
                # stray paper entries), producing false cooldowns that block
                # real trades and hurt turnover.
                status = str(res.get("status") or "")
                resp = res.get("response") or {}
                executed = status in ("filled", "ok") or (status == "submitted" and resp.get("success"))
                if not executed:
                    continue
                req = res.get("request") or {}
                sym = req.get("secid") or (res.get("order") or {}).get("symbol")
                ts_raw = row.get("ts")
                if not sym or not ts_raw:
                    continue
                try:
                    ts = _dt2.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                last_ts[sym] = ts if (sym not in last_ts or ts > last_ts[sym]) else last_ts[sym]
                if ts.date() == today:
                    today_count += 1
    _LIVE_TRACK_CACHE["last_ts"] = last_ts
    _LIVE_TRACK_CACHE["today_count"] = today_count
    _LIVE_TRACK_CACHE["computed_at"] = _t.time()
    return last_ts, today_count


def _recent_symbol_stats(symbol: str, now_dt) -> tuple[int, float | None]:
    """Count closed trades on this symbol in the last 60 minutes and average
    PnL%. Reads from data/logs/orders.jsonl with FIFO matching against the
    avg_price snapshot recorded in cycle_summary just before each SELL.
    Returns (n_closed_trades, avg_pnl_pct or None) — used by risk.py for
    anti-churn detection (block new BUYs after 4+ low-PnL trades on one symbol).

    Cache: computes all symbols at once on the first call per cycle, then
    reuses for the remaining 19. Reset every 60s so we don't go stale within
    a single cycle but don't re-scan orders.jsonl 20 times.
    """
    import json
    import time as _t
    from datetime import timedelta, timezone
    from datetime import datetime as _dt2

    from moex_agent.config import settings as _s

    # Cache-hit short circuit — return precomputed result without re-scanning files.
    if (_t.time() - _RECENT_STATS_CACHE["computed_at"]) < 60.0:
        cached = _RECENT_STATS_CACHE["by_symbol"].get(symbol)
        if cached is not None:
            return cached

    logs = _s.runtime_data_dir / "logs"
    orders_path = logs / "orders.jsonl"
    cycles_path = logs / "cycle_summary.jsonl"
    if not orders_path.exists():
        return 0, None
    cutoff = (now_dt if now_dt is not None else _dt2.now(tz=timezone.utc))
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    cutoff -= timedelta(minutes=60)
    # Pre-load cycle_summary tail for avg_price lookups.
    cycles: list[dict] = []
    if cycles_path.exists():
        with cycles_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    cycles.append(json.loads(line))
                except Exception:
                    continue
        cycles = cycles[-500:]

    def _parse_ts(value):
        if not value:
            return None
        try:
            dt = _dt2.fromisoformat(str(value).replace("Z", "+00:00").replace(" ", "T"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _avg_before(sym: str, when):
        if when is None:
            return 0.0
        best = 0.0
        for c in cycles:
            c_ts = _parse_ts(c.get("ts"))
            if c_ts is None or c_ts >= when:
                if c_ts is not None and c_ts >= when:
                    break
                continue
            pos = (c.get("portfolio") or {}).get("positions") or {}
            if sym in pos:
                best = float(pos[sym].get("avg_price") or 0.0)
        return best

    # Build stats for ALL symbols in one pass — cache for 60s so subsequent
    # symbols in the same cycle return instantly without rereading the file.
    by_sym_pnls: dict[str, list[float]] = {}
    with orders_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            ts = _parse_ts(row.get("ts"))
            if ts is None or ts < cutoff:
                continue
            result = row.get("result") or {}
            status = result.get("status") or ""
            response = result.get("response") or {}
            ok = status in ("filled", "ok") or (status == "submitted" and response.get("success"))
            if not ok:
                continue
            sym_row = (result.get("request") or {}).get("secid") or (result.get("order") or {}).get("symbol")
            if not sym_row:
                continue
            direction = (result.get("request") or {}).get("direction") or ""
            side = "sell" if direction.upper() == "S" else "buy"
            if side != "sell":
                continue
            price = float(response.get("price") or result.get("price") or 0.0)
            avg = _avg_before(sym_row, ts)
            if price > 0 and avg > 0:
                by_sym_pnls.setdefault(sym_row, []).append(price / avg - 1.0)

    cache_by_symbol: dict[str, tuple[int, float | None]] = {}
    for sym_row, pnls in by_sym_pnls.items():
        if pnls:
            cache_by_symbol[sym_row] = (len(pnls), sum(pnls) / len(pnls))
        else:
            cache_by_symbol[sym_row] = (0, None)
    _RECENT_STATS_CACHE["by_symbol"] = cache_by_symbol
    _RECENT_STATS_CACHE["computed_at"] = _t.time()

    return cache_by_symbol.get(symbol, (0, None))


def _recent_account_stats() -> tuple[int, float | None]:
    """Aggregate the per-symbol recent-trade cache (populated each cycle by
    _recent_symbol_stats) into an account-wide (n_closed_trades, avg_pnl_pct)
    over the last 60 min. Used by the scheduler for event-driven supervisor
    triggers — no extra file reads, reuses the cache already built this cycle.
    Returns (0, None) when there are no recent closed trades.
    """
    by_sym = (_RECENT_STATS_CACHE.get("by_symbol") or {})
    total_n = 0
    weighted = 0.0
    for value in by_sym.values():
        n, avg = value
        if n and avg is not None:
            total_n += n
            weighted += n * avg
    if total_n == 0:
        return 0, None
    return total_n, weighted / total_n


def run_once(
    client: ArenaClient,
    market_data: MarketDataClient,
    risk: RiskManager,
    store: JsonStore,
    memory: DecisionMemory,
    paper_ledger: PaperLedger | None = None,
    ml_data: MLDataCollector | None = None,
    ml_filter: MLBuyFilter | None = None,
    short_ml_filter: MLBuyFilter | None = None,
    brief_service: BriefService | None = None,
    explainer: ExplainerService | None = None,
    news_service: NewsService | None = None,
    memory_index: SimilarSetupsIndex | None = None,
    order_quota: dict | None = None,
    macro_cache: LiveMacroCache | None = None,
) -> None:
    real_portfolio = client.get_portfolio()
    portfolio = paper_ledger.portfolio(real_portfolio) if paper_ledger is not None else real_portfolio
    store.write_state("portfolio", portfolio_snapshot(portfolio))
    # Equity sanity: warn on an abnormal jump between cycles, which can signal a
    # broker-side glitch (e.g. a phantom cash_balance change) that would corrupt
    # the bot's risk math. Diagnostic only; never blocks trading.
    try:
        _prev_eq = store.read_state("last_equity")
        _eq = portfolio.equity
        if isinstance(_prev_eq, (int, float)) and _prev_eq > 0 and abs(_eq - _prev_eq) / _prev_eq > 0.15:
            logger.warning(
                "equity jumped abnormally between cycles — verify broker portfolio",
                extra={"extra": {"prev": round(_prev_eq, 2), "now": round(_eq, 2),
                                 "change_pct": round((_eq - _prev_eq) / _prev_eq * 100, 2)}},
            )
        store.write_state("last_equity", _eq)
    except Exception:
        pass

    # ── Loss circuit breakers (portfolio-level, OPT-IN) ─────────────────────
    # Daily: block new entries if today's drawdown exceeds max_daily_loss_pct.
    # Peak/drawdown: block new entries if equity fell max_drawdown_pct from its
    # all-time peak. Both use equity (realized+unrealized) and only stop NEW
    # entries — exits/covers always pass so risk can still be reduced.
    # Each breaker is DISABLED when its pct is set to 0 (the per-trade stop-loss
    # and position/exposure caps remain the active protection). Disabled here
    # by config decision: a portfolio-wide day lock throttles turnover and
    # contradicts the bot's fast-adaptation design.
    daily_loss_breaker = False
    try:
        from datetime import date as _date
        _today = _date.today().isoformat()
        _dl = store.read_state("daily_loss") or {}
        if _dl.get("date") != _today:
            _dl = {"date": _today, "start_equity": portfolio.equity}
            store.write_state("daily_loss", _dl)
        _start = float(_dl.get("start_equity") or portfolio.equity)
        if settings.max_daily_loss_pct > 0 and _start > 0 and (portfolio.equity - _start) / _start <= -abs(settings.max_daily_loss_pct):
            daily_loss_breaker = True
            logger.warning("DAILY LOSS LIMIT hit — blocking new entries",
                           extra={"extra": {"day_drawdown_pct": round((portfolio.equity - _start) / _start * 100, 2),
                                            "limit_pct": round(settings.max_daily_loss_pct * 100, 2)}})
        # All-time peak drawdown kill-switch (max_drawdown_pct).
        _peak = float(store.read_state("equity_peak") or portfolio.equity)
        # Guard the peak against the broker "phantom equity" glitch: a single
        # cycle can't ratchet the all-time peak up by >15%, otherwise a later
        # correction back to real equity would false-trip the drawdown switch.
        if portfolio.equity > _peak and (_peak <= 0 or (portfolio.equity - _peak) / _peak <= 0.15):
            _peak = portfolio.equity
            store.write_state("equity_peak", _peak)
        if settings.max_drawdown_pct > 0 and _peak > 0 and (portfolio.equity - _peak) / _peak <= -abs(settings.max_drawdown_pct):
            daily_loss_breaker = True
            logger.warning("MAX DRAWDOWN hit — blocking new entries",
                           extra={"extra": {"drawdown_from_peak_pct": round((portfolio.equity - _peak) / _peak * 100, 2),
                                            "limit_pct": round(settings.max_drawdown_pct * 100, 2)}})
    except Exception:
        pass
    latest_prices: dict[str, float] = {}
    decision_count = 0
    approved_count = 0
    order_count = 0
    blocked_reasons: Counter[str] = Counter()

    # Phase 1: collect features for the whole watchlist once. We pass them to
    # the LLM brief (so it sees real per-symbol state) and reuse them in the
    # decision loop (so we don't double-fetch from ALGOPACK).
    features_by_symbol: dict[str, tuple] = {}
    for symbol in WATCHLIST:
        # Per-symbol isolation: a hung/failed ALGOPACK fetch on ONE symbol must
        # not abort the whole cycle — that would blind the stops of every OTHER
        # held position. Skip just this symbol; it's re-evaluated next cycle.
        try:
            candles = market_data.get_candles(symbol)
        except Exception:
            logger.warning("get_candles failed; skipping symbol this cycle",
                           extra={"extra": {"symbol": symbol}})
            continue
        if not candles:
            logger.warning("no candles", extra={"extra": {"symbol": symbol}})
            continue
        try:
            super_features = market_data.get_super_features(symbol)
        except Exception:
            logger.warning("get_super_features failed; using candles only",
                           extra={"extra": {"symbol": symbol}})
            super_features = {}
        # Macro features for this symbol — cached in LiveMacroCache so the
        # 5-min refresh window is shared across all 20 tickers per cycle.
        # If the cache fails (network blip), build_features handles None
        # gracefully and ml_features.py imputes 0.0 — train/serve parity holds.
        macro_features: dict[str, float] | None = None
        if macro_cache is not None and candles:
            try:
                macro_features = macro_cache.features_for(
                    symbol,
                    candles[-1].ts,
                    candles_to_pairs(candles[-60:]),
                )
            except Exception:
                logger.exception("macro_cache.features_for failed", extra={"extra": {"symbol": symbol}})
        features = build_features(candles, super_features, macro_features)
        latest_prices[symbol] = features.price
        regime = detect_regime(features)
        features_by_symbol[symbol] = (features, regime)

    # Phase 2: LLM brief with full feature context + recent MOEX news.
    # News fetch is fail-soft (returns [] on error). Brief is cached 1h.
    headlines: list[str] = []
    if news_service is not None:
        try:
            news_items = news_service.get_or_refresh()
            headlines = compact_headlines(news_items, max_chars=1500)
        except Exception:
            logger.exception("news service crashed; brief continues without news")
    snapshot = _brief_snapshot(portfolio, features_by_symbol, macro_cache=macro_cache)
    if headlines:
        snapshot["news_headlines"] = headlines
    brief = brief_service.get_or_refresh(
        snapshot=snapshot,
        watchlist=list(WATCHLIST),
    ) if brief_service is not None else default_brief()

    # Peak-PnL tracker per long position, persisted across cycles, for the
    # trailing stop. Pruned to currently-held longs below.
    position_peaks: dict = store.read_state("position_peaks") or {}

    # Phase 3: per-symbol decision loop, reusing the pre-collected features.
    for symbol in WATCHLIST:
        if symbol not in features_by_symbol:
            continue
        now = utc_now()
        # Session close in minutes-of-day UTC: weekend session closes 16:00 UTC
        # (19:00 MSK), weekday 20:50 UTC (23:50 MSK). Drives the flatten window.
        _close_min = 960 if now.weekday() >= 5 else 1250
        features, regime = features_by_symbol[symbol]
        # Update the running peak PnL for an open long, used by the trailing stop.
        _p = portfolio.positions.get(symbol)
        _peak = None
        if _p is not None and _p.qty > 0 and _p.avg_price > 0:
            _cur_pnl = (features.price - _p.avg_price) / _p.avg_price
            _peak = max(float(position_peaks.get(symbol, _cur_pnl)), _cur_pnl)
            position_peaks[symbol] = _peak
        exit_signal = generate_exit_signal(
            features,
            regime,
            portfolio.positions.get(symbol),
            stop_loss_pct=risk.config.stop_loss_pct,
            take_profit_pct=risk.config.take_profit_pct,
            reversal_exit_min_pnl_pct=risk.config.reversal_exit_min_pnl_pct,
            now=now,
            position_opened_at=(
                paper_ledger.position_opened_at(symbol)
                if paper_ledger is not None
                else None
            ),
            time_exit_hours=risk.config.time_exit_hours,
            time_exit_min_profit_pct=risk.config.time_exit_min_profit_pct,
            peak_pnl_pct=_peak,
            trail_stop_pct=risk.config.trail_stop_pct,
        )
        signal = exit_signal or generate_signal(features, regime)

        # ── Daily-loss breaker: once the day's loss limit is hit, no new entries.
        # Exits, take-profits, trailing, covers (have a position / [cover]) pass.
        if daily_loss_breaker and exit_signal is None:
            _is_new_entry = signal.action == Action.BUY or "[short-entry]" in signal.reason
            if _is_new_entry and "[cover]" not in signal.reason:
                signal = Signal(symbol, Action.HOLD, 0.5, regime,
                                f"daily-loss breaker: no new entries; {signal.reason}", features)

        # ── Regime-directional bias: take the side of the trend ─────────────
        # In a falling market (brief risk_off/news_shock), suppress NEW long
        # entries so the bot doesn't keep buying dips into a downtrend — the
        # short routing below then takes over. Exits and covers always pass.
        _bearish = (
            settings.regime_directional
            and settings.llm_apply_brief_to_risk
            and str(brief.market_regime).lower() in settings.bearish_regimes_set
        )
        # A hard long-block in risk_off (combined with the short cap) can
        # deadlock the bot and collapse turnover. risk_off already raises the
        # long bar (ML +0.10, min-confidence 0.65); here we only suppress WEAK
        # longs (confidence < 0.66) so strong dip-buys still trade for turnover,
        # while shorts carry the trend side. Net: lean short, don't go idle.
        if (
            _bearish
            and signal.action == Action.BUY
            and exit_signal is None
            and "[cover]" not in signal.reason
            and signal.confidence < 0.66
        ):
            signal = Signal(symbol, Action.HOLD, 0.5, regime,
                            f"regime {brief.market_regime}: weak long suppressed (favor shorts); {signal.reason}",
                            features)

        # ── Short side routing (gated by enable_shorts) ─────────────────────
        _pos = portfolio.positions.get(symbol)
        if _pos is not None and _pos.qty < 0:
            # OPEN SHORT: managed only by cover signals (buy-to-cover); the
            # long entry/exit logic above doesn't apply. Force-cover near the
            # session close so we don't carry a short overnight (gap risk).
            # Peak PnL of the short (profit grows as price falls) — feeds the
            # trailing stop, mirroring the long side above.
            _short_pnl = (_pos.avg_price - features.price) / _pos.avg_price if _pos.avg_price > 0 else 0.0
            _short_peak = max(float(position_peaks.get(symbol, _short_pnl)), _short_pnl)
            position_peaks[symbol] = _short_peak
            cover = generate_cover_signal(
                features, regime, _pos,
                stop_loss_pct=settings.short_stop_loss_pct,
                take_profit_pct=settings.short_take_profit_pct,
                now=now,
                position_opened_at=(
                    paper_ledger.position_opened_at(symbol)
                    if paper_ledger is not None
                    else None
                ),
                time_exit_hours=settings.short_time_exit_hours,
                time_exit_min_profit_pct=settings.short_time_exit_min_profit_pct,
                peak_pnl_pct=_short_peak,
                trail_stop_pct=settings.short_trail_stop_pct,
            )
            if cover is None and (_close_min - (now.hour * 60 + now.minute)) <= settings.flatten_minutes_before_close:
                cover = Signal(symbol, Action.BUY, 0.9, regime,
                               "short force-cover before session close [cover]", features)
            signal = cover or Signal(symbol, Action.HOLD, 0.5, regime, "holding short", features)
        elif (_pos is None) and risk.config.enable_shorts and signal.action != Action.BUY:
            # FLAT and no long BUY this cycle → consider opening a SHORT,
            # gated by the short ML filter (P(down) from lgbm_short_filter).
            short_sig = generate_short_signal(features, regime)
            if short_sig.action == Action.SELL:
                if short_ml_filter is not None:
                    p_down = short_ml_filter.predict_up_probability(features)  # positive class = "down"
                    if p_down < settings.ml_min_down_probability:
                        short_sig = Signal(symbol, Action.HOLD, 0.5, regime,
                                           f"short ml filter blocked: P(down)={p_down:.3f} below {settings.ml_min_down_probability:.3f}; {short_sig.reason}", features)
                    else:
                        short_sig = Signal(symbol, short_sig.action, min(0.95, max(short_sig.confidence, p_down)),
                                           regime, f"{short_sig.reason}; ml P(down)={p_down:.3f}", features)
                elif settings.ml_short_model_abs_path.exists():
                    # Fail-CLOSED, mirroring the buy filter: the short model was
                    # supposed to load but didn't — don't short unfiltered (that
                    # would silently become a different, unvalidated strategy).
                    short_sig = Signal(symbol, Action.HOLD, 0.5, regime,
                                       f"short ml model unavailable — entry blocked (fail-closed); {short_sig.reason}", features)
                if short_sig.action == Action.SELL:
                    signal = short_sig

        # ── End-of-session flatten: no overnight gap risk ───────────────────
        # Close is _close_min (weekday 1250=20:50 UTC, weekend 960=16:00 UTC).
        # Force-close open LONGS in the flatten window (shorts are force-covered
        # above), and stop opening any NEW position in the last N minutes.
        if settings.flatten_before_close:
            _mins_to_close = _close_min - (now.hour * 60 + now.minute)
            _pos_now = portfolio.positions.get(symbol)
            if (
                _pos_now is not None and _pos_now.qty > 0
                and _mins_to_close <= settings.flatten_minutes_before_close
            ):
                signal = Signal(symbol, Action.SELL, 0.9, regime,
                                f"force-close long before session close ({_mins_to_close}min left)", features)
            elif (
                0 <= _mins_to_close <= settings.no_new_entry_minutes_before_close
                and (signal.action == Action.BUY or "[short-entry]" in signal.reason)
            ):
                signal = Signal(symbol, Action.HOLD, 0.5, regime,
                                f"no new entries {_mins_to_close}min before close; {signal.reason}", features)

        # ── Stale-data guard: never OPEN on a previous session's candle ──────
        # ALGOPACK can return the prior session's candles (e.g. on a weekend
        # when a symbol has no fresh weekend prints), so a new entry would act on
        # hours-old prices. Block NEW entries when the latest candle is older
        # than max_candle_age_minutes; exits/covers/flatten always pass.
        # feature_ts is a naive Moscow timestamp (UTC+3).
        _ft = features.feature_ts
        if (
            _ft is not None and not isinstance(_ft, str)
            and (signal.action == Action.BUY or "[short-entry]" in signal.reason)
            and "[cover]" not in signal.reason
        ):
            _age_min = (now.replace(tzinfo=None) - (_ft - timedelta(hours=3))).total_seconds() / 60.0
            if _age_min > settings.max_candle_age_minutes:
                signal = Signal(symbol, Action.HOLD, 0.5, regime,
                                f"stale data: latest candle {_age_min:.0f}min old (>{settings.max_candle_age_minutes}min) — no new entry; {signal.reason}",
                                features)

        # PnL-aware SELL override: when generate_signal returns SELL on an
        # existing position that is NOT at profit, suppress it. The "overbought
        # near upper Bollinger band" branch was originally meant for opening
        # short positions; on a long position underwater it just realizes a
        # loss. Real exits should come from exit_signal (stop_loss / take_profit
        # / reversal_exit), all of which are PnL-aware.
        position_for_pnl = portfolio.positions.get(symbol)
        if (
            exit_signal is None  # not from PnL-aware exit logic
            and signal.action == Action.SELL
            and position_for_pnl is not None
            and position_for_pnl.qty > 0
            and position_for_pnl.avg_price > 0
        ):
            pnl_pct = (features.price - position_for_pnl.avg_price) / position_for_pnl.avg_price
            # +0.1% required — covers commission/slippage uncertainty. Below
            # that, hold and let real risk machinery decide the exit.
            if pnl_pct < 0.001:
                signal = Signal(
                    symbol,
                    Action.HOLD,
                    0.5,
                    regime,
                    f"sell suppressed: position pnl {pnl_pct:.4%} below profit floor; {signal.reason}",
                    signal.features,
                )

        # Early-skip: SELL without an open position is dead air — neither
        # actionable nor informative. We still hand the observation to
        # ml_data (it dedupes by feature_ts) so offline training keeps
        # seeing these candles, then we skip the rest of the cycle for
        # this symbol: no decision row, no risk eval, no log spam.
        if (
            signal.action == Action.SELL
            and portfolio.positions.get(symbol) is None
            and "[short-entry]" not in signal.reason  # a flat short-entry SELL is intentional
        ):
            if ml_data is not None:
                ml_data.observe(
                    features=features,
                    regime=regime,
                    signal=signal,
                    approved=False,
                )
            continue
        # Macro trend gate: don't open a NEW long while IMOEX is in a strong 4h
        # downtrend — buying "oversold" into a broad selloff is the knife-catch
        # that bled the book. Shorts/covers pass (they want a downtrend).
        if (
            settings.macro_block_return_4h < 0
            and signal.action == Action.BUY
            and "[cover]" not in signal.reason
            and features.imoex_return_4h is not None
            and features.imoex_return_4h < settings.macro_block_return_4h
        ):
            signal = Signal(
                signal.symbol,
                Action.HOLD,
                min(signal.confidence, 0.5),
                signal.regime,
                f"macro downtrend blocks long: IMOEX 4h {features.imoex_return_4h:.2%} < {settings.macro_block_return_4h:.2%}; {signal.reason}",
                signal.features,
            )
        # LLM brief soft-block: if the brief flagged this symbol for the hour,
        # convert any BUY into HOLD with an explicit reason. Sells are not
        # affected — closing/risk-reducing actions always go through.
        if (
            settings.llm_apply_brief_to_risk
            and signal.action == Action.BUY
            and "[cover]" not in signal.reason  # never block a buy-to-cover
            and symbol.upper() in [s.upper() for s in brief.blocked_symbols]
        ):
            signal = Signal(
                signal.symbol,
                Action.HOLD,
                min(signal.confidence, 0.5),
                signal.regime,
                f"llm brief blocked symbol in regime '{brief.market_regime}': {brief.rationale[:200]}",
                signal.features,
            )

        # Fail-CLOSED: if the ML filter is enabled but the model didn't load,
        # block NEW entries instead of trading unfiltered (which would silently
        # become a different, unvalidated strategy). Exits/covers still pass.
        if (
            settings.ml_filter_enabled and ml_filter is None
            and (signal.action == Action.BUY or "[short-entry]" in signal.reason)
            and "[cover]" not in signal.reason
        ):
            signal = Signal(symbol, Action.HOLD, 0.5, regime,
                            f"ML enabled but model unavailable — entry blocked (fail-closed); {signal.reason}",
                            features)

        if ml_filter is not None and signal.action == Action.BUY and "[cover]" not in signal.reason:
            up_probability = ml_filter.predict_up_probability(features)
            # Regime-aware ML bar: stay active (base threshold) in a calm market,
            # get pickier in a deteriorating one so we don't catch falling knives.
            # Uses the brief's regime; offsets are small and bounded.
            _regime_offset = {
                "normal": 0.0,
                "macro_uncertain": 0.04,
                "news_shock": 0.08,
                "risk_off": 0.10,
            }.get(str(brief.market_regime).lower(), 0.0) if settings.llm_apply_brief_to_risk else 0.0
            ml_threshold = min(0.95, risk.config.ml_min_up_probability + _regime_offset)
            if up_probability < ml_threshold:
                signal = Signal(
                    signal.symbol,
                    Action.HOLD,
                    min(signal.confidence, 0.5),
                    signal.regime,
                    f"ml buy filter blocked: P(up)={up_probability:.3f} below {ml_threshold:.3f}; {signal.reason}",
                    signal.features,
                )
            else:
                signal = Signal(
                    signal.symbol,
                    signal.action,
                    min(0.95, max(signal.confidence, up_probability)),
                    signal.regime,
                    f"{signal.reason}; ml P(up)={up_probability:.3f}",
                    signal.features,
                )
        # Apply LLM brief's cash_multiplier + market_regime when the operator
        # opted in. risk.py clamps cash_multiplier to [0.5, 1.0] internally so
        # an "aggressive" brief can't shut trading off entirely (turnover
        # requirements still apply). market_regime raises the BUY confidence bar
        # in worse regimes but never blocks outright.
        cash_mult = brief.cash_multiplier if settings.llm_apply_brief_to_risk else 1.0
        regime_for_risk = brief.market_regime if settings.llm_apply_brief_to_risk else "normal"
        # mid_review mode is set by Opus every ~3h and stored in
        # data/state/mid_review.json with an expiry timestamp; risk.py reads
        # it via the parameter, and once expired we silently fall back to normal.
        from moex_agent.mid_review import read_active_mode
        mid_mode = read_active_mode(settings.runtime_data_dir)
        # ArenaGo lot size per ticker: 1 for most, 10 for MTSS/MOEX/GAZP/GMKN/
        # ALRS/AFLT/NLMK/SNGSP, 100 for VTBR. risk.py rounds qty DOWN to lot to
        # prevent below_lot_size rejections.
        lot_size = settings.arena_lot_sizes.get(symbol, 1)
        # Anti-churn signal: count closed trades on this symbol in the last 60
        # min and average PnL. risk.py blocks new BUYs if churning is detected.
        recent_trades_n, recent_avg_pnl = _recent_symbol_stats(symbol, now)
        # Cooldown + daily count: from paper ledger if present, else reconstruct
        # from orders.jsonl so these gates aren't dead in live mode.
        if paper_ledger is not None:
            _daily_count = paper_ledger.daily_trade_count()
            _last_at = paper_ledger.last_trade_at(symbol)
        else:
            _live_last_ts, _daily_count = _live_order_tracking(now)
            _last_at = _live_last_ts.get(symbol)
        approved, order, risk_reason = risk.evaluate(
            signal,
            portfolio,
            daily_trade_count=_daily_count,
            last_trade_at=_last_at,
            now=now,
            cash_multiplier=cash_mult,
            market_regime=regime_for_risk,
            mid_review_mode=mid_mode,
            lot_size=lot_size,
            recent_symbol_trades_60min=recent_trades_n,
            recent_symbol_avg_pnl_pct=recent_avg_pnl,
        )
        if ml_data is not None:
            ml_data.observe(
                features=features,
                regime=regime,
                signal=signal,
                approved=approved,
            )
        decision_count += 1
        if approved:
            approved_count += 1
        if not approved:
            for reason in split_risk_reasons(risk_reason):
                blocked_reasons[reason] += 1

        similar = None
        if memory_index is not None:
            try:
                similar_result = memory_index.query(features)
                if similar_result is not None:
                    similar = similar_result.to_payload()
            except Exception:
                logger.exception("memory query failed for symbol", extra={"extra": {"symbol": symbol}})

        decision = Decision(
            ts=now,
            symbol=symbol,
            signal=signal,
            order=order,
            approved=approved,
            reason=risk_reason,
            meta={
                "dry_run": client.dry_run,
                "paper_trading": client.paper_trading,
                "risk_reasons": split_risk_reasons(risk_reason) if not approved else [],
                "similar_setups": similar,
            },
        )
        payload = asdict(decision)
        memory.remember(payload)
        logger.info("decision", extra={"extra": payload})

        if approved and order is not None:
            # Process-level order quota for --max-orders runs. Once exhausted,
            # the bot still computes signals + logs everything, but never
            # actually places.
            if order_quota is not None and order_quota.get("limit", 0) > 0:
                if order_quota.get("placed", 0) >= order_quota["limit"]:
                    logger.warning(
                        "process order quota exhausted — skipping place_order",
                        extra={"extra": {"symbol": symbol, "placed": order_quota["placed"], "limit": order_quota["limit"]}},
                    )
                    continue
                order_quota["placed"] = order_quota.get("placed", 0) + 1
            order_count += 1
            try:
                result = client.place_order(order)
            except Exception:
                # A broker error on ONE order must not abort the whole cycle —
                # that would skip stop/cover/flatten evaluation for every symbol
                # later in the loop. Log it, record a failure row, move on.
                logger.exception("place_order raised; skipping symbol this cycle",
                                 extra={"extra": {"symbol": symbol}})
                store.append_jsonl("orders", {"ts": utc_now(),
                                              "result": {"status": "error", "symbol": symbol}})
                continue
            store.append_jsonl("orders", {"ts": utc_now(), "result": result})
            if paper_ledger is not None:
                paper_ledger.record_order(portfolio, order, utc_now())
            else:
                # LIVE: reflect the fill in the local portfolio so the REMAINING
                # symbols this cycle evaluate caps (cash buffer, exposure, sector,
                # short count) against post-order state — otherwise several orders
                # in one cycle can collectively breach a limit each passed alone.
                _st = str((result or {}).get("status") or "")
                _rsp = (result or {}).get("response") or {}
                if _st in ("filled", "ok") or (_st == "submitted" and _rsp.get("success")):
                    try:
                        portfolio.apply_fill(order.side, order.symbol, order.qty,
                                             order.limit_price, allow_short=settings.enable_shorts)
                    except Exception:
                        logger.exception("local apply_fill after live order failed",
                                         extra={"extra": {"symbol": symbol}})
            # Fire-and-forget LLM explanation. Never blocks. Result lands in
            # data/logs/llm_explanations.jsonl for review. Trades below the
            # min-value threshold are skipped: tiny tail scale-out sells would
            # flood the log with low-information entries and burn tokens, while
            # big trades are where post-hoc rationale matters for the retro.
            order_value_rub = (order.limit_price or 0) * (order.qty or 0)
            if explainer is not None and order_value_rub >= settings.llm_explain_min_value_rub:
                explainer.explain_async({
                    "symbol": symbol,
                    "order": {
                        "side": order.side.value,
                        "qty": order.qty,
                        "limit_price": order.limit_price,
                        "reason": order.reason,
                    },
                    "signal": {
                        "action": signal.action.value,
                        "confidence": round(signal.confidence, 4),
                        "regime": signal.regime.value,
                        "reason": signal.reason,
                    },
                    "features": {
                        "price": features.price,
                        "rsi": features.rsi,
                        "macd_diff": (features.macd or 0) - (features.macd_signal or 0),
                        "bb_low": features.bollinger_low,
                        "atr_pct": features.atr_pct,
                        "trade_imbalance": features.trade_imbalance,
                        "book_imbalance": features.book_imbalance,
                    },
                    "regime": regime.value,
                    "market_brief": {
                        "regime": brief.market_regime,
                        "rationale": brief.rationale,
                    },
                    "paper_trading": client.paper_trading,
                })

    # ── Safety sweep: protect positions whose data feed failed this cycle ───
    # The per-symbol loop skips any symbol whose candles didn't load, so a held
    # position with a flaky feed would get NO stop/cover/flatten that cycle and
    # could be carried overnight. The broker portfolio still reports its own
    # market_price, so here we run a feature-INDEPENDENT close for any open
    # position the main loop did not evaluate: a hard stop-loss breach, or the
    # end-of-session flatten window. Uses the broker's mark — no ALGOPACK needed.
    _now_sweep = utc_now()
    _sweep_close_min = 960 if _now_sweep.weekday() >= 5 else 1250
    _flatten_now = settings.flatten_before_close and (_sweep_close_min - (_now_sweep.hour * 60 + _now_sweep.minute)) <= settings.flatten_minutes_before_close
    for _sym, _pos in list(portfolio.positions.items()):
        if _sym in features_by_symbol:
            continue  # already handled with fresh features in the main loop
        if _pos.qty == 0 or _pos.avg_price <= 0 or _pos.market_price <= 0:
            continue
        _is_long = _pos.qty > 0
        if _is_long:
            _pnl = (_pos.market_price - _pos.avg_price) / _pos.avg_price
            _stop_hit = _pnl <= -abs(risk.config.stop_loss_pct)
        else:
            _pnl = (_pos.avg_price - _pos.market_price) / _pos.avg_price
            _stop_hit = _pnl <= -abs(settings.short_stop_loss_pct)
        if not (_stop_hit or _flatten_now):
            continue
        _reason = ("safety stop-loss (data feed down) " if _stop_hit
                   else "safety flatten before close (data feed down) ") + f"pnl={_pnl:.4%}"
        if not _is_long:
            _reason += " [cover]"
        _close = OrderIntent(
            symbol=_sym,
            side=OrderSide.SELL if _is_long else OrderSide.BUY,
            qty=abs(int(_pos.qty)),
            limit_price=_pos.market_price,
            reason=_reason,
        )
        logger.warning("safety sweep closing unattended position",
                       extra={"extra": {"symbol": _sym, "qty": _pos.qty,
                                        "pnl_pct": round(_pnl * 100, 3),
                                        "stop": _stop_hit, "flatten": _flatten_now}})
        try:
            _res = client.place_order(_close)
        except Exception:
            logger.exception("safety sweep place_order failed", extra={"extra": {"symbol": _sym}})
            store.append_jsonl("orders", {"ts": utc_now(), "result": {"status": "error", "symbol": _sym}})
            continue
        store.append_jsonl("orders", {"ts": utc_now(), "result": _res})
        order_count += 1
        if paper_ledger is not None:
            paper_ledger.record_order(portfolio, _close, utc_now())
        else:
            _st = str((_res or {}).get("status") or "")
            _rsp = (_res or {}).get("response") or {}
            if _st in ("filled", "ok") or (_st == "submitted" and _rsp.get("success")):
                try:
                    portfolio.apply_fill(_close.side, _close.symbol, _close.qty,
                                         _close.limit_price, allow_short=settings.enable_shorts)
                except Exception:
                    logger.exception("safety sweep apply_fill failed", extra={"extra": {"symbol": _sym}})

    if paper_ledger is not None:
        paper_ledger.mark_to_market(portfolio, latest_prices)
        store.write_state("portfolio", portfolio_snapshot(portfolio))

    # Persist trailing-stop peaks, pruned to currently-held longs (a closed
    # position must reset its peak so the next entry trails from scratch).
    held_longs = {s for s, p in portfolio.positions.items() if p.qty > 0}
    store.write_state("position_peaks", {s: v for s, v in position_peaks.items() if s in held_longs})

    # Persist latest market prices so monitor.py (and other tooling) can
    # compute unrealized PnL on open positions without re-hitting MOEX API.
    if latest_prices:
        store.write_state("latest_prices", {
            "ts": utc_now(),
            "prices": {sym: float(price) for sym, price in latest_prices.items()},
        })

    summary = {
        "portfolio": portfolio_snapshot(portfolio),
        "decisions": decision_count,
        "approved": approved_count,
        "orders": order_count,
        "blocked_reasons": dict(blocked_reasons),
        "paper_trading": client.paper_trading,
        "dry_run": client.dry_run,
        "llm_brief": {
            "regime": brief.market_regime,
            "cash_multiplier": brief.cash_multiplier,
            "blocked_symbols": brief.blocked_symbols,
        },
    }
    store.append_jsonl("cycle_summary", {"ts": utc_now(), **summary})
    logger.info("cycle summary", extra={"extra": summary})


def _brief_snapshot(
    portfolio,
    features_by_symbol: dict | None = None,
    *,
    macro_cache: LiveMacroCache | None = None,
) -> dict[str, object]:
    """Compact portfolio + per-symbol state to send to the LLM for the brief.

    With features_by_symbol the LLM sees not just "we have 4 positions" but
    "SBER RSI=20 oversold with -0.5% unrealized, GAZP overbought with +1.2%,
    market regime mixed". That's enough for it to recommend per-symbol blocks.

    Macro enrichment (when macro_cache is provided): IMOEX 60m/4h returns +
    sector breadth + FUTOI snapshot (physical-vs-juridical futures positioning
    on Si/RTSI/BR). Plus our own MegaAlerts emulator: volume_spike and
    price_jump flags per symbol so the LLM can correlate news to anomalies.
    """
    out: dict[str, object] = {
        "equity": round(float(portfolio.equity), 2),
        "cash": round(float(portfolio.cash), 2),
        "exposure": round(float(portfolio.exposure), 2),
        "positions_count": len(portfolio.positions),
        "open_symbols": sorted(portfolio.positions.keys()),
        "watchlist_size": len(WATCHLIST),
    }
    if not features_by_symbol:
        return out

    per_symbol: dict[str, dict] = {}
    regimes_counter: dict[str, int] = {}
    for symbol, (feat, reg) in features_by_symbol.items():
        position = portfolio.positions.get(symbol)
        has_position = position is not None and position.qty > 0
        pnl_pct = None
        if has_position and position.avg_price > 0:
            pnl_pct = (feat.price - position.avg_price) / position.avg_price

        bb_position = None
        if feat.bollinger_low is not None and feat.bollinger_high is not None:
            mid = (feat.bollinger_low + feat.bollinger_high) / 2.0
            half_range = (feat.bollinger_high - feat.bollinger_low) / 2.0
            if half_range > 0:
                # Normalized: -1 means at lower band, +1 means at upper band.
                bb_position = (feat.price - mid) / half_range

        per_symbol[symbol] = {
            "rsi": round(feat.rsi, 1) if feat.rsi is not None else None,
            "regime": reg.value,
            "bb_position": round(bb_position, 2) if bb_position is not None else None,
            "atr_pct": round((feat.atr_pct or 0) * 100, 3),
            "trade_imbalance": round(feat.trade_imbalance, 2) if feat.trade_imbalance is not None else None,
            "book_imbalance": round(feat.book_imbalance, 2) if feat.book_imbalance is not None else None,
            "has_position": has_position,
            "pnl_pct": round(pnl_pct * 100, 2) if pnl_pct is not None else None,
        }
        regimes_counter[reg.value] = regimes_counter.get(reg.value, 0) + 1

    out["per_symbol"] = per_symbol
    out["regime_distribution"] = regimes_counter

    # Market breadth — purely computed from features_by_symbol, no API call.
    # Gives the LLM a one-glance read of the tape: are most tickers above or
    # below their 20-period mid, is the market broadly oversold/overbought,
    # how many are in TREND vs RANGE.
    breadth: dict[str, object] = {
        "tickers_total": len(features_by_symbol),
        "above_mid": 0,
        "below_mid": 0,
        "oversold_rsi_lt_30": 0,
        "overbought_rsi_gt_70": 0,
        "in_trend": 0,
        "in_range": 0,
        "avg_atr_pct": 0.0,
        "median_rsi": None,
    }
    atr_sum = 0.0
    atr_n = 0
    rsis: list[float] = []
    for symbol, (feat, reg) in features_by_symbol.items():
        if feat.bollinger_low is not None and feat.bollinger_high is not None:
            mid = (feat.bollinger_low + feat.bollinger_high) / 2.0
            if feat.price > mid:
                breadth["above_mid"] = int(breadth["above_mid"]) + 1
            elif feat.price < mid:
                breadth["below_mid"] = int(breadth["below_mid"]) + 1
        if feat.rsi is not None:
            rsis.append(feat.rsi)
            if feat.rsi < 30.0:
                breadth["oversold_rsi_lt_30"] = int(breadth["oversold_rsi_lt_30"]) + 1
            elif feat.rsi > 70.0:
                breadth["overbought_rsi_gt_70"] = int(breadth["overbought_rsi_gt_70"]) + 1
        if feat.atr_pct is not None:
            atr_sum += feat.atr_pct
            atr_n += 1
        if reg.value == "trend":
            breadth["in_trend"] = int(breadth["in_trend"]) + 1
        elif reg.value == "range":
            breadth["in_range"] = int(breadth["in_range"]) + 1
    if atr_n:
        breadth["avg_atr_pct"] = round((atr_sum / atr_n) * 100, 3)
    if rsis:
        rsis_sorted = sorted(rsis)
        m = len(rsis_sorted)
        breadth["median_rsi"] = round(rsis_sorted[m // 2], 1) if m % 2 == 1 else round((rsis_sorted[m // 2 - 1] + rsis_sorted[m // 2]) / 2.0, 1)
    out["market_breadth"] = breadth

    # MegaAlerts emulator: flag tickers with volume_spike (volume_ratio > 3)
    # or stretched price (|bb_position| > 0.9). Risk filters already block
    # these for buy; the brief surfaces them so the LLM can correlate to news.
    anomalies: list[dict] = []
    for symbol, (feat, _reg) in features_by_symbol.items():
        flags: list[str] = []
        if feat.volume_ratio is not None and feat.volume_ratio > 3.0:
            flags.append(f"volume_spike(x{feat.volume_ratio:.1f})")
        if feat.bollinger_low is not None and feat.bollinger_high is not None:
            half = (feat.bollinger_high - feat.bollinger_low) / 2.0
            if half > 0:
                bb_pos = (feat.price - (feat.bollinger_low + feat.bollinger_high) / 2.0) / half
                if bb_pos > 0.9:
                    flags.append("stretched_upper_band")
                elif bb_pos < -0.9:
                    flags.append("stretched_lower_band")
        if feat.book_spread_pct is not None and feat.book_spread_pct > 0.015:
            flags.append(f"wide_spread({feat.book_spread_pct*100:.2f}%)")
        if flags:
            anomalies.append({"symbol": symbol, "flags": flags})
    if anomalies:
        out["anomalies"] = anomalies

    # Macro context — IMOEX state from LiveMacroCache. Cheap (cache lookup).
    if macro_cache is not None:
        macro_block: dict[str, object] = {}
        try:
            ts, value = macro_cache.latest_imoex()
            if value is not None:
                macro_block["imoex_close"] = round(value, 2)
                macro_block["imoex_ts"] = ts.isoformat() if ts else None
            # Pick any symbol's macro features (they share IMOEX state).
            for sym, (feat, _reg) in features_by_symbol.items():
                if feat.imoex_return_60m is not None:
                    macro_block["imoex_return_60m_pct"] = round(feat.imoex_return_60m * 100, 3)
                    macro_block["imoex_return_4h_pct"] = round((feat.imoex_return_4h or 0) * 100, 3)
                    break
        except Exception:
            logger.exception("macro_block compute failed")
        if macro_block:
            out["macro_context"] = macro_block

    # FUTOI snapshot — fail-soft fetch from ALGOPACK. Worst case returns {}
    # and the LLM proceeds without futures positioning context.
    try:
        from moex_agent.macro_data import get_futoi_snapshot
        futoi = get_futoi_snapshot()
        if futoi:
            out["futoi"] = futoi
    except Exception:
        logger.exception("futoi snapshot failed (non-fatal)")

    # Macro spot quotes — USDRUB and Brent active contract. Cheap (~1-2 calls
    # to ALGOPACK), fail-soft, gives LLM a current macro-frame anchor.
    try:
        from moex_agent.macro_data import get_macro_quotes_snapshot
        quotes = get_macro_quotes_snapshot()
        if quotes:
            out["macro_quotes"] = quotes
    except Exception:
        logger.exception("macro_quotes snapshot failed (non-fatal)")

    return out


def build_risk_manager() -> RiskManager:
    return RiskManager(
        RiskConfig(
            max_position_pct=settings.max_position_pct,
            max_portfolio_exposure_pct=settings.max_portfolio_exposure_pct,
            max_daily_loss_pct=settings.max_daily_loss_pct,
            max_drawdown_pct=settings.max_drawdown_pct,
            order_cash_pct=settings.order_cash_pct,
            min_cash_pct=settings.min_cash_pct,
            max_daily_trades=settings.max_daily_trades,
            symbol_cooldown_minutes=settings.symbol_cooldown_minutes,
            min_volume_ratio=settings.min_volume_ratio,
            min_trade_imbalance_buy=settings.min_trade_imbalance_buy,
            min_super_volume=settings.min_super_volume,
            min_book_imbalance_buy=settings.min_book_imbalance_buy,
            max_book_spread_pct=settings.max_book_spread_pct,
            add_position_min_confidence=settings.add_position_min_confidence,
            add_position_min_trade_imbalance=settings.add_position_min_trade_imbalance,
            add_position_min_pnl_pct=settings.add_position_min_pnl_pct,
            add_position_max_position_pct=settings.add_position_max_position_pct,
            conviction_sizing_enabled=settings.conviction_sizing_enabled,
            conviction_gain=settings.conviction_gain,
            conviction_max_mult=settings.conviction_max_mult,
            ml_min_up_probability=settings.ml_min_up_probability,
            stop_loss_pct=settings.stop_loss_pct,
            take_profit_pct=settings.take_profit_pct,
            reversal_exit_min_pnl_pct=settings.reversal_exit_min_pnl_pct,
            time_exit_hours=settings.time_exit_hours,
            time_exit_min_profit_pct=settings.time_exit_min_profit_pct,
            trail_stop_pct=settings.trail_stop_pct,
            enable_shorts=settings.enable_shorts,
            short_order_cash_pct=settings.short_order_cash_pct,
            max_concurrent_shorts=settings.max_concurrent_shorts,
            short_max_total_exposure_pct=settings.short_max_total_exposure_pct,
            max_positions_per_sector=settings.max_positions_per_sector,
        )
    )


def split_risk_reasons(reason: str) -> list[str]:
    return [part.strip() for part in reason.split(";") if part.strip()]


def build_market_data_client() -> MarketDataClient:
    provider = settings.market_data_provider.lower()
    if provider == "sample":
        return SampleMarketDataClient()
    if provider == "iss":
        return IssMarketDataClient(
            period=settings.candle_period,
            lookback_days=settings.candle_lookback_days,
        )
    return AlgopackMarketDataClient(
        settings.moex_algo_token,
        period=settings.candle_period,
        lookback_days=settings.candle_lookback_days,
        use_super_candles=settings.use_super_candles,
    )


def main() -> None:
    # Global network safety net: bound every socket op so a hung MOEX/ALGOPACK
    # request can't freeze the autonomous loop forever. moexalgo sets no
    # per-request timeout; this is the floor. (Polza sets its own, unaffected.)
    import socket as _socket
    _socket.setdefaulttimeout(settings.socket_timeout_seconds)
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one trading cycle and exit.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Disable dry-run/paper. Requires I_REALLY_WANT_LIVE_TRADING=yes env to acknowledge real orders.",
    )
    parser.add_argument(
        "--max-orders",
        type=int,
        default=0,
        help="Hard cap on orders this process can submit. 0 = unlimited. Useful for a controlled --live test.",
    )
    args = parser.parse_args()

    configure_logging(settings.log_level)

    # Safety: --live disables BOTH dry_run and paper_trading. To prevent an
    # accidental Ctrl+R recall from spraying real orders on ArenaGo, also
    # require an explicit env acknowledgement.
    if args.live:
        ack = os.getenv("I_REALLY_WANT_LIVE_TRADING", "").strip().lower()
        if ack != "yes":
            raise SystemExit(
                "Refusing to run --live without I_REALLY_WANT_LIVE_TRADING=yes. "
                "Set this env variable to acknowledge that real orders will be "
                "submitted to ArenaGo. Inspect order sizing and --max-orders first."
            )

    dry_run = settings.dry_run and not args.live
    paper_trading = settings.paper_trading and not args.live
    if (paper_trading or not dry_run) and not settings.sandbox_api_key:
        raise SystemExit("SANDBOX_API_KEY is required for ArenaGo portfolio/live mode.")

    store = JsonStore(settings.runtime_data_dir)
    memory = DecisionMemory(store)
    paper_ledger = PaperLedger(store) if paper_trading else None
    ml_data = MLDataCollector(store, horizon_steps=settings.ml_label_horizon_steps)
    ml_filter = MLBuyFilter.load(settings.ml_model_abs_path) if settings.ml_filter_enabled else None
    # Short ML filter (positive class = "down"); loaded only when shorts are on
    # and the model file exists. Same MLBuyFilter wrapper — predict_up_probability
    # returns P(down) for this model.
    short_ml_filter = None
    if settings.enable_shorts and settings.ml_short_model_abs_path.exists():
        try:
            short_ml_filter = MLBuyFilter.load(settings.ml_short_model_abs_path, positive_label="down")
            logger.info("short ml filter loaded", extra={"extra": {"path": str(settings.ml_short_model_abs_path)}})
        except Exception:
            logger.exception("failed to load short ml filter — shorts will use deterministic signal only")

    brief_service: BriefService | None = None
    explainer: ExplainerService | None = None
    news_service: NewsService | None = NewsService(
        store,
        ttl_seconds=settings.news_ttl_seconds,
        timeout=settings.news_timeout_seconds,
        max_items=settings.news_max_items,
    ) if settings.news_enabled else None

    memory_index: SimilarSetupsIndex | None = None
    if settings.memory_enabled:
        memory_index = SimilarSetupsIndex(
            settings.memory_dataset_abs_path,
            k=settings.memory_k_neighbors,
            max_rows=settings.memory_max_rows,
        )
        # Trigger load at startup so the first cycle isn't slowed down.
        _ = memory_index.available
        logger.info(
            "memory index init",
            extra={"extra": {"available": memory_index.available, "rows": memory_index.n_rows}},
        )
    if settings.llm_enabled and settings.polza_api_key:
        polza = PolzaClient(
            api_key=settings.polza_api_key,
            base_url=settings.polza_base_url,
            timeout=settings.llm_timeout_seconds,
        )
        # Brief fires once per hour while market is open (~10-15/day). Premium
        # tier (Opus) was overkill — DeepSeek Flash handles regime classification
        # and blocked-symbols extraction just as well at ~1/100th the cost.
        # Opus is now used only by daily_retro and auto_tuner (2 calls/day total).
        brief_service = BriefService(
            store,
            polza,
            model=settings.llm_model_brief,
            ttl_seconds=settings.llm_brief_ttl_seconds,
            max_tokens=settings.llm_brief_max_tokens,
            fallback_model=settings.llm_model_brief_fallback,
        )
        # Explainer fires per order — many calls. Use cheap workhorse model.
        explainer = ExplainerService(
            store,
            polza,
            model=settings.llm_model_fast,
            max_tokens=settings.llm_explain_max_tokens,
        )
    elif settings.llm_enabled and not settings.polza_api_key:
        logger.warning("LLM_ENABLED=true but POLZA_API_KEY is empty — LLM layer disabled")

    market_data = build_market_data_client()
    client = ArenaClient(
        settings.arenago_base_url,
        settings.sandbox_api_key,
        bot_name=settings.bot_name,
        dry_run=dry_run,
        paper_trading=paper_trading,
        max_retries=settings.arena_retry_attempts,
        retry_backoff_seconds=settings.arena_retry_backoff_seconds,
        max_consecutive_failures=settings.arena_max_consecutive_failures,
        lot_sizes=settings.arena_lot_sizes,
    )
    risk = build_risk_manager()

    logger.info(
        "agent started",
        extra={
            "extra": {
                "dry_run": dry_run,
                "paper_trading": paper_trading,
                "data_dir": str(settings.runtime_data_dir),
                "bot_name": settings.bot_name,
                "market_data_provider": settings.market_data_provider,
                "candle_period": settings.candle_period,
                "use_super_candles": settings.use_super_candles,
                "max_daily_trades": settings.max_daily_trades,
                "symbol_cooldown_minutes": settings.symbol_cooldown_minutes,
                "min_volume_ratio": settings.min_volume_ratio,
                "add_position_min_confidence": settings.add_position_min_confidence,
                "add_position_min_trade_imbalance": settings.add_position_min_trade_imbalance,
                "add_position_min_pnl_pct": settings.add_position_min_pnl_pct,
                "stop_loss_pct": settings.stop_loss_pct,
                "take_profit_pct": settings.take_profit_pct,
                "time_exit_hours": settings.time_exit_hours,
                "time_exit_min_profit_pct": settings.time_exit_min_profit_pct,
                "watchlist": WATCHLIST,
                "ml_filter_loaded": ml_filter is not None,
                "ml_model_path": str(settings.ml_model_abs_path),
                "ml_min_up_probability": settings.ml_min_up_probability,
                "llm_enabled": brief_service is not None,
                "llm_model_fast": settings.llm_model_fast,
                "llm_apply_brief_to_risk": settings.llm_apply_brief_to_risk,
            }
        },
    )

    order_quota: dict | None = None
    if args.max_orders and args.max_orders > 0:
        order_quota = {"limit": int(args.max_orders), "placed": 0}
        logger.info("process-level order quota armed", extra={"extra": order_quota})

    # Single shared macro cache across all cycles. 5-minute refresh means
    # at most 1 IMOEX+sector fetch every 5 minutes, regardless of how many
    # tickers we have on the watchlist.
    macro_cache: LiveMacroCache | None = None
    _macro_via_iss = settings.market_data_provider.lower() == "iss"
    if settings.moex_algo_token or _macro_via_iss:
        macro_cache = LiveMacroCache(
            refresh_seconds=300.0,
            lookback_hours=6,
            period=settings.candle_period,
            use_iss=_macro_via_iss,
        )

    # Scheduler for periodic LLM-driven supervisors. They run as subprocesses
    # so the trading loop is never blocked on a 5–30s LLM call. Each module
    # has its own internal guardrails (whitelisted parameters, cooldowns,
    # value clamps) so a misfired or hallucinated response cannot break the
    # bot. We just decide WHEN to fire them — and we don't fire them outside
    # MOEX trading hours (weekends, deep night) to save tokens.
    import subprocess
    import sys
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    # mid_review: interval-based (default 45 min, configurable via .env).
    # auto_tuner: time-of-day slots (default 08:00 + 16:00 UTC) — see
    # _auto_tuner_due below. AUTO_TUNER_INTERVAL is kept only for the
    # restore-timestamp default math; firing is decided by the slot check.
    MID_REVIEW_INTERVAL = _td(minutes=settings.mid_review_interval_min)
    AUTO_TUNER_INTERVAL = _td(hours=12)
    AUTO_TUNER_UTC_HOURS = settings.auto_tuner_utc_hours_list
    # Event-driven (fast) adaptation: react to a live bleed instead of only
    # waiting for the timer/slot. mid_review (cheap, reversible mode change)
    # reacts to any losing streak; auto_tuner (Opus, changes params) only on a
    # STRONG bleed and with a long cooldown so it can't thrash parameters.
    MID_REVIEW_EVENT_COOLDOWN = _td(minutes=15)
    AUTO_TUNER_EVENT_COOLDOWN = _td(minutes=90)
    BLEED_MIN_TRADES = 3            # ≥3 recent closed trades to call it a streak
    AUTO_TUNER_BLEED_TRADES = 5     # stronger evidence before a param change
    AUTO_TUNER_BLEED_AVG_PNL = -0.003  # avg recent PnL below -0.3% = real bleed

    def _auto_tuner_due(now: _dt, last_fire: _dt, hours: list[int]) -> bool:
        # Fire once per configured UTC-hour slot per day: when 'now' has passed
        # the most recent slot for today and we haven't fired since it opened.
        for h in sorted(hours, reverse=True):
            slot = now.replace(hour=h, minute=0, second=0, microsecond=0)
            if now >= slot:
                return last_fire < slot
        return False

    MIN_CYCLES_BEFORE_SCHEDULING = 3  # let the bot collect minimal data first (3 cycles ≈ 3-5 min)

    # Restore scheduler timestamps across restarts. Without this every restart
    # resets the clock to "now" and supervisors would never fire if we restart
    # more often than the interval (which we do during tuning).
    _scheduler_state = store.read_state("scheduler") or {}
    def _restore_ts(key: str, default_age: _td) -> _dt:
        raw = _scheduler_state.get(key)
        if raw:
            try:
                parsed = _dt.fromisoformat(str(raw).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=_tz.utc)
                return parsed
            except Exception:
                pass
        # No prior run recorded — schedule first fire shortly after the bot stabilises.
        return _dt.now(_tz.utc) - default_age

    # First fire 5 minutes after start for mid_review, 30 minutes for auto_tuner.
    last_mid_review_fire = _restore_ts("last_mid_review", MID_REVIEW_INTERVAL - _td(minutes=5))
    last_auto_tuner_fire = _restore_ts("last_auto_tuner", AUTO_TUNER_INTERVAL - _td(minutes=30))
    # On bot start, if recovered timestamps suggest the next mid_review is
    # MORE than its interval away (because we tightened from 3h to 90min),
    # snap forward so we don't wait the leftover from old interval.
    _now_init = _dt.now(_tz.utc)
    if _now_init - last_mid_review_fire > MID_REVIEW_INTERVAL * 2:
        last_mid_review_fire = _now_init - MID_REVIEW_INTERVAL + _td(minutes=5)
    if _now_init - last_auto_tuner_fire > AUTO_TUNER_INTERVAL * 2:
        last_auto_tuner_fire = _now_init - AUTO_TUNER_INTERVAL + _td(minutes=30)
    logger.info(
        "scheduler timestamps restored",
        extra={"extra": {
            "last_mid_review": last_mid_review_fire.isoformat(),
            "last_auto_tuner": last_auto_tuner_fire.isoformat(),
        }},
    )

    def _is_moex_active(now_utc: _dt) -> bool:
        """MOEX/ArenaGo trade Mon–Fri ~04:00–20:50 UTC (07:00–23:50 MSK, all
        sessions) AND on weekends ~07:00–16:00 UTC (10:00–19:00 MSK weekend
        session — organisers confirmed ArenaGo is open on weekends). Holidays
        not modeled. The stale-data guard in the decision loop blocks new
        entries when a symbol has no fresh candle, so this window can't cause
        trading on a previous session's stale prices.
        """
        minute_of_day = now_utc.hour * 60 + now_utc.minute
        if now_utc.weekday() >= 5:  # Sat/Sun — weekend session 07:00–16:00 UTC
            return 420 <= minute_of_day <= 960
        # 04:00 UTC = 240 min; 20:50 UTC = 1250 min
        return 240 <= minute_of_day <= 1250

    # Track in-flight subprocesses so we can check their exit code in later
    # cycles. Key = module name, value = (Popen, fire_time, log_path).
    inflight_supervisors: dict = {}

    def _fire_subprocess(module: str) -> bool:
        """Returns True if subprocess started (state should be persisted)."""
        log_dir = settings.runtime_data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"scheduler_{module.split('.')[-1]}.log"
        try:
            log_fh = log_path.open("a", encoding="utf-8", buffering=1)
            log_fh.write(f"\n=== {_dt.now(_tz.utc).isoformat()} START ===\n")
            popen_kwargs: dict = {
                "stdout": log_fh,
                "stderr": subprocess.STDOUT,
                "cwd": os.getcwd(),
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
                )
            proc = subprocess.Popen([sys.executable, "-m", module], **popen_kwargs)
            inflight_supervisors[module] = (proc, _dt.now(_tz.utc), log_path, log_fh)
            logger.info("scheduler fired", extra={"extra": {"module": module, "log": str(log_path)}})
            return True
        except Exception:
            logger.exception("scheduler fire failed", extra={"extra": {"module": module}})
            return False

    def _check_inflight_supervisors() -> dict:
        """Poll any running subprocesses; log result and return modules that
        completed FAILED (caller may want to rollback their timestamp)."""
        failed: list[str] = []
        completed: list[str] = []
        for module, (proc, started, log_path, fh) in list(inflight_supervisors.items()):
            rc = proc.poll()
            if rc is None:
                # Still running. Kill if it hangs more than 5 min.
                if (_dt.now(_tz.utc) - started).total_seconds() > 300:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    failed.append(module)
                    completed.append(module)
                    logger.warning("scheduler subprocess timeout, killed",
                                   extra={"extra": {"module": module, "log": str(log_path)}})
                continue
            completed.append(module)
            try:
                fh.write(f"=== END rc={rc} ===\n")
                fh.close()
            except Exception:
                pass
            if rc == 0:
                logger.info("scheduler subprocess ok",
                            extra={"extra": {"module": module, "rc": rc, "log": str(log_path)}})
            else:
                failed.append(module)
                logger.warning("scheduler subprocess FAILED",
                               extra={"extra": {"module": module, "rc": rc, "log": str(log_path)}})
        for m in completed:
            inflight_supervisors.pop(m, None)
        return {"failed": failed}

    # Hot-reload of .env so auto_tuner changes apply without restarting bot.
    # Tracks the .env mtime; when it changes, re-parses tunable params and
    # patches them into settings + risk.config (RiskConfig is frozen so we
    # rebuild it). Only params known to auto_tuner's TUNABLE_PARAMS list,
    # nothing else. Race-safe: file write is atomic via tmp+rename.
    from dataclasses import replace as _dc_replace

    TUNABLE_FLOAT_KEYS = {
        "MIN_TRADE_IMBALANCE_BUY": "min_trade_imbalance_buy",
        "MIN_BOOK_IMBALANCE_BUY": "min_book_imbalance_buy",
        "MAX_BOOK_SPREAD_PCT": "max_book_spread_pct",
        "MIN_VOLUME_RATIO": "min_volume_ratio",
        "MIN_SUPER_VOLUME": "min_super_volume",
        "STOP_LOSS_PCT": "stop_loss_pct",
        "TAKE_PROFIT_PCT": "take_profit_pct",
        "REVERSAL_EXIT_MIN_PNL_PCT": "reversal_exit_min_pnl_pct",
        "TRAIL_STOP_PCT": "trail_stop_pct",
        "ML_MIN_UP_PROBABILITY": "ml_min_up_probability",
        "ORDER_CASH_PCT": "order_cash_pct",
        "ADD_POSITION_MIN_CONFIDENCE": "add_position_min_confidence",
        "ADD_POSITION_MIN_TRADE_IMBALANCE": "add_position_min_trade_imbalance",
        "ADD_POSITION_MIN_PNL_PCT": "add_position_min_pnl_pct",
        "ADD_POSITION_MAX_POSITION_PCT": "add_position_max_position_pct",
        "MAX_POSITION_PCT": "max_position_pct",
        "MAX_PORTFOLIO_EXPOSURE_PCT": "max_portfolio_exposure_pct",
        "CONVICTION_GAIN": "conviction_gain",
        "CONVICTION_MAX_MULT": "conviction_max_mult",
        # MAX_DAILY_LOSS_PCT not exposed — see comment in auto_tuner.py
        # Short-side knobs. RiskConfig owns exposure/size; the rest live on
        # settings and are read directly at signal time.
        "SHORT_ORDER_CASH_PCT": "short_order_cash_pct",
        "SHORT_MAX_TOTAL_EXPOSURE_PCT": "short_max_total_exposure_pct",
    }
    # Short knobs that live on `settings`, not on RiskConfig.
    TUNABLE_SETTINGS_FLOAT_KEYS = {
        "SHORT_STOP_LOSS_PCT": "short_stop_loss_pct",
        "SHORT_TAKE_PROFIT_PCT": "short_take_profit_pct",
        "SHORT_TRAIL_STOP_PCT": "short_trail_stop_pct",
        "SHORT_TIME_EXIT_HOURS": "short_time_exit_hours",
        "ML_MIN_DOWN_PROBABILITY": "ml_min_down_probability",
    }
    TUNABLE_INT_KEYS = {
        "SYMBOL_COOLDOWN_MINUTES": "symbol_cooldown_minutes",
        "MAX_DAILY_TRADES": "max_daily_trades",
    }
    _env_path_for_reload = os.path.join(os.getcwd(), ".env")
    _last_env_mtime = {"ts": 0.0}

    def _maybe_reload_env(risk_mgr) -> RiskManager:
        try:
            mtime = os.path.getmtime(_env_path_for_reload)
        except OSError:
            return risk_mgr
        if mtime <= _last_env_mtime["ts"]:
            return risk_mgr
        # First call seeds the timestamp without reloading.
        if _last_env_mtime["ts"] == 0.0:
            _last_env_mtime["ts"] = mtime
            return risk_mgr
        _last_env_mtime["ts"] = mtime
        try:
            kv: dict[str, str] = {}
            with open(_env_path_for_reload, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    kv[k.strip()] = v.strip()
            risk_kwargs: dict = {}
            updated_keys: list[str] = []
            for env_key, attr in TUNABLE_FLOAT_KEYS.items():
                if env_key in kv:
                    try:
                        new_val = float(kv[env_key])
                    except ValueError:
                        continue
                    if getattr(risk_mgr.config, attr, None) != new_val:
                        risk_kwargs[attr] = new_val
                        updated_keys.append(f"{env_key}={new_val}")
            for env_key, attr in TUNABLE_INT_KEYS.items():
                if env_key in kv:
                    try:
                        new_val = int(float(kv[env_key]))
                    except ValueError:
                        continue
                    if getattr(risk_mgr.config, attr, None) != new_val:
                        risk_kwargs[attr] = new_val
                        updated_keys.append(f"{env_key}={new_val}")
            # Short knobs live on `settings` (not frozen) — patch in place so
            # the next signal cycle picks them up without a restart.
            for env_key, attr in TUNABLE_SETTINGS_FLOAT_KEYS.items():
                if env_key in kv:
                    try:
                        new_val = float(kv[env_key])
                    except ValueError:
                        continue
                    if getattr(settings, attr, None) != new_val:
                        setattr(settings, attr, new_val)
                        updated_keys.append(f"{env_key}={new_val}")
            if not risk_kwargs:
                if updated_keys:
                    logger.info("env hot-reload applied (settings only)",
                                extra={"extra": {"changes": updated_keys}})
                return risk_mgr
            new_config = _dc_replace(risk_mgr.config, **risk_kwargs)
            logger.info("env hot-reload applied", extra={"extra": {"changes": updated_keys}})
            return RiskManager(new_config)
        except Exception:
            logger.exception("env hot-reload failed")
            return risk_mgr

    def _persist_scheduler_state() -> None:
        store.write_state("scheduler", {
            "last_mid_review": last_mid_review_fire.isoformat(),
            "last_auto_tuner": last_auto_tuner_fire.isoformat(),
        })

    # Singleton guard: two bots on one ArenaGo portfolio = double/conflicting
    # orders. Each cycle writes a heartbeat (pid+ts). On startup we refuse only
    # if the heartbeat is FRESH (<90s) AND that pid is actually still running —
    # so a stopped previous instance never blocks a restart (the friction the
    # time-only check caused). Override: SINGLETON_DISABLE=1.
    def _pid_alive(pid: int) -> bool:
        if not pid or pid <= 0:
            return False
        try:
            if os.name == "nt":
                import ctypes
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                STILL_ACTIVE = 259
                k = ctypes.windll.kernel32
                h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
                if not h:
                    return False
                code = ctypes.c_ulong()
                ok = k.GetExitCodeProcess(h, ctypes.byref(code))
                k.CloseHandle(h)
                return bool(ok) and code.value == STILL_ACTIVE
            os.kill(int(pid), 0)
            return True
        except (OSError, ProcessLookupError):
            return False
        except Exception:
            return True  # unknown -> assume alive (safer than risking a double-run)

    if os.getenv("SINGLETON_DISABLE", "").lower() not in {"1", "true", "yes"}:
        _hb = store.read_state("bot_heartbeat") or {}
        _hb_ts = _hb.get("ts")
        _hb_pid = _hb.get("pid")
        if _hb_ts and _hb_pid != os.getpid():
            try:
                _ts = _dt.fromisoformat(str(_hb_ts).replace("Z", "+00:00"))
                if _ts.tzinfo is None:
                    _ts = _ts.replace(tzinfo=_tz.utc)
                _age = (_dt.now(_tz.utc) - _ts).total_seconds()
            except Exception:
                _age = 1e9
            if _age < 90 and _pid_alive(_hb_pid):
                logger.error(
                    "another moex_agent instance is LIVE (heartbeat %.0fs old, pid=%s, "
                    "process running) — refusing to start to avoid double-trading one "
                    "portfolio. Stop it first, or set SINGLETON_DISABLE=1 to override.",
                    _age, _hb_pid,
                )
                raise SystemExit(1)

    cycle = 0
    sleep_logged = False
    while True:
        cycle += 1
        now_utc_check = _dt.now(_tz.utc)
        # Heartbeat for the singleton guard (and external liveness checks).
        store.write_state("bot_heartbeat", {"pid": os.getpid(), "ts": now_utc_check.isoformat()})
        market_active_now = _is_moex_active(now_utc_check)
        if not market_active_now and not args.once:
            # Market closed — skip the full cycle to save MOEX API calls and
            # LLM tokens. We just idle and re-check periodically.
            if not sleep_logged:
                logger.info(
                    "market closed — entering idle mode",
                    extra={"extra": {"weekday": now_utc_check.weekday(), "utc_hour": now_utc_check.hour}},
                )
                sleep_logged = True
            time.sleep(300)  # check every 5 minutes
            continue
        if sleep_logged and market_active_now:
            logger.info("market opened — resuming trading cycles")
            sleep_logged = False
        # Hot-reload .env if auto_tuner/mid_review rewrote it. Rebuilds the
        # RiskManager in-place. ML threshold (ml_min_up_probability) is now a
        # RiskConfig field, so it reloads live too — the buy gate reads
        # risk.config.ml_min_up_probability, not the startup-cached settings.
        risk = _maybe_reload_env(risk)
        try:
            reset_cache = getattr(market_data, "reset_cycle_cache", None)
            if callable(reset_cache):
                reset_cache()
            run_once(
                client,
                market_data,
                risk,
                store,
                memory,
                paper_ledger,
                ml_data,
                ml_filter,
                short_ml_filter=short_ml_filter,
                brief_service=brief_service,
                explainer=explainer,
                news_service=news_service,
                memory_index=memory_index,
                order_quota=order_quota,
                macro_cache=macro_cache,
            )
        except Exception:
            logger.exception("cycle failed", extra={"extra": {"cycle": cycle}})

        # Periodic LLM supervisors. Fired only after the bot has produced some
        # data; mid_review needs recent trades, auto_tuner needs a full day.
        # Both gated by MOEX trading hours — no point burning tokens on a
        # market that isn't moving.
        # Each cycle: collect exit codes of in-flight supervisors. If one
        # failed, roll back its timestamp by 23h so the next cycle retries
        # rather than waiting 24h. The fail log is in data/logs/scheduler_*.log.
        if inflight_supervisors:
            check_result = _check_inflight_supervisors()
            if "moex_agent.mid_review" in check_result["failed"]:
                last_mid_review_fire -= _td(hours=2, minutes=50)
                _persist_scheduler_state()
                logger.warning("rolling back mid_review timestamp for retry")
            if "moex_agent.auto_tuner" in check_result["failed"]:
                last_auto_tuner_fire -= _td(hours=23, minutes=30)
                _persist_scheduler_state()
                logger.warning("rolling back auto_tuner timestamp for retry")

        if cycle >= MIN_CYCLES_BEFORE_SCHEDULING:
            now_utc = _dt.now(_tz.utc)
            market_active = _is_moex_active(now_utc)
            # Debug logging every 5 cycles to see exactly why scheduler decides
            # what it decides — silences false-alarm "scheduler is broken" hunts.
            if cycle % 5 == 0:
                logger.info("scheduler check", extra={"extra": {
                    "cycle": cycle,
                    "market_active": market_active,
                    "since_mid_review_min": round((now_utc - last_mid_review_fire).total_seconds() / 60, 1),
                    "since_auto_tuner_min": round((now_utc - last_auto_tuner_fire).total_seconds() / 60, 1),
                    "mid_review_interval_min": MID_REVIEW_INTERVAL.total_seconds() / 60,
                    "auto_tuner_interval_min": AUTO_TUNER_INTERVAL.total_seconds() / 60,
                    "inflight": list(inflight_supervisors.keys()),
                }})
            if market_active:
                # Account-wide recent-trade read (reuses this cycle's cache).
                recent_n, recent_avg = _recent_account_stats()
                bleeding = recent_n >= BLEED_MIN_TRADES and recent_avg is not None and recent_avg < 0.0
                strong_bleed = (
                    recent_n >= AUTO_TUNER_BLEED_TRADES
                    and recent_avg is not None
                    and recent_avg < AUTO_TUNER_BLEED_AVG_PNL
                )
                # Regime-flip trigger: when the brief regime changes (e.g.
                # normal -> risk_off), re-assess the session mode IMMEDIATELY
                # instead of waiting for the 45-min timer — the regime is the
                # single most important thing to adapt to fast. Cheap (Sonnet),
                # throttled by the same event cooldown. auto_tuner stays the sole
                # parameter writer (no two-writer race); it only event-fires on
                # bleeds, so a flip nudges the mode, not the params.
                # Read the regime from the persisted brief state — `brief` is
                # local to run_once() and NOT in scope here in main()'s loop.
                # (Referencing it directly was a NameError that crashed the loop
                # after MIN_CYCLES; --once never reaches this block so it slipped
                # past the smoke test.)
                _brief_state = store.read_state("llm_brief") or {}
                _regime_now = str(_brief_state.get("market_regime") or "normal").lower()
                _prev_regime = store.read_state("last_seen_regime")
                regime_flipped = _prev_regime is not None and _prev_regime != _regime_now
                if _prev_regime != _regime_now:
                    store.write_state("last_seen_regime", _regime_now)

                mid_review_interval_due = now_utc - last_mid_review_fire >= MID_REVIEW_INTERVAL
                mid_review_event_due = (
                    (bleeding or regime_flipped)
                    and now_utc - last_mid_review_fire >= MID_REVIEW_EVENT_COOLDOWN
                )
                if (
                    (mid_review_interval_due or mid_review_event_due)
                    and "moex_agent.mid_review" not in inflight_supervisors
                ):
                    last_mid_review_fire = now_utc
                    if mid_review_event_due and not mid_review_interval_due:
                        logger.info(
                            "mid_review fired EARLY",
                            extra={"extra": {"trigger": "regime_flip" if regime_flipped else "losing_streak",
                                             "regime": _regime_now, "prev_regime": _prev_regime,
                                             "recent_trades": recent_n,
                                             "recent_avg_pnl": round(recent_avg, 4) if recent_avg is not None else None}},
                        )
                    if _fire_subprocess("moex_agent.mid_review"):
                        _persist_scheduler_state()
                # The digest reads orders.jsonl (not paper_ledger.json, which
                # carries stale lifetime trades) so tuning acts on real numbers.
                auto_tuner_slot_due = _auto_tuner_due(now_utc, last_auto_tuner_fire, AUTO_TUNER_UTC_HOURS)
                auto_tuner_event_due = (
                    strong_bleed and now_utc - last_auto_tuner_fire >= AUTO_TUNER_EVENT_COOLDOWN
                )
                if (
                    (auto_tuner_slot_due or auto_tuner_event_due)
                    and "moex_agent.auto_tuner" not in inflight_supervisors
                ):
                    last_auto_tuner_fire = now_utc
                    if auto_tuner_event_due and not auto_tuner_slot_due:
                        logger.info(
                            "auto_tuner fired EARLY on strong bleed",
                            extra={"extra": {"recent_trades": recent_n, "recent_avg_pnl": round(recent_avg, 4)}},
                        )
                    if _fire_subprocess("moex_agent.auto_tuner"):
                        _persist_scheduler_state()

        if args.once:
            break
        if settings.max_cycles and cycle >= settings.max_cycles:
            logger.info("max_cycles reached", extra={"extra": {"cycle": cycle}})
            break
        time.sleep(settings.poll_interval_seconds)


if __name__ == "__main__":
    main()
