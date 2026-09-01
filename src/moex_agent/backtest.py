"""Historical backtester.

Replays the live decision pipeline against rows in `ml_dataset_offline.jsonl`,
producing a P&L curve and metrics. NOT a live trading replacement — purely an
offline evaluation tool.

Pipeline (matches `app.py` exactly):
    feature row → exit_signal? → generate_signal → ML filter (optional) →
    PnL-aware SELL override → risk.evaluate → simulated fill

Inputs:
    --dataset    path to the JSONL dataset
    --start/--end date range (inclusive)
    --symbols    comma-separated tickers or 'all'
    --ml-model   optional path to a trained LightGBM model (.joblib)
    --threshold  ml_min_up_probability override
    --use-trailing-stop / --trail-pct  enable trailing stop
    --use-per-symbol-stops             enable ATR-based per-symbol stops
    --output     where to write JSON report

Output (JSON):
    metrics: total_return, sharpe, max_drawdown, win_rate, n_trades, etc.
    per_symbol: same metrics per ticker
    equity_curve: downsampled (one point per day)
    trades_sample: last 100 trades

This module is independent of live trading — it does not fetch anything from
ALGOPACK; everything is replayed from the dataset built by ml_offline_dataset.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from moex_agent.config import settings
from moex_agent.models import (
    Action,
    MarketFeatures,
    MarketRegime,
    OrderSide,
    Portfolio,
    Position,
    Signal,
)
from moex_agent.risk import RiskConfig, RiskManager
from moex_agent.strategy import (
    generate_cover_signal,
    generate_exit_signal,
    generate_short_signal,
    generate_signal,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    starting_cash: float = 1_000_000.0
    # Slippage and commission as fraction (5 bps = 0.0005). Defaults are
    # conservative for MOEX shares (commission ~5 bps, slippage ~5 bps for
    # limit-near-market).
    slippage_bps: float = 5.0
    commission_bps: float = 5.0

    # Strategy/risk parameters — mirror settings defaults but overridable.
    order_cash_pct: float = 0.03
    min_cash_pct: float = 0.10
    max_position_pct: float = 0.15
    max_portfolio_exposure_pct: float = 0.75
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.08
    max_daily_trades: int = 80
    symbol_cooldown_minutes: int = 30
    min_volume_ratio: float = 0.30
    min_trade_imbalance_buy: float = -0.15
    min_super_volume: float = 1000.0
    min_book_imbalance_buy: float = -0.15
    max_book_spread_pct: float = 0.01
    add_position_min_confidence: float = 0.64
    add_position_min_trade_imbalance: float = 0.20
    add_position_min_pnl_pct: float = 0.0
    add_position_max_position_pct: float = 0.10

    # Exit logic.
    stop_loss_pct: float = 0.012
    take_profit_pct: float = 0.025
    reversal_exit_min_pnl_pct: float = -0.005

    # Time-based exit (force-close stuck positions). 0 = off.
    time_exit_hours: float = 0.0
    time_exit_min_profit_pct: float = 0.003

    # Trailing stop (new).
    trail_stop_pct: float | None = None  # e.g. 0.008 = exit if 0.8% below peak
    macro_block_return_4h: float = 0.0   # <0 blocks new longs when IMOEX 4h below it

    # Per-symbol stops (new). When True, use ATR-based stop per ticker.
    use_per_symbol_stops: bool = False
    atr_stop_multiplier: float = 2.0    # stop = K * median_atr
    atr_stop_floor: float = 0.006       # never below 0.6%
    atr_stop_ceiling: float = 0.025     # never above 2.5%

    # ML filter.
    use_ml_filter: bool = False
    ml_model_path: Path | None = None
    ml_min_up_probability: float = 0.65
    # Гейт по волатильности на вход в лонг. Издержки — фиксированный налог,
    # и бумага с малым ATR физически не успевает его отбить. 0 = выключен.
    min_atr_pct: float = 0.0

    # ── Short side ──────────────────────────────────────────────────────────
    # Off by default so existing long-only runs stay byte-for-byte comparable.
    enable_shorts: bool = False
    short_order_cash_pct: float = 0.02
    max_concurrent_shorts: int = 6
    short_max_total_exposure_pct: float = 0.06
    short_stop_loss_pct: float = 0.010
    short_take_profit_pct: float = 0.018
    # Trailing / time exit for shorts (0 = off) — the point of the exercise is
    # to measure whether they help, so they are separate knobs.
    short_trail_stop_pct: float = 0.0
    short_time_exit_hours: float = 0.0
    short_time_exit_min_profit_pct: float = 0.003
    # Short ML filter (P(down)).
    use_short_ml_filter: bool = False
    short_ml_model_path: Path | None = None
    ml_min_down_probability: float = 0.60


# --------------------------------------------------------------------------
# Dataset loader
# --------------------------------------------------------------------------

def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace(" ", "T"))


def _load_rows(
    path: Path,
    *,
    start_date: date | None,
    end_date: date | None,
    symbols: set[str] | None,
) -> list[dict[str, Any]]:
    """Stream-load JSONL rows in date range, return sorted by ts."""
    rows: list[dict[str, Any]] = []
    start_dt = (
        datetime.combine(start_date, datetime.min.time()) if start_date else None
    )
    end_dt = (
        datetime.combine(end_date, datetime.max.time()) if end_date else None
    )
    n_total = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            sym = str(row.get("symbol") or "")
            if symbols and sym not in symbols:
                continue
            ts_raw = row.get("feature_ts")
            if not ts_raw:
                continue
            try:
                ts = _parse_ts(ts_raw)
            except (ValueError, TypeError):
                continue
            if start_dt and ts < start_dt:
                continue
            if end_dt and ts > end_dt:
                continue
            rows.append({
                "symbol": sym,
                "ts": ts,
                "features": row.get("features") or {},
                "regime": str(row.get("regime") or "normal"),
                "signal_action": str(row.get("signal_action") or "hold"),
                "signal_confidence": float(row.get("signal_confidence") or 0.5),
                "signal_reason": str(row.get("signal_reason") or ""),
            })
    rows.sort(key=lambda r: (r["ts"], r["symbol"]))
    logger.info("loaded %d rows (of %d total) in date window", len(rows), n_total)
    return rows


def _market_features_from_row(row: dict[str, Any]) -> MarketFeatures:
    f = row["features"]
    def _opt(name: str) -> float | None:
        v = f.get(name)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return MarketFeatures(
        symbol=row["symbol"],
        feature_ts=row["ts"],
        price=float(f.get("price") or 0.0),
        rsi=_opt("rsi"),
        macd=_opt("macd"),
        macd_signal=_opt("macd_signal"),
        bollinger_low=_opt("bollinger_low"),
        bollinger_mid=_opt("bollinger_mid"),
        bollinger_high=_opt("bollinger_high"),
        atr_pct=_opt("atr_pct"),
        volume_ratio=_opt("volume_ratio"),
        trade_imbalance=_opt("trade_imbalance"),
        order_imbalance=_opt("order_imbalance"),
        book_imbalance=_opt("book_imbalance"),
        book_spread_pct=_opt("book_spread_pct"),
        super_volume=_opt("super_volume"),
        imoex_return_60m=_opt("imoex_return_60m"),
        imoex_return_4h=_opt("imoex_return_4h"),
        sector_return_60m=_opt("sector_return_60m"),
        corr_price_imoex_60=_opt("corr_price_imoex_60"),
        rel_strength_vs_imoex=_opt("rel_strength_vs_imoex"),
    )


# --------------------------------------------------------------------------
# Per-symbol stops via ATR
# --------------------------------------------------------------------------

def _compute_per_symbol_atr_stops(
    rows: list[dict[str, Any]],
    *,
    multiplier: float,
    floor: float,
    ceiling: float,
) -> dict[str, float]:
    """Per-ticker ATR-based stop. Uses median atr_pct from the dataset.

    The intuition: a ticker whose ATR is typically 0.3% needs a tighter stop
    than one with 1.0% ATR. Fixed 1.2% stop is right for the latter but
    invites random-noise exits on the former (or vice versa for noisy tickers
    like VTBR).
    """
    atrs: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        sym = row["symbol"]
        atr = row["features"].get("atr_pct")
        if atr is None:
            continue
        try:
            atr_f = float(atr)
        except (TypeError, ValueError):
            continue
        if 0 < atr_f < 0.10:  # sanity: discard outliers > 10%
            atrs[sym].append(atr_f)
    stops: dict[str, float] = {}
    for sym, values in atrs.items():
        if not values:
            continue
        median = statistics.median(values)
        stop = max(floor, min(ceiling, multiplier * median))
        stops[sym] = stop
    return stops


# --------------------------------------------------------------------------
# Simulation harness
# --------------------------------------------------------------------------

@dataclass
class TradeRecord:
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime | None = None
    exit_price: float | None = None
    symbol: str = ""
    qty: int = 0
    pnl_rub: float = 0.0
    pnl_pct: float = 0.0
    entry_reason: str = ""
    exit_reason: str = ""
    direction: str = "long"  # "long" | "short" — for per-side metrics
    # Сквозной id цепочки: вход и все его частичные выходы делят один id.
    # Без него позиционные метрики из трейд-лога не собрать — выходы
    # выглядят независимыми сделками.
    position_id: int = 0
    equity_at_entry: float = 0.0
    n_open_positions_at_entry: int = 0


def _build_risk(config: BacktestConfig) -> RiskManager:
    return RiskManager(RiskConfig(
        max_position_pct=config.max_position_pct,
        max_portfolio_exposure_pct=config.max_portfolio_exposure_pct,
        max_daily_loss_pct=config.max_daily_loss_pct,
        max_drawdown_pct=config.max_drawdown_pct,
        order_cash_pct=config.order_cash_pct,
        min_cash_pct=config.min_cash_pct,
        max_daily_trades=config.max_daily_trades,
        symbol_cooldown_minutes=config.symbol_cooldown_minutes,
        min_volume_ratio=config.min_volume_ratio,
        min_trade_imbalance_buy=config.min_trade_imbalance_buy,
        min_super_volume=config.min_super_volume,
        min_book_imbalance_buy=config.min_book_imbalance_buy,
        max_book_spread_pct=config.max_book_spread_pct,
        add_position_min_confidence=config.add_position_min_confidence,
        add_position_min_trade_imbalance=config.add_position_min_trade_imbalance,
        add_position_min_pnl_pct=config.add_position_min_pnl_pct,
        add_position_max_position_pct=config.add_position_max_position_pct,
        enable_shorts=config.enable_shorts,
        short_order_cash_pct=config.short_order_cash_pct,
        max_concurrent_shorts=config.max_concurrent_shorts,
        short_max_total_exposure_pct=config.short_max_total_exposure_pct,
    ))


def run_backtest(
    config: BacktestConfig,
    rows: list[dict[str, Any]],
    *,
    ml_filter: Any = None,
    short_ml_filter: Any = None,
    per_symbol_stops: dict[str, float] | None = None,
) -> dict[str, Any]:
    portfolio = Portfolio(cash=config.starting_cash, positions={})
    risk = _build_risk(config)

    last_trade_at: dict[str, datetime] = {}
    daily_trade_count = 0
    current_day: date | None = None
    peak_prices: dict[str, float] = {}  # for trailing stop
    short_peak_pnl: dict[str, float] = {}  # best PnL reached by an open short
    position_opened_at_bt: dict[str, datetime] = {}  # for time-based exit

    open_positions: dict[str, TradeRecord] = {}  # one per symbol
    closed_trades: list[TradeRecord] = []
    blocked_reasons: Counter[str] = Counter()

    total_buy_vol: float = 0.0   # оборот: сумма всех BUY-заявок в руб
    total_sell_vol: float = 0.0  # оборот: сумма всех SELL-заявок в руб

    equity_curve: list[dict[str, Any]] = []
    last_eq_day: date | None = None

    position_seq = 0
    slip = config.slippage_bps / 10000.0
    comm = config.commission_bps / 10000.0

    for row in rows:
        symbol = row["symbol"]
        ts = row["ts"]
        features = _market_features_from_row(row)
        if features.price <= 0:
            continue

        try:
            regime = MarketRegime(row["regime"])
        except ValueError:
            regime = MarketRegime.NORMAL

        # Reset daily trade count when day rolls over.
        day = ts.date()
        if current_day is None or day != current_day:
            daily_trade_count = 0
            current_day = day

        # Mark-to-market the current symbol's position.
        if symbol in portfolio.positions:
            existing = portfolio.positions[symbol]
            portfolio.positions[symbol] = Position(
                symbol=symbol, qty=existing.qty,
                avg_price=existing.avg_price, market_price=features.price,
            )
            # Update peak for trailing stop.
            if config.trail_stop_pct is not None:
                peak_prices[symbol] = max(
                    peak_prices.get(symbol, features.price),
                    features.price,
                )

        # Resolve per-symbol stop.
        if config.use_per_symbol_stops and per_symbol_stops:
            stop_for_symbol = per_symbol_stops.get(symbol, config.stop_loss_pct)
        else:
            stop_for_symbol = config.stop_loss_pct

        _pos_now = portfolio.positions.get(symbol)
        if _pos_now is not None and _pos_now.qty < 0:
            # ── OPEN SHORT: managed only by cover signals (mirrors app.py) ──
            _cur_pnl = (
                (_pos_now.avg_price - features.price) / _pos_now.avg_price
                if _pos_now.avg_price > 0 else 0.0
            )
            _peak = max(short_peak_pnl.get(symbol, _cur_pnl), _cur_pnl)
            short_peak_pnl[symbol] = _peak
            exit_signal = generate_cover_signal(
                features, regime, _pos_now,
                stop_loss_pct=config.short_stop_loss_pct,
                take_profit_pct=config.short_take_profit_pct,
                now=ts,
                position_opened_at=position_opened_at_bt.get(symbol),
                time_exit_hours=config.short_time_exit_hours,
                time_exit_min_profit_pct=config.short_time_exit_min_profit_pct,
                peak_pnl_pct=_peak,
                trail_stop_pct=config.short_trail_stop_pct,
            )
            signal = exit_signal or Signal(symbol, Action.HOLD, 0.5, regime, "holding short", features)
        else:
            # Compute exit signal (built-in stop/take/reversal/time-based).
            exit_signal = generate_exit_signal(
                features, regime, _pos_now,
                stop_loss_pct=stop_for_symbol,
                take_profit_pct=config.take_profit_pct,
                reversal_exit_min_pnl_pct=config.reversal_exit_min_pnl_pct,
                now=ts,
                position_opened_at=position_opened_at_bt.get(symbol),
                time_exit_hours=config.time_exit_hours,
                time_exit_min_profit_pct=config.time_exit_min_profit_pct,
            )

            # Trailing stop (separate, kicks in if not already exiting).
            if (
                exit_signal is None
                and config.trail_stop_pct is not None
                and symbol in portfolio.positions
            ):
                peak = peak_prices.get(symbol, features.price)
                if peak > 0:
                    drawdown_from_peak = (peak - features.price) / peak
                    if drawdown_from_peak > config.trail_stop_pct:
                        exit_signal = Signal(
                            symbol,
                            Action.SELL,
                            0.75,
                            regime,
                            f"trailing stop {drawdown_from_peak:.3%} below peak {peak:.2f}",
                            features,
                        )

            signal = exit_signal or generate_signal(features, regime)

        # Macro trend gate (same as app.py): block new longs in a strong 4h
        # IMOEX downtrend. Shorts/covers pass.
        if (
            config.macro_block_return_4h < 0
            and signal.action == Action.BUY
            and "[cover]" not in signal.reason
            and features.imoex_return_4h is not None
            and features.imoex_return_4h < config.macro_block_return_4h
        ):
            signal = Signal(symbol, Action.HOLD, min(signal.confidence, 0.5), regime,
                            f"macro downtrend blocks long; {signal.reason}", features)

        # PnL-aware SELL override — same as app.py.
        position_for_pnl = portfolio.positions.get(symbol)
        if (
            exit_signal is None
            and signal.action == Action.SELL
            and position_for_pnl is not None
            and position_for_pnl.qty > 0
            and position_for_pnl.avg_price > 0
        ):
            pnl_pct = (features.price - position_for_pnl.avg_price) / position_for_pnl.avg_price
            if pnl_pct < 0.001:
                signal = Signal(
                    symbol, Action.HOLD, 0.5, regime,
                    f"sell suppressed: pnl {pnl_pct:.4%}; {signal.reason}",
                    signal.features,
                )

        # ── Short entry: flat, shorts enabled, and no long BUY this cycle ────
        if (
            config.enable_shorts
            and exit_signal is None
            and portfolio.positions.get(symbol) is None
            and signal.action != Action.BUY
        ):
            short_sig = generate_short_signal(features, regime)
            if short_sig.action == Action.SELL:
                if config.use_short_ml_filter and short_ml_filter is not None:
                    p_down = short_ml_filter.predict_up_probability(features)  # positive class = "down"
                    if p_down < config.ml_min_down_probability:
                        short_sig = Signal(
                            symbol, Action.HOLD, min(short_sig.confidence, 0.5), regime,
                            f"short ml filter blocked: P(down)={p_down:.3f}; {short_sig.reason}", features,
                        )
                    else:
                        short_sig = Signal(
                            symbol, short_sig.action,
                            min(0.95, max(short_sig.confidence, p_down)), regime,
                            f"{short_sig.reason}; ml P(down)={p_down:.3f}", features,
                        )
                if short_sig.action == Action.SELL:
                    signal = short_sig

        # Early-skip SELL with no position (a tagged short entry opens one).
        if (
            signal.action == Action.SELL
            and portfolio.positions.get(symbol) is None
            and "[short-entry]" not in signal.reason
        ):
            continue

        # Volatility gate on long entries. Covers must never be blocked.
        if (
            config.min_atr_pct > 0.0
            and signal.action == Action.BUY
            and "[cover]" not in signal.reason
            and (features.atr_pct or 0.0) < config.min_atr_pct
        ):
            signal = Signal(
                symbol, Action.HOLD,
                min(signal.confidence, 0.5), regime,
                f"atr gate blocked: atr={(features.atr_pct or 0.0):.5f}; {signal.reason}",
                signal.features,
            )

        # ML filter on long BUY only — covers close risk and must never be blocked.
        if (
            config.use_ml_filter
            and ml_filter is not None
            and signal.action == Action.BUY
            and "[cover]" not in signal.reason
        ):
            p_up = ml_filter.predict_up_probability(features)
            if p_up < config.ml_min_up_probability:
                signal = Signal(
                    symbol, Action.HOLD,
                    min(signal.confidence, 0.5), regime,
                    f"ml filter blocked: P(up)={p_up:.3f}; {signal.reason}",
                    signal.features,
                )
            else:
                signal = Signal(
                    symbol, signal.action,
                    min(0.95, max(signal.confidence, p_up)),
                    regime,
                    f"{signal.reason}; ml P(up)={p_up:.3f}",
                    signal.features,
                )

        # Risk evaluation.
        approved, order, reason = risk.evaluate(
            signal,
            portfolio,
            daily_trade_count=daily_trade_count,
            last_trade_at=last_trade_at.get(symbol),
            now=ts,
        )
        if not approved:
            for piece in reason.split(";"):
                p = piece.strip()
                if p:
                    blocked_reasons[p] += 1
            continue

        if order is None or signal.action == Action.HOLD:
            continue

        # ── Short entry: SELL that opens a short position ────────────────────
        if order.side == OrderSide.SELL and "[short-entry]" in order.reason:
            fill_price = features.price * (1.0 - slip)
            commission = order.qty * fill_price * comm
            portfolio.apply_fill(OrderSide.SELL, symbol, order.qty, fill_price, allow_short=True)
            portfolio.cash -= commission
            total_sell_vol += order.qty * fill_price
            last_trade_at[symbol] = ts
            daily_trade_count += 1
            position_opened_at_bt[symbol] = ts
            short_peak_pnl[symbol] = 0.0
            position_seq += 1
            open_positions[symbol] = TradeRecord(
                entry_ts=ts,
                entry_price=fill_price,
                symbol=symbol,
                qty=order.qty,
                entry_reason=order.reason[:120],
                direction="short",
                position_id=position_seq,
                equity_at_entry=portfolio.equity,
                n_open_positions_at_entry=len(open_positions),
            )
            if last_eq_day is None or day != last_eq_day:
                equity_curve.append({
                    "date": day.isoformat(), "ts": ts.isoformat(),
                    "equity": round(portfolio.equity, 2), "cash": round(portfolio.cash, 2),
                    "exposure": round(portfolio.exposure, 2), "n_positions": len(portfolio.positions),
                })
                last_eq_day = day
            continue

        # ── Cover: BUY that closes an open short ────────────────────────────
        if order.side == OrderSide.BUY and "[cover]" in order.reason:
            existing = portfolio.positions.get(symbol)
            if existing is None or existing.qty >= 0:
                continue
            fill_price = features.price * (1.0 + slip)
            cover_qty = min(order.qty, -existing.qty)
            commission = cover_qty * fill_price * comm
            portfolio.apply_fill(OrderSide.BUY, symbol, cover_qty, fill_price, allow_short=True)
            portfolio.cash -= commission
            total_buy_vol += cover_qty * fill_price
            last_trade_at[symbol] = ts
            daily_trade_count += 1

            rec = open_positions.get(symbol)
            if rec is not None:
                # Short PnL: profit when the buy-back price is below entry.
                pnl_rub = cover_qty * (rec.entry_price - fill_price) - commission
                pnl_pct = (rec.entry_price - fill_price) / rec.entry_price
                closed_trades.append(TradeRecord(
                    entry_ts=rec.entry_ts, entry_price=rec.entry_price,
                    exit_ts=ts, exit_price=fill_price,
                    symbol=symbol, qty=cover_qty,
                    pnl_rub=pnl_rub, pnl_pct=pnl_pct,
                    entry_reason=rec.entry_reason, exit_reason=order.reason[:120],
                    direction="short",
                    position_id=rec.position_id,
                    equity_at_entry=rec.equity_at_entry,
                    n_open_positions_at_entry=rec.n_open_positions_at_entry,
                ))
                remaining = rec.qty - cover_qty
                if remaining <= 0:
                    open_positions.pop(symbol, None)
                    short_peak_pnl.pop(symbol, None)
                    position_opened_at_bt.pop(symbol, None)
                else:
                    rec.qty = remaining
            if last_eq_day is None or day != last_eq_day:
                equity_curve.append({
                    "date": day.isoformat(), "ts": ts.isoformat(),
                    "equity": round(portfolio.equity, 2), "cash": round(portfolio.cash, 2),
                    "exposure": round(portfolio.exposure, 2), "n_positions": len(portfolio.positions),
                })
                last_eq_day = day
            continue

        # Simulated fill.
        if order.side == OrderSide.BUY:
            fill_price = features.price * (1.0 + slip)
            cost = order.qty * fill_price
            commission = cost * comm
            if portfolio.cash < cost + commission:
                continue
            portfolio.apply_fill(OrderSide.BUY, symbol, order.qty, fill_price)
            portfolio.cash -= commission  # commission on top of fill
            total_buy_vol += order.qty * fill_price
            last_trade_at[symbol] = ts
            daily_trade_count += 1
            peak_prices[symbol] = features.price
            # Record position open time only when transitioning from no
            # position to a new one. Scale-ins don't reset the clock.
            if symbol not in position_opened_at_bt:
                position_opened_at_bt[symbol] = ts
            # Track or update open trade record. If already open, this is an
            # add-on — record as separate trade entry would be cleaner but
            # backward-compat with paper bot: we still track avg_price.
            if symbol not in open_positions:
                position_seq += 1
                open_positions[symbol] = TradeRecord(
                    entry_ts=ts,
                    entry_price=fill_price,
                    symbol=symbol,
                    qty=order.qty,
                    entry_reason=order.reason[:120],
                    position_id=position_seq,
                    equity_at_entry=portfolio.equity,
                    n_open_positions_at_entry=len(open_positions),
                )
            else:
                # Average up
                rec = open_positions[symbol]
                new_qty = rec.qty + order.qty
                rec.entry_price = (rec.entry_price * rec.qty + fill_price * order.qty) / new_qty
                rec.qty = new_qty
        else:
            # SELL — close (or scale) existing position.
            existing = portfolio.positions.get(symbol)
            if existing is None or existing.qty <= 0:
                continue
            fill_price = features.price * (1.0 - slip)
            sell_qty = min(order.qty, existing.qty)
            portfolio.apply_fill(OrderSide.SELL, symbol, sell_qty, fill_price)
            commission = sell_qty * fill_price * comm
            portfolio.cash -= commission
            total_sell_vol += sell_qty * fill_price
            last_trade_at[symbol] = ts
            daily_trade_count += 1

            rec = open_positions.get(symbol)
            if rec is not None:
                # P&L for the sold portion at trade-record avg cost.
                pnl_rub = sell_qty * (fill_price - rec.entry_price) - commission
                pnl_pct = (fill_price - rec.entry_price) / rec.entry_price
                closed = TradeRecord(
                    entry_ts=rec.entry_ts,
                    entry_price=rec.entry_price,
                    exit_ts=ts,
                    exit_price=fill_price,
                    symbol=symbol,
                    qty=sell_qty,
                    pnl_rub=pnl_rub,
                    pnl_pct=pnl_pct,
                    entry_reason=rec.entry_reason,
                    exit_reason=order.reason[:120],
                    position_id=rec.position_id,
                    equity_at_entry=rec.equity_at_entry,
                    n_open_positions_at_entry=rec.n_open_positions_at_entry,
                )
                closed_trades.append(closed)
                # Update remaining open position (scale-out).
                remaining = rec.qty - sell_qty
                if remaining <= 0:
                    open_positions.pop(symbol, None)
                    peak_prices.pop(symbol, None)
                    position_opened_at_bt.pop(symbol, None)
                else:
                    rec.qty = remaining

        # Equity snapshot: once per day (avoid 1.4M points in the curve).
        if last_eq_day is None or day != last_eq_day:
            equity_curve.append({
                "date": day.isoformat(),
                "ts": ts.isoformat(),
                "equity": round(portfolio.equity, 2),
                "cash": round(portfolio.cash, 2),
                "exposure": round(portfolio.exposure, 2),
                "n_positions": len(portfolio.positions),
            })
            last_eq_day = day

    # Final snapshot.
    if equity_curve and equity_curve[-1]["date"] != (current_day.isoformat() if current_day else ""):
        equity_curve.append({
            "date": current_day.isoformat() if current_day else None,
            "equity": round(portfolio.equity, 2),
            "cash": round(portfolio.cash, 2),
            "exposure": round(portfolio.exposure, 2),
            "n_positions": len(portfolio.positions),
        })

    return {
        "closed_trades": closed_trades,
        "total_buy_vol": total_buy_vol,
        "total_sell_vol": total_sell_vol,
        "open_positions_at_end": {
            sym: {
                "qty": pos.qty,
                "avg_price": pos.avg_price,
                "market_price": pos.market_price,
                "unrealized_pnl_pct": (
                    (pos.market_price - pos.avg_price) / pos.avg_price
                    if pos.avg_price > 0 else 0.0
                ),
            }
            for sym, pos in portfolio.positions.items()
        },
        "equity_curve": equity_curve,
        "final_cash": portfolio.cash,
        "final_equity": portfolio.equity,
        "blocked_reasons": dict(blocked_reasons),
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _max_drawdown(equity_curve: list[dict[str, Any]]) -> float:
    peak = -math.inf
    max_dd = 0.0
    for pt in equity_curve:
        v = float(pt["equity"])
        peak = max(peak, v)
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def _annualized_sharpe(equity_curve: list[dict[str, Any]]) -> float | None:
    if len(equity_curve) < 5:
        return None
    daily_eq = [float(pt["equity"]) for pt in equity_curve]
    rets = [daily_eq[i] / daily_eq[i - 1] - 1.0 for i in range(1, len(daily_eq)) if daily_eq[i - 1] > 0]
    if len(rets) < 2:
        return None
    sigma = statistics.stdev(rets)
    if sigma <= 0:
        return None
    mu = statistics.mean(rets)
    # 252 trading days/year for daily-sampled returns.
    return (mu / sigma) * math.sqrt(252)


def compute_metrics(result: dict[str, Any], config: BacktestConfig) -> dict[str, Any]:
    trades: list[TradeRecord] = result["closed_trades"]
    eq = result["equity_curve"]

    starting = config.starting_cash
    final = result["final_equity"]
    total_ret = (final - starting) / starting if starting > 0 else 0.0

    wins = [t for t in trades if t.pnl_rub > 0]
    losses = [t for t in trades if t.pnl_rub <= 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_pnl_pct = statistics.mean([t.pnl_pct for t in trades]) if trades else 0.0
    avg_win_pct = statistics.mean([t.pnl_pct for t in wins]) if wins else 0.0
    avg_loss_pct = statistics.mean([t.pnl_pct for t in losses]) if losses else 0.0
    profit_factor = (
        sum(t.pnl_rub for t in wins) / abs(sum(t.pnl_rub for t in losses))
        if losses and sum(t.pnl_rub for t in losses) != 0
        else None
    )

    # Hold time.
    hold_times = []
    for t in trades:
        if t.exit_ts and t.entry_ts:
            hold_times.append((t.exit_ts - t.entry_ts).total_seconds() / 60.0)
    avg_hold_min = statistics.mean(hold_times) if hold_times else 0.0

    # Оборот
    total_oborot = result.get("total_buy_vol", 0.0) + result.get("total_sell_vol", 0.0)
    n_days = max(len(eq), 1)
    avg_daily_oborot = total_oborot / n_days
    projected_14d_oborot = avg_daily_oborot * 14

    return {
        "starting_cash": starting,
        "final_equity": round(final, 2),
        "total_return_pct": round(total_ret * 100, 4),
        "max_drawdown_pct": round(_max_drawdown(eq) * 100, 4),
        "sharpe_annualized": (
            round(_annualized_sharpe(eq), 4) if _annualized_sharpe(eq) is not None else None
        ),
        "n_trades": len(trades),
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_pnl_pct": round(avg_pnl_pct * 100, 4),
        "avg_win_pct": round(avg_win_pct * 100, 4),
        "avg_loss_pct": round(avg_loss_pct * 100, 4),
        "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "avg_hold_minutes": round(avg_hold_min, 1),
        "total_oborot_rub": round(total_oborot, 0),
        "avg_daily_oborot_rub": round(avg_daily_oborot, 0),
        "projected_14d_oborot_rub": round(projected_14d_oborot, 0),
        "n_blocked_signals_total": sum(result["blocked_reasons"].values()),
        "blocked_reasons_top5": dict(Counter(result["blocked_reasons"]).most_common(5)),
        "by_direction": _direction_metrics(trades),
    }


def _direction_metrics(trades: list[TradeRecord]) -> dict[str, dict[str, Any]]:
    """Per-side breakdown. The whole point of short support: long and short
    must be judged separately, because a healthy long book can hide a short
    book that only loses (and vice versa)."""
    out: dict[str, dict[str, Any]] = {}
    for side in ("long", "short"):
        side_trades = [t for t in trades if (t.direction or "long") == side]
        if not side_trades:
            # Полный набор ключей даже при нуле сделок — иначе любой потребитель
            # (отчёт, тюнер, свип) падает на KeyError ровно в тот момент, когда
            # сторона отключена и сравнивать интереснее всего.
            out[side] = {
                "n_trades": 0, "pnl_rub": 0.0, "win_rate_pct": 0.0,
                "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
                "loss_win_ratio": None, "profit_factor": None, "avg_hold_minutes": 0.0,
            }
            continue
        wins = [t for t in side_trades if t.pnl_rub > 0]
        losses = [t for t in side_trades if t.pnl_rub <= 0]
        avg_win = statistics.mean([t.pnl_pct for t in wins]) if wins else 0.0
        avg_loss = statistics.mean([t.pnl_pct for t in losses]) if losses else 0.0
        holds = [
            (t.exit_ts - t.entry_ts).total_seconds() / 60.0
            for t in side_trades if t.exit_ts and t.entry_ts
        ]
        out[side] = {
            "n_trades": len(side_trades),
            "pnl_rub": round(sum(t.pnl_rub for t in side_trades), 2),
            "win_rate_pct": round(len(wins) / len(side_trades) * 100, 2),
            "avg_win_pct": round(avg_win * 100, 4),
            "avg_loss_pct": round(avg_loss * 100, 4),
            # The number that sank the hackathon run: how many times bigger the
            # average loss is than the average win.
            "loss_win_ratio": round(abs(avg_loss) / avg_win, 2) if avg_win > 0 else None,
            "profit_factor": (
                round(sum(t.pnl_rub for t in wins) / abs(sum(t.pnl_rub for t in losses)), 3)
                if losses and sum(t.pnl_rub for t in losses) != 0 else None
            ),
            "avg_hold_minutes": round(statistics.mean(holds), 1) if holds else 0.0,
        }
    return out


def compute_per_symbol_metrics(result: dict[str, Any]) -> dict[str, dict]:
    trades: list[TradeRecord] = result["closed_trades"]
    by_symbol: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        by_symbol[t.symbol].append(t)
    out: dict[str, dict] = {}
    for sym, sym_trades in by_symbol.items():
        wins = [t for t in sym_trades if t.pnl_rub > 0]
        total_pnl = sum(t.pnl_rub for t in sym_trades)
        out[sym] = {
            "n_trades": len(sym_trades),
            "total_pnl_rub": round(total_pnl, 2),
            "win_rate_pct": round(len(wins) / len(sym_trades) * 100, 2) if sym_trades else 0,
            "avg_pnl_pct": round(statistics.mean([t.pnl_pct for t in sym_trades]) * 100, 4) if sym_trades else 0,
        }
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _default_dataset_path() -> Path:
    return settings.runtime_data_dir / "logs" / "ml_dataset_offline.jsonl"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay trading pipeline on historical dataset.")
    p.add_argument("--dataset", type=Path, default=_default_dataset_path())
    p.add_argument("--symbols", type=str, default="all",
                   help="Comma-separated tickers or 'all'.")
    p.add_argument("--start", type=str, default="",
                   help="YYYY-MM-DD inclusive. Empty = no lower bound.")
    p.add_argument("--end", type=str, default="",
                   help="YYYY-MM-DD inclusive. Empty = no upper bound.")
    p.add_argument("--starting-cash", type=float, default=1_000_000.0)
    p.add_argument("--ml-model", type=Path, default=None,
                   help="Optional path to a LightGBM .joblib model. If set, ML filter is applied.")
    p.add_argument("--ml-threshold", type=float, default=0.65)
    p.add_argument("--stop-loss-pct", type=float, default=0.012)
    p.add_argument("--take-profit-pct", type=float, default=0.025)
    p.add_argument("--trail-stop-pct", type=float, default=None,
                   help="Enable trailing stop at this fraction (e.g. 0.008 = 0.8%%).")
    p.add_argument("--macro-block-return-4h", type=float, default=0.0,
                   help="Block new longs when IMOEX 4h return < this (e.g. -0.015). 0 = off.")
    p.add_argument("--use-per-symbol-stops", action="store_true",
                   help="Use ATR-based per-symbol stops instead of fixed stop_loss_pct.")
    p.add_argument("--time-exit-hours", type=float, default=0.0,
                   help="Force-close positions older than this many hours when pnl is below "
                        "--time-exit-min-profit-pct. 0 = disabled.")
    p.add_argument("--time-exit-min-profit-pct", type=float, default=0.003,
                   help="PnL floor below which the time-based exit fires (default 0.003 = +0.3%%).")
    p.add_argument("--order-cash-pct", type=float, default=None,
                   help="Fraction of equity per order (e.g. 0.05 = 5%%). Default: BacktestConfig default (0.03).")
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--commission-bps", type=float, default=5.0)
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write the JSON report. Default: data/backtests/<timestamp>.json")
    p.add_argument("--name", type=str, default="run",
                   help="Tag for the output filename.")
    # ── Short side ──────────────────────────────────────────────────────────
    p.add_argument("--min-atr-pct", type=float, default=0.0,
                   help="Block long entries when atr_pct is below this (e.g. 0.001915 = median). 0 = off.")
    p.add_argument("--dump-trades", type=Path, default=None,
                   help="Write the full per-exit trade log to this CSV path.")
    p.add_argument("--no-scale-in", action="store_true",
                   help="Disable adding to an open position (scale-in).")
    p.add_argument("--enable-shorts", action="store_true",
                   help="Enable the short side (entry, cover, per-side metrics).")
    p.add_argument("--short-ml-model", type=Path, default=None,
                   help="Path to the short LightGBM filter (P(down)).")
    p.add_argument("--short-ml-threshold", type=float, default=0.60)
    p.add_argument("--short-stop-loss-pct", type=float, default=0.010)
    p.add_argument("--short-take-profit-pct", type=float, default=0.018)
    p.add_argument("--short-trail-stop-pct", type=float, default=0.0,
                   help="Trailing stop for shorts (0 = off).")
    p.add_argument("--short-time-exit-hours", type=float, default=0.0,
                   help="Force-cover a stuck short after N hours (0 = off).")
    p.add_argument("--short-order-cash-pct", type=float, default=0.02)
    p.add_argument("--short-max-exposure-pct", type=float, default=0.06,
                   help="Cap on total gross short exposure as a share of equity.")
    p.add_argument("--short-max-concurrent", type=int, default=6)
    return p.parse_args()


def _dump_trades_csv(trades: list[TradeRecord], path: Path) -> None:
    """Полный трейд-лог: строка на каждый выход.

    position_id связывает вход со всеми его частичными выходами — без него
    позиционные метрики (реальный win rate, размер позиции, время удержания)
    из лога не восстановить: частичные выходы выглядят отдельными сделками.
    """
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "position_id", "symbol", "side", "entry_ts", "exit_ts", "qty",
        "entry_price", "exit_price", "size_rub", "pnl_rub", "pnl_pct",
        "entry_reason", "exit_reason", "equity_at_entry", "n_open_positions_at_entry",
        "hold_minutes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for t in trades:
            hold = (
                (t.exit_ts - t.entry_ts).total_seconds() / 60.0
                if t.exit_ts and t.entry_ts else ""
            )
            writer.writerow([
                t.position_id, t.symbol, t.direction,
                t.entry_ts.isoformat() if t.entry_ts else "",
                t.exit_ts.isoformat() if t.exit_ts else "",
                t.qty,
                round(t.entry_price, 4),
                round(t.exit_price, 4) if t.exit_price else "",
                round(t.qty * t.entry_price, 2),
                round(t.pnl_rub, 2),
                round(t.pnl_pct * 100, 4),
                t.entry_reason, t.exit_reason,
                round(t.equity_at_entry, 2),
                t.n_open_positions_at_entry,
                round(hold, 1) if hold != "" else "",
            ])
    print(f"Trade log saved: {path} ({len(trades)} rows)")


def _resolve_symbols(raw: str) -> set[str] | None:
    raw = raw.strip()
    if raw.lower() == "all":
        return None
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


def _resolve_date(raw: str) -> date | None:
    raw = raw.strip()
    if not raw:
        return None
    return date.fromisoformat(raw)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    config = BacktestConfig(
        starting_cash=args.starting_cash,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        trail_stop_pct=args.trail_stop_pct,
        macro_block_return_4h=args.macro_block_return_4h,
        use_per_symbol_stops=args.use_per_symbol_stops,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        use_ml_filter=args.ml_model is not None,
        ml_model_path=args.ml_model,
        ml_min_up_probability=args.ml_threshold,
        min_atr_pct=args.min_atr_pct,
        time_exit_hours=args.time_exit_hours,
        time_exit_min_profit_pct=args.time_exit_min_profit_pct,
        enable_shorts=args.enable_shorts,
        use_short_ml_filter=args.short_ml_model is not None,
        short_ml_model_path=args.short_ml_model,
        ml_min_down_probability=args.short_ml_threshold,
        short_stop_loss_pct=args.short_stop_loss_pct,
        short_take_profit_pct=args.short_take_profit_pct,
        short_trail_stop_pct=args.short_trail_stop_pct,
        short_time_exit_hours=args.short_time_exit_hours,
        short_order_cash_pct=args.short_order_cash_pct,
        short_max_total_exposure_pct=args.short_max_exposure_pct,
        max_concurrent_shorts=args.short_max_concurrent,
        **({"add_position_min_confidence": 0.99, "add_position_max_position_pct": 0.0}
           if args.no_scale_in else {}),
        **({"order_cash_pct": args.order_cash_pct} if args.order_cash_pct is not None else {}),
    )

    print("=== Backtest configuration ===")
    for k, v in asdict(config).items():
        print(f"  {k}: {v}")
    print()

    rows = _load_rows(
        args.dataset,
        start_date=_resolve_date(args.start),
        end_date=_resolve_date(args.end),
        symbols=_resolve_symbols(args.symbols),
    )
    if not rows:
        raise SystemExit("No rows matched the date/symbol filter. Check --dataset and --start/--end.")

    print(f"Loaded {len(rows):,} rows  |  {rows[0]['ts'].date()} → {rows[-1]['ts'].date()}")

    per_symbol_stops: dict[str, float] | None = None
    if config.use_per_symbol_stops:
        per_symbol_stops = _compute_per_symbol_atr_stops(
            rows,
            multiplier=config.atr_stop_multiplier,
            floor=config.atr_stop_floor,
            ceiling=config.atr_stop_ceiling,
        )
        print("\nPer-symbol ATR-based stops:")
        for sym, stop in sorted(per_symbol_stops.items()):
            print(f"  {sym}: {stop*100:.3f}%")
        print()

    ml_filter = None
    if config.use_ml_filter and config.ml_model_path:
        from moex_agent.ml_filter import MLBuyFilter
        ml_filter = MLBuyFilter.load(Path(config.ml_model_path))
        print(f"ML filter loaded: {config.ml_model_path}, threshold={config.ml_min_up_probability}\n")

    short_ml_filter = None
    if config.use_short_ml_filter and config.short_ml_model_path:
        from moex_agent.ml_filter import MLBuyFilter
        short_ml_filter = MLBuyFilter.load(Path(config.short_ml_model_path), positive_label="down")
        print(f"Short ML filter loaded: {config.short_ml_model_path}, "
              f"threshold={config.ml_min_down_probability}\n")

    print("Running backtest...\n")
    result = run_backtest(config, rows, ml_filter=ml_filter,
                          short_ml_filter=short_ml_filter, per_symbol_stops=per_symbol_stops)
    metrics = compute_metrics(result, config)
    per_symbol = compute_per_symbol_metrics(result)

    print("=== Aggregate metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()
    print("  OBOROT total: " + str(int(metrics["total_oborot_rub"])) + " rub"
          + "  | avg/day: " + str(int(metrics["avg_daily_oborot_rub"])) + " rub"
          + "  | 14d proj: " + str(int(metrics["projected_14d_oborot_rub"])) + " rub")
    print()

    if config.enable_shorts:
        print("=== By direction ===")
        for side, m in metrics["by_direction"].items():
            if not m.get("n_trades"):
                print(f"  {side}: сделок нет")
                continue
            print(f"  {side}: trades={m['n_trades']}  pnl={m['pnl_rub']} RUB"
                  f"  win_rate={m['win_rate_pct']}%"
                  f"  avg_win={m['avg_win_pct']}%  avg_loss={m['avg_loss_pct']}%"
                  f"  loss/win={m['loss_win_ratio']}  PF={m['profit_factor']}"
                  f"  hold={m['avg_hold_minutes']}min")
        print()

    print("=== Per-symbol (top 10 by total P&L) ===")
    ranked = sorted(per_symbol.items(), key=lambda kv: kv[1]["total_pnl_rub"], reverse=True)
    for sym, m in ranked[:10]:
        print("  " + sym.rjust(6) + ": trades=" + str(m["n_trades"]).rjust(3)
              + "  total_pnl=" + str(round(m["total_pnl_rub"], 2)).rjust(10) + " RUB"
              + "  win_rate=" + str(m["win_rate_pct"]).rjust(5) + "%"
              + "  avg=" + str(m["avg_pnl_pct"]).rjust(7) + "%")
    if len(ranked) > 10:
        print("  ... and " + str(len(ranked) - 10) + " more")
    print()

    # Build output
    out_dir = (args.output.parent if args.output else (settings.runtime_data_dir / "backtests"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.output or (out_dir / (args.name + "_" + stamp + ".json"))

    payload = {
        "name": args.name,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "data": {
            "dataset": str(args.dataset),
            "start": str(_resolve_date(args.start)) if args.start else None,
            "end": str(_resolve_date(args.end)) if args.end else None,
            "symbols": args.symbols,
            "rows_count": len(rows),
            "first_ts": rows[0]["ts"].isoformat(),
            "last_ts": rows[-1]["ts"].isoformat(),
        },
        "metrics": metrics,
        "per_symbol": per_symbol,
        "per_symbol_stops": per_symbol_stops,
        "equity_curve": result["equity_curve"],
        "open_positions_at_end": result["open_positions_at_end"],
        "trades_sample_last100": [
            {
                "entry_ts": t.entry_ts.isoformat(),
                "entry_price": round(t.entry_price, 4),
                "exit_ts": t.exit_ts.isoformat() if t.exit_ts else None,
                "exit_price": round(t.exit_price, 4) if t.exit_price else None,
                "symbol": t.symbol,
                "qty": t.qty,
                "pnl_rub": round(t.pnl_rub, 2),
                "pnl_pct": round(t.pnl_pct * 100, 4),
                "entry_reason": t.entry_reason,
                "exit_reason": t.exit_reason,
            }
            for t in result["closed_trades"][-100:]
        ],
        "blocked_reasons": result["blocked_reasons"],
    }

    if args.dump_trades:
        _dump_trades_csv(result["closed_trades"], args.dump_trades)

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Report saved: " + str(out_path))


if __name__ == "__main__":
    main()
