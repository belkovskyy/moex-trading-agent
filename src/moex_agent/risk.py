from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from moex_agent.models import Action, MarketRegime, OrderIntent, OrderSide, Portfolio, Signal
from moex_agent.strategy import SECTOR_MAP


@dataclass(frozen=True)
class RiskConfig:
    max_position_pct: float
    max_portfolio_exposure_pct: float
    max_daily_loss_pct: float
    max_drawdown_pct: float
    order_cash_pct: float
    min_cash_pct: float
    max_daily_trades: int
    symbol_cooldown_minutes: int
    min_volume_ratio: float
    min_trade_imbalance_buy: float
    min_super_volume: float
    min_book_imbalance_buy: float
    max_book_spread_pct: float
    add_position_min_confidence: float
    add_position_min_trade_imbalance: float
    add_position_min_pnl_pct: float
    add_position_max_position_pct: float
    # Conviction-based sizing (defaults inert; live values injected from .env).
    conviction_sizing_enabled: bool = False
    conviction_gain: float = 0.0
    conviction_max_mult: float = 1.0
    # ML buy-filter threshold lives here (not just settings) so the hot-reload
    # path can update it live. The gate in app.py reads risk.config.
    ml_min_up_probability: float = 0.70
    # Exit thresholds live here too: they're in TUNABLE_FLOAT_KEYS so the
    # hot-reload's _dc_replace needs real fields, the overbought-sell guard and
    # generate_exit_signal read live values from self.config.
    stop_loss_pct: float = 0.012
    take_profit_pct: float = 0.028
    reversal_exit_min_pnl_pct: float = 0.005
    time_exit_hours: float = 0.0
    time_exit_min_profit_pct: float = 0.003
    trail_stop_pct: float = 0.012  # let winners run; lock gain on pullback from peak
    # Shorts (inert unless enable_shorts=True). Conservative by design.
    enable_shorts: bool = False
    short_order_cash_pct: float = 0.02
    max_concurrent_shorts: int = 2
    short_max_total_exposure_pct: float = 0.06
    max_positions_per_sector: int = 0  # 0 = off (soft diversification cap)


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config

    def evaluate(
        self,
        signal: Signal,
        portfolio: Portfolio,
        *,
        daily_trade_count: int = 0,
        last_trade_at: datetime | None = None,
        now: datetime | None = None,
        cash_multiplier: float = 1.0,
        market_regime: str = "normal",
        mid_review_mode: str = "normal",
        lot_size: int = 1,
        recent_symbol_trades_60min: int = 0,
        recent_symbol_avg_pnl_pct: float | None = None,
    ) -> tuple[bool, OrderIntent | None, str]:
        equity = max(portfolio.equity, 1.0)
        cash_after_min = portfolio.cash - equity * self.config.min_cash_pct
        now = now or datetime.now(tz=last_trade_at.tzinfo if last_trade_at else None)
        reasons: list[str] = []

        if signal.action == Action.HOLD:
            return True, None, "hold signal"

        # ── Short side (tagged signals; inert unless enable_shorts) ──────────
        # Cover ([cover]) must ALWAYS be allowed (closing risk), even if shorts
        # are disabled — we never want to be stuck unable to close a short.
        if signal.action == Action.BUY and "[cover]" in signal.reason:
            pos = portfolio.positions.get(signal.symbol)
            if pos is None or pos.qty >= 0:
                return False, None, "no short to cover"
            qty = -pos.qty  # buy back the whole short
            order = OrderIntent(
                symbol=signal.symbol,
                side=OrderSide.BUY,
                qty=qty,
                limit_price=round(signal.features.price * 1.001, 2),
                reason=signal.reason,
            )
            return True, order, "cover approved"

        if signal.action == Action.SELL and "[short-entry]" in signal.reason:
            if not self.config.enable_shorts:
                return False, None, "shorts disabled"
            existing = portfolio.positions.get(signal.symbol)
            if existing is not None and existing.qty != 0:
                return False, None, "already have a position in symbol"
            open_shorts = sum(1 for p in portfolio.positions.values() if p.qty < 0)
            if open_shorts >= self.config.max_concurrent_shorts:
                return False, None, f"max concurrent shorts reached ({open_shorts})"
            # Shared limits that also apply to the short path:
            if daily_trade_count >= self.config.max_daily_trades:
                return False, None, "daily trade limit reached"
            effective_cooldown = self.config.symbol_cooldown_minutes
            if mid_review_mode == "accelerate":
                effective_cooldown = max(5, effective_cooldown // 2)
            if last_trade_at is not None and now - last_trade_at < timedelta(minutes=effective_cooldown):
                return False, None, "symbol cooldown active"
            # Total gross short exposure cap.
            short_exposure = sum(abs(p.market_value) for p in portfolio.positions.values() if p.qty < 0)
            if short_exposure / equity >= self.config.short_max_total_exposure_pct:
                return False, None, (
                    f"max short exposure reached ({short_exposure / equity:.1%} >= "
                    f"{self.config.short_max_total_exposure_pct:.1%})"
                )
            if signal.confidence < 0.55:
                return False, None, "short signal confidence below threshold"
            if (
                signal.features.book_spread_pct is not None
                and signal.features.book_spread_pct > self.config.max_book_spread_pct
            ):
                return False, None, "book spread above threshold"
            # Short size: small fraction of equity, capped by max_position_pct.
            cash_to_use = equity * self.config.short_order_cash_pct * max(0.5, min(1.0, float(cash_multiplier)))
            headroom = max(0.0, equity * self.config.max_position_pct)
            cash_to_use = min(cash_to_use, headroom)
            raw_qty = int(cash_to_use // max(signal.features.price, 1.0))
            lot = max(1, int(lot_size))
            qty = (raw_qty // lot) * lot
            if qty <= 0:
                return False, None, f"short size below lot ({raw_qty} < {lot})"
            order = OrderIntent(
                symbol=signal.symbol,
                side=OrderSide.SELL,
                qty=qty,
                limit_price=round(signal.features.price * 0.999, 2),
                reason=signal.reason,
            )
            return True, order, "short approved"

        if signal.confidence < 0.55:
            reasons.append("signal confidence below threshold")

        # Regime-based behavior: the LLM brief classifies the market every hour
        # into normal/macro_uncertain/risk_off/news_shock. Raise the bar on BUY
        # confidence in worse regimes, but never block all buys outright (turnover
        # requirements still apply on tough days). SELLs always pass (they close
        # positions). The cash_multiplier already shrinks size in bad regimes
        # (0.5x floor), so the ML filter and cash_mult provide the main defense.
        regime_lower = str(market_regime or "normal").lower()
        if signal.action == Action.BUY:
            min_conf_for_regime = {
                "normal": 0.55,
                "macro_uncertain": 0.60,
                "risk_off": 0.65,
                "news_shock": 0.72,
            }.get(regime_lower, 0.55)
            if signal.confidence < min_conf_for_regime:
                reasons.append(f"regime {regime_lower} requires confidence >= {min_conf_for_regime}")

        # Mid-session mode (Opus every ~3h decides session mode):
        #   normal      → no overrides
        #   cautious    → ML bar +0.05, size 0.7x (size handled below)
        #   pause_buys  → block all BUYs for the active window
        #   accelerate  → cooldown halved, size 1.2x (size below)
        mid_mode = str(mid_review_mode or "normal").lower()
        if signal.action == Action.BUY:
            if mid_mode == "pause_buys":
                reasons.append("mid_review mode=pause_buys blocks new buys")
            elif mid_mode == "cautious" and signal.confidence < (min_conf_for_regime + 0.03):
                # Small +0.03 bump: combined with the ML threshold and regime
                # gating, a larger bump leaves the bot stuck with zero trades.
                reasons.append("mid_review mode=cautious requires +0.03 confidence")

        if signal.action == Action.BUY:
            position = portfolio.positions.get(signal.symbol)
            if daily_trade_count >= self.config.max_daily_trades:
                reasons.append("daily trade limit reached")
            # Cooldown is halved when mid_review mode is 'accelerate'.
            effective_cooldown = self.config.symbol_cooldown_minutes
            if mid_mode == "accelerate":
                effective_cooldown = max(5, effective_cooldown // 2)
            if last_trade_at is not None and now - last_trade_at < timedelta(minutes=effective_cooldown):
                reasons.append("symbol cooldown active")

            # Anti-churn: if we've already traded this symbol 4+ times in the
            # last hour AND average realised PnL on those trades is below 0.5%,
            # we're churning — block new BUY for 30 min so we don't keep
            # collecting micro-wins that net out near zero after commission.
            if (
                recent_symbol_trades_60min >= 4
                and recent_symbol_avg_pnl_pct is not None
                and recent_symbol_avg_pnl_pct < 0.005
            ):
                reasons.append(
                    f"anti-churn: {recent_symbol_trades_60min} trades in 60min "
                    f"with avg PnL {recent_symbol_avg_pnl_pct:.3%}"
                )
            has_super_volume = signal.features.super_volume is not None
            if has_super_volume and signal.features.super_volume < self.config.min_super_volume:
                reasons.append("super volume below threshold")
            if (
                not has_super_volume
                and signal.features.volume_ratio is not None
                and signal.features.volume_ratio < self.config.min_volume_ratio
            ):
                reasons.append("volume ratio below threshold")
            if (
                signal.features.book_spread_pct is not None
                and signal.features.book_spread_pct > self.config.max_book_spread_pct
            ):
                reasons.append("book spread above threshold")
            if (
                signal.features.trade_imbalance is not None
                and signal.features.trade_imbalance < self.config.min_trade_imbalance_buy
            ):
                reasons.append("trade imbalance blocks buy")
            if (
                signal.features.book_imbalance is not None
                and signal.features.book_imbalance < self.config.min_book_imbalance_buy
            ):
                reasons.append("negative book imbalance blocks buy")
            if signal.regime == MarketRegime.RANGE:
                trade_imbalance = signal.features.trade_imbalance
                order_imbalance = signal.features.order_imbalance
                book_imbalance = signal.features.book_imbalance
                weak_trade_imbalance = trade_imbalance is None or trade_imbalance < max(
                    self.config.min_trade_imbalance_buy + 0.15,
                    0.35,
                )
                negative_order_imbalance = order_imbalance is not None and order_imbalance < 0.0
                negative_book_imbalance = book_imbalance is not None and book_imbalance < 0.0
                if (weak_trade_imbalance and negative_order_imbalance) or (
                    negative_book_imbalance and negative_order_imbalance
                ):
                    reasons.append("range buy lacks order-flow confirmation")
            if cash_after_min <= 0:
                reasons.append("min cash buffer blocks buy")
            current_position_pct = portfolio.position_value(signal.symbol) / equity
            if current_position_pct >= self.config.max_position_pct:
                reasons.append("max position limit reached")
            if portfolio.exposure / equity >= self.config.max_portfolio_exposure_pct:
                reasons.append("max portfolio exposure reached")
            # Soft sector diversification: only for NEW names (adds to a held
            # name are fine). Defers opening an Nth name in a sector we're
            # already concentrated in, so a sector drop can't take out the book.
            if self.config.max_positions_per_sector > 0 and signal.symbol not in portfolio.positions:
                sector = SECTOR_MAP.get(signal.symbol)
                if sector is not None:
                    held_in_sector = sum(
                        1 for sym, pos in portfolio.positions.items()
                        if pos.qty != 0 and SECTOR_MAP.get(sym) == sector
                    )
                    if held_in_sector >= self.config.max_positions_per_sector:
                        reasons.append(f"sector cap: already {held_in_sector} {sector} positions")
            if position is not None and position.qty > 0:
                pnl_pct = (signal.features.price - position.avg_price) / position.avg_price if position.avg_price > 0 else 0.0
                if current_position_pct >= self.config.add_position_max_position_pct:
                    reasons.append("add-position cap reached")
                if pnl_pct < self.config.add_position_min_pnl_pct:
                    reasons.append("add blocked in weak position")
                if signal.confidence < self.config.add_position_min_confidence:
                    reasons.append("add requires stronger confidence")
                if (
                    signal.features.trade_imbalance is None
                    or signal.features.trade_imbalance < self.config.add_position_min_trade_imbalance
                ):
                    reasons.append("add requires stronger trade imbalance")
            if reasons:
                return False, None, "; ".join(reasons)

            # Combined sizing: LLM brief cash_multiplier (clamped 0.5-1.0) ×
            # mid_review mode multiplier (cautious 0.7x, accelerate 1.2x).
            # Final size is still bounded by available cash after min buffer.
            safe_multiplier = max(0.5, min(1.0, float(cash_multiplier)))
            mid_size_mult = {
                "normal": 1.0,
                "cautious": 0.85,   # gentler shrink than 0.7 to keep some flow
                "accelerate": 1.2,
                "pause_buys": 1.0,  # already blocked above, but keep map complete
            }.get(mid_mode, 1.0)
            # Conviction multiplier: scale size by ML edge. signal.confidence
            # carries P(up) for ML-passed buys. Bounded by conviction_max_mult
            # here and by max_position_pct below — neither can breach the cap.
            conviction_mult = 1.0
            if self.config.conviction_sizing_enabled:
                edge = max(0.0, min(1.0, (signal.confidence - 0.5) / 0.5))
                conviction_mult = min(
                    self.config.conviction_max_mult,
                    1.0 + self.config.conviction_gain * edge,
                )
            cash_to_use = min(
                equity * self.config.order_cash_pct * safe_multiplier * mid_size_mult * conviction_mult,
                cash_after_min,
            )
            # Inviolable ceiling: never let this order push the position past
            # max_position_pct of equity. Conviction sizing must respect it.
            headroom = max(0.0, equity * self.config.max_position_pct - portfolio.position_value(signal.symbol))
            cash_to_use = min(cash_to_use, headroom)
            raw_qty = int(cash_to_use // signal.features.price)
            # Round DOWN to the nearest lot (some MOEX tickers have lots of 10
            # or 100). Without this we'd send odd qty like 7 which the exchange
            # rejects as below_lot_size.
            lot = max(1, int(lot_size))
            qty = (raw_qty // lot) * lot
            if qty <= 0:
                return False, None, f"order size is below lot ({raw_qty} < {lot})"
            order = OrderIntent(
                symbol=signal.symbol,
                side=OrderSide.BUY,
                qty=qty,
                limit_price=round(signal.features.price * 1.001, 2),
                reason=signal.reason,
            )
            return True, order, "buy approved"

        position = portfolio.positions.get(signal.symbol)
        if position is None or position.qty <= 0:
            reasons.append("no position to sell")
        if reasons:
            return False, None, "; ".join(reasons)
        reason_text = signal.reason.lower()
        # Entry-side SELL signals ("overbought near upper Bollinger band" and
        # "MACD bearish with elevated RSI") come from generate_signal and bypass
        # the profitable_floor in generate_exit_signal. Apply the same PnL gate
        # here so they don't dump winners early. Both are short-entry logic that,
        # in a long-only book, must NOT close a position before it has matured —
        # real exits belong to generate_exit_signal (stop / take-profit /
        # reversal), all PnL-aware.
        entry_side_sell = (
            "overbought near upper" in reason_text or "macd bearish" in reason_text
        )
        if entry_side_sell:
            pnl_pct = 0.0
            if position.avg_price > 0:
                pnl_pct = (signal.features.price - position.avg_price) / position.avg_price
            # Use the same floor as profitable reversal: >=0.5% by default,
            # or whatever reversal_exit_min_pnl_pct is currently set to.
            floor = max(0.0015, self.config.reversal_exit_min_pnl_pct)
            if pnl_pct < floor:
                return False, None, (
                    f"entry-side sell blocked: pnl {pnl_pct:.4%} below floor {floor:.4%}"
                )
        # Exit sizing. Partial profit-taking is kept for the choppy tape: it locks
        # gains before frequent reversals, and full-position exits everywhere
        # backtest worse here. The real lever for the win-small/lose-big asymmetry
        # is entry edge, not exit sizing.
        if (
            "stop-loss exit" in reason_text
            or "trailing stop" in reason_text
            or "early-cut" in reason_text
            or "force-close" in reason_text
            or "take-profit exit" in reason_text  # target hit (+TP%) → take it all
        ):
            # Full-position exits. Banking the whole position on a take-profit
            # target (like the stop closes the whole loss) reduces the
            # win-small/lose-big asymmetry. "scale-out" (0.33) and "reversal exit"
            # (0.75) stay partial — they're weaker, earlier signals.
            raw_qty = position.qty
        elif "reversal exit" in reason_text:
            raw_qty = max(1, min(position.qty, int(position.qty * 0.75)))
        elif "scale-out" in reason_text:
            raw_qty = max(1, min(position.qty, int(position.qty * 0.33)))
        else:
            raw_qty = max(1, min(position.qty, int(position.qty * 0.5)))
        # Round DOWN to lot.
        lot = max(1, int(lot_size))
        qty = (raw_qty // lot) * lot
        if qty <= 0:
            if position.qty >= lot:
                qty = lot
            else:
                # Remainder below one lot cannot be sold on the main board:
                # the position stays open with no working stop. Say that
                # plainly instead of repeating a generic size reject.
                return (
                    False,
                    None,
                    f"position stuck: {position.qty} below lot {lot}, "
                    "cannot exit — position has no working stop",
                )
        qty = min(qty, position.qty)
        # Micro-sell guard: if the resulting sale value is below 5000 RUB, the
        # commission ratio and the contribution to required turnover are both
        # poor. Promote the partial exit to a full position close instead.
        sale_value = qty * signal.features.price
        if sale_value < 5000.0 and qty < position.qty:
            # Promote to a full close, but keep the lot grid: position.qty may
            # itself be a non-multiple remainder (partial fill, legacy buy), and
            # sending it raw is what produced repeating below_lot_size rejects.
            qty = max(qty, (position.qty // lot) * lot)
        order = OrderIntent(
            symbol=signal.symbol,
            side=OrderSide.SELL,
            qty=qty,
            limit_price=round(signal.features.price * 0.999, 2),
            reason=signal.reason,
        )
        return True, order, "sell approved"
