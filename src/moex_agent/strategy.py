from __future__ import annotations

from datetime import datetime

from moex_agent.models import Action, MarketFeatures, MarketRegime, Position, Signal


WATCHLIST = [
    # ── ML-trained tier (17 stable tickers, >12 months of history) ──────
    # Bot analyzes these AND ML filter has high confidence on them.
    "LKOH", "SBER", "ROSN", "GAZP", "YDEX", "NVTK", "GMKN", "MOEX", "MTSS",
    "PLZL", "MGNT", "ALRS", "AFLT", "CHMF", "NLMK", "SNGSP", "PIKK",
    # ── Coverage tier (3 tickers excluded from ML training) ─────────────
    # Bot analyzes these but ML predictions are less reliable (out-of-
    # distribution data). The deterministic strategy (Bollinger/RSI) and
    # the risk layer (volume/spread/imbalance filters) still apply. In
    # practice these tickers will get traded less often, which is fine.
    # - VTBR: lot=100, share ~0.90 RUB → tick noise eats the stop on hot
    #         moments; risk layer's MAX_BOOK_SPREAD_PCT will mostly block.
    # - X5:   ~4 months of history (redomicilation Jan 2026).
    # - T:    same, redomicilation from TCS Group.
    "VTBR", "X5", "T",
]


# Sector map for soft diversification (risk.py). A SOFT per-sector cap (default
# 3) avoids piling the whole book into one sector (e.g. all-oil longs that fall
# together on a sector drop) WITHOUT hard-blocking — adds to names already held
# still pass, only NEW names beyond the cap are deferred.
SECTOR_MAP: dict[str, str] = {
    # Oil & gas
    "LKOH": "oil_gas", "ROSN": "oil_gas", "GAZP": "oil_gas", "NVTK": "oil_gas", "SNGSP": "oil_gas",
    # Banks / finance
    "SBER": "finance", "VTBR": "finance", "T": "finance", "MOEX": "finance",
    # Metals & mining
    "GMKN": "metals", "CHMF": "metals", "NLMK": "metals", "PLZL": "metals", "ALRS": "metals",
    # Consumer / retail
    "MGNT": "consumer", "X5": "consumer",
    # Tech / telecom / transport / other
    "YDEX": "tech", "MTSS": "telecom", "AFLT": "transport", "PIKK": "realestate",
}


def _negative(value: float | None, threshold: float = 0.0) -> bool:
    return value is not None and value < threshold


def _positive(value: float | None, threshold: float = 0.0) -> bool:
    return value is not None and value > threshold


def generate_signal(features: MarketFeatures, regime: MarketRegime) -> Signal:
    rsi = features.rsi
    macd = features.macd
    macd_signal = features.macd_signal
    price = features.price

    if regime == MarketRegime.CRISIS:
        return Signal(features.symbol, Action.HOLD, 0.2, regime, "crisis mode blocks new risk", features)

    if (
        rsi is not None
        and features.bollinger_low is not None
        and price <= features.bollinger_low * 1.004
        and rsi < 40
    ):
        bearish_momentum = macd is not None and macd_signal is not None and macd < macd_signal
        weak_order_flow = _negative(features.trade_imbalance, -0.2) or _negative(features.order_imbalance, -0.2)
        weak_book = _negative(features.book_imbalance, -0.15)
        if bearish_momentum and weak_order_flow and weak_book:
            return Signal(
                features.symbol,
                Action.HOLD,
                0.5,
                regime,
                "oversold setup blocked by bearish momentum and weak order-flow",
                features,
            )
        # TREND / HIGH_VOLATILITY: never fade the trend with a counter-trend
        # dip-buy. Buying an oversold dip while momentum is bearish is the
        # knife-catch that bled the book in a falling market.
        if regime in (MarketRegime.TREND, MarketRegime.HIGH_VOLATILITY) and bearish_momentum:
            return Signal(features.symbol, Action.HOLD, 0.5, regime,
                          "oversold long suppressed: counter-trend dip-buy in trend/high-vol", features)
        # HIGH_VOLATILITY: bands are wide and a near-touch is mostly noise —
        # require a TRUE touch of the lower band (drop the +0.4% slack).
        if regime == MarketRegime.HIGH_VOLATILITY and price > features.bollinger_low:
            return Signal(features.symbol, Action.HOLD, 0.5, regime,
                          "high-volatility: oversold long needs a true band touch", features)
        return Signal(features.symbol, Action.BUY, 0.68, regime, "oversold near lower Bollinger band", features)

    if rsi is not None and features.bollinger_high is not None and price >= features.bollinger_high and rsi > 68:
        return Signal(features.symbol, Action.SELL, 0.65, regime, "overbought near upper Bollinger band", features)

    if macd is not None and macd_signal is not None:
        if macd > macd_signal and rsi is not None and 43 <= rsi <= 67:
            if regime in (MarketRegime.RANGE, MarketRegime.HIGH_VOLATILITY):
                strong_trade_flow = _positive(features.trade_imbalance, 0.28)
                supportive_order_flow = not _negative(features.order_imbalance)
                supportive_book = not _negative(features.book_imbalance)
                toxic_order_flow = _negative(features.order_imbalance, -0.25) and _negative(features.book_imbalance)
                if not strong_trade_flow:
                    return Signal(
                        features.symbol,
                        Action.HOLD,
                        0.5,
                        regime,
                        "range setup lacks strong trade-flow confirmation",
                        features,
                    )
                if toxic_order_flow or (
                    not supportive_order_flow and not supportive_book and not _positive(features.trade_imbalance, 0.45)
                ):
                    return Signal(
                        features.symbol,
                        Action.HOLD,
                        0.5,
                        regime,
                        "range setup blocked by negative order-flow context",
                        features,
                    )
            confidence = 0.58
            reason = "MACD bullish with neutral RSI"
            if features.trade_imbalance is not None and features.trade_imbalance > 0.2:
                confidence += 0.06
                reason += "; buyer trade imbalance confirms"
            if features.book_imbalance is not None and features.book_imbalance > 0.1:
                confidence += 0.04
                reason += "; order book imbalance confirms"
            return Signal(features.symbol, Action.BUY, min(confidence, 0.75), regime, reason, features)
        if macd < macd_signal and rsi is not None and rsi > 55:
            confidence = 0.56
            reason = "MACD bearish with elevated RSI"
            if features.trade_imbalance is not None and features.trade_imbalance < -0.2:
                confidence += 0.06
                reason += "; seller trade imbalance confirms"
            if features.book_imbalance is not None and features.book_imbalance < -0.1:
                confidence += 0.04
                reason += "; order book imbalance confirms"
            return Signal(features.symbol, Action.SELL, min(confidence, 0.75), regime, reason, features)

    return Signal(features.symbol, Action.HOLD, 0.5, regime, "no strong deterministic edge", features)


def generate_exit_signal(
    features: MarketFeatures,
    regime: MarketRegime,
    position: Position | None,
    *,
    stop_loss_pct: float,
    take_profit_pct: float,
    reversal_exit_min_pnl_pct: float,
    now: datetime | None = None,
    position_opened_at: datetime | None = None,
    time_exit_hours: float = 0.0,
    time_exit_min_profit_pct: float = 0.0,
    peak_pnl_pct: float | None = None,
    trail_stop_pct: float = 0.0,
) -> Signal | None:
    if position is None or position.qty <= 0 or position.avg_price <= 0:
        return None

    pnl_pct = (features.price - position.avg_price) / position.avg_price
    if pnl_pct <= -abs(stop_loss_pct):
        return Signal(
            features.symbol,
            Action.SELL,
            0.82,
            regime,
            f"stop-loss exit at {pnl_pct:.4%}",
            features,
        )
    if pnl_pct >= abs(take_profit_pct):
        return Signal(
            features.symbol,
            Action.SELL,
            0.74,
            regime,
            f"take-profit exit at {pnl_pct:.4%}",
            features,
        )

    # Trailing stop: let a winner run, lock the gain when it gives back
    # trail_stop_pct from its peak. This replaces the old habit of dumping
    # winners at the +0.3% reversal floor — a position that peaks at +1.5% now
    # exits near +1.1% instead of +0.3%. Activates once peak >= trail_stop_pct.
    if (
        trail_stop_pct > 0.0
        and peak_pnl_pct is not None
        and peak_pnl_pct >= abs(trail_stop_pct)
        and pnl_pct <= peak_pnl_pct - abs(trail_stop_pct)
    ):
        return Signal(
            features.symbol,
            Action.SELL,
            0.75,
            regime,
            f"trailing stop: locked {pnl_pct:.4%} (peak {peak_pnl_pct:.4%})",
            features,
        )

    # Time-based exit: force-close stuck positions that aren't making progress.
    # Fires when (a) we know when the position opened, (b) it's been open longer
    # than time_exit_hours, and (c) pnl is below time_exit_min_profit_pct. The
    # backtest showed ~30-40% of return leaks through mark-to-market drift on
    # positions stuck between SL and TP — this releases that capital.
    if (
        time_exit_hours > 0.0
        and position_opened_at is not None
        and now is not None
    ):
        age_hours = (now - position_opened_at).total_seconds() / 3600.0
        if age_hours >= time_exit_hours and pnl_pct < time_exit_min_profit_pct:
            return Signal(
                features.symbol,
                Action.SELL,
                0.70,
                regime,
                f"time-based exit at age {age_hours:.1f}h, pnl {pnl_pct:.4%}",
                features,
            )

    macd = features.macd
    macd_signal = features.macd_signal
    if macd is None or macd_signal is None:
        return None

    price = max(features.price, 1.0)
    macd_gap_ratio = abs(macd - macd_signal) / price
    bearish_flow = features.trade_imbalance is not None and features.trade_imbalance < -0.35
    weak_book = features.book_imbalance is not None and features.book_imbalance < -0.20
    bearish_momentum = macd < macd_signal and features.rsi is not None and features.rsi < 45
    fading_momentum = macd < macd_signal or (
        features.trade_imbalance is not None and features.trade_imbalance < -0.10
    )
    if pnl_pct >= abs(take_profit_pct) * 0.5 and fading_momentum and (bearish_flow or weak_book):
        return Signal(
            features.symbol,
            Action.SELL,
            0.68,
            regime,
            f"take-profit scale-out at {pnl_pct:.4%}",
            features,
        )
    # ── Loss-side ONLY: early-cut a reversing position that is NOT yet a real
    # winner. Profit-locking belongs exclusively to the trailing stop /
    # take-profit above, so "reversal" never dumps winners and loss-cutting
    # stays decoupled from profit-taking. This fires only BELOW the
    # trailing-profit zone, when three independent bearish signals confirm a
    # real turn, to cut a marginal/losing position before it reaches the full
    # stop. NOTE: reversal_exit_min_pnl_pct is intentionally not a profit floor.
    strong_reversal = bearish_flow and weak_book and bearish_momentum
    in_profit_zone = pnl_pct >= abs(trail_stop_pct)  # owned by the trailing stop
    if strong_reversal and not in_profit_zone and pnl_pct > -abs(stop_loss_pct):
        return Signal(
            features.symbol,
            Action.SELL,
            0.76,
            regime,
            f"early-cut on confirmed reversal at {pnl_pct:.4%}",
            features,
        )

    return None


# ── Short side (gated by ENABLE_SHORTS; inert until app routes it) ──────────
# Mirror image of the long logic: we open a SHORT on overbought-at-upper-band
# with bearish confirmation, and COVER (buy-to-cover) on stop (price rose),
# take-profit (price fell), or a bullish reversal. A short SELL is tagged
# "[short-entry]" so the risk/exec layers can tell it apart from a long-reducing
# SELL. Short PnL is (entry - price)/entry: profit when price FALLS.


def generate_short_signal(features: MarketFeatures, regime: MarketRegime) -> Signal:
    """Entry signal for opening a SHORT. Returns a SELL tagged [short-entry],
    or HOLD. Symmetric to generate_signal's oversold-long branch."""
    rsi = features.rsi
    macd = features.macd
    macd_signal = features.macd_signal
    price = features.price

    if regime == MarketRegime.CRISIS:
        return Signal(features.symbol, Action.HOLD, 0.2, regime, "crisis mode blocks new risk", features)

    if (
        rsi is not None
        and features.bollinger_high is not None
        and price >= features.bollinger_high * 0.99
        and rsi > 70
    ):
        # Safety block: don't short a strong, well-bid rally. Blocks only when
        # momentum AND order-flow are strongly bullish, letting overbought-but-
        # fading setups through so the short side actually fires.
        bullish_momentum = macd is not None and macd_signal is not None and macd > macd_signal
        strong_buy_flow = _positive(features.trade_imbalance, 0.25) or _positive(features.order_imbalance, 0.25)
        if bullish_momentum and strong_buy_flow:
            return Signal(
                features.symbol,
                Action.HOLD,
                0.5,
                regime,
                "overbought short setup blocked by bullish momentum and strong order-flow",
                features,
            )
        # TREND / HIGH_VOLATILITY: don't fade the trend — shorting the top of a
        # rally while momentum is still bullish gets run over. Mirror of the
        # long-side counter-trend block above.
        if regime in (MarketRegime.TREND, MarketRegime.HIGH_VOLATILITY) and bullish_momentum:
            return Signal(features.symbol, Action.HOLD, 0.5, regime,
                          "overbought short suppressed: counter-trend in trend/high-vol [short-entry]", features)
        # HIGH_VOLATILITY: require a TRUE touch of the upper band (drop the
        # -1% slack) — wide bands make near-touches noise.
        if regime == MarketRegime.HIGH_VOLATILITY and price < features.bollinger_high:
            return Signal(features.symbol, Action.HOLD, 0.5, regime,
                          "high-volatility: overbought short needs a true band touch [short-entry]", features)
        return Signal(features.symbol, Action.SELL, 0.68, regime, "overbought near upper Bollinger band [short-entry]", features)

    # ── Trend / breakdown short (the real bear-market tool) ─────────────────
    # The branch above shorts a TOP (mean-reversion). This one shorts a
    # confirmed DOWNTREND continuation: bearish MACD, price below the middle
    # band, RSI falling but NOT yet oversold (38-52 — above 38 avoids shorting
    # into a bounce), with seller pressure. Together the short side now fires in
    # BOTH a topping market and a falling one. Gated further by the short ML
    # P(down) filter in app.py.
    if (
        macd is not None
        and macd_signal is not None
        and macd < macd_signal
        and features.bollinger_mid is not None
        and price < features.bollinger_mid
        and rsi is not None
        and 38.0 <= rsi <= 52.0
        and _negative(features.trade_imbalance, -0.20)
    ):
        confidence = 0.64
        reason = "downtrend breakdown [short-entry]"
        if _negative(features.book_imbalance, -0.10):
            confidence += 0.04
            reason = "downtrend breakdown; weak book [short-entry]"
        return Signal(features.symbol, Action.SELL, min(confidence, 0.72), regime, reason, features)

    return Signal(features.symbol, Action.HOLD, 0.5, regime, "no short edge", features)


def generate_cover_signal(
    features: MarketFeatures,
    regime: MarketRegime,
    position: Position | None,
    *,
    stop_loss_pct: float,
    take_profit_pct: float,
    now: datetime | None = None,
    position_opened_at: datetime | None = None,
    time_exit_hours: float = 0.0,
    time_exit_min_profit_pct: float = 0.0,
    peak_pnl_pct: float | None = None,
    trail_stop_pct: float = 0.0,
) -> Signal | None:
    """Exit signal for an OPEN short (position.qty < 0). Returns a BUY tagged
    [cover], or None. PnL for a short is (entry - price)/entry: a price RISE is
    a loss (stop), a price FALL is profit (take-profit).

    Mirrors the long-side exit machinery: hard stop, take-profit, trailing stop
    and time-based exit. The trailing stop and time exit are the two mechanisms
    that fixed the long side's loss asymmetry; without them a short only ever
    exits at a fixed target or a full stop, so winners give the move back and
    stuck shorts bleed. Both are opt-in (0 = off) so they can be A/B measured.
    """
    if position is None or position.qty >= 0 or position.avg_price <= 0:
        return None

    pnl_pct = (position.avg_price - features.price) / position.avg_price
    if pnl_pct <= -abs(stop_loss_pct):
        return Signal(features.symbol, Action.BUY, 0.82, regime, f"short stop-loss cover at {pnl_pct:.4%} [cover]", features)
    if pnl_pct >= abs(take_profit_pct):
        return Signal(features.symbol, Action.BUY, 0.74, regime, f"short take-profit cover at {pnl_pct:.4%} [cover]", features)

    # Trailing stop: lock the gain when a profitable short gives back
    # trail_stop_pct from its best PnL. Activates once the peak reaches the
    # trail distance, same as the long side.
    if (
        trail_stop_pct > 0.0
        and peak_pnl_pct is not None
        and peak_pnl_pct >= abs(trail_stop_pct)
        and pnl_pct <= peak_pnl_pct - abs(trail_stop_pct)
    ):
        return Signal(
            features.symbol,
            Action.BUY,
            0.75,
            regime,
            f"short trailing stop: locked {pnl_pct:.4%} (peak {peak_pnl_pct:.4%}) [cover]",
            features,
        )

    # Time-based exit: release capital from a short that is going nowhere,
    # instead of waiting for the full stop.
    if (
        time_exit_hours > 0.0
        and position_opened_at is not None
        and now is not None
    ):
        age_hours = (now - position_opened_at).total_seconds() / 3600.0
        if age_hours >= time_exit_hours and pnl_pct < time_exit_min_profit_pct:
            return Signal(
                features.symbol,
                Action.BUY,
                0.70,
                regime,
                f"short time-based cover at age {age_hours:.1f}h, pnl {pnl_pct:.4%} [cover]",
                features,
            )

    # Bullish reversal against the short → cover to protect profit.
    macd = features.macd
    macd_signal = features.macd_signal
    if macd is not None and macd_signal is not None:
        bullish_momentum = macd > macd_signal and features.rsi is not None and features.rsi > 55
        strong_buy_flow = _positive(features.trade_imbalance, 0.35) or _positive(features.book_imbalance, 0.20)
        if pnl_pct >= 0.0015 and bullish_momentum and strong_buy_flow:
            return Signal(features.symbol, Action.BUY, 0.72, regime, f"short reversal cover at {pnl_pct:.4%} [cover]", features)

    return None
