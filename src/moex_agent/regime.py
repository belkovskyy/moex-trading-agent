from __future__ import annotations

from moex_agent.models import MarketFeatures, MarketRegime


def detect_regime(features: MarketFeatures) -> MarketRegime:
    atr = features.atr_pct
    rsi = features.rsi
    macd = features.macd
    macd_signal = features.macd_signal
    price = features.price

    if atr is not None and atr >= 0.045:
        return MarketRegime.CRISIS
    if atr is not None and atr >= 0.025:
        return MarketRegime.HIGH_VOLATILITY

    if macd is not None and macd_signal is not None and abs(macd - macd_signal) / max(price, 1.0) > 0.002:
        return MarketRegime.TREND

    if rsi is not None and 42.0 <= rsi <= 58.0:
        return MarketRegime.RANGE

    return MarketRegime.NORMAL

