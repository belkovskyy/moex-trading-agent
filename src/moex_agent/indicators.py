from __future__ import annotations

from statistics import mean, pstdev

from moex_agent.models import Candle, MarketFeatures


def sma(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return mean(values[-window:])


def ema_series(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (window + 1.0)
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1.0 - alpha) * out[-1])
    return out


def rsi(values: list[float], window: int = 14) -> float | None:
    if len(values) <= window:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for prev, cur in zip(values[-window - 1 : -1], values[-window:]):
        diff = cur - prev
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_loss = mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = mean(gains) / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 35:
        return None, None
    ema12 = ema_series(values, 12)
    ema26 = ema_series(values, 26)
    macd_line = [fast - slow for fast, slow in zip(ema12[-len(ema26) :], ema26)]
    signal_line = ema_series(macd_line, 9)
    return macd_line[-1], signal_line[-1]


def bollinger(values: list[float], window: int = 20, width: float = 2.0) -> tuple[float | None, float | None, float | None]:
    if len(values) < window:
        return None, None, None
    sample = values[-window:]
    mid = mean(sample)
    sigma = pstdev(sample)
    return mid - width * sigma, mid, mid + width * sigma


def atr_pct(candles: list[Candle], window: int = 14) -> float | None:
    if len(candles) <= window:
        return None
    ranges: list[float] = []
    recent = candles[-window:]
    previous_close = candles[-window - 1].close
    for candle in recent:
        tr = max(
            candle.high - candle.low,
            abs(candle.high - previous_close),
            abs(candle.low - previous_close),
        )
        ranges.append(tr)
        previous_close = candle.close
    price = candles[-1].close
    return None if price <= 0 else mean(ranges) / price


def volume_ratio(candles: list[Candle], window: int = 20) -> float | None:
    if len(candles) < window + 1:
        return None
    baseline = mean([c.volume for c in candles[-window - 1 : -1]])
    return None if baseline <= 0 else candles[-1].volume / baseline


def build_features(
    candles: list[Candle],
    super_features: dict | None = None,
    macro_features: dict | None = None,
) -> MarketFeatures:
    if not candles:
        raise ValueError("Cannot build features from empty candles.")
    super_features = super_features or {}
    macro_features = macro_features or {}
    closes = [c.close for c in candles]
    low, mid, high = bollinger(closes)
    macd_value, macd_signal = macd(closes)
    return MarketFeatures(
        symbol=candles[-1].symbol,
        feature_ts=candles[-1].ts,
        price=candles[-1].close,
        rsi=rsi(closes),
        macd=macd_value,
        macd_signal=macd_signal,
        bollinger_low=low,
        bollinger_mid=mid,
        bollinger_high=high,
        atr_pct=atr_pct(candles),
        volume_ratio=volume_ratio(candles),
        trade_imbalance=super_features.get("trade_imbalance"),
        order_imbalance=super_features.get("order_imbalance"),
        book_imbalance=super_features.get("book_imbalance"),
        book_spread_pct=super_features.get("book_spread_pct"),
        super_volume=super_features.get("super_volume"),
        imoex_return_60m=macro_features.get("imoex_return_60m"),
        imoex_return_4h=macro_features.get("imoex_return_4h"),
        sector_return_60m=macro_features.get("sector_return_60m"),
        corr_price_imoex_60=macro_features.get("corr_price_imoex_60"),
        rel_strength_vs_imoex=macro_features.get("rel_strength_vs_imoex"),
    )
