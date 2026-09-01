from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from random import Random
from typing import Any, Protocol

from moex_agent.models import Candle, utc_now

logger = logging.getLogger(__name__)


class MarketDataError(RuntimeError):
    pass


class MarketDataClient(Protocol):
    def get_candles(self, symbol: str, limit: int = 120) -> list[Candle]:
        raise NotImplementedError

    def get_super_features(self, symbol: str) -> dict[str, float]:
        raise NotImplementedError


class AlgopackMarketDataClient:
    def __init__(self, token: str, *, period: str = "10min", lookback_days: int = 5, use_super_candles: bool = True):
        self.token = token
        self.period = period
        self.lookback_days = lookback_days
        self.use_super_candles = use_super_candles
        self._super_features_cache: dict[str, dict[str, float]] = {}

    def get_candles(self, symbol: str, limit: int = 120) -> list[Candle]:
        if not self.token:
            raise MarketDataError("MOEX_ALGO_TOKEN is empty.")
        try:
            from moexalgo import Ticker, session
        except ImportError as exc:
            raise MarketDataError("moexalgo is not installed. Run: pip install -r requirements.txt") from exc

        session.TOKEN = self.token
        end = date.today()
        start = end - timedelta(days=self.lookback_days)
        rows = Ticker(symbol).candles(
            start=start.isoformat(),
            end=end.isoformat(),
            period=self.period,
            native=True,
        )
        candles = [_row_to_candle(symbol, row) for row in rows]
        candles = [candle for candle in candles if candle is not None]
        if len(candles) > limit:
            return candles[-limit:]
        return candles

    def get_super_features(self, symbol: str) -> dict[str, float]:
        if symbol in self._super_features_cache:
            return self._super_features_cache[symbol]
        if not self.use_super_candles:
            return {}
        if not self.token:
            raise MarketDataError("MOEX_ALGO_TOKEN is empty.")
        try:
            from moexalgo import Ticker, session
        except ImportError as exc:
            raise MarketDataError("moexalgo is not installed. Run: pip install moexalgo") from exc

        session.TOKEN = self.token
        end = date.today()
        start = end - timedelta(days=1)
        ticker = Ticker(symbol)
        features: dict[str, float] = {}

        trade = _latest_row(ticker.tradestats(start=start.isoformat(), end=end.isoformat(), native=True))
        if trade:
            features["trade_imbalance"] = _safe_float(trade.get("disb"))
            features["super_volume"] = _safe_float(trade.get("vol"))

        order = _latest_row(ticker.orderstats(start=start.isoformat(), end=end.isoformat(), native=True))
        if order:
            put_b = _safe_float(order.get("put_vol_b"))
            put_s = _safe_float(order.get("put_vol_s"))
            features["order_imbalance"] = _ratio_diff(put_b, put_s)

        book = _latest_row(ticker.obstats(start=start.isoformat(), end=end.isoformat(), native=True))
        if book:
            features["book_imbalance"] = _safe_float(book.get("imbalance_vol"))
            spread = _safe_float(book.get("spread_bbo"))
            mid = _mid_price_from_book(book)
            if mid > 0:
                features["book_spread_pct"] = spread / mid

        result = {key: value for key, value in features.items() if value is not None}
        self._super_features_cache[symbol] = result
        return result

    def reset_cycle_cache(self) -> None:
        self._super_features_cache.clear()


class SampleMarketDataClient:
    def get_candles(self, symbol: str, limit: int = 120) -> list[Candle]:
        return _sample_candles(symbol, limit)

    def get_super_features(self, symbol: str) -> dict[str, float]:
        return {}

    def reset_cycle_cache(self) -> None:
        return None


def _row_to_candle(symbol: str, row: dict[str, Any]) -> Candle | None:
    try:
        ts_raw = row.get("begin") or row.get("datetime") or row.get("time")
        if isinstance(ts_raw, datetime):
            ts = ts_raw
        else:
            ts = datetime.fromisoformat(str(ts_raw))
        return Candle(
            symbol=symbol,
            ts=ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0) or 0.0),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("failed to parse candle", extra={"extra": {"symbol": symbol, "row": row, "error": str(exc)}})
        return None


def _latest_row(rows: Any) -> dict[str, Any] | None:
    collected = list(rows)
    return collected[-1] if collected else None


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio_diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    total = left + right
    if total <= 0:
        return None
    return (left - right) / total


def _mid_price_from_book(row: dict[str, Any]) -> float:
    bid = _safe_float(row.get("vwap_b_1mio")) or _safe_float(row.get("vwap_b")) or 0.0
    ask = _safe_float(row.get("vwap_s_1mio")) or _safe_float(row.get("vwap_s")) or 0.0
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return max(bid, ask)


def _sample_candles(symbol: str, limit: int) -> list[Candle]:
    rng = Random(symbol)
    now = utc_now()
    price = 100.0 + rng.random() * 500.0
    candles: list[Candle] = []
    for idx in range(limit):
        drift = 0.02 if symbol in {"SBER", "LKOH", "YDEX"} else 0.0
        shock = rng.uniform(-1.5, 1.5)
        close = max(1.0, price * (1.0 + (drift + shock) / 100.0))
        high = max(price, close) * (1.0 + rng.random() * 0.005)
        low = min(price, close) * (1.0 - rng.random() * 0.005)
        candles.append(
            Candle(
                symbol=symbol,
                ts=now - timedelta(minutes=limit - idx),
                open=price,
                high=high,
                low=low,
                close=close,
                volume=10_000 + rng.random() * 20_000,
            )
        )
        price = close
    return candles
