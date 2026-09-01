from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class MarketRegime(str, Enum):
    NORMAL = "normal"
    TREND = "trend"
    RANGE = "range"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"


@dataclass(frozen=True)
class Candle:
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: int
    avg_price: float
    market_price: float

    @property
    def market_value(self) -> float:
        return self.qty * self.market_price


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    @property
    def equity(self) -> float:
        return self.cash + sum(position.market_value for position in self.positions.values())

    @property
    def exposure(self) -> float:
        # Gross exposure: both long and short legs carry market risk. For a
        # long-only book this is identical to the old sum(max(0, mv)) since
        # long market_value is always >= 0. Shorts (qty < 0) add |market_value|.
        return sum(abs(position.market_value) for position in self.positions.values())

    def position_value(self, symbol: str) -> float:
        position = self.positions.get(symbol)
        return 0.0 if position is None else position.market_value

    def mark_to_market(self, prices: dict[str, float]) -> None:
        for symbol, price in prices.items():
            position = self.positions.get(symbol)
            if position is None or price <= 0:
                continue
            self.positions[symbol] = Position(
                symbol=symbol,
                qty=position.qty,
                avg_price=position.avg_price,
                market_price=price,
            )

    def apply_fill(
        self,
        side: OrderSide,
        symbol: str,
        qty: int,
        price: float,
        *,
        allow_short: bool = False,
    ) -> None:
        """Apply a fill.

        Long-only behaviour (allow_short=False, the default) is byte-for-byte
        identical to the original implementation: BUY grows a long, SELL reduces
        a long capped at the held quantity, and a SELL with no long is a no-op.

        With allow_short=True the position quantity is signed: a SELL beyond the
        long flips/opens a short (qty < 0), and a BUY covers/flips a short.
        """
        if qty <= 0 or price <= 0:
            return
        existing = self.positions.get(symbol)
        existing_qty = existing.qty if existing else 0

        if side == OrderSide.BUY:
            if existing_qty < 0:
                # Covering a short (buy-to-cover). Pay cash to buy back.
                self.cash -= qty * price
                new_qty = existing_qty + qty
                if new_qty == 0:
                    self.positions.pop(symbol, None)
                elif new_qty < 0:
                    # Still short — short entry avg_price unchanged.
                    self.positions[symbol] = Position(symbol, qty=new_qty, avg_price=existing.avg_price, market_price=price)
                else:
                    # Flipped through zero to a long; remainder bought at price.
                    self.positions[symbol] = Position(symbol, qty=new_qty, avg_price=price, market_price=price)
                return
            # Long increase / open — original behaviour (cash-checked).
            cost = qty * price
            if cost > self.cash:
                return
            if existing is None:
                self.positions[symbol] = Position(symbol, qty=qty, avg_price=price, market_price=price)
            else:
                new_qty = existing.qty + qty
                avg_price = ((existing.avg_price * existing.qty) + cost) / new_qty
                self.positions[symbol] = Position(symbol, qty=new_qty, avg_price=avg_price, market_price=price)
            self.cash -= cost
            return

        # SELL
        if existing_qty > 0:
            if not allow_short:
                # Original long-only path: reduce long, capped at held qty.
                sell_qty = min(qty, existing_qty)
                self.cash += sell_qty * price
                remaining_qty = existing_qty - sell_qty
                if remaining_qty <= 0:
                    self.positions.pop(symbol, None)
                else:
                    self.positions[symbol] = Position(symbol, qty=remaining_qty, avg_price=existing.avg_price, market_price=price)
                return
            # Shorts allowed: sell can exceed the long and flip to short.
            self.cash += qty * price
            new_qty = existing_qty - qty
            if new_qty == 0:
                self.positions.pop(symbol, None)
            elif new_qty > 0:
                self.positions[symbol] = Position(symbol, qty=new_qty, avg_price=existing.avg_price, market_price=price)
            else:
                self.positions[symbol] = Position(symbol, qty=new_qty, avg_price=price, market_price=price)
            return

        # existing_qty <= 0: opening / increasing a short.
        if not allow_short:
            return  # long-only: SELL with no long is a no-op (original behaviour)
        self.cash += qty * price
        if existing is None or existing_qty == 0:
            self.positions[symbol] = Position(symbol, qty=-qty, avg_price=price, market_price=price)
        else:
            new_qty = existing_qty - qty  # more negative
            # Weighted average of the short entry price.
            avg_price = ((existing.avg_price * (-existing_qty)) + qty * price) / (-new_qty)
            self.positions[symbol] = Position(symbol, qty=new_qty, avg_price=avg_price, market_price=price)


@dataclass(frozen=True)
class MarketFeatures:
    symbol: str
    feature_ts: datetime | None
    price: float
    rsi: float | None
    macd: float | None
    macd_signal: float | None
    bollinger_low: float | None
    bollinger_mid: float | None
    bollinger_high: float | None
    atr_pct: float | None
    volume_ratio: float | None
    trade_imbalance: float | None = None
    order_imbalance: float | None = None
    book_imbalance: float | None = None
    book_spread_pct: float | None = None
    super_volume: float | None = None
    # Macro context features (filled by macro_data.compute_macro_features).
    # Train/serve parity: when None at inference time the ml_features module
    # imputes 0.0, matching how the offline dataset stores missing values.
    imoex_return_60m: float | None = None
    imoex_return_4h: float | None = None
    sector_return_60m: float | None = None
    corr_price_imoex_60: float | None = None
    rel_strength_vs_imoex: float | None = None


@dataclass(frozen=True)
class Signal:
    symbol: str
    action: Action
    confidence: float
    regime: MarketRegime
    reason: str
    features: MarketFeatures


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: OrderSide
    qty: int
    limit_price: float
    reason: str


@dataclass(frozen=True)
class Decision:
    ts: datetime
    symbol: str
    signal: Signal
    order: OrderIntent | None
    approved: bool
    reason: str
    meta: dict[str, Any] = field(default_factory=dict)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def match_round_trips(orders: list[tuple[str, float, float]]) -> list[dict[str, float]]:
    """Match opens↔closes for ONE symbol via a signed-lot FIFO, handling BOTH
    longs (BUY opens, SELL closes) AND shorts (SELL opens, BUY covers).

    `orders` is a time-sorted list of (side, qty, price) where side is "buy"/
    "sell" and qty>0. Returns a list of closed round-trips with realized
    {pnl_abs, pnl_pct} (pnl_pct in PERCENT). A position flipping through zero
    is split into a close + a new open.

    Previously the supervisory layers (mid_review/auto_tuner/monitor) only
    matched BUY→SELL, so short round-trips (SELL→BUY cover) were invisible —
    the LLM/tuner saw an incomplete PnL picture once shorts went live.
    """
    closed: list[dict[str, float]] = []
    pos = 0.0          # signed open quantity (+long / -short)
    lots: list[list[float]] = []  # FIFO of [qty_signed, price]
    for side, qty, price in orders:
        if qty <= 0 or price <= 0:
            continue
        signed = qty if side == "buy" else -qty
        # Same direction as current position (or flat) → OPEN (append lot).
        if pos == 0 or (pos > 0) == (signed > 0):
            lots.append([signed, price])
            pos += signed
            continue
        # Opposite direction → CLOSE against existing lots (FIFO).
        remaining = qty  # absolute size to close
        while remaining > 1e-9 and lots:
            lot_qty, lot_price = lots[0]
            take = min(remaining, abs(lot_qty))
            if lot_qty > 0:   # closing a LONG with a SELL
                pnl_abs = (price - lot_price) * take
            else:             # covering a SHORT with a BUY
                pnl_abs = (lot_price - price) * take
            # %-return against ENTRY, symmetric for both sides:
            # long = (exit-entry)/entry, short = (entry-exit)/entry.
            pnl_pct = ((price / lot_price - 1.0) if lot_qty > 0
                       else (lot_price - price) / lot_price) * 100.0
            closed.append({
                "pnl_abs": pnl_abs,
                "pnl_pct": pnl_pct,
                # Which book this round-trip belongs to: a positive lot was a
                # long being sold, a negative lot was a short being covered.
                "side": "long" if lot_qty > 0 else "short",
            })
            lot_qty = (abs(lot_qty) - take) * (1 if lot_qty > 0 else -1)
            if abs(lot_qty) <= 1e-9:
                lots.pop(0)
            else:
                lots[0][0] = lot_qty
            remaining -= take
            pos += (take if signed > 0 else -take)
        # Any leftover opens a NEW position in the opposite direction.
        if remaining > 1e-9:
            new_signed = remaining if signed > 0 else -remaining
            lots.append([new_signed, price])
            pos += new_signed
    return closed
