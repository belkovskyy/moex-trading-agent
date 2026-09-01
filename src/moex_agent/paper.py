from __future__ import annotations

from datetime import date, datetime
from typing import Any

from moex_agent.models import OrderIntent, OrderSide, Portfolio
from moex_agent.portfolio import portfolio_from_snapshot, portfolio_snapshot
from moex_agent.storage import JsonStore


class PaperLedger:
    def __init__(self, store: JsonStore):
        self.store = store
        self.payload = store.read_state("paper_ledger") or {}
        self._reset_daily_if_needed()

    def portfolio(self, real_portfolio: Portfolio) -> Portfolio:
        paper = self.payload.get("portfolio")
        if paper:
            portfolio = portfolio_from_snapshot(paper)
            for symbol, position in real_portfolio.positions.items():
                portfolio.positions.setdefault(symbol, position)
            return portfolio
        self.payload["portfolio"] = portfolio_snapshot(real_portfolio)
        self._persist()
        return portfolio_from_snapshot(self.payload["portfolio"])

    def daily_trade_count(self) -> int:
        # Called every cycle; ensure stale counter from previous day is reset.
        # Without this the bot's daily_trade_limit gates lock up if the bot
        # runs through midnight without restart.
        self._reset_daily_if_needed()
        return int(self.payload.get("daily_trade_count", 0))

    def last_trade_at(self, symbol: str) -> datetime | None:
        raw = self.payload.get("last_trade_at", {}).get(symbol)
        if not raw:
            return None
        return datetime.fromisoformat(raw)

    def position_opened_at(self, symbol: str) -> datetime | None:
        """When the current position in `symbol` was opened (qty 0 → positive).

        Returns None if the position pre-existed our tracking (e.g. carried
        over from a bot restart before this field was added) or if no position
        is open. Used by generate_exit_signal for time-based force-close.
        """
        raw = self.payload.get("position_opened_at", {}).get(symbol)
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except (TypeError, ValueError):
            return None

    def record_order(self, portfolio: Portfolio, order: OrderIntent, executed_at: datetime) -> None:
        # Make sure the counter belongs to today before we increment it.
        self._reset_daily_if_needed()
        existing = portfolio.positions.get(order.symbol)
        position_before_qty = existing.qty if existing is not None else 0
        avg_price_before = existing.avg_price if existing is not None else 0.0
        sell_qty = 0
        realized_pnl = 0.0
        if order.side == OrderSide.SELL and existing is not None and existing.qty > 0:
            # Closing/reducing a LONG.
            sell_qty = min(order.qty, existing.qty)
            realized_pnl = (order.limit_price - existing.avg_price) * sell_qty
        elif order.side == OrderSide.BUY and existing is not None and existing.qty < 0:
            # Covering a SHORT (buy-to-cover): profit when cover price < entry.
            sell_qty = min(order.qty, -existing.qty)
            realized_pnl = (existing.avg_price - order.limit_price) * sell_qty

        # allow_short=True so the paper ledger tracks short opens/covers. For
        # long-only ops this is identical (risk sizes sells <= held qty, so no
        # accidental flip to short).
        portfolio.apply_fill(order.side, order.symbol, order.qty, order.limit_price, allow_short=True)
        self.payload["portfolio"] = portfolio_snapshot(portfolio)
        self.payload["daily_trade_count"] = self.daily_trade_count() + 1
        self.payload.setdefault("last_trade_at", {})[order.symbol] = executed_at.isoformat()
        current_position = portfolio.positions.get(order.symbol)

        # Track position open time: record when qty crosses 0 → positive,
        # clear when fully closed. Scale-ins do NOT reset the clock — we
        # measure how long capital has been deployed in this trade, not
        # when the most recent add-on happened.
        # Track open time by qty crossing zero in EITHER direction (long or
        # short), so shorts also get an opened_at.
        opened_at_map = self.payload.setdefault("position_opened_at", {})
        if current_position is None or current_position.qty == 0:
            opened_at_map.pop(order.symbol, None)
        elif position_before_qty == 0:
            opened_at_map[order.symbol] = executed_at.isoformat()

        self.payload.setdefault("trades", []).append(
            {
                "ts": executed_at.isoformat(),
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": order.qty,
                "limit_price": order.limit_price,
                "reason": order.reason,
                "position_before_qty": position_before_qty,
                "position_after_qty": current_position.qty if current_position is not None else 0,
                "avg_price_before": avg_price_before,
                "realized_pnl": round(realized_pnl, 2),
                "closed_qty": sell_qty,
            }
        )
        self._persist()

    def mark_to_market(self, portfolio: Portfolio, prices: dict[str, float]) -> None:
        portfolio.mark_to_market(prices)
        self.payload["portfolio"] = portfolio_snapshot(portfolio)
        self._persist()

    def _reset_daily_if_needed(self) -> None:
        today = date.today().isoformat()
        if self.payload.get("trade_date") != today:
            self.payload["trade_date"] = today
            self.payload["daily_trade_count"] = 0
            self.payload.setdefault("last_trade_at", {})
            self._persist()

    def _persist(self) -> None:
        self.store.write_state("paper_ledger", self.payload)
