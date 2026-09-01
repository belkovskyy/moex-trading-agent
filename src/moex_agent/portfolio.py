from __future__ import annotations

from dataclasses import asdict

from moex_agent.models import Portfolio, Position


def portfolio_snapshot(portfolio: Portfolio) -> dict:
    return {
        "cash": portfolio.cash,
        "equity": portfolio.equity,
        "exposure": portfolio.exposure,
        "positions": {symbol: asdict(position) for symbol, position in portfolio.positions.items()},
    }


def portfolio_from_snapshot(payload: dict) -> Portfolio:
    positions = {
        symbol: Position(
            symbol=symbol,
            qty=int(raw.get("qty", 0)),
            avg_price=float(raw.get("avg_price", 0.0)),
            market_price=float(raw.get("market_price", raw.get("avg_price", 0.0))),
        )
        for symbol, raw in payload.get("positions", {}).items()
    }
    return Portfolio(cash=float(payload.get("cash", 0.0)), positions=positions)
