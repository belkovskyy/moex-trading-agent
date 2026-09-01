"""Лотность на выходе из позиции.

Живой инцидент: бот пытался продать 7 акций MTSS при лоте 10 — заявка
отбивалась как below_lot_size каждый цикл, позиция висела без работающего
стопа. Здесь закрыты оба места, где количество может разъехаться с лотом.

Запуск:  PYTHONPATH=src python -m pytest tests -q
"""
from __future__ import annotations

from datetime import datetime, timezone

from moex_agent.models import (
    Action,
    MarketFeatures,
    MarketRegime,
    Portfolio,
    Position,
    Signal,
)
from moex_agent.risk import RiskConfig, RiskManager

NOW = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)


def _features(price: float) -> MarketFeatures:
    return MarketFeatures(
        symbol="MTSS", feature_ts=NOW, price=price,
        rsi=50.0, macd=0.0, macd_signal=0.0,
        bollinger_low=price * 0.98, bollinger_mid=price, bollinger_high=price * 1.02,
        atr_pct=0.01, volume_ratio=1.0, book_spread_pct=0.001,
    )


def _risk() -> RiskManager:
    return RiskManager(RiskConfig(
        max_position_pct=0.2, max_portfolio_exposure_pct=0.75, max_daily_loss_pct=0.03,
        max_drawdown_pct=0.08, order_cash_pct=0.1, min_cash_pct=0.1, max_daily_trades=80,
        symbol_cooldown_minutes=30, min_volume_ratio=0.3, min_trade_imbalance_buy=-0.15,
        min_super_volume=1000.0, min_book_imbalance_buy=-0.15, max_book_spread_pct=0.01,
        add_position_min_confidence=0.64, add_position_min_trade_imbalance=0.2,
        add_position_min_pnl_pct=0.0, add_position_max_position_pct=0.1,
    ))


def _exit_signal(reason: str, price: float) -> Signal:
    return Signal("MTSS", Action.SELL, 0.8, MarketRegime.NORMAL, reason, _features(price))


def test_remainder_below_lot_reports_stuck_position():
    """7 акций при лоте 10 продать нельзя — говорим об этом прямо."""
    risk = _risk()
    portfolio = Portfolio(cash=100_000.0, positions={"MTSS": Position("MTSS", 7, 200.0, 210.0)})
    approved, order, reason = risk.evaluate(
        _exit_signal("take-profit exit", 210.0), portfolio,
        daily_trade_count=0, last_trade_at=None, now=NOW, lot_size=10,
    )
    assert not approved and order is None
    assert "stuck" in reason and "no working stop" in reason


def test_micro_sell_guard_keeps_lot_grid():
    """Промоушен частичного выхода в полный не должен ломать кратность лоту."""
    risk = _risk()
    # 17 акций — остаток не кратен лоту; частичный выход дорастает до полного,
    # но продать можно только 10.
    portfolio = Portfolio(cash=100_000.0, positions={"MTSS": Position("MTSS", 17, 200.0, 210.0)})
    approved, order, _ = risk.evaluate(
        _exit_signal("scale-out at +0.5R", 210.0), portfolio,
        daily_trade_count=0, last_trade_at=None, now=NOW, lot_size=10,
    )
    assert approved and order is not None
    assert order.qty % 10 == 0, f"qty {order.qty} не кратно лоту"
    assert order.qty == 10


def test_full_position_exit_is_lot_aligned():
    """Полный выход из кратной позиции остаётся полным."""
    risk = _risk()
    portfolio = Portfolio(cash=100_000.0, positions={"MTSS": Position("MTSS", 30, 200.0, 210.0)})
    approved, order, _ = risk.evaluate(
        _exit_signal("take-profit exit", 210.0), portfolio,
        daily_trade_count=0, last_trade_at=None, now=NOW, lot_size=10,
    )
    assert approved and order is not None and order.qty == 30
