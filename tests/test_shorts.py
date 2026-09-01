"""Тесты шорт-стороны: учёт PnL, выходы, риск-лимиты.

Проект жил без тестов, а шорты — самая тонкая часть: знак PnL, переворот
позиции через ноль, лимиты экспозиции. Здесь закрыто то, что молча ломается
и не видно в логах.

Запуск:  PYTHONPATH=src python -m pytest tests -q
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from moex_agent.models import (
    Action,
    MarketFeatures,
    MarketRegime,
    OrderSide,
    Portfolio,
    Position,
    match_round_trips,
)
from moex_agent.risk import RiskConfig, RiskManager
from moex_agent.strategy import generate_cover_signal

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)


def _features(price: float, **kw) -> MarketFeatures:
    base = dict(
        symbol="GAZP", feature_ts=NOW, price=price,
        rsi=50.0, macd=0.0, macd_signal=0.0,
        bollinger_low=price * 0.98, bollinger_mid=price, bollinger_high=price * 1.02,
        atr_pct=0.01, volume_ratio=1.0,
    )
    base.update(kw)
    return MarketFeatures(**base)


def _risk(**kw) -> RiskManager:
    cfg = dict(
        max_position_pct=0.2, max_portfolio_exposure_pct=0.75, max_daily_loss_pct=0.03,
        max_drawdown_pct=0.08, order_cash_pct=0.1, min_cash_pct=0.1, max_daily_trades=80,
        symbol_cooldown_minutes=30, min_volume_ratio=0.3, min_trade_imbalance_buy=-0.15,
        min_super_volume=1000.0, min_book_imbalance_buy=-0.15, max_book_spread_pct=0.01,
        add_position_min_confidence=0.64, add_position_min_trade_imbalance=0.2,
        add_position_min_pnl_pct=0.0, add_position_max_position_pct=0.1,
        enable_shorts=True, short_order_cash_pct=0.05,
        max_concurrent_shorts=6, short_max_total_exposure_pct=0.35,
    )
    cfg.update(kw)
    return RiskManager(RiskConfig(**cfg))


# ── Учёт позиции и PnL ──────────────────────────────────────────────────────

def test_sell_opens_short_only_when_allowed():
    """Без allow_short продажа без позиции — no-op, а не случайный шорт."""
    p = Portfolio(cash=100_000.0, positions={})
    p.apply_fill(OrderSide.SELL, "GAZP", 10, 100.0)
    assert p.positions == {}

    p2 = Portfolio(cash=100_000.0, positions={})
    p2.apply_fill(OrderSide.SELL, "GAZP", 10, 100.0, allow_short=True)
    assert p2.positions["GAZP"].qty == -10
    assert p2.cash == pytest.approx(101_000.0)  # продажа приносит кэш


def test_short_profit_when_price_falls():
    p = Portfolio(cash=100_000.0, positions={})
    p.apply_fill(OrderSide.SELL, "GAZP", 10, 100.0, allow_short=True)   # шорт по 100
    p.apply_fill(OrderSide.BUY, "GAZP", 10, 90.0, allow_short=True)     # откуп по 90
    assert "GAZP" not in p.positions
    assert p.cash == pytest.approx(100_100.0)  # +10 * 10 руб прибыли


def test_short_loss_when_price_rises():
    p = Portfolio(cash=100_000.0, positions={})
    p.apply_fill(OrderSide.SELL, "GAZP", 10, 100.0, allow_short=True)
    p.apply_fill(OrderSide.BUY, "GAZP", 10, 110.0, allow_short=True)
    assert p.cash == pytest.approx(99_900.0)


def test_sell_beyond_long_flips_to_short():
    """Продажа больше лонга переворачивает позицию, а не обрезается молча."""
    p = Portfolio(cash=100_000.0, positions={"GAZP": Position("GAZP", 10, 100.0, 100.0)})
    p.apply_fill(OrderSide.SELL, "GAZP", 15, 105.0, allow_short=True)
    assert p.positions["GAZP"].qty == -5
    assert p.positions["GAZP"].avg_price == 105.0  # шорт открыт по цене сделки


def test_short_adds_to_gross_exposure():
    """Шорт несёт рыночный риск: экспозиция считает модуль стоимости."""
    p = Portfolio(cash=100_000.0, positions={"GAZP": Position("GAZP", -10, 100.0, 100.0)})
    assert p.exposure == pytest.approx(1000.0)


def test_round_trips_tag_side():
    """Разбивка по сторонам — основа для тюнера: шорт-цикл не должен
    записываться в лонги."""
    long_rt = match_round_trips([("buy", 10, 100.0), ("sell", 10, 110.0)])
    short_rt = match_round_trips([("sell", 10, 100.0), ("buy", 10, 90.0)])
    assert [r["side"] for r in long_rt] == ["long"]
    assert [r["side"] for r in short_rt] == ["short"]
    assert long_rt[0]["pnl_abs"] == pytest.approx(100.0)
    assert short_rt[0]["pnl_abs"] == pytest.approx(100.0)


# ── Выходы из шорта ─────────────────────────────────────────────────────────

def _short_pos(entry: float = 100.0) -> Position:
    return Position("GAZP", -10, entry, entry)


def test_cover_on_stop_and_take():
    pos = _short_pos()
    stop = generate_cover_signal(_features(101.5), MarketRegime.NORMAL, pos,
                                 stop_loss_pct=0.01, take_profit_pct=0.02)
    assert stop is not None and stop.action == Action.BUY and "[cover]" in stop.reason

    take = generate_cover_signal(_features(97.5), MarketRegime.NORMAL, pos,
                                 stop_loss_pct=0.01, take_profit_pct=0.02)
    assert take is not None and "take-profit" in take.reason


def test_no_cover_signal_for_long_position():
    """Функция покрытия не должна трогать лонг."""
    long_pos = Position("GAZP", 10, 100.0, 100.0)
    assert generate_cover_signal(_features(90.0), MarketRegime.NORMAL, long_pos,
                                 stop_loss_pct=0.01, take_profit_pct=0.02) is None


def test_short_trailing_stop_off_by_default():
    """Трейлинг для шортов замерен как вредный — по умолчанию выключен (0)."""
    pos = _short_pos()
    # цена 99: прибыль 1%, пик был 2% — при trail=0 выхода быть не должно
    assert generate_cover_signal(_features(99.0), MarketRegime.NORMAL, pos,
                                 stop_loss_pct=0.05, take_profit_pct=0.05,
                                 peak_pnl_pct=0.02, trail_stop_pct=0.0) is None


def test_short_trailing_stop_fires_when_enabled():
    pos = _short_pos()
    sig = generate_cover_signal(_features(99.0), MarketRegime.NORMAL, pos,
                                stop_loss_pct=0.05, take_profit_pct=0.05,
                                peak_pnl_pct=0.02, trail_stop_pct=0.008)
    assert sig is not None and "trailing" in sig.reason


def test_short_time_exit_respects_min_profit():
    """Выход по времени срабатывает только на непрогрессирующей позиции."""
    pos = _short_pos()
    opened = NOW - timedelta(hours=10)
    stuck = generate_cover_signal(_features(100.0), MarketRegime.NORMAL, pos,
                                  stop_loss_pct=0.05, take_profit_pct=0.05,
                                  now=NOW, position_opened_at=opened,
                                  time_exit_hours=6.0, time_exit_min_profit_pct=0.003)
    assert stuck is not None and "time-based" in stuck.reason

    working = generate_cover_signal(_features(99.0), MarketRegime.NORMAL, pos,
                                    stop_loss_pct=0.05, take_profit_pct=0.05,
                                    now=NOW, position_opened_at=opened,
                                    time_exit_hours=6.0, time_exit_min_profit_pct=0.003)
    assert working is None  # прибыль 1% > порога — держим


# ── Риск-слой ───────────────────────────────────────────────────────────────

def _short_signal(price: float = 100.0, confidence: float = 0.68):
    from moex_agent.models import Signal
    return Signal("GAZP", Action.SELL, confidence, MarketRegime.NORMAL,
                  "overbought near upper Bollinger band [short-entry]",
                  _features(price, book_spread_pct=0.001))


def test_cover_allowed_even_when_shorts_disabled():
    """Закрытие риска нельзя блокировать — иначе шорт не выпустить."""
    from moex_agent.models import Signal
    risk = _risk(enable_shorts=False)
    portfolio = Portfolio(cash=100_000.0, positions={"GAZP": Position("GAZP", -10, 100.0, 100.0)})
    sig = Signal("GAZP", Action.BUY, 0.82, MarketRegime.NORMAL,
                 "short stop-loss cover at -1.2% [cover]", _features(101.0))
    approved, order, _ = risk.evaluate(sig, portfolio, daily_trade_count=0, last_trade_at=None, now=NOW)
    assert approved and order is not None and order.qty == 10


def test_short_entry_blocked_when_disabled():
    risk = _risk(enable_shorts=False)
    approved, _, reason = risk.evaluate(_short_signal(), Portfolio(cash=100_000.0, positions={}),
                                        daily_trade_count=0, last_trade_at=None, now=NOW)
    assert not approved and "disabled" in reason


def test_short_exposure_cap_enforced():
    """Суммарная короткая экспозиция ограничена долей капитала."""
    risk = _risk(short_max_total_exposure_pct=0.05)
    positions = {f"S{i}": Position(f"S{i}", -100, 100.0, 100.0) for i in range(2)}  # 20k шортов
    portfolio = Portfolio(cash=100_000.0, positions=positions)
    approved, _, reason = risk.evaluate(_short_signal(), portfolio,
                                        daily_trade_count=0, last_trade_at=None, now=NOW)
    assert not approved and "short exposure" in reason


def test_max_concurrent_shorts_enforced():
    risk = _risk(max_concurrent_shorts=1)
    portfolio = Portfolio(cash=1_000_000.0,
                          positions={"SBER": Position("SBER", -10, 100.0, 100.0)})
    approved, _, reason = risk.evaluate(_short_signal(), portfolio,
                                        daily_trade_count=0, last_trade_at=None, now=NOW)
    assert not approved and "concurrent shorts" in reason


def test_short_entry_approved_and_sized():
    risk = _risk()
    portfolio = Portfolio(cash=1_000_000.0, positions={})
    approved, order, _ = risk.evaluate(_short_signal(), portfolio,
                                       daily_trade_count=0, last_trade_at=None, now=NOW)
    assert approved and order is not None
    assert order.side == OrderSide.SELL and order.qty > 0
    # Размер не должен превышать долю капитала под шорт.
    assert order.qty * 100.0 <= 1_000_000.0 * 0.05 + 1e-6


# ── Стыковка времени (регрессия) ────────────────────────────────────────────

def test_row_ts_prefers_market_time_over_systime():
    """`systime` — время публикации записи биржей, а не время сделки: у
    январских данных оно бывает августовским. Если брать его первым, вся
    микроструктура пристыковывается к свечам мимо на месяцы."""
    from moex_agent.ml_offline_dataset import _parse_row_ts

    row = {
        "tradedate": "2024-01-03",
        "tradetime": "10:05:00",
        "systime": "2024-08-13 17:34:00",
        "disb": 0.1,
    }
    assert _parse_row_ts(row) == datetime(2024, 1, 3, 10, 5)

    # systime остаётся запасным вариантом, когда рыночного времени нет
    assert _parse_row_ts({"systime": "2024-08-13 17:34:00"}) == datetime(2024, 8, 13, 17, 34)


# ── ML-фильтр: набор признаков (регрессия) ──────────────────────────────────

def test_ml_filter_uses_model_own_feature_set(tmp_path):
    """Модель может быть обучена на подмножестве признаков (без микроструктуры).
    Раньше фильтр требовал ровно полный список и при несовпадении молча
    возвращал None — бот в этот момент торговал вообще без ML-отбора."""
    import json

    import joblib
    from lightgbm import LGBMClassifier

    from moex_agent.ml_features import FEATURE_COLUMNS, MICROSTRUCTURE_FEATURE_COLUMNS
    from moex_agent.ml_filter import MLBuyFilter

    columns = [c for c in FEATURE_COLUMNS if c not in MICROSTRUCTURE_FEATURE_COLUMNS]
    x = [[float(i + j) for j in range(len(columns))] for i in range(40)]
    y = ["up" if i % 2 else "not_up" for i in range(40)]
    model = LGBMClassifier(n_estimators=5, min_child_samples=2, verbosity=-1)
    model.fit(x, y, feature_name=columns)

    path = tmp_path / "m.joblib"
    joblib.dump(model, path)
    path.with_suffix(".meta.json").write_text(
        json.dumps({"feature_columns": columns}), encoding="utf-8"
    )

    flt = MLBuyFilter.load(path)
    assert flt is not None, "модель с урезанным набором признаков должна грузиться"
    assert len(flt.columns) == len(columns)
    p = flt.predict_up_probability(_features(100.0))
    assert 0.0 <= p <= 1.0
