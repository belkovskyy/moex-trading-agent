"""Single source of truth for ML feature engineering.

Both `ml_offline_dataset` (training) and `ml_filter` (inference) must compute
features identically — otherwise train/serving skew silently destroys edge.

This module defines:
- BASE_FEATURE_COLUMNS: raw features already present in MarketFeatures.
- DERIVED_FEATURE_COLUMNS: cheap transformations computable from MarketFeatures
  alone (no candle history needed).
- TIME_FEATURE_COLUMNS: encoded time-of-day signals from feature_ts.
- MACRO_FEATURE_COLUMNS: cross-instrument features from macro_data.py.
- FEATURE_COLUMNS: the full ordered list used by the model.

The full vector is what the trained LightGBM expects.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from moex_agent.models import MarketFeatures


BASE_FEATURE_COLUMNS = [
    "price",
    "rsi",
    "macd",
    "macd_signal",
    "bollinger_low",
    "bollinger_mid",
    "bollinger_high",
    "atr_pct",
    "volume_ratio",
    "trade_imbalance",
    "order_imbalance",
    "book_imbalance",
    "book_spread_pct",
    "super_volume",
]

DERIVED_FEATURE_COLUMNS = [
    "rsi_centered",
    "macd_diff",
    "macd_diff_norm",
    "bb_position",
    "bb_width_pct",
    "price_to_bb_low",
]

TIME_FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "is_morning_session",
    "is_main_session",
    "is_evening_session",
]

MACRO_FEATURE_COLUMNS = [
    "imoex_return_60m",
    "imoex_return_4h",
    "sector_return_60m",
    "corr_price_imoex_60",
    "rel_strength_vs_imoex",
]

# Микроструктура — единственная группа, которой нет в бесплатных источниках
# (ISS даёт свечи и индексы, а дисбалансы и стакан — только платный AlgoPack).
# Выделена отдельно, чтобы можно было обучить модель без неё и увидеть в
# цифрах, сколько она реально добавляет — то есть за что берут деньги.
MICROSTRUCTURE_FEATURE_COLUMNS = [
    "trade_imbalance",
    "order_imbalance",
    "book_imbalance",
    "book_spread_pct",
    "super_volume",
]

FEATURE_COLUMNS: list[str] = (
    BASE_FEATURE_COLUMNS
    + DERIVED_FEATURE_COLUMNS
    + TIME_FEATURE_COLUMNS
    + MACRO_FEATURE_COLUMNS
)

FEATURE_COLUMNS_LEGACY = BASE_FEATURE_COLUMNS


_EPS = 1e-9
_MSK_OFFSET = timedelta(hours=3)


def _safe(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if f != f:
        return 0.0
    return f


def _derived_from_base(base: Mapping[str, float]) -> dict[str, float]:
    price = _safe(base.get("price"))
    rsi = _safe(base.get("rsi"))
    macd = _safe(base.get("macd"))
    macd_sig = _safe(base.get("macd_signal"))
    bb_low = _safe(base.get("bollinger_low"))
    bb_mid = _safe(base.get("bollinger_mid"))
    bb_high = _safe(base.get("bollinger_high"))
    bb_range = bb_high - bb_low
    return {
        "rsi_centered": rsi - 50.0,
        "macd_diff": macd - macd_sig,
        "macd_diff_norm": (macd - macd_sig) / max(price, _EPS),
        "bb_position": (price - bb_mid) / (bb_range + _EPS) if bb_range > 0 else 0.0,
        "bb_width_pct": bb_range / max(bb_mid, _EPS) if bb_mid > 0 else 0.0,
        "price_to_bb_low": (price - bb_low) / max(bb_low, _EPS) if bb_low > 0 else 0.0,
    }


def _time_features(feature_ts: datetime | str | None) -> dict[str, float]:
    if feature_ts is None:
        return {name: 0.0 for name in TIME_FEATURE_COLUMNS}
    if isinstance(feature_ts, str):
        try:
            feature_ts = datetime.fromisoformat(feature_ts.replace(" ", "T"))
        except ValueError:
            return {name: 0.0 for name in TIME_FEATURE_COLUMNS}
    if feature_ts.tzinfo is None:
        feature_ts = feature_ts.replace(tzinfo=timezone.utc)
    msk = feature_ts.astimezone(timezone(_MSK_OFFSET))
    hour = msk.hour + msk.minute / 60.0
    import math
    angle = 2.0 * math.pi * hour / 24.0
    total_min = msk.hour * 60 + msk.minute
    is_morning = 1.0 if 6 * 60 + 50 <= total_min < 9 * 60 + 50 else 0.0
    is_main = 1.0 if 9 * 60 + 50 <= total_min < 19 * 60 else 0.0
    is_evening = 1.0 if total_min >= 19 * 60 or total_min < 3 * 60 + 50 else 0.0
    return {
        "hour_sin": math.sin(angle),
        "hour_cos": math.cos(angle),
        "is_morning_session": is_morning,
        "is_main_session": is_main,
        "is_evening_session": is_evening,
    }


def _macro_from_features(features: MarketFeatures) -> dict[str, float]:
    return {name: _safe(getattr(features, name, 0.0)) for name in MACRO_FEATURE_COLUMNS}


def _macro_from_mapping(source: Mapping[str, Any]) -> dict[str, float]:
    return {name: _safe(source.get(name)) for name in MACRO_FEATURE_COLUMNS}


def feature_dict_from_market_features(features: MarketFeatures) -> dict[str, float]:
    """Полный словарь признаков — из него берётся ровно тот набор, на котором
    обучена конкретная модель (он может быть уже полного, см. --no-microstructure)."""
    base = {name: _safe(getattr(features, name, 0.0)) for name in BASE_FEATURE_COLUMNS}
    derived = _derived_from_base(base)
    time_feats = _time_features(features.feature_ts)
    macro = _macro_from_features(features)
    return {**base, **derived, **time_feats, **macro}


def feature_vector_from_market_features(
    features: MarketFeatures, columns: list[str] | None = None
) -> list[float]:
    full = feature_dict_from_market_features(features)
    return [full[name] for name in (columns or FEATURE_COLUMNS)]


def feature_dict_from_base(
    base_features: Mapping[str, Any],
    feature_ts: datetime | str | None,
) -> dict[str, float]:
    safe_base = {name: _safe(base_features.get(name)) for name in BASE_FEATURE_COLUMNS}
    derived = _derived_from_base(safe_base)
    time_feats = _time_features(feature_ts)
    macro = _macro_from_mapping(base_features)
    return {**safe_base, **derived, **time_feats, **macro}


def feature_vector_from_base(
    base_features: Mapping[str, Any],
    feature_ts: datetime | str | None,
) -> list[float]:
    full = feature_dict_from_base(base_features, feature_ts)
    return [full[name] for name in FEATURE_COLUMNS]
