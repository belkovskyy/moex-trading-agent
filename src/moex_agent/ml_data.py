from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from moex_agent.models import Action, MarketFeatures, MarketRegime, Signal
from moex_agent.storage import JsonStore

TARGET_UP_THRESHOLD = 0.0015
TARGET_DOWN_THRESHOLD = -0.0015


@dataclass
class PendingObservation:
    symbol: str
    feature_ts: datetime
    price: float
    regime: str
    signal_action: str
    signal_confidence: float
    signal_reason: str
    approved: bool
    features: dict[str, Any]


class MLDataCollector:
    def __init__(self, store: JsonStore, *, horizon_steps: int = 3):
        self.store = store
        self.horizon_steps = max(1, int(horizon_steps))
        state = store.read_state("ml_dataset_pending") or {}
        raw_pending = state.get("pending", {}) if isinstance(state, dict) else {}
        self.pending: dict[str, list[PendingObservation]] = {}
        for symbol, items in raw_pending.items():
            restored: list[PendingObservation] = []
            for item in items or []:
                try:
                    restored.append(
                        PendingObservation(
                            symbol=str(item["symbol"]),
                            feature_ts=datetime.fromisoformat(str(item["feature_ts"])),
                            price=float(item["price"]),
                            regime=str(item["regime"]),
                            signal_action=str(item["signal_action"]),
                            signal_confidence=float(item["signal_confidence"]),
                            signal_reason=str(item["signal_reason"]),
                            approved=bool(item["approved"]),
                            features=dict(item["features"]),
                        )
                    )
                except Exception:
                    continue
            if restored:
                self.pending[str(symbol)] = restored

    def observe(
        self,
        *,
        features: MarketFeatures,
        regime: MarketRegime,
        signal: Signal,
        approved: bool,
    ) -> None:
        if features.feature_ts is None or features.price <= 0:
            return
        symbol = features.symbol
        queue = self.pending.setdefault(symbol, [])
        if queue and queue[-1].feature_ts == features.feature_ts:
            return

        observation = PendingObservation(
            symbol=symbol,
            feature_ts=features.feature_ts,
            price=features.price,
            regime=regime.value,
            signal_action=signal.action.value,
            signal_confidence=signal.confidence,
            signal_reason=signal.reason,
            approved=approved,
            features={
                "price": features.price,
                "rsi": features.rsi,
                "macd": features.macd,
                "macd_signal": features.macd_signal,
                "bollinger_low": features.bollinger_low,
                "bollinger_mid": features.bollinger_mid,
                "bollinger_high": features.bollinger_high,
                "atr_pct": features.atr_pct,
                "volume_ratio": features.volume_ratio,
                "trade_imbalance": features.trade_imbalance,
                "order_imbalance": features.order_imbalance,
                "book_imbalance": features.book_imbalance,
                "book_spread_pct": features.book_spread_pct,
                "super_volume": features.super_volume,
            },
        )
        queue.append(observation)
        if len(queue) > self.horizon_steps:
            anchor = queue.pop(0)
            forward_return = (features.price - anchor.price) / anchor.price
            label = label_forward_return(forward_return)
            self.store.append_jsonl(
                "ml_dataset",
                {
                    "symbol": anchor.symbol,
                    "feature_ts": anchor.feature_ts,
                    "horizon_steps": self.horizon_steps,
                    "target_ts": features.feature_ts,
                    "target_price": features.price,
                    "forward_return": round(forward_return, 6),
                    "target_label": label,
                    "regime": anchor.regime,
                    "signal_action": anchor.signal_action,
                    "signal_confidence": round(anchor.signal_confidence, 4),
                    "signal_reason": anchor.signal_reason,
                    "approved": anchor.approved,
                    "features": anchor.features,
                },
            )
        self._persist()

    def _persist(self) -> None:
        payload = {
            "horizon_steps": self.horizon_steps,
            "pending": {
                symbol: [
                    {
                        **asdict(item),
                        "feature_ts": item.feature_ts.isoformat(),
                    }
                    for item in items
                ]
                for symbol, items in self.pending.items()
            },
        }
        self.store.write_state("ml_dataset_pending", payload)


def label_forward_return(forward_return: float) -> str:
    if forward_return > TARGET_UP_THRESHOLD:
        return "up"
    if forward_return < TARGET_DOWN_THRESHOLD:
        return "down"
    return "flat"
