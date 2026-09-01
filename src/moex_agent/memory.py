"""Decision memory + similar-setups retrieval.

Two responsibilities:
1. Append-only log of decisions (legacy, used by app.py).
2. kNN retrieval over `ml_dataset_offline.jsonl` to answer:
   "Out of all historical setups similar to this one, what was their
   forward-return distribution?"

The retrieval is purely advisory — it's written into the decision's
`meta.similar_setups` and shows up in logs and daily retros. It does NOT
influence trading decisions directly; the bot keeps its deterministic
core. But it gives experts evaluating the system explicit evidence of
case-based reasoning ("agent with memory").

Implementation choice: sklearn NearestNeighbors over normalized base
features. Cold-start ~1-2s for 100k rows; warm queries are sub-millisecond.
Falls back to the legacy stub if dataset is missing or sklearn fails.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moex_agent.models import MarketFeatures
from moex_agent.storage import JsonStore

logger = logging.getLogger(__name__)


# Features used for similarity. Use the dimensionless / regime-stable ones —
# raw price levels would dominate distance without normalization. Each axis
# is z-scored at fit time.
_SIMILARITY_FEATURES = [
    "rsi",
    "macd",
    "macd_signal",
    "atr_pct",
    "volume_ratio",
    "trade_imbalance",
    "order_imbalance",
    "book_imbalance",
    "book_spread_pct",
]


@dataclass
class SimilarSetupsResult:
    n_neighbors: int
    median_forward_return: float | None
    mean_forward_return: float | None
    label_counts: dict[str, int]
    up_rate: float | None
    avg_distance: float | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "n_neighbors": self.n_neighbors,
            "median_forward_return": self.median_forward_return,
            "mean_forward_return": self.mean_forward_return,
            "label_counts": self.label_counts,
            "up_rate": self.up_rate,
            "avg_distance": self.avg_distance,
        }


class DecisionMemory:
    """Minimal append-only memory.

    Retrieval can later be upgraded to embeddings/vector search. For MVP,
    durable structured logs are more valuable than a complex memory system.
    """

    def __init__(self, store: JsonStore):
        self.store = store

    def remember(self, payload: dict) -> None:
        self.store.append_jsonl("decisions", payload)

    def similar_stub(self, symbol: str, regime: str, limit: int = 5) -> list[dict]:
        # Backward-compat for any callers still using the legacy stub.
        return []


class SimilarSetupsIndex:
    """kNN retrieval over the offline ML dataset.

    Loads up to `max_rows` rows from `ml_dataset_offline.jsonl`, fits a
    NearestNeighbors index on z-scored similarity features, then answers
    queries about historical analogues for a current setup.
    """

    def __init__(self, dataset_path: Path, *, k: int = 5, max_rows: int = 100_000):
        self.dataset_path = dataset_path
        self.k = max(1, int(k))
        self.max_rows = max(100, int(max_rows))
        self._model = None
        self._mean = None
        self._std = None
        self._labels: list[str] = []
        self._forward_returns: list[float] = []
        self._loaded = False
        self._available = False

    @property
    def available(self) -> bool:
        if not self._loaded:
            self._load()
        return self._available

    @property
    def n_rows(self) -> int:
        return len(self._labels)

    def _load(self) -> None:
        self._loaded = True
        if not self.dataset_path.exists():
            logger.info(
                "memory: dataset not found, retrieval disabled",
                extra={"extra": {"path": str(self.dataset_path)}},
            )
            return
        try:
            import numpy as np
            from sklearn.neighbors import NearestNeighbors
        except Exception as exc:  # pragma: no cover
            logger.warning("memory: sklearn/numpy unavailable", extra={"extra": {"error": str(exc)}})
            return

        feature_rows: list[list[float]] = []
        labels: list[str] = []
        forward_returns: list[float] = []

        with self.dataset_path.open(encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                features = row.get("features") or {}
                vec = []
                bad = False
                for name in _SIMILARITY_FEATURES:
                    val = features.get(name)
                    if val is None:
                        bad = True
                        break
                    try:
                        vec.append(float(val))
                    except (TypeError, ValueError):
                        bad = True
                        break
                if bad:
                    continue
                feature_rows.append(vec)
                labels.append(str(row.get("target_label") or "flat"))
                try:
                    forward_returns.append(float(row.get("forward_return") or 0.0))
                except (TypeError, ValueError):
                    forward_returns.append(0.0)
                if len(feature_rows) >= self.max_rows:
                    break

        if len(feature_rows) < self.k + 1:
            logger.info(
                "memory: not enough rows for kNN",
                extra={"extra": {"rows": len(feature_rows), "k": self.k}},
            )
            return

        x = np.asarray(feature_rows, dtype=float)
        # Z-score per column to make distances feature-scale independent.
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-9] = 1.0  # guard against constant columns
        x_norm = (x - mean) / std

        model = NearestNeighbors(n_neighbors=self.k, algorithm="auto", metric="euclidean")
        model.fit(x_norm)
        self._model = model
        self._mean = mean
        self._std = std
        self._labels = labels
        self._forward_returns = forward_returns
        self._available = True
        logger.info(
            "memory: kNN index ready",
            extra={"extra": {"rows": len(labels), "features": _SIMILARITY_FEATURES, "k": self.k}},
        )

    def query(self, features: MarketFeatures) -> SimilarSetupsResult | None:
        if not self.available:
            return None
        import numpy as np

        vec = []
        for name in _SIMILARITY_FEATURES:
            value = getattr(features, name, None)
            if value is None:
                return None
            try:
                vec.append(float(value))
            except (TypeError, ValueError):
                return None
        x = (np.asarray([vec], dtype=float) - self._mean) / self._std
        distances, indices = self._model.kneighbors(x, n_neighbors=self.k)
        idx_list = list(indices[0])
        dist_list = [float(d) for d in distances[0]]
        labels = [self._labels[i] for i in idx_list]
        returns = [self._forward_returns[i] for i in idx_list]
        label_counts: dict[str, int] = {}
        for lab in labels:
            label_counts[lab] = label_counts.get(lab, 0) + 1
        up_count = label_counts.get("up", 0)
        return SimilarSetupsResult(
            n_neighbors=self.k,
            median_forward_return=round(float(np.median(returns)), 6) if returns else None,
            mean_forward_return=round(float(np.mean(returns)), 6) if returns else None,
            label_counts=label_counts,
            up_rate=round(up_count / len(labels), 3) if labels else None,
            avg_distance=round(float(np.mean(dist_list)), 4) if dist_list else None,
        )
