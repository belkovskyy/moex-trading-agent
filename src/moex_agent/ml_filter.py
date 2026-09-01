from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

from moex_agent.ml_features import FEATURE_COLUMNS, feature_vector_from_market_features
from moex_agent.models import MarketFeatures

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)


class MLBuyFilter:
    def __init__(self, model: Any, *, up_label: str = "up", columns: list[str] | None = None):
        self.model = model
        self.up_label = up_label
        self.classes_ = [str(value) for value in getattr(model, "classes_", [])]
        # Набор признаков берётся у самой модели: она может быть обучена на
        # подмножестве (например, без микроструктуры). Раньше здесь жёстко
        # стоял полный список, и такая модель молча отвергалась как
        # «feature count mismatch» — фильтр отключался, а бот торговал без него.
        self.columns = columns or list(FEATURE_COLUMNS)

    @classmethod
    def load(cls, path: Path, *, positive_label: str = "up") -> "MLBuyFilter | None":
        """Load a probability filter. positive_label is the class whose
        probability predict_up_probability returns — "up" for the buy filter,
        "down" for the short filter (lgbm_short_filter)."""
        if not path.exists():
            logger.info("ml filter model not found", extra={"extra": {"path": str(path)}})
            return None
        try:
            import joblib
        except Exception as exc:
            logger.warning("joblib import failed for ml filter", extra={"extra": {"error": str(exc)}})
            return None
        try:
            model = joblib.load(path)
        except Exception as exc:
            logger.warning(
                "failed to load ml filter model",
                extra={"extra": {"path": str(path), "error": str(exc)}},
            )
            return None
        classes = [str(value) for value in getattr(model, "classes_", [])]
        if positive_label not in classes:
            logger.warning(
                "ml filter model missing positive class",
                extra={"extra": {"path": str(path), "positive_label": positive_label, "classes": classes}},
            )
            return None
        columns = cls._resolve_columns(path, model)
        n_model = int(getattr(model, "n_features_", getattr(model, "n_features_in_", -1)))
        if n_model > 0 and n_model != len(columns):
            logger.warning(
                "ml filter feature count mismatch. Retrain required.",
                extra={"extra": {"path": str(path), "model_n": n_model, "resolved_n": len(columns)}},
            )
            return None
        return cls(model, up_label=positive_label, columns=columns)

    @staticmethod
    def _resolve_columns(path: Path, model: Any) -> list[str]:
        """Признаки модели: сначала из meta-файла рядом, затем из самой модели,
        и лишь в последнюю очередь — полный список по умолчанию."""
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                import json
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                columns = meta.get("feature_columns")
                if isinstance(columns, list) and columns:
                    return [str(c) for c in columns]
            except Exception as exc:
                logger.warning("meta-файл модели не прочитан", extra={"extra": {"path": str(meta_path), "error": str(exc)}})
        names = getattr(model, "feature_name_", None) or getattr(model, "feature_names_in_", None)
        if names is not None and len(names) > 0:
            return [str(c) for c in names]
        return list(FEATURE_COLUMNS)

    def predict_up_probability(self, features: MarketFeatures) -> float:
        vector = [feature_vector_from_market_features(features, self.columns)]
        probabilities = self.model.predict_proba(vector)[0]
        class_probs = {str(label): float(prob) for label, prob in zip(self.model.classes_, probabilities, strict=False)}
        return class_probs.get(self.up_label, 0.0)
