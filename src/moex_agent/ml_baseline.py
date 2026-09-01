"""Baseline ML buy-filter trainer.

Trains a LightGBM binary classifier P(up) where the positive class is defined
by the dataset's `target_label`. Supports two labeling schemes (controlled at
dataset-build time, not here):

- `forward_return`: target_label in {up, flat, down} based on forward_return.
- `triple_barrier`: target_label in {up, flat, down}, where 'up' means the
  upper barrier was hit before the lower barrier within a horizon — this is
  what we actually want for a buy filter.

This module trains the filter as 'up' vs 'not_up'. Evaluation is purged
walk-forward CV (Lopez de Prado lite): we split the timeline into K folds
and embargo a window between train and test to prevent target leakage.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from moex_agent.config import settings
from moex_agent.ml_features import (
    FEATURE_COLUMNS,
    MICROSTRUCTURE_FEATURE_COLUMNS,
    feature_dict_from_base,
)

# Набор признаков для текущего обучения. Меняется только флагом
# --no-microstructure: он выкидывает группу, доступную лишь по платной
# подписке, чтобы измерить её вклад (обучить с ней и без неё и сравнить).
ACTIVE_FEATURE_COLUMNS: list[str] = list(FEATURE_COLUMNS)


def _dataset_path() -> Path:
    return settings.runtime_data_dir / "logs" / "ml_dataset.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=_dataset_path())
    parser.add_argument(
        "--task",
        choices=["binary_up", "binary_down", "multiclass"],
        default="binary_up",
        help="Binary 'up vs not_up' for a buy filter, 'down vs not_down' for a "
             "short filter (train on a dataset built with mirrored barriers).",
    )
    parser.add_argument("--entry-only", action="store_true", default=True,
                        help="Only train on rows where the strategy generated a buy signal. "
                             "Default True — see audit note on train/serving distribution match.")
    parser.add_argument("--all-rows", dest="entry_only", action="store_false",
                        help="Disable --entry-only and train on every candle (legacy).")
    parser.add_argument("--approved-only", action="store_true",
                        help="Restrict to candles that risk-layer would have approved.")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of purged walk-forward folds. 0 disables CV (single split).")
    parser.add_argument("--embargo-candles", type=int, default=10,
                        help="Candles to skip between train and test in CV — prevents target leakage.")
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None,
                        help="If set, save model + metadata JSON next to it.")
    parser.add_argument("--no-microstructure", action="store_true",
                        help="Обучить без микроструктурных признаков (дисбалансы, "
                             "стакан, super_volume). Нужно, чтобы измерить их вклад: "
                             "они доступны только по платной подписке ALGOPACK.")
    return parser.parse_args()


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace(" ", "T"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            feature_ts = row.get("feature_ts")
            if not feature_ts:
                continue
            try:
                row["_feature_ts_dt"] = _parse_ts(feature_ts)
            except Exception:
                continue
            rows.append(row)
    rows.sort(key=lambda item: item["_feature_ts_dt"])
    return rows


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    entry_only: bool,
    approved_only: bool,
) -> list[dict[str, Any]]:
    filtered = rows
    if entry_only:
        filtered = [row for row in filtered if str(row.get("signal_action") or "") == "buy"]
    if approved_only:
        filtered = [row for row in filtered if bool(row.get("approved"))]
    return filtered


def build_matrix(rows: list[dict[str, Any]], *, task: str) -> tuple[list[list[float]], list[str], list[float]]:
    """Return (X, y, sample_weights).

    sample_weights down-weights candles that share their forward window with
    neighboring rows — a lite version of Lopez de Prado's uniqueness weights.
    Without this, overlapping windows give the model 5-10x effective duplicate
    rows from the same forward move and overfit the validation metric.
    """
    x: list[list[float]] = []
    y: list[str] = []
    horizons: list[int] = []
    for row in rows:
        features = row.get("features") or {}
        full = feature_dict_from_base(features, row.get("feature_ts"))
        x.append([float(full[name]) for name in ACTIVE_FEATURE_COLUMNS])
        label = str(row.get("target_label") or "flat")
        if task == "binary_up":
            y.append("up" if label == "up" else "not_up")
        elif task == "binary_down":
            y.append("down" if label == "down" else "not_down")
        else:
            y.append(label)
        horizons.append(int(row.get("horizon_steps") or 1))
    # Uniqueness weight ~ 1 / horizon; longer windows overlap more so they
    # count less per row. For uniform horizon this is just a constant; for
    # mixed horizons (e.g. triple-barrier early hits) it does meaningful work.
    weights = [1.0 / max(1, h) for h in horizons]
    return x, y, weights


def _purged_walk_forward_indices(
    n: int,
    *,
    n_folds: int,
    embargo: int,
) -> list[tuple[list[int], list[int]]]:
    """Yield (train_idx, test_idx) for each purged walk-forward fold.

    Forward-expanding train, fixed-size test, embargo gap between them. This
    matches how the model will be used in production: train on past, predict
    on future, no leakage from overlapping label windows.
    """
    if n_folds < 2 or n < n_folds * 20:
        return [
            (list(range(0, int(n * 0.8))),
             list(range(int(n * 0.8) + embargo, n))),
        ]
    fold_size = n // (n_folds + 1)
    splits: list[tuple[list[int], list[int]]] = []
    for k in range(n_folds):
        train_end = fold_size * (k + 1)
        test_start = train_end + embargo
        test_end = test_start + fold_size
        if test_end > n:
            break
        splits.append(
            (list(range(0, train_end)), list(range(test_start, test_end)))
        )
    return splits


def _train_lightgbm(x: list[list[float]], y: list[str], weights: list[float], *, task: str):
    from lightgbm import LGBMClassifier

    params: dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbosity": -1,
        "feature_name": ACTIVE_FEATURE_COLUMNS,
    }
    if task in ("binary_up", "binary_down"):
        params["objective"] = "binary"
        params["class_weight"] = "balanced"
    else:
        params["objective"] = "multiclass"
        params["class_weight"] = "balanced"
    model = LGBMClassifier(**params)
    model.fit(x, y, sample_weight=weights, feature_name=ACTIVE_FEATURE_COLUMNS)
    return model


def _classification_metrics(model, x_test: list[list[float]], y_test: list[str]) -> dict[str, Any]:
    from sklearn.metrics import classification_report, roc_auc_score
    preds = model.predict(x_test)
    report_dict = classification_report(y_test, preds, output_dict=True, zero_division=0)
    auc = None
    classes = list(model.classes_)
    # Positive class is "up" for a buy filter, "down" for a short filter.
    pos = "up" if "up" in classes else ("down" if "down" in classes else None)
    if pos is not None:
        try:
            pos_idx = classes.index(pos)
            probs = model.predict_proba(x_test)[:, pos_idx]
            y_bin = [1 if v == pos else 0 for v in y_test]
            if 0 < sum(y_bin) < len(y_bin):
                auc = float(roc_auc_score(y_bin, probs))
        except Exception:
            auc = None
    return {"classification_report": report_dict, "roc_auc": auc, "positive_class": pos}


def train_baseline(
    path: Path,
    *,
    task: str,
    entry_only: bool,
    approved_only: bool,
    test_size: float,
    cv_folds: int,
    embargo: int,
    min_rows: int,
    output: Path | None,
) -> None:
    rows = load_rows(path)
    rows = filter_rows(rows, entry_only=entry_only, approved_only=approved_only)
    if len(rows) < min_rows:
        raise SystemExit(f"Not enough rows after filters: {len(rows)} < {min_rows}")

    try:
        import joblib  # noqa: F401
        from sklearn.metrics import classification_report  # noqa: F401
        import lightgbm  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"lightgbm, scikit-learn and joblib are required: {exc}") from exc

    x, y, weights = build_matrix(rows, task=task)
    n = len(x)
    print(f"Dataset: {path}")
    print(f"Rows after filters: {n}")
    print(f"Label distribution: {dict(Counter(y))}")
    print(f"Task: {task} | entry_only={entry_only} | approved_only={approved_only}")
    print(f"Feature columns ({len(ACTIVE_FEATURE_COLUMNS)}):")
    for col in ACTIVE_FEATURE_COLUMNS:
        print(f"  - {col}")
    print()

    # Walk-forward CV (purged).
    fold_metrics: list[dict[str, Any]] = []
    if cv_folds and cv_folds >= 2:
        print(f"Purged walk-forward CV: folds={cv_folds}, embargo={embargo} candles")
        splits = _purged_walk_forward_indices(n, n_folds=cv_folds, embargo=embargo)
        for k, (tr_idx, te_idx) in enumerate(splits, 1):
            if len(set(y[i] for i in tr_idx)) < 2 or len(set(y[i] for i in te_idx)) < 2:
                print(f"  Fold {k}: skipped (single-class slice)")
                continue
            x_tr = [x[i] for i in tr_idx]
            y_tr = [y[i] for i in tr_idx]
            w_tr = [weights[i] for i in tr_idx]
            x_te = [x[i] for i in te_idx]
            y_te = [y[i] for i in te_idx]
            model_k = _train_lightgbm(x_tr, y_tr, w_tr, task=task)
            metrics_k = _classification_metrics(model_k, x_te, y_te)
            pos = metrics_k.get("positive_class") or "up"
            base_pos = sum(1 for v in y_te if v == pos) / len(y_te)
            metrics_k["base_rate_pos"] = base_pos
            metrics_k["n_train"] = len(x_tr)
            metrics_k["n_test"] = len(x_te)
            fold_metrics.append(metrics_k)
            print(
                f"  Fold {k}: train={len(x_tr)} test={len(x_te)}  "
                f"AUC={metrics_k['roc_auc']}  base_{pos}={base_pos:.3f}"
            )
        print()

    # Final fit on the full series with a clean held-out tail for the saved
    # classification report. This is the model we actually ship.
    split_index = max(1, min(n - 1, int(n * (1.0 - test_size))))
    x_train, x_test = x[:split_index], x[split_index + embargo:]
    y_train, y_test = y[:split_index], y[split_index + embargo:]
    w_train = weights[:split_index]
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        raise SystemExit("Train or test slice is single-class. Adjust filters / time window.")

    model = _train_lightgbm(x_train, y_train, w_train, task=task)
    final_metrics = _classification_metrics(model, x_test, y_test)
    print("=== Final held-out report ===")
    from sklearn.metrics import classification_report as cr
    print(cr(y_test, model.predict(x_test), digits=4, zero_division=0))
    _pos = final_metrics.get("positive_class") or "up"
    print(f"ROC-AUC ({_pos}) : {final_metrics['roc_auc']}")
    print(f"Base rate {_pos} : {sum(1 for v in y_test if v == _pos) / len(y_test):.3f}")

    try:
        importances = sorted(
            zip(ACTIVE_FEATURE_COLUMNS, model.feature_importances_, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )
        print("\nFeature importances:")
        for name, importance in importances:
            print(f"  {name}: {importance}")
    except Exception:
        importances = []

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(model, output)
        meta = {
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "dataset_path": str(path),
            "rows_total": n,
            "rows_train": len(x_train),
            "rows_test": len(x_test),
            "task": task,
            "entry_only": entry_only,
            "approved_only": approved_only,
            "embargo_candles": embargo,
            "cv_folds": cv_folds,
            "feature_columns": ACTIVE_FEATURE_COLUMNS,
            "label_distribution": dict(Counter(y)),
            "final_metrics": final_metrics,
            "cv_metrics": fold_metrics,
            "feature_importances": [
                {"name": name, "importance": int(importance)} for name, importance in importances
            ],
            "lightgbm_params": model.get_params(),
        }
        meta_path = output.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\nModel saved : {output}")
        print(f"Metadata    : {meta_path}")


def main() -> None:
    args = _parse_args()
    if args.no_microstructure:
        global ACTIVE_FEATURE_COLUMNS
        ACTIVE_FEATURE_COLUMNS = [
            c for c in FEATURE_COLUMNS if c not in MICROSTRUCTURE_FEATURE_COLUMNS
        ]
        print(f"Микроструктура исключена: обучаю на {len(ACTIVE_FEATURE_COLUMNS)} "
              f"признаках вместо {len(FEATURE_COLUMNS)}")
    train_baseline(
        args.dataset,
        task=args.task,
        entry_only=args.entry_only,
        approved_only=args.approved_only,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        embargo=args.embargo_candles,
        min_rows=args.min_rows,
        output=args.output,
    )


if __name__ == "__main__":
    main()
