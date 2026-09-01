"""Offline dataset builder.

Pulls historical candles + super-features from ALGOPACK for a watchlist
window and produces a labeled jsonl dataset suitable for training the
ml_filter LightGBM model.

Two labeling modes:

- `--target-mode return` (legacy): label by forward return over N candles.
  Labels are 'up' / 'flat' / 'down' based on +/- TARGET_UP_THRESHOLD.
  Problem: a forward return through 30 minutes ignores how the trade would
  actually be exited. A position that briefly hits +1% then closes flat
  gets labeled 'flat' even though the trade would have been profitable.

- `--target-mode triple_barrier` (recommended for buy-filter): for each
  candidate moment, walk forward up to H candles and check whether the
  upper barrier (price * (1+up)) was hit BEFORE the lower barrier
  (price * (1-down)). Labels are 'up' (TP first), 'down' (SL first), or
  'flat' (timeout). This matches how the trade is actually managed.

The full PDF watchlist is exposed via `--symbols all`. Going back >12 months
is supported via `--days N` (ALGOPACK serves history from 2020).
"""
from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from moex_agent.config import settings
from moex_agent.indicators import build_features
from moex_agent.macro_data import (
    candles_to_pairs,
    compute_macro_features,
    fetch_macro_bundle,
)
from moex_agent.market_data import _mid_price_from_book, _ratio_diff, _row_to_candle, _safe_float
from moex_agent.ml_data import label_forward_return
from moex_agent.ml_features import MACRO_FEATURE_COLUMNS, feature_dict_from_base
from moex_agent.models import Action, Candle
from moex_agent.regime import detect_regime
from moex_agent.strategy import WATCHLIST, generate_signal


# Full set of tradable MOEX tickers in the instrument universe.
FULL_WATCHLIST = [
    "LKOH", "SBER", "ROSN", "GAZP", "VTBR", "YDEX", "PLZL", "T",
    "NVTK", "X5", "GMKN", "MGNT", "ALRS", "AFLT", "CHMF", "NLMK",
    "MOEX", "SNGSP", "MTSS", "PIKK",
]


def _default_dataset_path() -> Path:
    return settings.runtime_data_dir / "logs" / "ml_dataset_offline.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=_default_dataset_path())
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(WATCHLIST),
        help="Comma-separated tickers, or 'all' for the full PDF watchlist.",
    )
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--period", type=str, default=settings.candle_period)
    parser.add_argument(
        "--target-mode",
        choices=["return", "triple_barrier"],
        default="triple_barrier",
        help="Labeling scheme. 'triple_barrier' is recommended for a buy filter.",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=settings.ml_label_horizon_steps,
        help="Number of forward candles considered. For triple-barrier this is "
             "the max horizon before timeout-flat label.",
    )
    parser.add_argument(
        "--barrier-up",
        type=float,
        default=0.012,
        help="Upper barrier as fraction of entry price (e.g. 0.012 = +1.2%, matches TAKE_PROFIT_PCT).",
    )
    parser.add_argument(
        "--barrier-down",
        type=float,
        default=0.008,
        help="Lower barrier as fraction of entry price (matches STOP_LOSS_PCT * a bit of slack).",
    )
    parser.add_argument("--min-candles", type=int, default=35)
    parser.add_argument("--use-super-features", action="store_true", default=settings.use_super_candles)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--local-parquet", type=str, default="",
        help="Строить из локальной выгрузки ALGOPACK (папка с stocks/futures/currency), "
             "а не через API. Не требует токена.",
    )
    parser.add_argument(
        "--iss", action="store_true",
        help="Свечи и макро-индексы брать с MOEX ISS (бесплатно, без токена). "
             "Совмещается с --local-parquet: оттуда возьмётся микроструктура.",
    )
    return parser.parse_args()


def _resolve_window(args: argparse.Namespace) -> tuple[date, date]:
    if args.start:
        start = date.fromisoformat(args.start)
    else:
        start = date.today() - timedelta(days=max(1, int(args.days)))
    end = date.fromisoformat(args.end) if args.end else date.today()
    if end < start:
        raise SystemExit("end date must not be earlier than start date")
    return start, end


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    raw = args.symbols.strip()
    if raw.lower() == "all":
        return list(FULL_WATCHLIST)
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _iter_rows(rows: Iterable[Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _parse_row_ts(row: dict[str, Any]) -> datetime | None:
    """Рыночное время строки.

    ВАЖНО: `systime` — это момент ПУБЛИКАЦИИ записи биржей, а не время сделки
    (у январских данных systime бывает августовским). Раньше он стоял первым в
    списке кандидатов, и микроструктура пристыковывалась к свечам по метке,
    промахивающейся на месяцы. Поэтому пара `tradedate`+`tradetime` идёт до
    него, а `systime` остаётся последним запасным вариантом.
    """
    candidates = [
        row.get("begin"),
        row.get("datetime"),
        row.get("time"),
        row.get("moment"),
    ]
    tradedate = row.get("tradedate") or row.get("date")
    tradetime = row.get("tradetime") or row.get("updatetime")
    if tradedate and tradetime:
        candidates.append(f"{str(tradedate)[:10]}T{tradetime}")
        candidates.append(f"{str(tradedate)[:10]} {tradetime}")
    candidates.append(row.get("systime"))

    for raw in candidates:
        if raw is None or raw == "":
            continue
        if isinstance(raw, datetime):
            return raw
        text = str(raw).strip()
        if not text:
            continue
        normalized = text.replace(" ", "T")
        for variant in (normalized, text):
            try:
                return datetime.fromisoformat(variant)
            except ValueError:
                continue
    return None


def _series_by_ts(rows: list[dict[str, Any]]) -> tuple[list[datetime], list[dict[str, Any]]]:
    """Sort rows by ts; return parallel (times, rows) for O(log N) bisect.

    The previous implementation returned `list[(ts, row)]` and `_latest_before`
    did a linear O(N) scan from the start. With 100k+ super-feature rows and
    70k+ candles, that was O(N*M) ~ 7 billion ops per ticker ~ 45 min/ticker.
    Bisect drops it to O(M log N) ~ 7s/ticker.
    """
    points: list[tuple[datetime, dict[str, Any]]] = []
    for row in rows:
        ts = _parse_row_ts(row)
        if ts is None:
            continue
        points.append((ts, row))
    points.sort(key=lambda item: item[0])
    times = [p[0] for p in points]
    parsed_rows = [p[1] for p in points]
    return times, parsed_rows


def _latest_before(series_index: tuple[list[datetime], list[dict[str, Any]]], ts: datetime) -> dict[str, Any] | None:
    times, rows = series_index
    if not times:
        return None
    # Find rightmost time <= ts. bisect_right gives insertion point after equals.
    idx = bisect.bisect_right(times, ts) - 1
    if idx < 0:
        return None
    return rows[idx]


_EMPTY_SERIES: tuple[list[datetime], list[dict[str, Any]]] = ([], [])


def _build_super_features(
    ts: datetime,
    *,
    trade_series: tuple[list[datetime], list[dict[str, Any]]],
    order_series: tuple[list[datetime], list[dict[str, Any]]],
    book_series: tuple[list[datetime], list[dict[str, Any]]],
) -> dict[str, float]:
    features: dict[str, float] = {}

    trade = _latest_before(trade_series, ts)
    if trade:
        trade_imbalance = _safe_float(trade.get("disb"))
        super_volume = _safe_float(trade.get("vol"))
        if trade_imbalance is not None:
            features["trade_imbalance"] = trade_imbalance
        if super_volume is not None:
            features["super_volume"] = super_volume

    order = _latest_before(order_series, ts)
    if order:
        put_b = _safe_float(order.get("put_vol_b"))
        put_s = _safe_float(order.get("put_vol_s"))
        order_imbalance = _ratio_diff(put_b, put_s)
        if order_imbalance is not None:
            features["order_imbalance"] = order_imbalance

    book = _latest_before(book_series, ts)
    if book:
        book_imbalance = _safe_float(book.get("imbalance_vol"))
        spread = _safe_float(book.get("spread_bbo"))
        mid = _mid_price_from_book(book)
        if book_imbalance is not None:
            features["book_imbalance"] = book_imbalance
        if spread is not None and mid > 0:
            features["book_spread_pct"] = spread / mid

    return features


def _retry(fn, *, attempts: int = 4, delay: float = 2.0, label: str = ""):
    """Small retry helper for ALGOPACK transient SSL/EOF errors."""
    import time
    last_exc = None
    for k in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if k == attempts:
                break
            wait = delay * (2 ** (k - 1))
            print(f"  {label}: attempt {k}/{attempts} failed ({exc.__class__.__name__}: {exc}); sleep {wait:.0f}s")
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def _date_chunks(start: date, end: date, *, chunk_days: int = 90) -> list[tuple[date, date]]:
    """Yield (chunk_start, chunk_end) pairs covering [start, end]."""
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=chunk_days - 1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _fetch_chunked(fetch_fn, *, start: date, end: date, label: str, chunk_days: int = 90) -> list:
    """Call fetch_fn(start_s, end_s) on each ~chunk_days slice and concat.

    ALGOPACK caps responses at ~10k rows regardless of date range. For 1+ year
    of 10min candles or super-candle stats this means we miss ~80% of the data
    without chunking. 90-day chunks fit well under the cap (~5-6k rows max)
    while halving the request count vs 45-day chunks.

    Per-chunk progress is printed to stdout so the user can see what's
    happening during the long fetch. ALGOPACK is slow (1-3s per request) and
    silent for 10+ min is anxiety-inducing.
    """
    all_rows: list = []
    chunks = _date_chunks(start, end, chunk_days=chunk_days)
    total = len(chunks)
    import sys
    for idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
        sub = _retry(
            lambda cs=chunk_start, ce=chunk_end: fetch_fn(cs.isoformat(), ce.isoformat()),
            label=f"{label} {chunk_start}..{chunk_end}",
        )
        rows = _iter_rows(sub)
        all_rows.extend(rows)
        # Compact one-line progress: "[3/13] LKOH candles 2024-02-15..2024-05-14  +2840 rows  total=8520"
        print(
            f"  [{idx:>2}/{total}] {label}  +{len(rows):>5} rows  total={len(all_rows):>6}",
            flush=True,
        )
        sys.stdout.flush()
    return all_rows


def _fetch_symbol_rows(symbol: str, *, start: date, end: date, period: str) -> tuple[list[Candle], list, list, list]:
    try:
        from moexalgo import Ticker, session
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("moexalgo is not installed. Run: pip install -r requirements.txt") from exc

    if not settings.moex_algo_token:
        raise SystemExit("MOEX_ALGO_TOKEN is empty.")

    session.TOKEN = settings.moex_algo_token
    ticker = Ticker(symbol)

    # Each endpoint is chunked individually. 45-day chunks fit comfortably
    # below ALGOPACK's ~10k-row cap even at 10min granularity (~3-4k rows).
    candles_rows = _fetch_chunked(
        lambda cs, ce: ticker.candles(start=cs, end=ce, period=period, native=True),
        start=start, end=end, label=f"{symbol} candles",
    )
    candles = [_row_to_candle(symbol, row) for row in candles_rows]
    candles = [candle for candle in candles if candle is not None]
    # Defensive dedup by timestamp (chunks can overlap on boundary dates).
    seen_ts: set = set()
    deduped: list[Candle] = []
    for c in candles:
        if c.ts in seen_ts:
            continue
        seen_ts.add(c.ts)
        deduped.append(c)
    candles = sorted(deduped, key=lambda c: c.ts)

    trade_rows = _fetch_chunked(
        lambda cs, ce: ticker.tradestats(start=cs, end=ce, native=True),
        start=start, end=end, label=f"{symbol} tradestats",
    )
    order_rows = _fetch_chunked(
        lambda cs, ce: ticker.orderstats(start=cs, end=ce, native=True),
        start=start, end=end, label=f"{symbol} orderstats",
    )
    book_rows = _fetch_chunked(
        lambda cs, ce: ticker.obstats(start=cs, end=ce, native=True),
        start=start, end=end, label=f"{symbol} obstats",
    )
    return candles, trade_rows, order_rows, book_rows


def triple_barrier_label(
    *,
    candles: list[Candle],
    entry_idx: int,
    horizon: int,
    barrier_up: float,
    barrier_down: float,
) -> tuple[str, int, float, datetime, float]:
    """Walk forward up to `horizon` candles. Return whichever barrier hits first.

    Returns (label, steps_used, forward_return_at_exit, exit_ts, exit_price).
    Labels: 'up' (upper barrier hit), 'down' (lower barrier hit), 'flat' (timeout).
    """
    entry = candles[entry_idx]
    entry_price = entry.close
    upper = entry_price * (1.0 + barrier_up)
    lower = entry_price * (1.0 - barrier_down)
    last_idx = min(entry_idx + horizon, len(candles) - 1)
    for j in range(entry_idx + 1, last_idx + 1):
        c = candles[j]
        # Intra-candle high/low so the model learns the actual path a position
        # would have taken, not just close-to-close.
        hit_up = c.high >= upper
        hit_down = c.low <= lower
        if hit_up and hit_down:
            # Both barriers within one candle — direction of close vs open is a
            # reasonable proxy for which side was hit first. Down-close ->
            # likely went up then down (hit_down last) -> hit_up first.
            # Up-close   -> likely went down then up (hit_up last) -> hit_down first.
            # Flat close -> mark as 'flat' (uncertain), still consume the horizon.
            if c.close > c.open:
                return "up", j - entry_idx, (upper - entry_price) / entry_price, c.ts, upper
            if c.close < c.open:
                return "down", j - entry_idx, (lower - entry_price) / entry_price, c.ts, lower
            return "flat", j - entry_idx, (c.close - entry_price) / entry_price, c.ts, c.close
        if hit_up:
            return "up", j - entry_idx, (upper - entry_price) / entry_price, c.ts, upper
        if hit_down:
            return "down", j - entry_idx, (lower - entry_price) / entry_price, c.ts, lower
    # Timeout: use close of the last candle in the window.
    final = candles[last_idx]
    return "flat", last_idx - entry_idx, (final.close - entry_price) / entry_price, final.ts, final.close


def build_offline_dataset(args: argparse.Namespace) -> None:
    start, end = _resolve_window(args)
    symbols = _resolve_symbols(args)
    dataset_path: Path = args.dataset
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if args.replace and dataset_path.exists():
        dataset_path.unlink()

    label_counts: Counter[str] = Counter()
    rows_by_symbol: Counter[str] = Counter()
    total_rows = 0
    print(
        f"Building dataset:\n"
        f"  symbols   : {symbols}\n"
        f"  window    : {start} -> {end} ({(end-start).days} days)\n"
        f"  period    : {args.period}\n"
        f"  target    : {args.target_mode}\n"
        f"  horizon   : {args.horizon_steps} candles\n"
        f"  barriers  : +{args.barrier_up*100:.2f}% / -{args.barrier_down*100:.2f}% (triple_barrier only)\n"
        f"  super     : {args.use_super_features}\n"
        f"  output    : {dataset_path}\n"
    )

    # Fetch macro bundle once for the whole window (IMOEX + each unique
    # sector index referenced by the symbol list). This is much cheaper than
    # per-symbol re-fetching, and gives us O(log N) ts lookup later.
    print("Fetching macro bundle (IMOEX + sector indices)...", flush=True)
    try:
        if getattr(args, "iss", False):
            from moex_agent.moex_iss import fetch_macro_bundle_iss
            macro_bundle = fetch_macro_bundle_iss(symbols, start=start, end=end)
            source_label = "macro (ISS)"
        else:
            macro_bundle = fetch_macro_bundle(
                symbols,
                start=start,
                end=end,
                period=args.period,
                chunk_days=90,
            )
            source_label = "macro"
        print(f"  {source_label}: { {k: len(v) for k, v in macro_bundle.items()} }", flush=True)
    except Exception as exc:
        print(f"  macro fetch failed ({exc}); proceeding with zero macro features.")
        macro_bundle = {}

    with dataset_path.open("a", encoding="utf-8") as output:
        for symbol in symbols:
            try:
                if getattr(args, "iss", False):
                    # Свечи с ISS (бесплатно, все тикеры), микроструктура — из
                    # локальной выгрузки ALGOPACK, если она есть по этой бумаге.
                    from moex_agent.moex_iss import fetch_candles
                    candles = fetch_candles(symbol, start=start, end=end)
                    trade_rows = order_rows = book_rows = []
                    if getattr(args, "local_parquet", None):
                        from moex_agent.algopack_local import fetch_symbol_rows_local
                        _, trade_rows, order_rows, book_rows = fetch_symbol_rows_local(
                            symbol, start=start, end=end, root=Path(args.local_parquet)
                        )
                elif getattr(args, "local_parquet", None):
                    from moex_agent.algopack_local import fetch_symbol_rows_local
                    candles, trade_rows, order_rows, book_rows = fetch_symbol_rows_local(
                        symbol, start=start, end=end, root=Path(args.local_parquet)
                    )
                else:
                    candles, trade_rows, order_rows, book_rows = _fetch_symbol_rows(
                        symbol, start=start, end=end, period=args.period
                    )
            except Exception as exc:
                print(f"{symbol}: fetch failed: {exc}")
                continue
            if len(candles) < max(args.min_candles, args.horizon_steps + 2):
                print(f"{symbol}: skipped, not enough candles ({len(candles)})")
                continue

            trade_series = _series_by_ts(trade_rows) if args.use_super_features else _EMPTY_SERIES
            order_series = _series_by_ts(order_rows) if args.use_super_features else _EMPTY_SERIES
            book_series = _series_by_ts(book_rows) if args.use_super_features else _EMPTY_SERIES

            symbol_rows = 0
            last_index = len(candles) - args.horizon_steps - 1
            # All indicators need at most ~36 prior candles (MACD=35, Bollinger=20,
            # RSI=14, ATR=14). Slicing the whole history each iteration was O(N²)
            # — for 70k candles that's billions of ops. Constant-size window is
            # all we need.
            FEATURE_LOOKBACK = 60
            # Macro features need a (ts, close) view of the symbol's own
            # recent history for correlation/rel_strength calculations. Build
            # the full sorted pairs once per symbol — it's already O(N), and
            # we'll slice into it cheaply inside the loop via the trailing
            # FEATURE_LOOKBACK window.
            symbol_pairs_full = candles_to_pairs(candles)
            for idx in range(args.min_candles - 1, last_index):
                history_start = max(0, idx + 1 - FEATURE_LOOKBACK)
                history = candles[history_start: idx + 1]
                super_features = _build_super_features(
                    history[-1].ts,
                    trade_series=trade_series,
                    order_series=order_series,
                    book_series=book_series,
                )
                # Macro features at this row's timestamp. The trailing
                # FEATURE_LOOKBACK window of ticker pairs is enough for the
                # 60-min correlation and rel-strength calcs.
                macro_features = compute_macro_features(
                    ts=history[-1].ts,
                    symbol=symbol,
                    symbol_closes_recent=symbol_pairs_full[history_start: idx + 1],
                    bundle=macro_bundle,
                ) if macro_bundle else {name: 0.0 for name in MACRO_FEATURE_COLUMNS}
                features = build_features(history, super_features, macro_features)
                regime = detect_regime(features)
                signal = generate_signal(features, regime)

                # Triple-barrier labeling walks forward through HLC, not just close.
                if args.target_mode == "triple_barrier":
                    label, steps_used, forward_return, target_ts, target_price = triple_barrier_label(
                        candles=candles,
                        entry_idx=idx,
                        horizon=args.horizon_steps,
                        barrier_up=args.barrier_up,
                        barrier_down=args.barrier_down,
                    )
                    horizon_used = steps_used
                else:
                    future = candles[idx + args.horizon_steps]
                    forward_return = (future.close - features.price) / features.price
                    label = label_forward_return(forward_return)
                    horizon_used = args.horizon_steps
                    target_ts = future.ts
                    target_price = future.close

                # Base features dict — includes BASE columns (microstructure +
                # candle indicators) plus MACRO columns (cross-instrument
                # context). feature_dict_from_base() picks both up from this
                # mapping during training; for inference, MarketFeatures carries
                # the same fields and feature_vector_from_market_features()
                # reads them through getattr().
                base_features_dict = {
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
                    **macro_features,
                }

                row = {
                    "source": "offline_algopack",
                    "symbol": symbol,
                    "feature_ts": features.feature_ts.isoformat() if features.feature_ts else None,
                    "horizon_steps": horizon_used,
                    "target_ts": target_ts.isoformat() if isinstance(target_ts, datetime) else None,
                    "target_price": target_price,
                    "forward_return": round(forward_return, 6),
                    "target_label": label,
                    "regime": regime.value,
                    "signal_action": signal.action.value,
                    "signal_confidence": round(signal.confidence, 4),
                    "signal_reason": signal.reason,
                    "approved": signal.action != Action.HOLD,
                    "features": base_features_dict,
                    "meta": {
                        "period": args.period,
                        "use_super_features": bool(args.use_super_features),
                        "window_start": start.isoformat(),
                        "window_end": end.isoformat(),
                        "target_mode": args.target_mode,
                        "barrier_up": args.barrier_up if args.target_mode == "triple_barrier" else None,
                        "barrier_down": args.barrier_down if args.target_mode == "triple_barrier" else None,
                    },
                }
                output.write(json.dumps(row, ensure_ascii=False))
                output.write("\n")
                label_counts[label] += 1
                rows_by_symbol[symbol] += 1
                symbol_rows += 1
                total_rows += 1

            print(f"{symbol}: {symbol_rows} rows")

    print(f"\nDataset: {dataset_path}")
    print(f"Total rows: {total_rows}")
    print("Label distribution:", dict(label_counts))
    print("Rows by symbol  :", dict(rows_by_symbol))


def main() -> None:
    args = _parse_args()
    build_offline_dataset(args)


if __name__ == "__main__":
    main()
