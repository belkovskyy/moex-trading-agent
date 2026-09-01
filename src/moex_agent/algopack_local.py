"""Локальные данные ALGOPACK вместо обращений к API.

Выгрузка лежит на диске в parquet, корень задаётся через ALGOPACK_LOCAL_ROOT. Структура:

    <root>/<market>/<dataset>/<TICKER>.parquet
    market  ∈ stocks | futures | currency
    dataset ∈ tradestats | orderstats | obstats

Поля в файлах — те же, что отдаёт API (`disb`, `vol`, `put_vol_b`,
`imbalance_vol`, `spread_bbo`, `tradedate`/`tradetime`), поэтому модуль
возвращает данные в том же виде, что `ml_offline_dataset._fetch_symbol_rows`,
и подменяется без изменений в остальном коде.

Зачем: подписка на ALGOPACK истекает, а история — нет. Бэктест и сборка
датасета не должны зависеть от живого токена.

Свечи собираются из tradestats: там уже есть OHLC по пятиминуткам
(`pr_open/pr_high/pr_low/pr_close`) — отдельный запрос candles не нужен.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

from moex_agent.models import Candle

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path(os.getenv("ALGOPACK_LOCAL_ROOT", "data/algopack"))
MARKETS = ("stocks", "futures", "currency")
DATASETS = ("tradestats", "orderstats", "obstats")


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    """Читает parquet в список словарей. Пустой файл — маркер «данных нет»."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("нужен pandas + pyarrow: conda install -c conda-forge pandas pyarrow") from exc
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("не прочитал %s: %s", path.name, exc)
        return []
    return df.to_dict("records")


def _in_window(row: dict[str, Any], start: date | None, end: date | None) -> bool:
    raw = row.get("tradedate")
    if raw is None:
        return True
    if isinstance(raw, datetime):
        day = raw.date()
    elif isinstance(raw, date):
        day = raw
    else:
        try:
            day = date.fromisoformat(str(raw)[:10])
        except ValueError:
            return True
    if start and day < start:
        return False
    if end and day > end:
        return False
    return True


def find_symbol_file(symbol: str, dataset: str, *, root: Path = DEFAULT_ROOT) -> Path | None:
    """Ищет файл тикера по рынкам (акции → фьючерсы → валюта)."""
    for market in MARKETS:
        candidate = root / market / dataset / f"{symbol}.parquet"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _rows(symbol: str, dataset: str, *, root: Path, start: date | None, end: date | None) -> list[dict[str, Any]]:
    path = find_symbol_file(symbol, dataset, root=root)
    if path is None:
        return []
    return [r for r in _read_parquet(path) if _in_window(r, start, end)]


def _to_candle(symbol: str, row: dict[str, Any]) -> Candle | None:
    """Свеча из строки tradestats (OHLC уже посчитан биржей)."""
    tradedate, tradetime = row.get("tradedate"), row.get("tradetime")
    if tradedate is None or tradetime is None:
        return None
    try:
        ts = datetime.fromisoformat(f"{str(tradedate)[:10]}T{str(tradetime)}")
    except ValueError:
        return None
    try:
        close = float(row.get("pr_close") or 0.0)
        if close <= 0:
            return None
        return Candle(
            symbol=symbol,
            ts=ts,
            open=float(row.get("pr_open") or close),
            high=float(row.get("pr_high") or close),
            low=float(row.get("pr_low") or close),
            close=close,
            volume=float(row.get("vol") or 0.0),
        )
    except (TypeError, ValueError):
        return None


def fetch_symbol_rows_local(
    symbol: str,
    *,
    start: date | None = None,
    end: date | None = None,
    root: Path = DEFAULT_ROOT,
) -> tuple[list[Candle], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Аналог `_fetch_symbol_rows`, но из локальных parquet.

    Возвращает (candles, trade_rows, order_rows, book_rows).
    """
    trade_rows = _rows(symbol, "tradestats", root=root, start=start, end=end)
    order_rows = _rows(symbol, "orderstats", root=root, start=start, end=end)
    book_rows = _rows(symbol, "obstats", root=root, start=start, end=end)

    candles = [c for c in (_to_candle(symbol, r) for r in trade_rows) if c is not None]
    seen: set[datetime] = set()
    unique: list[Candle] = []
    for c in sorted(candles, key=lambda x: x.ts):
        if c.ts in seen:
            continue
        seen.add(c.ts)
        unique.append(c)

    logger.info(
        "%s: свечей %d, tradestats %d, orderstats %d, obstats %d",
        symbol, len(unique), len(trade_rows), len(order_rows), len(book_rows),
    )
    return unique, trade_rows, order_rows, book_rows


def available_symbols(*, root: Path = DEFAULT_ROOT, market: str = "stocks") -> list[str]:
    """Тикеры, по которым есть непустой tradestats."""
    directory = root / market / "tradestats"
    if not directory.exists():
        return []
    return sorted(
        p.stem for p in directory.glob("*.parquet") if p.stat().st_size > 0
    )


def coverage_report(symbols: list[str], *, root: Path = DEFAULT_ROOT) -> dict[str, dict[str, Any]]:
    """Что реально есть по каждому тикеру — чтобы не гадать перед сборкой."""
    out: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        info: dict[str, Any] = {}
        for dataset in DATASETS:
            path = find_symbol_file(symbol, dataset, root=root)
            info[dataset] = path is not None
        out[symbol] = info
    return out
