"""Свечи с MOEX ISS — бесплатный официальный источник, без токена.

Зачем: подписка ALGOPACK даёт микроструктуру (дисбалансы, стакан), но обычные
свечи и индексы биржа отдаёт открыто. Это закрывает две дыры разом — тикеры,
которых нет в локальной выгрузке (SBER, LKOH и др.), и макро-признаки (IMOEX,
отраслевые индексы), которые без API подставлялись нулями.

Ограничение API: 500 свечей на запрос, поэтому пагинация по `start`.
Скачанное кладётся в parquet-кэш, чтобы не дёргать биржу повторно.

    from moex_agent.moex_iss import fetch_candles, fetch_macro_bundle_iss
    candles = fetch_candles("SBER", start=date(2023,1,1), end=date(2026,7,1))
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from moex_agent.models import Candle

logger = logging.getLogger(__name__)

ISS_BASE = "https://iss.moex.com/iss"
PAGE_SIZE = 500          # жёсткий предел ISS на один ответ
REQUEST_PAUSE = 0.12     # вежливая пауза между запросами
INTERVAL_10MIN = 10

CACHE_DIR = Path("data/iss_cache")

# Индексы, которые нужны как макро-контекст.
INDEX_TICKERS = {"IMOEX", "MOEXOG", "MOEXFN", "MOEXMM", "MOEXCN", "MOEXTL", "MOEXTN", "MOEXRE", "MOEXIT"}


def _endpoint(security: str) -> str:
    market = "index" if security.upper() in INDEX_TICKERS else "shares"
    return f"{ISS_BASE}/engines/stock/markets/{market}/securities/{security}/candles.json"


def _fetch_page(
    security: str, *, start: date, end: date, offset: int, interval: int,
    attempts: int = 4,
) -> list[list[Any]]:
    """Одна страница свечей с повторами.

    Домашняя сеть/VPN отваливается на длинных выкачках (getaddrinfo failed),
    и без повторов половина тикеров молча остаётся без данных.
    """
    import requests

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                _endpoint(security),
                params={
                    "from": start.isoformat(),
                    "till": end.isoformat(),
                    "interval": interval,
                    "start": offset,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["candles"]["data"]
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2.0 * attempt, 10.0))
    raise RuntimeError(f"{security}: страница offset={offset} не скачалась: {last_error}")


def _rows_to_candles(security: str, rows: Iterable[list[Any]]) -> list[Candle]:
    """Строки ISS: [open, close, high, low, value, volume, begin, end]."""
    out: list[Candle] = []
    for row in rows:
        try:
            ts = datetime.fromisoformat(str(row[6]))
            close = float(row[1])
            if close <= 0:
                continue
            out.append(Candle(
                symbol=security,
                ts=ts,
                open=float(row[0]),
                high=float(row[2]),
                low=float(row[3]),
                close=close,
                volume=float(row[5] or 0.0),
            ))
        except (IndexError, TypeError, ValueError):
            continue
    return out


def _cache_path(security: str, start: date, end: date, interval: int) -> Path:
    return CACHE_DIR / f"{security}_{interval}m_{start.isoformat()}_{end.isoformat()}.parquet"


def fetch_candles(
    security: str,
    *,
    start: date,
    end: date,
    interval: int = INTERVAL_10MIN,
    use_cache: bool = True,
) -> list[Candle]:
    """Все свечи инструмента за период, с пагинацией и кэшем на диск."""
    cache_file = _cache_path(security, start, end, interval)
    if use_cache and cache_file.exists() and cache_file.stat().st_size > 0:
        try:
            import pandas as pd
            frame = pd.read_parquet(cache_file)
            return [
                Candle(symbol=security, ts=row.ts, open=row.open, high=row.high,
                       low=row.low, close=row.close, volume=row.volume)
                for row in frame.itertuples()
            ]
        except Exception as exc:
            logger.warning("кэш %s не прочитан (%s), качаю заново", cache_file.name, exc)

    candles: list[Candle] = []
    offset = 0
    complete = False
    while True:
        try:
            rows = _fetch_page(security, start=start, end=end, offset=offset, interval=interval)
        except Exception as exc:
            # Обрыв посреди пагинации: данные неполные, и кэшировать их нельзя —
            # иначе следующий запуск примет огрызок за готовую историю.
            logger.warning("%s: выкачка прервана на offset=%d: %s", security, offset, exc)
            break
        if not rows:
            complete = True
            break
        candles.extend(_rows_to_candles(security, rows))
        # Короткая страница = данные кончились. Тот же самый признак когда-то
        # оборвал выкачку ALGOPACK на середине, поэтому здесь он безопасен
        # только потому, что ISS отдаёт ровно PAGE_SIZE, пока есть что отдавать.
        if len(rows) < PAGE_SIZE:
            complete = True
            break
        offset += len(rows)
        time.sleep(REQUEST_PAUSE)

    unique: dict[datetime, Candle] = {c.ts: c for c in candles}
    result = [unique[ts] for ts in sorted(unique)]

    if use_cache and result and complete:
        try:
            import pandas as pd
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{
                "ts": c.ts, "open": c.open, "high": c.high,
                "low": c.low, "close": c.close, "volume": c.volume,
            } for c in result]).to_parquet(cache_file, index=False)
        except Exception as exc:
            logger.warning("кэш не записан: %s", exc)

    logger.info("%s: %d свечей %s — %s", security, len(result), start, end)
    return result


def fetch_macro_bundle_iss(
    symbols: list[str],
    *,
    start: date,
    end: date,
    interval: int = INTERVAL_10MIN,
) -> dict[str, Any]:
    """IMOEX + отраслевые индексы по списку бумаг, в формате MacroSeries.

    Возвращает то же, что `macro_data.fetch_macro_bundle`, поэтому подставляется
    в сборщик датасета без правок на стороне потребителя.
    """
    from moex_agent.macro_data import INDEX_TICKER, SECTOR_INDEX_MAP, MacroSeries

    needed = {INDEX_TICKER}
    for symbol in symbols:
        sector = SECTOR_INDEX_MAP.get(symbol.upper())
        if sector:
            needed.add(sector)

    bundle: dict[str, Any] = {}
    for index_name in sorted(needed):
        candles = fetch_candles(index_name, start=start, end=end, interval=interval)
        bundle[index_name] = MacroSeries(
            name=index_name,
            times=[c.ts for c in candles],
            closes=[c.close for c in candles],
        )
    return bundle
