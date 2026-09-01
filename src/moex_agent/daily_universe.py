"""Дневные свечи по широкой вселенной — сырьё для теста длинного горизонта.

Двадцати бумаг хватает интрадею, но кросс-секционная стратегия — это
ранжирование бумаг друг относительно друга, и на двадцати именах ранжировать
нечего. Здесь вселенная строится по обороту из истории ISS, дальше по каждой
бумаге качаются дневки и складываются в одну панель.

    python -m moex_agent.daily_universe --top 100 --start 2015-01-01
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from moex_agent.moex_iss import ISS_BASE, REQUEST_PAUSE, fetch_candles

logger = logging.getLogger(__name__)

INTERVAL_DAILY = 24
OUT_DIR = Path("data/daily")

HISTORY_URL = f"{ISS_BASE}/history/engines/stock/markets/shares/boards/TQBR/securities.json"


def _history_page(day: date, offset: int, attempts: int = 3) -> list[dict[str, Any]]:
    import requests

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                HISTORY_URL,
                params={"date": day.isoformat(), "start": offset, "iss.meta": "off",
                        "history.columns": "SECID,VALUE,NUMTRADES"},
                timeout=30,
            )
            response.raise_for_status()
            block = response.json()["history"]
            return [dict(zip(block["columns"], row)) for row in block["data"]]
        except Exception as exc:
            logger.warning("история за %s (offset %d), попытка %d: %s", day, offset, attempt, exc)
            time.sleep(2 * attempt)
    return []


def _history_day(day: date, shift_limit: int = 6) -> list[dict[str, Any]]:
    """ISS режет ответ по 100 строк и сортирует по тикеру: без пагинации
    вселенная получается алфавитной, а не по обороту. Выходные и праздники
    отдают пустой ответ, поэтому дата сдвигается вперёд до ближайших торгов."""
    for shift in range(shift_limit):
        probe = day + timedelta(days=shift)
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = _history_page(probe, offset)
            if not page:
                break
            rows.extend(page)
            offset += len(page)
            time.sleep(REQUEST_PAUSE)
        if rows:
            return rows
    logger.warning("нет торгов в окне с %s", day)
    return []


def build_universe(*, top: int, probe_dates: list[date]) -> list[str]:
    """Объединение топов по обороту на каждую пробную дату.

    Не «топ по среднему за весь период»: такой отбор оставляет только доживших
    до конца и выкидывает всё, что размещалось позже (YDEX, T, X5) или умерло
    раньше. Объединение годовых срезов даёт бумагу в выборке тогда, когда она
    реально была ликвидной, а фильтр по дате торгов делается уже в бэктесте.
    """
    best: dict[str, float] = {}
    for day in probe_dates:
        rows = _history_day(day)
        day_turnover = {
            row["SECID"]: float(row["VALUE"])
            for row in rows if row.get("SECID") and row.get("VALUE")
        }
        picked = sorted(day_turnover, key=lambda s: day_turnover[s], reverse=True)[:top]
        for secid in picked:
            best[secid] = max(best.get(secid, 0.0), day_turnover[secid])
        logger.info("%s: торгуется %d, в топ-%d отобрано %d, вселенная %d",
                    day, len(rows), top, len(picked), len(best))
        time.sleep(REQUEST_PAUSE)

    ranked = sorted(best, key=lambda s: best[s], reverse=True)
    logger.info("вселенная: %d бумаг за %d срезов", len(ranked), len(probe_dates))
    return ranked


def _probe_dates(start: date, end: date, per_year: int = 2) -> list[date]:
    months = [1, 7] if per_year == 2 else [1, 4, 7, 10]
    out = []
    for year in range(start.year, end.year + 1):
        for month in months:
            day = date(year, month, 15)
            if start <= day <= end:
                out.append(day)
    return out


def download(symbols: list[str], *, start: date, end: date, out_dir: Path) -> Path:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    missing: list[str] = []
    for i, symbol in enumerate(symbols, 1):
        try:
            candles = fetch_candles(symbol, start=start, end=end, interval=INTERVAL_DAILY)
        except Exception as exc:
            logger.warning("%s: не скачалось (%s)", symbol, exc)
            missing.append(symbol)
            continue
        if not candles:
            missing.append(symbol)
            continue
        frames.append(pd.DataFrame([{
            "symbol": symbol, "ts": c.ts, "open": c.open, "high": c.high,
            "low": c.low, "close": c.close, "volume": c.volume,
        } for c in candles]))
        logger.info("[%d/%d] %s: %d дней", i, len(symbols), symbol, len(candles))

    if missing:
        logger.warning("без данных: %s", ", ".join(missing))
    panel = pd.concat(frames, ignore_index=True).sort_values(["ts", "symbol"])
    path = out_dir / f"panel_{start.isoformat()}_{end.isoformat()}.parquet"
    panel.to_parquet(path, index=False)
    logger.info("панель: %d строк, %d бумаг → %s",
                len(panel), panel["symbol"].nunique(), path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Дневные свечи по широкой вселенной")
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--symbols", default="", help="готовый список через запятую")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start = datetime.fromisoformat(args.start).date()
    end = datetime.fromisoformat(args.end).date()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = build_universe(top=args.top, probe_dates=_probe_dates(start, end))
        (args.out).mkdir(parents=True, exist_ok=True)
        (args.out / "universe.txt").write_text("\n".join(symbols), encoding="utf-8")

    download(symbols, start=start, end=end, out_dir=args.out)


if __name__ == "__main__":
    main()
