"""Рекордер стакана: пишет снимки котировок в JSONL, ни с чем в боте не связан.

Задача одна — накопить данные, по которым цена круга считается офлайн для
любого объёма. Решений не принимает, в paper.py не лезет.

Источники:
  iss_l1 — бесплатный ISS: лучший бид/аск, спред, суммарная глубина по стороне.
           Десять уровней ISS отдаёт только подписчикам, поэтому здесь L1.
  alor   — заглушка под L2 через Alor OpenAPI, включается когда появится токен.

    python -m moex_agent.l2_recorder --estimate
    python -m moex_agent.l2_recorder --out data/l2
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import signal
import time
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MSK = ZoneInfo("Europe/Moscow")

ISS_BOARD_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"

# Двадцать бумаг замороженной конфигурации (config.arena_lot_sizes).
UNIVERSE = [
    "LKOH", "SBER", "ROSN", "GAZP", "YDEX", "NVTK", "GMKN", "MOEX", "MTSS", "PLZL",
    "MGNT", "ALRS", "AFLT", "CHMF", "NLMK", "SNGSP", "PIKK", "VTBR", "T", "X5",
]

MAIN_SESSION = (dtime(9, 55), dtime(18, 50))
EVENING_SESSION = (dtime(19, 5), dtime(23, 50))


class Source(Protocol):
    name: str

    def poll(self, symbols: list[str]) -> list[dict[str, Any]]:
        ...


class IssL1Source:
    """Один запрос на весь пул: ISS отдаёт marketdata списком."""

    name = "iss_l1"

    FIELDS = (
        "SECID", "BID", "BIDDEPTHT", "OFFER", "OFFERDEPTHT", "SPREAD",
        "LAST", "LASTCHANGE", "VALTODAY", "VOLTODAY", "NUMTRADES", "UPDATETIME",
    )

    def __init__(self, timeout: float = 10.0) -> None:
        import requests

        self._session = requests.Session()
        self._timeout = timeout

    def poll(self, symbols: list[str]) -> list[dict[str, Any]]:
        response = self._session.get(
            ISS_BOARD_URL,
            params={
                "iss.meta": "off",
                "iss.only": "marketdata",
                "securities": ",".join(symbols),
                "marketdata.columns": ",".join(self.FIELDS),
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        block = response.json()["marketdata"]
        columns = block["columns"]
        captured = datetime.now(MSK).isoformat(timespec="milliseconds")

        rows: list[dict[str, Any]] = []
        for values in block["data"]:
            raw = dict(zip(columns, values))
            secid = raw.get("SECID")
            if secid not in symbols:
                continue
            rows.append({
                "schema_version": SCHEMA_VERSION,
                "source": self.name,
                "ts": captured,
                "exchange_time": raw.get("UPDATETIME"),
                "symbol": secid,
                "bids": [[raw.get("BID"), raw.get("BIDDEPTHT")]],
                "asks": [[raw.get("OFFER"), raw.get("OFFERDEPTHT")]],
                "spread": raw.get("SPREAD"),
                "last": {
                    "price": raw.get("LAST"),
                    "change": raw.get("LASTCHANGE"),
                    "value_today": raw.get("VALTODAY"),
                    "volume_today": raw.get("VOLTODAY"),
                    "num_trades": raw.get("NUMTRADES"),
                },
            })
        return rows


class AlorL2Source:
    """Настоящие десять уровней. Ждёт refresh-токен, пока не реализован."""

    name = "alor_l2"

    def poll(self, symbols: list[str]) -> list[dict[str, Any]]:
        raise NotImplementedError("нужен ALOR_REFRESH_TOKEN в .env")


class DayWriter:
    """JSONL с ротацией по торговому дню. Разрывы — отдельным файлом.

    День пишется несжатым и пакуется при ротации: в gzip нельзя заглянуть,
    пока в него пишут, а если процесс убьют — файл останется без маркера конца
    и не прочитается штатным ридером целиком.
    """

    def __init__(self, out_dir: Path) -> None:
        self._dir = out_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._day: date | None = None
        self._handle: Any = None
        self._path: Path | None = None
        self._gaps = (self._dir / "gaps.jsonl").open("a", encoding="utf-8")

    def _compress_finished(self) -> None:
        if self._handle is None or self._path is None:
            return
        self._handle.close()
        target = self._path.with_suffix(".jsonl.gz")
        try:
            with self._path.open("rb") as src, gzip.open(target, "wb") as dst:
                dst.writelines(src)
            self._path.unlink()
            logger.info("упакован %s", target.name)
        except Exception as exc:
            logger.warning("не упаковался %s: %s", self._path.name, exc)

    def _rotate(self, day: date) -> None:
        self._compress_finished()
        self._path = self._dir / f"{day.isoformat()}.jsonl"
        self._handle = self._path.open("a", encoding="utf-8")
        self._day = day
        logger.info("пишем в %s", self._path)

    def write(self, rows: list[dict[str, Any]]) -> None:
        today = datetime.now(MSK).date()
        if today != self._day:
            self._rotate(today)
        assert self._handle is not None
        for row in rows:
            self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._handle.flush()

    def gap(self, reason: str, seconds: float, detail: str = "") -> None:
        """Без этого потом не отличить 'спреда не было' от 'нас не было'."""
        record = {
            "schema_version": SCHEMA_VERSION,
            "ts": datetime.now(MSK).isoformat(timespec="milliseconds"),
            "reason": reason,
            "seconds": round(seconds, 1),
            "detail": detail[:400],
        }
        self._gaps.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._gaps.flush()
        logger.warning("разрыв %s: %.1f с (%s)", reason, seconds, detail[:120])

    def close(self) -> None:
        self._compress_finished()
        self._gaps.close()


def in_session(moment: datetime, *, evening: bool) -> bool:
    if moment.weekday() >= 5:
        return False
    now = moment.time()
    windows = [MAIN_SESSION] + ([EVENING_SESSION] if evening else [])
    return any(start <= now <= end for start, end in windows)


def estimate_disk(symbols: int, interval: int, evening: bool) -> str:
    seconds = (8 * 3600 + 55 * 60) + (4 * 3600 + 45 * 60 if evening else 0)
    per_day = seconds // interval * symbols
    raw_mb = per_day * 320 / 1024 / 1024
    return (
        f"{per_day} строк в день, ~{raw_mb:.0f} МБ сырых "
        f"(~{raw_mb / 8:.0f} МБ под gzip), за месяц (21 торговый день) "
        f"~{raw_mb * 21 / 1024:.1f} ГБ сырых / ~{raw_mb * 21 / 8:.0f} МБ сжатых"
    )


def run(
    *, out_dir: Path, symbols: list[str], interval: int, evening: bool,
    source: Source, max_backoff: float = 300.0,
) -> None:
    writer = DayWriter(out_dir)
    stopping = False

    def _stop(*_: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    backoff = float(interval)
    last_ok: float | None = None

    while not stopping:
        now = datetime.now(MSK)
        if not in_session(now, evening=evening):
            last_ok = None
            time.sleep(min(60, interval))
            continue

        started = time.monotonic()
        try:
            rows = source.poll(symbols)
            if not rows:
                raise RuntimeError("пустой ответ")
            writer.write(rows)
            if last_ok is not None and started - last_ok > interval * 2:
                writer.gap("delay", started - last_ok, "снимки шли реже интервала")
            last_ok = started
            backoff = float(interval)
        except Exception as exc:  # сеть, ISS, что угодно — падать нельзя
            since = started - last_ok if last_ok else float(interval)
            writer.gap("poll_failed", since, f"{type(exc).__name__}: {exc}")
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
            continue

        time.sleep(max(0.0, interval - (time.monotonic() - started)))

    writer.close()
    logger.info("остановлен")


def main() -> None:
    parser = argparse.ArgumentParser(description="Рекордер котировок MOEX")
    parser.add_argument("--out", type=Path, default=Path("data/l2"))
    parser.add_argument("--interval", type=int, default=10, help="секунд между снимками")
    parser.add_argument("--symbols", default=",".join(UNIVERSE))
    parser.add_argument("--source", choices=("iss_l1", "alor"), default="iss_l1")
    parser.add_argument("--evening", action="store_true", help="писать и вечернюю сессию")
    parser.add_argument("--estimate", action="store_true", help="только оценка объёма")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.estimate:
        print(estimate_disk(len(symbols), args.interval, args.evening))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.out / "recorder.log", encoding="utf-8"),
        ],
    )

    source: Source = AlorL2Source() if args.source == "alor" else IssL1Source()
    logger.info("источник %s, бумаг %d, интервал %d с", source.name, len(symbols), args.interval)
    run(out_dir=args.out, symbols=symbols, interval=args.interval,
        evening=args.evening, source=source)


if __name__ == "__main__":
    main()
