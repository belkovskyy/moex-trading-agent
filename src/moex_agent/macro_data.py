"""Macro market context — IMOEX and sector indices.

This is the cross-instrument layer that turns "this ticker's RSI is 32" into
"this ticker's RSI is 32 and IMOEX is down -1.2% in the last hour, sector is
flat, ticker is outperforming by +0.4%".

We keep this DELIBERATELY narrow:

- IMOEX and 8 sector indices only. No futures (Si/BR) — they require quarterly
  contract roll which adds complexity without clear ML payoff. Si/BR live in
  LLM brief instead.
- 10-min candles, same period as the per-symbol candle series.
- Bisect-based timestamp alignment, same pattern as super-feature alignment in
  ml_offline_dataset.

Both the offline dataset builder and the live market_data client use this
module. Train/serve parity is critical.
"""
from __future__ import annotations

import bisect
import logging
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from moex_agent.models import Candle

logger = logging.getLogger(__name__)


# Sector-index map. MOEX sector indices are computed by Moscow Exchange from
# member-stocks. Empty/missing here = ticker has no clean sector index.
# Sources: https://www.moex.com/ru/index/MOEXBC, MOEXOG, etc.
SECTOR_INDEX_MAP: dict[str, str] = {
    # Oil & gas
    "LKOH": "MOEXOG",
    "ROSN": "MOEXOG",
    "SNGSP": "MOEXOG",
    "NVTK": "MOEXOG",
    "GAZP": "MOEXOG",
    # Financials
    "SBER": "MOEXFN",
    "VTBR": "MOEXFN",
    "MOEX": "MOEXFN",
    "T": "MOEXFN",
    # Metals & mining
    "GMKN": "MOEXMM",
    "ALRS": "MOEXMM",
    "PLZL": "MOEXMM",
    "CHMF": "MOEXMM",
    "NLMK": "MOEXMM",
    # Consumer
    "MGNT": "MOEXCN",
    "X5": "MOEXCN",
    # Telecom
    "MTSS": "MOEXTL",
    # Transportation
    "AFLT": "MOEXTN",
    # Real estate
    "PIKK": "MOEXRE",
    # Information technology (Yandex)
    "YDEX": "MOEXIT",
}

INDEX_TICKER = "IMOEX"

# Time windows for derived features (in minutes).
RETURN_WINDOW_60M = 60
RETURN_WINDOW_4H = 240
CORR_WINDOW_60M = 60


@dataclass
class MacroSeries:
    """Time-indexed candle series for a single macro instrument.

    Sorted by ts ascending. Stored as parallel arrays for O(log N) bisect.
    """

    name: str
    times: list[datetime]
    closes: list[float]

    def __len__(self) -> int:
        return len(self.times)

    def latest_before(self, ts: datetime) -> tuple[datetime | None, float | None]:
        if not self.times:
            return None, None
        idx = bisect.bisect_right(self.times, ts) - 1
        if idx < 0:
            return None, None
        return self.times[idx], self.closes[idx]

    def return_over_minutes(self, ts: datetime, minutes: int) -> float | None:
        """Return ((close_at_ts - close_at_ts-minutes) / close_at_ts-minutes)."""
        _, end_close = self.latest_before(ts)
        if end_close is None or end_close <= 0:
            return None
        start_ts_target = ts - timedelta(minutes=minutes)
        _, start_close = self.latest_before(start_ts_target)
        if start_close is None or start_close <= 0:
            return None
        return (end_close - start_close) / start_close

    def closes_in_window(self, end_ts: datetime, minutes: int) -> list[float]:
        """Closes within [end_ts - minutes, end_ts], sorted by ts ascending."""
        if not self.times:
            return []
        start_ts = end_ts - timedelta(minutes=minutes)
        lo = bisect.bisect_left(self.times, start_ts)
        hi = bisect.bisect_right(self.times, end_ts)
        return self.closes[lo:hi]


def _row_to_index_close(row: dict[str, Any]) -> tuple[datetime, float] | None:
    """Index candles use the same 'begin' + 'close' schema as stock candles."""
    ts_raw = row.get("begin") or row.get("datetime") or row.get("time")
    close_raw = row.get("close")
    try:
        if isinstance(ts_raw, datetime):
            ts = ts_raw
        else:
            ts = datetime.fromisoformat(str(ts_raw))
        close = float(close_raw)
    except (TypeError, ValueError):
        return None
    return ts, close


def _date_chunks(start: date, end: date, *, chunk_days: int = 90) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=chunk_days - 1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _retry(fn, *, attempts: int = 4, delay: float = 2.0, label: str = ""):
    last_exc: BaseException | None = None
    for k in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - depends on network
            last_exc = exc
            if k == attempts:
                break
            wait = delay * (2 ** (k - 1))
            logger.warning("macro fetch %s: attempt %d/%d failed (%s); sleep %.0fs", label, k, attempts, exc, wait)
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def fetch_index_series(
    index_name: str,
    *,
    start: date,
    end: date,
    period: str = "10min",
    moex_algo_token: str | None = None,
    chunk_days: int = 90,
) -> MacroSeries:
    """Fetch 10-min candles for a MOEX index and return aligned MacroSeries.

    Args:
        index_name: 'IMOEX', 'MOEXOG', 'MOEXMM', etc.
        start, end: inclusive date range.
        period: candle period understood by moexalgo (e.g. '10min', '1h').
        moex_algo_token: ALGOPACK token. If None, settings.moex_algo_token is used.
        chunk_days: split the range into chunks of this size to bypass
            ALGOPACK's ~10k-row cap.
    """
    try:
        from moexalgo import session, Ticker
    except ImportError as exc:
        raise RuntimeError("moexalgo is not installed") from exc

    if moex_algo_token is None:
        from moex_agent.config import settings as _settings
        moex_algo_token = _settings.moex_algo_token
    if not moex_algo_token:
        raise RuntimeError("MOEX_ALGO_TOKEN is empty.")
    session.TOKEN = moex_algo_token

    # MOEX indices are accessed via Ticker in the installed moexalgo version.
    # IMOEX/MOEXOG/etc are recognized by secid and routed to the right board
    # internally. (Docs mention `moexalgo.indices.Index`, but that submodule
    # is absent in the version we have installed.)
    index_obj = Ticker(index_name)
    all_rows: list[tuple[datetime, float]] = []
    chunks = _date_chunks(start, end, chunk_days=chunk_days)
    for idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
        def _do(cs=chunk_start, ce=chunk_end):
            # Installed moexalgo uses `native=True` (returns list of dicts),
            # not `use_dataframe=False` (newer doc-version kwarg).
            return list(index_obj.candles(
                start=cs.isoformat(),
                end=ce.isoformat(),
                period=period,
                native=True,
            ))
        rows = _retry(_do, label=f"{index_name} {chunk_start}..{chunk_end}")
        chunk_parsed = 0
        for raw in rows:
            # moexalgo returns either Candle namedtuple, dict-like, or pd row.
            if hasattr(raw, "_asdict"):
                rec = raw._asdict()
            elif isinstance(raw, dict):
                rec = raw
            else:
                # last resort: use vars()
                rec = {k: getattr(raw, k) for k in dir(raw) if not k.startswith("_")}
            parsed = _row_to_index_close(rec)
            if parsed is not None:
                all_rows.append(parsed)
                chunk_parsed += 1
        # Compact progress matching the per-symbol fetcher in ml_offline_dataset:
        # "  [ 1/13] IMOEX candles  + 5312 rows  total=  5312"
        import sys
        print(
            f"  [{idx:>2}/{len(chunks)}] {index_name} candles  +{chunk_parsed:>5} rows  total={len(all_rows):>6}",
            flush=True,
        )
        sys.stdout.flush()

    # Sort + dedupe by ts.
    seen: set[datetime] = set()
    unique: list[tuple[datetime, float]] = []
    for ts, close in sorted(all_rows, key=lambda x: x[0]):
        if ts in seen:
            continue
        seen.add(ts)
        unique.append((ts, close))
    return MacroSeries(name=index_name, times=[u[0] for u in unique], closes=[u[1] for u in unique])


def fetch_macro_bundle(
    symbols: list[str],
    *,
    start: date,
    end: date,
    period: str = "10min",
    moex_algo_token: str | None = None,
    chunk_days: int = 90,
) -> dict[str, MacroSeries]:
    """Fetch IMOEX + all unique sector indices referenced by `symbols`.

    Returns dict keyed by index name. Tickers without a sector-index entry in
    SECTOR_INDEX_MAP simply don't pull a sector series.
    """
    needed: set[str] = {INDEX_TICKER}
    for symbol in symbols:
        sector = SECTOR_INDEX_MAP.get(symbol.upper())
        if sector:
            needed.add(sector)

    bundle: dict[str, MacroSeries] = {}
    for index_name in sorted(needed):
        try:
            bundle[index_name] = fetch_index_series(
                index_name,
                start=start,
                end=end,
                period=period,
                moex_algo_token=moex_algo_token,
                chunk_days=chunk_days,
            )
            logger.info("macro %s: %d candles", index_name, len(bundle[index_name]))
        except Exception as exc:  # pragma: no cover
            logger.warning("macro %s: fetch failed (%s); skipping", index_name, exc)
            bundle[index_name] = MacroSeries(name=index_name, times=[], closes=[])
    return bundle


def compute_macro_features(
    *,
    ts: datetime,
    symbol: str,
    symbol_closes_recent: list[tuple[datetime, float]],
    bundle: dict[str, MacroSeries],
) -> dict[str, float]:
    """Compute the 5 macro features for one (symbol, ts).

    Args:
        ts: the candle's begin timestamp.
        symbol: ticker (used for sector lookup).
        symbol_closes_recent: ticker's recent (ts, close) points, ASCENDING,
            containing AT LEAST the last 60 min before `ts`. Used for
            corr_price_imoex_60 and rel_strength.
        bundle: output of fetch_macro_bundle().

    Returns dict with keys present in `MACRO_FEATURE_COLUMNS`. All values are
    safe floats (0.0 instead of None/NaN) so feature_vector_from_base parity
    is preserved.
    """
    imoex = bundle.get(INDEX_TICKER)
    sector_name = SECTOR_INDEX_MAP.get(symbol.upper())
    sector = bundle.get(sector_name) if sector_name else None

    imoex_60m = imoex.return_over_minutes(ts, RETURN_WINDOW_60M) if imoex else None
    imoex_4h = imoex.return_over_minutes(ts, RETURN_WINDOW_4H) if imoex else None
    sector_60m = sector.return_over_minutes(ts, RETURN_WINDOW_60M) if sector else None

    # Ticker 60m return for rel_strength.
    ticker_60m: float | None = None
    if symbol_closes_recent:
        last_ts, last_close = symbol_closes_recent[-1]
        target_ts = ts - timedelta(minutes=RETURN_WINDOW_60M)
        # bisect on ticker history
        times = [t for t, _ in symbol_closes_recent]
        idx = bisect.bisect_right(times, target_ts) - 1
        if 0 <= idx < len(symbol_closes_recent):
            start_close = symbol_closes_recent[idx][1]
            if start_close > 0 and last_close > 0:
                ticker_60m = (last_close - start_close) / start_close

    rel_strength = None
    if ticker_60m is not None and imoex_60m is not None:
        rel_strength = ticker_60m - imoex_60m

    # Correlation of ticker returns vs IMOEX returns on the last 60-minute window.
    corr_price_imoex_60 = _rolling_correlation(symbol_closes_recent, imoex, ts, CORR_WINDOW_60M)

    def _safe(value: float | None) -> float:
        if value is None:
            return 0.0
        try:
            f = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f

    return {
        "imoex_return_60m": _safe(imoex_60m),
        "imoex_return_4h": _safe(imoex_4h),
        "sector_return_60m": _safe(sector_60m),
        "corr_price_imoex_60": _safe(corr_price_imoex_60),
        "rel_strength_vs_imoex": _safe(rel_strength),
    }


def _rolling_correlation(
    symbol_closes: list[tuple[datetime, float]],
    imoex: MacroSeries | None,
    end_ts: datetime,
    window_minutes: int,
) -> float | None:
    """Pearson correlation of period-over-period returns inside a time window."""
    if not symbol_closes or imoex is None or len(imoex) < 6:
        return None
    start_ts = end_ts - timedelta(minutes=window_minutes)
    # Pull symbol points within window.
    sym_window = [(ts, close) for ts, close in symbol_closes if start_ts <= ts <= end_ts]
    if len(sym_window) < 4:
        return None
    # Align by ts: for each symbol ts, find imoex close at that ts.
    pairs: list[tuple[float, float]] = []
    for ts, sym_close in sym_window:
        _, imoex_close = imoex.latest_before(ts)
        if imoex_close is None or imoex_close <= 0 or sym_close <= 0:
            continue
        pairs.append((sym_close, imoex_close))
    if len(pairs) < 4:
        return None
    # Convert prices to returns (skip first row).
    sym_ret = [(pairs[i][0] / pairs[i - 1][0]) - 1.0 for i in range(1, len(pairs)) if pairs[i - 1][0] > 0]
    idx_ret = [(pairs[i][1] / pairs[i - 1][1]) - 1.0 for i in range(1, len(pairs)) if pairs[i - 1][1] > 0]
    n = min(len(sym_ret), len(idx_ret))
    if n < 3:
        return None
    sym_ret, idx_ret = sym_ret[:n], idx_ret[:n]
    mean_s = sum(sym_ret) / n
    mean_i = sum(idx_ret) / n
    num = sum((sym_ret[k] - mean_s) * (idx_ret[k] - mean_i) for k in range(n))
    den_s = math.sqrt(sum((sym_ret[k] - mean_s) ** 2 for k in range(n)))
    den_i = math.sqrt(sum((idx_ret[k] - mean_i) ** 2 for k in range(n)))
    if den_s <= 0 or den_i <= 0:
        return None
    return num / (den_s * den_i)


def candles_to_pairs(candles: list[Candle]) -> list[tuple[datetime, float]]:
    """Convenience: convert Candle list to (ts, close) pairs sorted ASC."""
    return [(c.ts, c.close) for c in sorted(candles, key=lambda c: c.ts)]


# ----- live (low-latency) cached client ----------------------------------

class LiveMacroCache:
    """Thin cache for live trading loop.

    The trading loop runs every POLL_INTERVAL_SECONDS (default 30s). Fetching
    IMOEX + sector indices every cycle would add 5-9 API calls per cycle.
    Instead we cache the last 4h of each macro series and refresh once every
    `refresh_seconds`.
    """

    def __init__(self, *, refresh_seconds: float = 300.0, lookback_hours: int = 6, period: str = "10min",
                 use_iss: bool = False):
        self._refresh_seconds = float(refresh_seconds)
        self._lookback_hours = int(lookback_hours)
        self._period = period
        self._use_iss = use_iss
        self._cache: dict[str, MacroSeries] = {}
        self._last_refresh: float = 0.0
        self._symbols_seen: set[str] = set()

    def features_for(
        self,
        symbol: str,
        ts: datetime,
        symbol_closes_recent: list[tuple[datetime, float]],
    ) -> dict[str, float]:
        self._symbols_seen.add(symbol.upper())
        self._maybe_refresh()
        return compute_macro_features(
            ts=ts,
            symbol=symbol,
            symbol_closes_recent=symbol_closes_recent,
            bundle=self._cache,
        )

    def latest_imoex(self) -> tuple[datetime | None, float | None]:
        imoex = self._cache.get(INDEX_TICKER)
        if not imoex or not imoex.times:
            return None, None
        return imoex.times[-1], imoex.closes[-1]

    def force_refresh(self) -> None:
        self._last_refresh = 0.0
        self._maybe_refresh()

    def _maybe_refresh(self) -> None:
        now = time.time()
        if now - self._last_refresh < self._refresh_seconds and self._cache:
            return
        try:
            end = date.today()
            start = end - timedelta(days=max(2, self._lookback_hours // 24 + 1))
            symbols = sorted(self._symbols_seen) if self._symbols_seen else list(SECTOR_INDEX_MAP.keys())
            if self._use_iss:
                from moex_agent.moex_iss import fetch_macro_bundle_iss

                interval = {"1min": 1, "10min": 10, "60min": 60, "1h": 60}.get(self._period, 10)
                self._cache = fetch_macro_bundle_iss(
                    symbols, start=start, end=end, interval=interval
                )
            else:
                self._cache = fetch_macro_bundle(
                    symbols,
                    start=start,
                    end=end,
                    period=self._period,
                    chunk_days=30,
                )
            self._last_refresh = now
            logger.info("LiveMacroCache refreshed: %s", {k: len(v) for k, v in self._cache.items()})
        except Exception as exc:  # pragma: no cover - depends on network
            logger.warning("LiveMacroCache refresh failed: %s", exc)


# ----- LLM brief enrichment (FUTOI snapshot) ----------------------------

def get_futoi_snapshot(*, moex_algo_token: str | None = None) -> dict[str, Any]:
    """Pull latest FUTOI (open positions of phys/jur entities) on key futures.

    Returns a fail-soft dict. On any error returns {}, so the LLM brief never
    breaks on a transient ALGOPACK hiccup.

    Schema returned (when successful):
        {
            "Si": {"phys_long": int, "phys_short": int,
                   "jur_long": int, "jur_short": int,
                   "phys_net": int, "jur_net": int},
            "RTSI": {...},
            "BR": {...},   # optional
        }
    """
    try:
        from moexalgo import session, Market
    except ImportError:
        return {}
    if moex_algo_token is None:
        from moex_agent.config import settings as _settings
        moex_algo_token = _settings.moex_algo_token
    if not moex_algo_token:
        return {}
    session.TOKEN = moex_algo_token

    try:
        futures_market = Market("futures")
    except Exception:
        return {}

    interesting_secids = {"Si", "RTSI", "RTS", "BR", "GAZR", "SBRF", "LKOH"}
    rows: list[Any] = []
    # Installed moexalgo doesn't accept `latest`/`use_dataframe` kwargs on
    # futures_market.futoi() — use the canonical `date=<today>` + `native=True`
    # combo that matches the rest of the codebase.
    today = date.today()
    try:
        result = futures_market.futoi(date=today.isoformat(), native=True)
        rows = list(result)
    except TypeError:
        # Older moexalgo signature: positional or different kwargs. Try
        # date-less fallback (returns today's data by default in some versions).
        try:
            result = futures_market.futoi(native=True)
            rows = list(result)
        except Exception as exc:
            logger.warning("FUTOI snapshot failed (fallback): %s", exc)
            return {}
    except Exception as exc:
        logger.warning("FUTOI snapshot failed: %s", exc)
        return {}

    out: dict[str, Any] = {}
    for raw in rows:
        if hasattr(raw, "_asdict"):
            rec = raw._asdict()
        elif isinstance(raw, dict):
            rec = raw
        else:
            continue
        secid = str(rec.get("secid") or rec.get("symbol") or "").upper()
        if not secid:
            continue
        # Match by prefix because ALGOPACK FUTOI uses the underlying root
        # contract (e.g. "Si" or "Si-12.26").
        matched_root = None
        for root in interesting_secids:
            if secid.startswith(root.upper()):
                matched_root = root
                break
        if matched_root is None:
            continue

        phys_long = _safe_int(rec.get("phys_pos_long") or rec.get("fiz_long"))
        phys_short = _safe_int(rec.get("phys_pos_short") or rec.get("fiz_short"))
        jur_long = _safe_int(rec.get("jur_pos_long") or rec.get("ur_long"))
        jur_short = _safe_int(rec.get("jur_pos_short") or rec.get("ur_short"))
        if phys_long is None and jur_long is None:
            continue
        entry = out.setdefault(matched_root, {
            "phys_long": 0, "phys_short": 0,
            "jur_long": 0, "jur_short": 0,
            "phys_net": 0, "jur_net": 0,
            "secid": secid,
        })
        entry["phys_long"] = phys_long or 0
        entry["phys_short"] = phys_short or 0
        entry["jur_long"] = jur_long or 0
        entry["jur_short"] = jur_short or 0
        entry["phys_net"] = (phys_long or 0) - (phys_short or 0)
        entry["jur_net"] = (jur_long or 0) - (jur_short or 0)
    return out


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


# ----- LLM brief enrichment: macro spot quotes -------------------------

# Hardcoded candidate secids for the LLM brief macro snapshot. The first
# candidate that returns a non-empty candle wins. Update quarterly when
# active futures contracts roll. For LLM-only use — we just want a current
# value, no historical alignment.
_MACRO_QUOTE_CANDIDATES = {
    "usd_rub_spot": ["USD000UTSTOM", "USDRUB_TOM", "USD000000TOD"],
    "eur_rub_spot": ["EUR_RUB__TOM", "EURRUB_TOM"],
    "usd_rub_fut": ["Si-6.26", "Si-9.26", "Si-12.26"],
    "brent_fut":   ["BR-6.26", "BR-7.26", "BR-8.26", "BR-9.26"],
}


def get_macro_quotes_snapshot(*, moex_algo_token: str | None = None) -> dict[str, Any]:
    """Spot-and-active-futures snapshot for LLM brief macro_context.

    Returns dict like:
        {
            "usd_rub_spot": {"secid": "USD000UTSTOM", "close": 92.15, "ts": "..."},
            "brent_fut":    {"secid": "BR-6.26",      "close": 72.4,  "ts": "..."},
            ...
        }

    Fail-soft: missing keys mean the candidate either didn't resolve or didn't
    return data. The LLM brief proceeds without that quote.
    """
    try:
        from moexalgo import session, Ticker
    except ImportError:
        return {}
    if moex_algo_token is None:
        from moex_agent.config import settings as _settings
        moex_algo_token = _settings.moex_algo_token
    if not moex_algo_token:
        return {}
    session.TOKEN = moex_algo_token

    end = date.today()
    start = end - timedelta(days=7)  # 7 days is enough to find at least one close
    out: dict[str, Any] = {}
    for label, candidates in _MACRO_QUOTE_CANDIDATES.items():
        for secid in candidates:
            try:
                rows = list(Ticker(secid).candles(
                    start=start.isoformat(),
                    end=end.isoformat(),
                    period="1d",
                    native=True,
                ))
            except Exception:
                continue
            # Find the most recent row with a numeric close.
            latest_close: float | None = None
            latest_ts: datetime | None = None
            for raw in rows:
                rec = raw._asdict() if hasattr(raw, "_asdict") else (raw if isinstance(raw, dict) else None)
                if rec is None:
                    continue
                parsed = _row_to_index_close(rec)
                if parsed is None:
                    continue
                ts, close = parsed
                if latest_ts is None or ts > latest_ts:
                    latest_ts = ts
                    latest_close = close
            if latest_close is not None and latest_ts is not None:
                out[label] = {
                    "secid": secid,
                    "close": round(latest_close, 4),
                    "ts": latest_ts.isoformat(),
                }
                break
    return out
