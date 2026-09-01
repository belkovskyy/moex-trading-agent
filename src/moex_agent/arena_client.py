from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import requests

from moex_agent.models import OrderIntent, Portfolio, Position

logger = logging.getLogger(__name__)


class ArenaAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class KillSwitchTripped(RuntimeError):
    """Raised by ArenaClient when consecutive failures exceed threshold.

    Bot should treat this as a hard stop for new orders. Existing positions
    and market data reads are not affected — only place_order is gated.
    """


class ArenaClient:
    """Thin ArenaGo adapter.

    ArenaGo provides execution and portfolio endpoints. Market data is handled
    by a separate ALGOPACK/MOEX client.

    Resilience features:
    - Retry on network/5xx with exponential backoff
    - Consecutive-failure counter ("kill switch"): after N failures place_order
      refuses to even attempt a submission, returning a dry_run-like response.
      This prevents burning order slots on a broken ArenaGo session during
      long autonomous runs.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        bot_name: str,
        dry_run: bool = True,
        paper_trading: bool = True,
        timeout: int = 20,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.5,
        max_consecutive_failures: int = 10,
        lot_sizes: dict[str, int] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.bot_name = bot_name
        self.dry_run = dry_run
        self.paper_trading = paper_trading
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_seconds = max(0.5, float(retry_backoff_seconds))
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self.consecutive_failures = 0
        self.kill_switch_armed = False
        # When the kill switch arms, we record the moment. After
        # `kill_switch_cooldown_seconds` we let the next place_order attempt
        # through to probe whether ArenaGo recovered. This avoids the bot
        # staying frozen through a long autonomous run because of a brief
        # infrastructure hiccup. If the probe fails too, we re-arm.
        self.kill_switch_armed_at: datetime | None = None
        self.kill_switch_cooldown_seconds = 30 * 60  # 30 minutes
        # ArenaGo `quantity` field is in LOTS, not shares. For some tickers
        # (GAZP, GMKN, MOEX, MTSS, ...) 1 lot = 10 shares; for others it's 1.
        # Without this conversion, an order intended for 247 shares of GAZP
        # actually buys 247 lots = 2470 shares = 10× position size.
        self.lot_sizes: dict[str, int] = dict(lot_sizes or {})
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": api_key})
        self.session.headers.update({"Content-Type": "application/json"})

    def lot_size(self, symbol: str) -> int:
        """Return the ArenaGo lot size for a symbol. Default 1."""
        return max(1, int(self.lot_sizes.get(symbol, 1)))

    def shares_to_lots(self, symbol: str, shares: int) -> int:
        """Round shares down to nearest whole lot. Always returns >= 1 if
        the input is positive, since ArenaGo doesn't accept fractional lots."""
        lot = self.lot_size(symbol)
        return max(1, int(shares) // lot) if shares > 0 else 0

    def lots_to_shares(self, symbol: str, lots: int) -> int:
        return int(lots) * self.lot_size(symbol)

    def get_portfolio(self) -> Portfolio:
        if self.dry_run and not self.paper_trading:
            return Portfolio(
                cash=1_000_000.0,
                positions={
                    "SBER": Position("SBER", qty=100, avg_price=300.0, market_price=305.0),
                    "LKOH": Position("LKOH", qty=10, avg_price=7200.0, market_price=7180.0),
                },
            )
        bot = self._get_bot()
        positions = self.get_positions()
        return Portfolio(cash=float(bot.get("cash_balance", 0.0)), positions=positions)

    def place_order(self, order: OrderIntent) -> dict[str, Any]:
        if self.dry_run or self.paper_trading:
            logger.info("dry-run order", extra={"extra": {"order": order}})
            return {"status": "paper" if self.paper_trading else "dry_run", "order": order}
        if self.kill_switch_armed:
            # Auto-recovery: after the cooldown, let one probe order through.
            # If it succeeds we'll reset consecutive_failures in _request. If it
            # fails we re-arm with a fresh timestamp.
            now_utc = datetime.now(timezone.utc)
            armed_at = self.kill_switch_armed_at
            if armed_at is not None and (now_utc - armed_at).total_seconds() >= self.kill_switch_cooldown_seconds:
                logger.warning(
                    "kill switch cooldown elapsed — releasing one probe order",
                    extra={"extra": {
                        "cooldown_seconds": self.kill_switch_cooldown_seconds,
                        "armed_for_seconds": int((now_utc - armed_at).total_seconds()),
                    }},
                )
                self.kill_switch_armed = False
                # Halve the failure counter so a recovered API doesn't immediately
                # re-arm on the next single failure.
                self.consecutive_failures = max(0, self.consecutive_failures // 2)
            else:
                logger.warning(
                    "kill switch armed — refusing place_order",
                    extra={"extra": {
                        "consecutive_failures": self.consecutive_failures,
                        "threshold": self.max_consecutive_failures,
                        "order": order,
                    }},
                )
                return {"status": "kill_switch_blocked", "order": order}
        # Strategy/risk operate in SHARES. ArenaGo `quantity` is in LOTS.
        # Convert here at the boundary; if there aren't enough shares to fill
        # a whole lot, skip the order rather than silently buying too much.
        lot_size = self.lot_size(order.symbol)
        lots = int(order.qty) // lot_size
        if lots < 1:
            logger.warning(
                "order skipped: shares below one lot",
                extra={"extra": {
                    "symbol": order.symbol, "shares": order.qty, "lot_size": lot_size,
                }},
            )
            return {"status": "below_lot_size", "order": order, "lot_size": lot_size}
        payload = {
            "direction": "B" if order.side.value == "buy" else "S",
            "secid": order.symbol,
            "quantity": lots,                   # ArenaGo expects LOTS
            "bot": self.bot_name,
        }
        data = self._request("POST", "/api/submit_order", json=payload)
        if isinstance(data, dict) and data.get("error"):
            raise ArenaAPIError(str(data["error"]))
        return {
            "status": "submitted",
            "request": payload,
            "response": data,
            "intended_shares": order.qty,
            "lot_size": lot_size,
            "actual_shares": lots * lot_size,   # what was actually filled
        }

    def reset_kill_switch(self) -> None:
        """Manual reset hook. Call after verifying ArenaGo is healthy again."""
        self.consecutive_failures = 0
        self.kill_switch_armed = False
        logger.info("kill switch manually reset")

    def get_bots(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/bots")
        if not isinstance(data, list):
            raise ArenaAPIError(f"Unexpected bots response: {data}")
        return data

    def get_positions(self) -> dict[str, Position]:
        # URL-encode bot_name: it can contain spaces, Cyrillic, commas and '?'
        # ("Бу, испугался?"). Without encoding the '?' becomes a query separator
        # and the server gets a different path → returns empty.
        bot_path = quote(self.bot_name, safe="")
        try:
            data = self._request("GET", f"/api/positions/{bot_path}")
        except ArenaAPIError as exc:
            if exc.status_code == 404:
                logger.info("no open positions", extra={"extra": {"bot_name": self.bot_name}})
                return {}
            raise
        if not isinstance(data, list):
            raise ArenaAPIError(f"Unexpected positions response: {data}")
        positions: dict[str, Position] = {}
        for raw in data:
            symbol = str(raw.get("secid", ""))
            # ArenaGo reports `position` in LOTS. Convert to shares so the
            # rest of the bot (risk, strategy, paper_ledger snapshot) sees
            # everything in shares — consistent with how candles + prices
            # are denominated.
            qty_lots = int(float(raw.get("position", 0) or 0))
            qty_shares = self.lots_to_shares(symbol, qty_lots)
            avg_price = float(raw.get("average_price", 0.0) or 0.0)
            market_price = float(raw.get("price", avg_price) or avg_price)
            if symbol and qty_shares:
                positions[symbol] = Position(symbol, qty=qty_shares, avg_price=avg_price, market_price=market_price)
        return positions

    def get_trades(self) -> list[dict[str, Any]]:
        bot_path = quote(self.bot_name, safe="")
        try:
            data = self._request("GET", f"/api/trades/{bot_path}")
        except ArenaAPIError as exc:
            if exc.status_code == 404:
                logger.info("no trades", extra={"extra": {"bot_name": self.bot_name}})
                return []
            raise
        if not isinstance(data, list):
            raise ArenaAPIError(f"Unexpected trades response: {data}")
        return data

    def healthcheck(self) -> bool:
        if self.dry_run:
            return True
        try:
            response = self.session.get(self.base_url, timeout=self.timeout)
            return response.status_code < 500
        except requests.RequestException:
            return False

    def _get_bot(self) -> dict[str, Any]:
        for bot in self.get_bots():
            if bot.get("name") == self.bot_name:
                return bot
        raise ArenaAPIError(f"Bot not found in /api/bots: {self.bot_name}")

    def _record_success(self) -> None:
        if self.consecutive_failures:
            logger.info(
                "arena call recovered after failures",
                extra={"extra": {"recovered_after": self.consecutive_failures}},
            )
        self.consecutive_failures = 0

    def _record_failure(self, reason: str) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive_failures and not self.kill_switch_armed:
            self.kill_switch_armed = True
            self.kill_switch_armed_at = datetime.now(timezone.utc)
            logger.error(
                "kill switch ARMED — too many consecutive ArenaGo failures",
                extra={"extra": {
                    "consecutive_failures": self.consecutive_failures,
                    "threshold": self.max_consecutive_failures,
                    "last_reason": reason,
                }},
            )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        if not self.api_key:
            raise ArenaAPIError("SANDBOX_API_KEY is empty.")
        url = f"{self.base_url}{path}"
        last_exc: ArenaAPIError | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs,
                )
                if response.status_code >= 500:
                    body = response.text[:300] if response is not None else ""
                    last_exc = ArenaAPIError(
                        f"HTTP {response.status_code}: {body}",
                        status_code=response.status_code,
                    )
                    # 5xx retried with backoff
                    if attempt < self.max_retries:
                        self._sleep_backoff(attempt)
                        continue
                    self._record_failure(f"5xx: {response.status_code}")
                    raise last_exc
                response.raise_for_status()
                self._record_success()
                return response.json()
            except requests.HTTPError as exc:
                response = exc.response
                body = response.text[:500] if response is not None else ""
                status_code = response.status_code if response is not None else None
                # 4xx is treated as legitimate API response (e.g. 404 = no positions).
                # NOT counted as failure for kill-switch purposes — those are
                # logical responses, not infrastructure problems.
                self._record_success()
                raise ArenaAPIError(f"{exc}; body={body}", status_code=status_code) from exc
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = ArenaAPIError(f"{type(exc).__name__}: {exc}")
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt)
                    continue
                self._record_failure(f"{type(exc).__name__}: {exc}")
                raise last_exc from exc
            except requests.RequestException as exc:
                # Other transient network issue. Treat as retryable.
                last_exc = ArenaAPIError(str(exc))
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt)
                    continue
                self._record_failure(f"{type(exc).__name__}: {exc}")
                raise last_exc from exc
        # Shouldn't reach here, but be defensive.
        self._record_failure("exhausted retries")
        raise last_exc if last_exc is not None else ArenaAPIError("unknown failure")

    def _sleep_backoff(self, attempt: int) -> None:
        delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
        time.sleep(delay)
