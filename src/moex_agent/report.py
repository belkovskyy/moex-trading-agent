from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from moex_agent.config import settings


@dataclass(frozen=True)
class Report:
    decisions: int
    total_polls: int
    unique_candles: int
    orders: int
    blocked: int
    approved: int
    action_counts: dict[str, int]
    symbol_counts: dict[str, int]
    blocked_reasons: dict[str, int]
    order_symbols: dict[str, int]
    order_sides: dict[str, int]
    approved_by_symbol: dict[str, int]
    blocked_by_symbol: dict[str, int]
    blocked_reasons_by_symbol: dict[str, dict[str, int]]
    avg_confidence_by_action: dict[str, float]
    avg_confidence_by_symbol: dict[str, float]
    session_decisions: dict[str, int]
    session_orders: dict[str, int]
    order_turnover_by_symbol: dict[str, float]
    trade_count_by_symbol: dict[str, int]
    trade_realized_pnl_by_symbol: dict[str, float]
    trade_realized_pnl_by_exit: dict[str, float]
    trade_realized_pnl_total: float | None
    trade_win_rate: float | None
    trade_avg_win: float | None
    trade_avg_loss: float | None
    closed_trades: int
    turnover: float
    paper_pnl: float | None
    paper_pnl_pct: float | None
    exposure_pct: float | None
    first_ts: str | None
    last_ts: str | None
    latest_portfolio: dict[str, Any] | None


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_report(data_dir: Path, *, paper_only: bool = True) -> Report:
    logs_dir = data_dir / "logs"
    decisions = list(load_jsonl(logs_dir / "decisions.jsonl"))
    orders = list(load_jsonl(logs_dir / "orders.jsonl"))
    summaries = list(load_jsonl(logs_dir / "cycle_summary.jsonl"))
    ledger_state = _load_state(data_dir / "state" / "paper_ledger.json")
    trades = ledger_state.get("trades", []) if isinstance(ledger_state, dict) else []

    if paper_only:
        decisions = [row for row in decisions if row.get("meta", {}).get("paper_trading") is True]
        orders = [row for row in orders if _order_status(row) == "paper"]
        summaries = [row for row in summaries if row.get("paper_trading") is True]
        trades = trades if isinstance(trades, list) else []

    # Dedupe: the agent polls every 30s but candles update every 10min.
    # Without this, "Decisions: 2266" is really ~80 unique candles polled
    # 25-30 times each. Stats below operate on unique (symbol, feature_ts)
    # pairs. Orders are NOT deduped (real events).
    total_polls = len(decisions)
    decisions = _dedupe_by_candle(decisions)
    unique_candles = len(decisions)

    action_counts: Counter[str] = Counter()
    symbol_counts: Counter[str] = Counter()
    blocked_reasons: Counter[str] = Counter()
    order_symbols: Counter[str] = Counter()
    order_sides: Counter[str] = Counter()
    approved_by_symbol: Counter[str] = Counter()
    blocked_by_symbol: Counter[str] = Counter()
    session_decisions: Counter[str] = Counter()
    session_orders: Counter[str] = Counter()
    order_turnover_by_symbol: defaultdict[str, float] = defaultdict(float)
    blocked_reasons_by_symbol: defaultdict[str, Counter[str]] = defaultdict(Counter)
    confidence_sum_by_action: defaultdict[str, float] = defaultdict(float)
    confidence_count_by_action: Counter[str] = Counter()
    confidence_sum_by_symbol: defaultdict[str, float] = defaultdict(float)
    confidence_count_by_symbol: Counter[str] = Counter()
    timestamps: list[str] = []
    blocked_count = 0
    approved_count = 0

    for row in decisions:
        timestamps.append(str(row.get("ts")))
        symbol = str(row.get("symbol", "unknown"))
        action = str(row.get("signal", {}).get("action", "unknown"))
        confidence = _safe_float(row.get("signal", {}).get("confidence"))
        symbol_counts[symbol] += 1
        action_counts[action] += 1
        confidence_sum_by_action[action] += confidence
        confidence_count_by_action[action] += 1
        confidence_sum_by_symbol[symbol] += confidence
        confidence_count_by_symbol[symbol] += 1
        session_decisions[_session_bucket(row.get("ts"))] += 1
        if row.get("approved") is True:
            approved_count += 1
            approved_by_symbol[symbol] += 1
        if row.get("approved") is False:
            blocked_count += 1
            blocked_by_symbol[symbol] += 1
            for item in _split_risk_reasons(row):
                blocked_reasons[item] += 1
                blocked_reasons_by_symbol[symbol][item] += 1

    turnover = 0.0
    for row in orders:
        order = row.get("result", {}).get("order") or row.get("result", {}).get("request") or {}
        symbol = str(order.get("symbol") or order.get("secid") or "unknown")
        side = str(order.get("side") or order.get("direction") or "unknown")
        qty = _safe_float(order.get("qty") or order.get("quantity"))
        price = _safe_float(order.get("limit_price") or order.get("price"))
        order_symbols[symbol] += 1
        order_sides[side] += 1
        session_orders[_session_bucket(row.get("ts"))] += 1
        order_value = abs(qty * price)
        turnover += order_value
        order_turnover_by_symbol[symbol] += order_value

    trade_count_by_symbol: Counter[str] = Counter()
    trade_realized_pnl_by_symbol: defaultdict[str, float] = defaultdict(float)
    trade_realized_pnl_by_exit: defaultdict[str, float] = defaultdict(float)
    realized_pnls: list[float] = []
    for trade in trades:
        if str(trade.get("side")) != "sell":
            continue
        closed_qty = int(_safe_float(trade.get("closed_qty")))
        if closed_qty <= 0:
            continue
        symbol = str(trade.get("symbol", "unknown"))
        realized_pnl = _safe_float(trade.get("realized_pnl"))
        exit_bucket = _exit_bucket(str(trade.get("reason", "")))
        trade_count_by_symbol[symbol] += 1
        trade_realized_pnl_by_symbol[symbol] += realized_pnl
        trade_realized_pnl_by_exit[exit_bucket] += realized_pnl
        realized_pnls.append(realized_pnl)

    latest_portfolio = summaries[-1].get("portfolio") if summaries else None
    paper_pnl = None
    paper_pnl_pct = None
    exposure_pct = None
    if latest_portfolio:
        equity = _safe_float(latest_portfolio.get("equity"))
        exposure = _safe_float(latest_portfolio.get("exposure"))
        if equity > 0:
            paper_pnl = round(equity - 1_000_000.0, 2)
            paper_pnl_pct = round((paper_pnl / 1_000_000.0) * 100.0, 4)
            exposure_pct = round((exposure / equity) * 100.0, 2)

    avg_confidence_by_action = {
        key: round(confidence_sum_by_action[key] / confidence_count_by_action[key], 4)
        for key in confidence_count_by_action
    }
    avg_confidence_by_symbol = {
        key: round(confidence_sum_by_symbol[key] / confidence_count_by_symbol[key], 4)
        for key in confidence_count_by_symbol
    }
    winning_trades = [pnl for pnl in realized_pnls if pnl > 0]
    losing_trades = [pnl for pnl in realized_pnls if pnl < 0]
    trade_realized_pnl_total = round(sum(realized_pnls), 2) if realized_pnls else None
    trade_win_rate = round((len(winning_trades) / len(realized_pnls)) * 100.0, 2) if realized_pnls else None
    trade_avg_win = round(sum(winning_trades) / len(winning_trades), 2) if winning_trades else None
    trade_avg_loss = round(sum(losing_trades) / len(losing_trades), 2) if losing_trades else None

    return Report(
        decisions=len(decisions),
        total_polls=total_polls,
        unique_candles=unique_candles,
        orders=len(orders),
        blocked=blocked_count,
        approved=approved_count,
        action_counts=dict(action_counts),
        symbol_counts=dict(symbol_counts),
        blocked_reasons=dict(blocked_reasons),
        order_symbols=dict(order_symbols),
        order_sides=dict(order_sides),
        approved_by_symbol=dict(approved_by_symbol),
        blocked_by_symbol=dict(blocked_by_symbol),
        blocked_reasons_by_symbol={key: dict(value) for key, value in blocked_reasons_by_symbol.items()},
        avg_confidence_by_action=avg_confidence_by_action,
        avg_confidence_by_symbol=avg_confidence_by_symbol,
        session_decisions=dict(session_decisions),
        session_orders=dict(session_orders),
        order_turnover_by_symbol={key: round(value, 2) for key, value in order_turnover_by_symbol.items()},
        trade_count_by_symbol=dict(trade_count_by_symbol),
        trade_realized_pnl_by_symbol={key: round(value, 2) for key, value in trade_realized_pnl_by_symbol.items()},
        trade_realized_pnl_by_exit={key: round(value, 2) for key, value in trade_realized_pnl_by_exit.items()},
        trade_realized_pnl_total=trade_realized_pnl_total,
        trade_win_rate=trade_win_rate,
        trade_avg_win=trade_avg_win,
        trade_avg_loss=trade_avg_loss,
        closed_trades=len(realized_pnls),
        turnover=round(turnover, 2),
        paper_pnl=paper_pnl,
        paper_pnl_pct=paper_pnl_pct,
        exposure_pct=exposure_pct,
        first_ts=min(timestamps) if timestamps else None,
        last_ts=max(timestamps) if timestamps else None,
        latest_portfolio=latest_portfolio,
    )


def render_markdown(report: Report) -> str:
    lines = [
        "# Paper Report",
        "",
        f"- Period: `{report.first_ts}` -> `{report.last_ts}`",
        f"- Unique candles evaluated: `{report.unique_candles}` (from `{report.total_polls}` raw polls)",
        f"- Decisions: `{report.decisions}`",
        f"- Approved decisions: `{report.approved}`",
        f"- Orders: `{report.orders}`",
        f"- Blocked decisions: `{report.blocked}`",
        f"- Estimated turnover: `{report.turnover}`",
        f"- Paper PnL: `{report.paper_pnl}`",
        f"- Paper PnL %: `{report.paper_pnl_pct}`",
        f"- Exposure %: `{report.exposure_pct}`",
        f"- Closed sell trades: `{report.closed_trades}`",
        f"- Realized PnL from closed sells: `{report.trade_realized_pnl_total}`",
        f"- Trade win rate %: `{report.trade_win_rate}`",
        f"- Average win: `{report.trade_avg_win}`",
        f"- Average loss: `{report.trade_avg_loss}`",
        "",
        "## Session Breakdown",
        "Decisions:",
        _format_counter(report.session_decisions),
        "",
        "Orders:",
        _format_counter(report.session_orders),
        "",
        "## Actions",
        _format_counter(report.action_counts),
        "",
        "## Average Confidence",
        "By action:",
        _format_float_map(report.avg_confidence_by_action),
        "",
        "By symbol:",
        _format_float_map(report.avg_confidence_by_symbol),
        "",
        "## Orders by Symbol",
        _format_counter(report.order_symbols),
        "",
        "## Orders by Side",
        _format_counter(report.order_sides),
        "",
        "## Turnover by Symbol",
        _format_float_map(report.order_turnover_by_symbol),
        "",
        "## Blocked Reasons",
        "One blocked decision can have several reasons.",
        _format_counter(report.blocked_reasons),
        "",
        "## Decisions by Symbol",
        _format_counter(report.symbol_counts),
        "",
        "## Approval by Symbol",
        "Approved decisions:",
        _format_counter(report.approved_by_symbol),
        "",
        "Blocked decisions:",
        _format_counter(report.blocked_by_symbol),
        "",
        "## Blocked Reasons by Symbol",
        _format_nested_counter(report.blocked_reasons_by_symbol),
        "",
        "## Closed Trade Diagnostics",
        "Closed sells by symbol:",
        _format_counter(report.trade_count_by_symbol),
        "",
        "Realized PnL by symbol:",
        _format_float_map(report.trade_realized_pnl_by_symbol),
        "",
        "Realized PnL by exit type:",
        _format_float_map(report.trade_realized_pnl_by_exit),
    ]
    if report.latest_portfolio:
        portfolio = report.latest_portfolio
        lines.extend(
            [
                "",
                "## Latest Portfolio",
                f"- Cash: `{round(float(portfolio.get('cash', 0.0)), 2)}`",
                f"- Equity: `{round(float(portfolio.get('equity', 0.0)), 2)}`",
                f"- Exposure: `{round(float(portfolio.get('exposure', 0.0)), 2)}`",
                f"- Positions: `{len(portfolio.get('positions', {}))}`",
            ]
        )
        for symbol, position in sorted(portfolio.get("positions", {}).items()):
            qty = _safe_float(position.get("qty"))
            avg_price = _safe_float(position.get("avg_price"))
            market_price = _safe_float(position.get("market_price"))
            pnl = round((market_price - avg_price) * qty, 2)
            lines.append(f"- `{symbol}` qty `{int(qty)}` avg `{round(avg_price, 4)}` market `{round(market_price, 4)}` pnl `{pnl}`")
    return "\n".join(lines) + "\n"


def write_report(data_dir: Path, *, paper_only: bool = True) -> Path:
    report = build_report(data_dir, paper_only=paper_only)
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"paper_report_{stamp}.md"
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Include non-paper dry-run records.")
    args = parser.parse_args()
    path = write_report(settings.runtime_data_dir, paper_only=not args.all)
    print(path)


def _order_status(row: dict[str, Any]) -> str:
    return str(row.get("result", {}).get("status", ""))


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _format_counter(counter: dict[str, int]) -> str:
    if not counter:
        return "- none"
    return "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _format_float_map(values: dict[str, float]) -> str:
    if not values:
        return "- none"
    return "\n".join(
        f"- `{key}`: `{round(value, 4)}`" for key, value in sorted(values.items(), key=lambda item: (-abs(item[1]), item[0]))
    )


def _format_nested_counter(values: dict[str, dict[str, int]]) -> str:
    if not values:
        return "- none"
    lines: list[str] = []
    for symbol, reasons in sorted(values.items(), key=lambda item: (-sum(item[1].values()), item[0])):
        lines.append(f"- `{symbol}`")
        for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"  - `{reason}`: `{count}`")
    return "\n".join(lines)


def _split_risk_reasons(row: dict[str, Any]) -> list[str]:
    meta_reasons = row.get("meta", {}).get("risk_reasons")
    if isinstance(meta_reasons, list) and meta_reasons:
        return [str(reason) for reason in meta_reasons if str(reason).strip()]
    reason = str(row.get("reason", "unknown"))
    return [part.strip() for part in reason.split(";") if part.strip()]


def _dedupe_by_candle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first decision per (symbol, feature_ts).

    The agent polls every 30s but 10min candles update only every 10min.
    Many raw decision rows are the same (symbol, feature_ts) re-evaluated.
    For meaningful stats we keep the earliest decision per candle — that's
    the one that actually drove the order (if any). Later polls of the
    same candle are dropped.
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: str(r.get("ts", ""))):
        symbol = str(row.get("symbol", "unknown"))
        feature_ts = str(
            row.get("signal", {}).get("features", {}).get("feature_ts", "")
        ) or "unknown"
        key = (symbol, feature_ts)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _session_bucket(value: Any) -> str:
    ts = _parse_ts(value)
    if ts is None:
        return "unknown"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    moscow_time = ts.astimezone(timezone(timedelta(hours=3)))
    moscow_hour = moscow_time.hour
    moscow_minute = moscow_time.minute
    total = (moscow_hour * 60) + moscow_minute
    if 6 * 60 + 50 <= total < 9 * 60 + 50:
        return "morning"
    if 9 * 60 + 50 <= total < 19 * 60:
        return "main"
    if total >= 19 * 60 or total < 3 * 60 + 50:
        return "evening"
    return "off_session"


def _exit_bucket(reason: str) -> str:
    normalized = reason.strip().lower()
    if "stop-loss" in normalized:
        return "stop_loss"
    if "take-profit" in normalized and "scale-out" in normalized:
        return "take_profit_scale_out"
    if "take-profit" in normalized:
        return "take_profit"
    if "reversal" in normalized:
        return "reversal"
    if "sell" in normalized:
        return "sell"
    return "other"


if __name__ == "__main__":
    main()
