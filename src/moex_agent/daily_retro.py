"""Daily retrospective generator.

Reads today's trading artifacts (paper_ledger, decisions, llm_explanations,
cycle_summaries) and asks an LLM to produce an analytical retrospective in
Markdown for expert review.

Produces the agent's self-reflection in plain Russian for review.

Usage:
    python -m moex_agent.daily_retro                      # today
    python -m moex_agent.daily_retro --date 2026-05-18    # specific date
    python -m moex_agent.daily_retro --output some/path.md

Does NOT modify the running bot. Safe to run any time.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from moex_agent.config import settings
from moex_agent.llm import PolzaClient


def _date_arg(value: str) -> date:
    return date.fromisoformat(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=_date_arg, default=None,
                        help="Date to summarize, ISO format. Default: today UTC.")
    parser.add_argument("--data-dir", type=Path, default=settings.runtime_data_dir)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-trades", type=int, default=20,
                        help="Cap on closed trades fed to the LLM (latest first).")
    parser.add_argument("--max-explanations", type=int, default=12,
                        help="Cap on per-trade LLM explanations included.")
    parser.add_argument("--model", type=str, default=None,
                        help="Override LLM model. Default: settings.llm_model_premium.")
    return parser.parse_args()


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace(" ", "T"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _within_day(ts_value: Any, target: date) -> bool:
    dt = _parse_ts(ts_value)
    if dt is None:
        return False
    return dt.astimezone(timezone.utc).date() == target


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _build_digest(
    data_dir: Path,
    target_date: date,
    *,
    max_trades: int,
    max_explanations: int,
) -> dict[str, Any]:
    """Produce a compact summary the LLM can chew on without context-stuffing."""
    paper_state = _load_json(data_dir / "state" / "paper_ledger.json")
    all_trades = paper_state.get("trades", []) if isinstance(paper_state, dict) else []
    day_trades = [t for t in all_trades if _within_day(t.get("ts"), target_date)]

    decisions = _load_jsonl(data_dir / "logs" / "decisions.jsonl")
    day_decisions = [d for d in decisions if _within_day(d.get("ts"), target_date)]

    explanations = _load_jsonl(data_dir / "logs" / "llm_explanations.jsonl")
    day_explanations = [e for e in explanations if _within_day(e.get("ts"), target_date)]

    summaries = _load_jsonl(data_dir / "logs" / "cycle_summary.jsonl")
    day_summaries = [s for s in summaries if _within_day(s.get("ts"), target_date)]

    closed_sells = [
        t for t in day_trades
        if t.get("side") == "sell" and int(t.get("closed_qty", 0) or 0) > 0
    ]
    pnls = [float(t.get("realized_pnl", 0) or 0) for t in closed_sells]
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p < 0]

    actions = Counter(str(d.get("signal", {}).get("action", "?")) for d in day_decisions)

    blocked_reasons: Counter[str] = Counter()
    for d in day_decisions:
        if not d.get("approved"):
            for reason in d.get("meta", {}).get("risk_reasons", []) or []:
                blocked_reasons[str(reason)] += 1

    orders_by_symbol = Counter(str(t.get("symbol", "?")) for t in day_trades)
    pnl_by_symbol: dict[str, float] = {}
    for t in closed_sells:
        sym = str(t.get("symbol", "?"))
        pnl_by_symbol[sym] = pnl_by_symbol.get(sym, 0.0) + float(t.get("realized_pnl", 0) or 0)
    pnl_by_symbol = {k: round(v, 2) for k, v in pnl_by_symbol.items()}

    brief_regimes: Counter[str] = Counter()
    for s in day_summaries:
        brief = s.get("llm_brief") or {}
        regime = str(brief.get("regime", "unknown"))
        brief_regimes[regime] += 1

    last_portfolio = None
    if day_summaries:
        last_portfolio = day_summaries[-1].get("portfolio")
    elif paper_state.get("portfolio"):
        last_portfolio = paper_state["portfolio"]

    digest: dict[str, Any] = {
        "date": target_date.isoformat(),
        "summary": {
            "total_decisions": len(day_decisions),
            "total_orders": len(day_trades),
            "buys": int(orders_by_symbol.total() and sum(1 for t in day_trades if t.get("side") == "buy")) or 0,
            "sells": sum(1 for t in day_trades if t.get("side") == "sell"),
            "closed_trades": len(closed_sells),
            "realized_pnl_total": round(sum(pnls), 2),
            "win_rate_pct": round(len(winning) / len(pnls) * 100, 1) if pnls else None,
            "avg_win": round(sum(winning) / len(winning), 2) if winning else None,
            "avg_loss": round(sum(losing) / len(losing), 2) if losing else None,
            "actions_distribution": dict(actions),
            "blocked_reasons_top": dict(blocked_reasons.most_common(8)),
            "brief_regimes_observed": dict(brief_regimes),
            "orders_by_symbol": dict(orders_by_symbol.most_common(15)),
            "realized_pnl_by_symbol": pnl_by_symbol,
        },
        "closed_trades_sample": [
            {
                "ts": t.get("ts"),
                "symbol": t.get("symbol"),
                "qty": int(t.get("closed_qty", 0) or 0),
                "limit_price": t.get("limit_price"),
                "avg_price_before": t.get("avg_price_before"),
                "realized_pnl": t.get("realized_pnl"),
                "reason": t.get("reason"),
            }
            for t in closed_sells[-max_trades:]
        ],
        "explanations_sample": [
            {
                "symbol": e.get("symbol"),
                "side": (e.get("order") or {}).get("side"),
                "qty": (e.get("order") or {}).get("qty"),
                "summary": e.get("summary"),
                "primary_signal": e.get("primary_signal"),
                "key_risks": e.get("key_risks"),
            }
            for e in day_explanations[-max_explanations:]
            if e.get("status") == "ok" and e.get("summary")
        ],
        "latest_portfolio": last_portfolio,
    }
    return digest


def _build_messages(digest: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "Ты — старший аналитик автономного торгового агента на Московской бирже (MOEX). "
        "Тебе передают сводку за один торговый день: статистику решений, исполненные сделки, "
        "пер-сделочные объяснения от LLM, причины блокировок ордеров. Твоя задача — "
        "написать ретроспективу 1-2 страницы в Markdown для экспертной оценки.\n\n"
        "Жёсткие требования:\n"
        "1. Пиши на русском, фактически, без воды.\n"
        "2. Опирайся на конкретные числа и тикеры ИЗ данных. Не выдумывай.\n"
        "3. Если данных в разделе мало — напиши это явно, не размазывай.\n"
        "4. Конкретные технические термины (RSI, Bollinger, oversold, reversal exit) — допустимы.\n"
        "5. Не льсти и не извиняйся за бот. Критикуй там, где есть основания.\n\n"
        "Структура ответа (строго):\n\n"
        "# Daily Retrospective — {YYYY-MM-DD}\n\n"
        "## Резюме\n"
        "3-5 строк с ключевыми цифрами: PnL, число сделок, win rate, экспозиция, оборот, "
        "наблюдённый LLM-режим. Без оценочных эпитетов.\n\n"
        "## Что работало\n"
        "2-4 пункта. Для каждого: какой сетап (oversold/MACD/reversal exit), на каких тикерах, "
        "с какими цифрами PnL. Например: \"Reversal exits на GAZP закрыли 5 сделок с суммарным "
        "+224 ₽\".\n\n"
        "## Что не работало\n"
        "1-3 пункта. Какие блокировки чаще всего срабатывали (из blocked_reasons_top), на каких "
        "тикерах вход не получился, есть ли тикеры с минусом.\n\n"
        "## Паттерны\n"
        "1-3 наблюдения. Например: \"Бот ориентируется на upper-Bollinger sell — RSI 80+ на "
        "GAZP/ROSN надёжно даёт reversal\". Или \"GMKN режется book_spread filter — стоит проверить, "
        "не слишком ли консервативно\".\n\n"
        "## Что попробовать завтра\n"
        "1-3 конкретных предложения с обоснованием. Подкрутить пороги? Расширить watchlist? "
        "Отключить какой-то сигнал?\n\n"
        "Если данных за день почти нет (например, 0 сделок) — честно скажи, что ретроспектива "
        "ограничена, и опиши, что бот делал/решал."
    )
    user = json.dumps(digest, ensure_ascii=False, default=str, indent=2)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_retro(args: argparse.Namespace) -> Path:
    target_date = args.date or datetime.now(timezone.utc).date()
    print(f"Building daily retro for {target_date} (UTC)...")

    digest = _build_digest(
        args.data_dir,
        target_date,
        max_trades=args.max_trades,
        max_explanations=args.max_explanations,
    )
    s = digest["summary"]
    print(f"  decisions     : {s['total_decisions']}")
    print(f"  orders        : {s['total_orders']}")
    print(f"  closed trades : {s['closed_trades']}")
    print(f"  realized PnL  : {s['realized_pnl_total']}")
    print(f"  win rate %    : {s['win_rate_pct']}")
    print(f"  explanations  : {len(digest['explanations_sample'])} sample (ok+nonempty)")

    if not settings.polza_api_key:
        raise SystemExit("POLZA_API_KEY is empty — cannot generate retro.")

    client = PolzaClient(
        api_key=settings.polza_api_key,
        base_url=settings.polza_base_url,
        timeout=60.0,           # retro is not latency-sensitive
        max_retries=2,
    )
    model = args.model or settings.llm_model_premium
    messages = _build_messages(digest)
    approx_prompt_tokens = len(messages[0]["content"] + messages[1]["content"]) // 4
    print(f"  model         : {model}")
    print(f"  approx prompt : ~{approx_prompt_tokens} tokens")

    resp = client.chat(
        model=model,
        messages=messages,
        max_tokens=2500,
        temperature=0.3,
        json_mode=False,         # we want Markdown, not JSON
    )
    if resp is None:
        raise SystemExit("LLM call failed (see logs).")

    output_path = args.output or args.data_dir / "reports" / f"daily_retro_{target_date.isoformat()}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = (resp.content or "").strip()
    if not content:
        raise SystemExit("LLM returned empty content.")
    if not content.startswith("#"):
        content = f"# Daily Retrospective — {target_date.isoformat()}\n\n" + content

    # Footer with metadata so experts know provenance.
    footer = (
        f"\n\n---\n\n"
        f"_Generated by `python -m moex_agent.daily_retro --date {target_date.isoformat()}` "
        f"on {datetime.now(timezone.utc).isoformat()}. "
        f"Model: `{resp.model}`. Tokens: prompt={resp.prompt_tokens}, completion={resp.completion_tokens}, "
        f"total={resp.total_tokens}._"
    )
    output_path.write_text(content + footer, encoding="utf-8")

    # Rough RUB cost (DeepSeek prices baked in; replace if you use a different model).
    deepseek_cost = (resp.prompt_tokens * 12.93 + resp.completion_tokens * 25.86) / 1_000_000
    print(f"\nSaved retro: {output_path}")
    print(f"Tokens: prompt={resp.prompt_tokens} completion={resp.completion_tokens} total={resp.total_tokens}")
    print(f"Cost estimate (if DeepSeek V4 Flash): {deepseek_cost:.4f} RUB")
    return output_path


def main() -> None:
    args = _parse_args()
    generate_retro(args)


if __name__ == "__main__":
    main()
