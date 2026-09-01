"""Quick sweep: order_cash_pct {0.03, 0.05, 0.07} on full offline dataset.
Run: PYTHONPATH=src python sweep_oborot.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pathlib import Path
from datetime import date
from moex_agent.backtest import BacktestConfig, _load_rows, _compute_per_symbol_atr_stops, run_backtest, compute_metrics
from moex_agent.ml_filter import MLBuyFilter

DATASET = Path("data/logs/ml_dataset_offline.jsonl")
MODEL   = Path("data/models/lgbm_buy_filter_v4.joblib")
START   = date(2024, 1, 1)

print("Loading dataset...")
rows = _load_rows(DATASET, start_date=START, end_date=None, symbols=None)
print(f"Rows: {len(rows)}  {rows[0]['ts'].date()} -> {rows[-1]['ts'].date()}\n")

ml_filter = MLBuyFilter.load(MODEL)
if ml_filter:
    print(f"ML filter loaded: {MODEL}\n")
else:
    print("WARNING: ML filter not loaded!\n")

results = []
for pct in [0.03, 0.05, 0.07]:
    print(f"--- order_cash_pct={pct} ---")
    cfg = BacktestConfig(
        order_cash_pct=pct,
        use_ml_filter=True,
        ml_model_path=MODEL,
        ml_min_up_probability=0.85,
        stop_loss_pct=0.012,
        take_profit_pct=0.025,
    )
    result = run_backtest(cfg, rows, ml_filter=ml_filter)
    m = compute_metrics(result, cfg)
    results.append((pct, m))
    print(f"  n_trades={m['n_trades']}  return={m['total_return_pct']}%  "
          f"max_dd={m['max_drawdown_pct']}%  win_rate={m['win_rate_pct']}%  "
          f"sharpe={m['sharpe_annualized']}")
    print(f"  profit_factor={m['profit_factor']}  avg_hold={m['avg_hold_minutes']}min")
    print(f"  OBOROT: {int(m['total_oborot_rub']):,} rub  "
          f"| 14d proj: {int(m['projected_14d_oborot_rub']):,} rub")
    print()

print("=" * 70)
print(f"{'pct':>6}  {'trades':>7}  {'return%':>8}  {'maxDD%':>7}  {'14d_oborot':>12}  {'sharpe':>7}")
print("-" * 70)
for pct, m in results:
    print(f"{pct:>6.2f}  {m['n_trades']:>7}  {m['total_return_pct']:>8.3f}  "
          f"{m['max_drawdown_pct']:>7.3f}  {int(m['projected_14d_oborot_rub']):>12,}  "
          f"{m['sharpe_annualized'] or 0:>7.3f}")
print("=" * 70)
print("Note: dataset = LKOH only. Live bot trades 20 tickers -> multiply ~15-20x.")
