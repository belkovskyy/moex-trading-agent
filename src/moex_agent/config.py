from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Deployment may inject empty placeholder keys via ENV; with override=False they
# shadow the real keys from .env. Force our own keys to win (SANDBOX_API_KEY is
# intentionally left to the environment).
_ENV_FILE = dotenv_values(PROJECT_ROOT / ".env")
for _own_key in ("POLZA_API_KEY", "MOEX_ALGO_TOKEN"):
    _own_val = _ENV_FILE.get(_own_key)
    if _own_val:
        os.environ[_own_key] = _own_val


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    sandbox_api_key: str = os.getenv("SANDBOX_API_KEY", "")
    # Default portfolio name so the cloud pod works without a BOT_NAME var
    # (the .env doesn't ship to the cluster). Still overridable.
    bot_name: str = os.getenv("BOT_NAME", "Бу, испугался?")
    arenago_base_url: str = os.getenv("ARENAGO_BASE_URL", "https://arenago.ru")
    dry_run: bool = _bool_env("DRY_RUN", True)
    paper_trading: bool = _bool_env("PAPER_TRADING", True)

    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    poll_interval_seconds: int = _int_env("POLL_INTERVAL_SECONDS", 30)
    max_cycles: int = _int_env("MAX_CYCLES", 0)

    market_data_provider: str = os.getenv("MARKET_DATA_PROVIDER", "algopack")
    moex_algo_token: str = os.getenv("MOEX_ALGO_TOKEN", "")
    candle_period: str = os.getenv("CANDLE_PERIOD", "10min")
    candle_lookback_days: int = _int_env("CANDLE_LOOKBACK_DAYS", 5)
    use_super_candles: bool = _bool_env("USE_SUPER_CANDLES", True)

    max_position_pct: float = _float_env("MAX_POSITION_PCT", 0.15)
    max_portfolio_exposure_pct: float = _float_env("MAX_PORTFOLIO_EXPOSURE_PCT", 0.75)
    max_daily_loss_pct: float = _float_env("MAX_DAILY_LOSS_PCT", 0.03)
    max_drawdown_pct: float = _float_env("MAX_DRAWDOWN_PCT", 0.08)
    order_cash_pct: float = _float_env("ORDER_CASH_PCT", 0.03)
    min_cash_pct: float = _float_env("MIN_CASH_PCT", 0.10)
    max_daily_trades: int = _int_env("MAX_DAILY_TRADES", 30)
    symbol_cooldown_minutes: int = _int_env("SYMBOL_COOLDOWN_MINUTES", 30)
    min_volume_ratio: float = _float_env("MIN_VOLUME_RATIO", 0.30)
    min_trade_imbalance_buy: float = _float_env("MIN_TRADE_IMBALANCE_BUY", -0.15)
    min_super_volume: float = _float_env("MIN_SUPER_VOLUME", 1000.0)
    min_book_imbalance_buy: float = _float_env("MIN_BOOK_IMBALANCE_BUY", -0.15)
    max_book_spread_pct: float = _float_env("MAX_BOOK_SPREAD_PCT", 0.01)
    add_position_min_confidence: float = _float_env("ADD_POSITION_MIN_CONFIDENCE", 0.64)
    add_position_min_trade_imbalance: float = _float_env("ADD_POSITION_MIN_TRADE_IMBALANCE", 0.20)
    add_position_min_pnl_pct: float = _float_env("ADD_POSITION_MIN_PNL_PCT", 0.0)
    add_position_max_position_pct: float = _float_env("ADD_POSITION_MAX_POSITION_PCT", 0.10)
    stop_loss_pct: float = _float_env("STOP_LOSS_PCT", 0.007)
    take_profit_pct: float = _float_env("TAKE_PROFIT_PCT", 0.012)
    reversal_exit_min_pnl_pct: float = _float_env("REVERSAL_EXIT_MIN_PNL_PCT", -0.003)
    time_exit_hours: float = _float_env("TIME_EXIT_HOURS", 0.0)
    time_exit_min_profit_pct: float = _float_env("TIME_EXIT_MIN_PROFIT_PCT", 0.003)
    # Trailing stop: let winners run, then lock the gain on a pullback from the
    # peak (instead of dumping at the fixed +0.3% reversal floor). Once a
    # position's peak PnL >= trail_stop_pct, exit when it gives back trail_stop_pct
    # from that peak. 0 = off. auto_tuner-tunable.
    trail_stop_pct: float = _float_env("TRAIL_STOP_PCT", 0.012)  # tighter values churn in backtest

    # Conviction-based position sizing. When enabled, order size scales with
    # ML edge above the buy threshold:
    #   edge = (P(up) - ml_min_up_probability) / (1 - ml_min_up_probability)
    #   mult = clamp(1 + conviction_gain * edge, 1.0, conviction_max_mult)
    #   size = order_cash_pct * mult,  then HARD-capped by max_position_pct.
    # Default OFF (gain active only when enabled) so behaviour is unchanged
    # until deliberately switched on. gain/max_mult are auto_tuner-tunable
    # within bounds; max_position_pct remains the inviolable ceiling.
    conviction_sizing_enabled: bool = _bool_env("CONVICTION_SIZING_ENABLED", False)
    conviction_gain: float = _float_env("CONVICTION_GAIN", 0.5)
    conviction_max_mult: float = _float_env("CONVICTION_MAX_MULT", 1.6)

    polza_api_key: str = os.getenv("POLZA_API_KEY", "")
    polza_base_url: str = os.getenv("POLZA_BASE_URL", "https://api.polza.ai")
    llm_enabled: bool = _bool_env("LLM_ENABLED", True)
    llm_model_fast: str = os.getenv("LLM_MODEL_FAST", "deepseek/deepseek-v4-flash")
    # Dedicated brief model. A non-reasoning flash model answers the structured
    # json_mode brief reliably; reasoning models sometimes spend their whole
    # token budget on hidden reasoning and return empty. Falls back to
    # llm_model_fast if unset.
    llm_model_brief: str = os.getenv("LLM_MODEL_BRIEF", "") or os.getenv("LLM_MODEL_FAST", "deepseek/deepseek-v4-flash")
    # Fallback brief model, tried in the SAME refresh if the primary returns
    # empty/invalid JSON. Deliberately a different open family (Qwen) so two
    # unrelated engines rarely fail on the same call.
    llm_model_brief_fallback: str = os.getenv("LLM_MODEL_BRIEF_FALLBACK", "qwen/qwen3-30b-a3b-instruct-2507")
    # Premium = auto_tuner / daily_retro (single call, no fallback). A
    # non-reasoning model with reliable JSON output is used to avoid empty
    # responses silently no-op'ing these layers.
    llm_model_premium: str = os.getenv("LLM_MODEL_PREMIUM", "qwen/qwen3-30b-a3b-instruct-2507")
    # Mid-review classifier (4 modes, single call, no fallback) — same reliable
    # non-reasoning model.
    llm_model_mid_review: str = os.getenv("LLM_MODEL_MID_REVIEW", "qwen/qwen3-30b-a3b-instruct-2507")
    llm_brief_ttl_seconds: int = _int_env("LLM_BRIEF_TTL_SECONDS", 3600)
    # Default floor 1000 (.env overrides to 1500). Empty-response handling is
    # done via retry-on-empty + JSON-extraction in polza_client.py, not here.
    llm_brief_max_tokens: int = _int_env("LLM_BRIEF_MAX_TOKENS", 1000)
    llm_explain_max_tokens: int = _int_env("LLM_EXPLAIN_MAX_TOKENS", 350)
    # Min order value (RUB) to trigger an LLM explanation. 0 = explain EVERY
    # trade (a full decision log aids review).
    llm_explain_min_value_rub: float = _float_env("LLM_EXPLAIN_MIN_VALUE_RUB", 0.0)
    llm_timeout_seconds: float = _float_env("LLM_TIMEOUT_SECONDS", 15.0)
    llm_apply_brief_to_risk: bool = _bool_env("LLM_APPLY_BRIEF_TO_RISK", True)
    news_enabled: bool = _bool_env("NEWS_ENABLED", True)
    news_ttl_seconds: int = _int_env("NEWS_TTL_SECONDS", 1200)
    news_timeout_seconds: float = _float_env("NEWS_TIMEOUT_SECONDS", 8.0)
    news_max_items: int = _int_env("NEWS_MAX_ITEMS", 12)
    memory_enabled: bool = _bool_env("MEMORY_ENABLED", True)
    memory_dataset_path: Path = Path(os.getenv("MEMORY_DATASET_PATH", "./data/logs/ml_dataset_offline.jsonl"))
    memory_k_neighbors: int = _int_env("MEMORY_K_NEIGHBORS", 5)
    memory_max_rows: int = _int_env("MEMORY_MAX_ROWS", 100000)
    arena_max_consecutive_failures: int = _int_env("ARENA_MAX_CONSECUTIVE_FAILURES", 10)
    arena_retry_attempts: int = _int_env("ARENA_RETRY_ATTEMPTS", 3)
    arena_retry_backoff_seconds: float = _float_env("ARENA_RETRY_BACKOFF_SECONDS", 1.5)

    # ArenaGo lot sizes
    arena_lot_sizes: dict[str, int] = field(default_factory=lambda: {
        "LKOH": 1, "SBER": 1, "ROSN": 1, "GAZP": 10, "YDEX": 1,
        "NVTK": 1, "GMKN": 10, "MOEX": 10, "MTSS": 10, "PLZL": 1,
        "MGNT": 1, "ALRS": 10, "AFLT": 10, "CHMF": 1, "NLMK": 10,
        "SNGSP": 10, "PIKK": 1, "VTBR": 100, "T": 1, "X5": 1,
    })
    # LLM supervisor scheduling.
    #  - mid_review: interval-based, every N minutes (regime/mode adaptation).
    #  - auto_tuner: time-of-day, fires once per listed UTC hour per day. We do
    #    NOT run it at the open (no fresh same-day trades → it gates out on the
    #    ">=5 closed trades" rule). Defaults: 08:00 UTC (11:00 MSK, mid first
    #    half) and 16:00 UTC (19:00 MSK, second half).
    mid_review_interval_min: int = _int_env("MID_REVIEW_INTERVAL_MIN", 45)
    auto_tuner_utc_hours: str = os.getenv("AUTO_TUNER_UTC_HOURS", "8,16")

    @property
    def auto_tuner_utc_hours_list(self) -> list[int]:
        out: list[int] = []
        for part in str(self.auto_tuner_utc_hours).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                h = int(part)
            except ValueError:
                continue
            if 0 <= h <= 23:
                out.append(h)
        return sorted(set(out)) or [8, 16]

    # Shorts (flag-gated; inert until ENABLE_SHORTS=true and paper-tested).
    # Small size, few concurrent, tight stop — shorts on MOEX equities carry
    # gap/borrow risk, so they're deliberately more conservative than longs.
    enable_shorts: bool = _bool_env("ENABLE_SHORTS", True)
    short_order_cash_pct: float = _float_env("SHORT_ORDER_CASH_PCT", 0.02)
    # Favourable stop/take ratio: a tighter stop than take gets shorts stopped
    # out before target on a noisy tape, so let a working short run.
    short_stop_loss_pct: float = _float_env("SHORT_STOP_LOSS_PCT", 0.010)
    short_take_profit_pct: float = _float_env("SHORT_TAKE_PROFIT_PCT", 0.018)
    # Trailing stop / time exit for shorts — the two mechanisms that fixed the
    # long side's loss asymmetry. 0 = off; set from backtest, not by guess.
    short_trail_stop_pct: float = _float_env("SHORT_TRAIL_STOP_PCT", 0.0)
    short_time_exit_hours: float = _float_env("SHORT_TIME_EXIT_HOURS", 0.0)
    short_time_exit_min_profit_pct: float = _float_env("SHORT_TIME_EXIT_MIN_PROFIT_PCT", 0.003)
    # Concurrent-shorts cap high enough that the short side can express itself
    # (a low cap rejects most generated short signals).
    max_concurrent_shorts: int = _int_env("SHORT_MAX_CONCURRENT", 6)
    # Total gross short exposure cap (sum of |short value| / equity), enforced
    # in risk.py.
    short_max_total_exposure_pct: float = _float_env("SHORT_MAX_TOTAL_EXPOSURE_PCT", 0.06)
    # Soft diversification: max distinct positions per sector (0 = off). New
    # names beyond the cap are deferred; adds to names already held still pass.
    max_positions_per_sector: int = _int_env("MAX_POSITIONS_PER_SECTOR", 3)
    ml_short_model_path: Path = Path(os.getenv("ML_SHORT_MODEL_PATH", "./data/models/lgbm_short_filter_v2.joblib"))
    ml_min_down_probability: float = _float_env("ML_MIN_DOWN_PROBABILITY", 0.70)

    @property
    def ml_short_model_abs_path(self) -> Path:
        return self.ml_short_model_path if self.ml_short_model_path.is_absolute() else PROJECT_ROOT / self.ml_short_model_path

    # Macro trend gate: block NEW long entries when IMOEX is in a strong 4h
    # downtrend (don't buy "oversold" into a broad selloff). Fraction, e.g.
    # -0.015 = -1.5%. 0 = off. Shorts are NOT gated (they should fire in a
    # downtrend).
    macro_block_return_4h: float = _float_env("MACRO_BLOCK_RETURN_4H", -0.015)
    # Regime-directional bias: in a falling market (brief regime risk_off /
    # news_shock) take the side of the trend — suppress NEW longs and let the
    # short side fire, rather than fighting the downtrend with long-biased
    # mean-reversion. 'on' (default) only acts when the brief is applied to
    # risk. Comma list of regimes that flip us short.
    regime_directional: bool = _bool_env("REGIME_DIRECTIONAL", True)
    bearish_regimes: str = os.getenv("BEARISH_REGIMES", "risk_off,news_shock")

    @property
    def bearish_regimes_set(self) -> set[str]:
        return {r.strip().lower() for r in self.bearish_regimes.split(",") if r.strip()}
    # Global socket timeout (sec) so a hung MOEX/ALGOPACK request can't freeze
    # the whole trading loop during autonomous operation. Applies to any request
    # that doesn't set its own timeout (moexalgo's do not). Legit calls take 1-3s.
    socket_timeout_seconds: float = _float_env("SOCKET_TIMEOUT_SECONDS", 30.0)
    # Flatten policy: close positions and stop opening new ones near the 20:50
    # UTC close so nothing is carried overnight (gap risk).
    flatten_before_close: bool = _bool_env("FLATTEN_BEFORE_CLOSE", True)
    no_new_entry_minutes_before_close: int = _int_env("NO_NEW_ENTRY_MINUTES_BEFORE_CLOSE", 10)
    # Force-flatten longs / force-cover shorts only in the final N minutes before
    # the 20:50 UTC close. Minute-based (not hour-based) so positions run until
    # the last N minutes, then close to avoid overnight gap risk.
    flatten_minutes_before_close: int = _int_env("FLATTEN_MINUTES_BEFORE_CLOSE", 10)
    # Stale-data guard: don't OPEN a new position if the latest candle is older
    # than this (minutes). Protects weekend/illiquid sessions where ALGOPACK may
    # serve the prior session's candles. Exits/covers/flatten are never gated.
    max_candle_age_minutes: int = _int_env("MAX_CANDLE_AGE_MINUTES", 30)

    ml_label_horizon_steps: int = _int_env("ML_LABEL_HORIZON_STEPS", 3)
    ml_filter_enabled: bool = _bool_env("ML_FILTER_ENABLED", True)
    ml_min_up_probability: float = _float_env("ML_MIN_UP_PROBABILITY", 0.58)
    ml_model_path: Path = Path(os.getenv("ML_MODEL_PATH", "./data/models/lgbm_buy_filter_v5.joblib"))

    @property
    def runtime_data_dir(self) -> Path:
        return self.data_dir if self.data_dir.is_absolute() else PROJECT_ROOT / self.data_dir

    @property
    def ml_model_abs_path(self) -> Path:
        return self.ml_model_path if self.ml_model_path.is_absolute() else PROJECT_ROOT / self.ml_model_path

    @property
    def memory_dataset_abs_path(self) -> Path:
        return self.memory_dataset_path if self.memory_dataset_path.is_absolute() else PROJECT_ROOT / self.memory_dataset_path


settings = Settings()
