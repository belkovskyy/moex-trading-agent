# MOEX AI Agent

Автономный торговый агент для MOEX (ArenaGo). Построен как операционная система
для торговли: детерминированное ядро принимает торговые решения по количественным
правилам и ML-фильтру, независимый риск-слой контролирует потери, а LLM работает
вне горячего пути как слой интерпретации контекста и адаптации параметров.

Подробный обзор архитектуры и ключевых идей — в [SOLUTION_OVERVIEW.md](SOLUTION_OVERVIEW.md).
Лицензии зависимостей — в [LICENSES.md](LICENSES.md).

## Запуск

```powershell
cd "D:\Биржа"
conda activate moex-agent          # либо venv c requirements.txt
python -m moex_agent
```

Конфигурация — в `.env` (загружается автоматически). Ключевые переменные:

| Переменная | Назначение |
|---|---|
| `SANDBOX_API_KEY` | ключ ArenaGo (на автономном этапе подставляется через ENV) |
| `MOEX_ALGO_TOKEN` | токен ALGOPACK для рыночных данных |
| `POLZA_API_KEY` | ключ агрегатора LLM Polza.ai |
| `DRY_RUN`, `PAPER_TRADING` | `false` — реальная торговля; `true` — без отправки заявок |

## Структура

| Модуль | Роль |
|---|---|
| `app.py` | главный цикл, планировщик LLM-слоёв, восстановление после рестарта |
| `market_data.py` | адаптер ALGOPACK (свечи, tradestats, orderstats, obstats) |
| `indicators.py` | RSI, MACD, Bollinger Bands, ATR, volume_ratio |
| `regime.py` | определение режима рынка (normal/trend/range/high_volatility/crisis) |
| `strategy.py` | сигналы входа и выхода (long, short, cover) по явным правилам |
| `ml_features.py`, `ml_filter.py` | вектор признаков и LightGBM-фильтр сигналов |
| `risk.py` | риск-менеджер: стопы, лимиты, sizing, kill-switch |
| `arena_client.py` | адаптер ArenaGo с ретраями и kill-switch |
| `models.py` | модели данных, учёт PnL (long и short) |
| `llm/`, `mid_review.py`, `auto_tuner.py`, `daily_retro.py` | LLM-слой (см. ниже) |
| `memory.py` | retrieval похожих исторических ситуаций (kNN) |
| `ml_offline_dataset.py`, `ml_baseline.py` | сборка датасета и обучение модели |
| `backtest.py` | прогон стратегии на исторических данных |
| `monitor.py`, `report.py` | мониторинг состояния и отчёты |

## LLM-слой

LLM не принимает торговых решений и не находится в горячем пути. Четыре роли,
все кэшируются и не блокируют выставление заявки. Модель под каждую роль подобрана
по стоимости/качеству:

- **Brief** (DeepSeek V4 Flash, резерв Qwen3 30B; каждые 5 мин) —
  классифицирует режим рынка, задаёт множитель размера и список бумаг к
  исключению. Учитывает индикаторы, макроконтекст и новости MOEX.
- **Mid-review** (Qwen3 30B Instruct; каждые 45 мин + при смене режима) — выбирает
  режим сессии (обычный / осторожный / пауза покупок / ускоренный).
- **Auto-tuner** (Qwen3 30B Instruct; 8 раз в день + при просадке) — корректирует
  параметры в пределах жёстких границ.
- **Explainer / Daily-retro** (DeepSeek V4 Flash / Qwen3 30B Instruct) — объяснение каждой сделки и
  разбор торгового дня для журнала и экспертной оценки.

Детали архитектуры и список ML-признаков — в [SOLUTION_OVERVIEW.md](SOLUTION_OVERVIEW.md).

Отказоустойчивость: при недоступности LLM brief переходит в осторожный режим,
торговля продолжается на детерминированном слое.

## Команды

```powershell
python -m moex_agent                 # основной торговый цикл
python -m moex_agent.monitor         # сводка состояния
python -m moex_agent.report          # отчёт за день
python -m moex_agent.daily_retro     # LLM-разбор дня
python -m moex_agent.backtest --ml-model data/models/lgbm_buy_filter_v4.joblib --ml-threshold 0.65
```

### Бэктест шорт-стороны

Бэктест прогоняет обе стороны и печатает метрики раздельно (`by_direction`):
win rate, отношение среднего убытка к среднему выигрышу, profit factor и
время удержания отдельно для лонгов и шортов. Без этой разбивки прибыльная
сторона прячется за убыточной.

```powershell
python -m moex_agent.backtest --dataset data/logs/ml_dataset_local.jsonl `
  --enable-shorts --short-order-cash-pct 0.08 --short-max-exposure-pct 0.35 `
  --short-ml-model data/models/lgbm_short_filter_v1.joblib --short-ml-threshold 0.60 `
  --ml-model data/models/lgbm_buy_filter_v4.joblib --ml-threshold 0.58
```

Трейлинг-стоп и выход по времени для шортов (`--short-trail-stop-pct`,
`--short-time-exit-hours`) по умолчанию выключены: на замере 2023–2026 они
ухудшали результат — шорту нужно ~30 часов, ранний выход срезает эдж.

### Работа без подписки ALGOPACK

Датасет собирается из локальной выгрузки parquet, токен не нужен:

```powershell
python -m moex_agent.ml_offline_dataset --local-parquet "D:\DS\algopack" `
  --symbols GMKN,VTBR,MTSS --start 2023-01-01 --end 2026-07-01 `
  --target-mode triple_barrier --use-super-features --replace `
  --dataset data/logs/ml_dataset_local.jsonl
```

Макро-признаки (IMOEX, отраслевые индексы) при таком запуске нулевые — они
тянутся только через API.

## Тесты

```powershell
python -m pytest tests -q
```

## Воспроизводимость ML-модели

```powershell
# 1. Сборка офлайн-датасета (20 тикеров, 3 года, triple-barrier разметка)
python -m moex_agent.ml_offline_dataset --days 1095 --target-mode triple_barrier `
  --horizon-steps 9 --barrier-up 0.010 --barrier-down 0.007 --use-super-features --replace

# 2. Обучение модели с purged walk-forward CV
python -m moex_agent.ml_baseline --dataset data/logs/ml_dataset_offline.jsonl `
  --task binary_up --cv-folds 5 --embargo-candles 10 --output data/models/lgbm_buy_filter_v4.joblib
```

Активная модель: `lgbm_buy_filter_v4` — 30 признаков (TA + макро + время суток),
walk-forward ROC-AUC ≈ 0.75. Скрипты, параметры и метаданные (`*.meta.json`) —
в репозитории.

## Развёртывание

```powershell
docker build -t moex-agent .
docker run --rm -v ${PWD}\data:/data moex-agent
```

Контейнер хранит состояние в `/data` (постоянный том) и пишет JSON-логи в stdout.
Развёртывание — через GitLab CI на Yandex Cloud; `SANDBOX_API_KEY` подставляется
средой исполнения.

## Источники данных

| Источник | Назначение |
|---|---|
| ArenaGo (`arenago.ru/api`) | портфель, позиции, сделки, выставление заявок |
| ALGOPACK / MOEX (`apim.moex.com`) | свечи, Super Candles, макроданные |
| MOEX ISS news | заголовки новостей для brief |
| Polza.ai | вызовы LLM |
