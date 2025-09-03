FolderStructure.md

Version 1.0

This document describes the repository folder layout and the responsibility of each directory/file. Use this as a navigation guide. For technical details of functions and variables, see Modules.md.

Root
trading-bot-python/
├─ tradingbot/
├─ scripts/
├─ state/
├─ models/
├─ metrics/
├─ ui/
├─ logs/
└─ artifacts/

tradingbot/

Main source code for the bot.

core/

runtime_api.py → Aggregates broker & paper state for UI.

order_router.py → Central order placement with guardrails and idempotency.

budget_manager.py → Budget logic, risk % sizing, rollover, and equity ladder.

history_store.py → Reads/writes trade history JSONL (8 tables).

paper_state.py → Paper trading wallet and positions.

retry.py → Retry/backoff utility.

loggerconfig.py → Logging configuration (JSONL + console).

Responsibility: Market-neutral core logic. Never depends on UI, only consumed by API/adapters.

brokers/

bybit_adapter.py → Bybit integration via ccxt. Wallet, positions, trades.

ibkr_adapter.py → IBKR integration via ib_insync. Wallet, portfolio, executions.

Responsibility: Abstraction of broker APIs. Must only expose normalized data.

strategies/

manager.py → Strategy registry & lifecycle (start/stop).

strategy_*.py → Individual strategy implementations (e.g., SMA cross, RSI).

Responsibility: Define signals; never execute orders directly. Emit (side, conf) only.

training/

train_manager.py → Orchestrates ML/RL runners, persistence.

ml_trainer.py (planned) → Supervised indicator learner (XGBoost/LightGBM).

rl_trainer.py (planned) → PPO agent (Stable-Baselines3).

validation_manager.py (planned) → Promotion/degrade logic for strategies.

Responsibility: Model training & validation pipeline.

scripts/

cleanup_logs.py → Archives logs to artifacts/logs/YYYY-MM-DD/.

(future) backtest_validator.py → Run validation on offline tick data.

Responsibility: Operational utilities.

state/

budgets.json → Asset budgets, risk %, sizing thresholds, rollover, leverage caps.

runtime.json → Live/paper enable flags.

paper/ → JSONL trade logs for paper mode (per asset).

live/

bybit/ → JSONL trade logs for Bybit live.

ibkr/ → JSONL trade logs for IBKR live.

Responsibility: Ephemeral bot state (positions, trades, budgets).
Never commit these to Git.

models/

{asset}/ml/ → ML checkpoints.

{asset}/rl/ → RL checkpoints.

Responsibility: Persisted models per asset.

metrics/

{asset}/train_ml.jsonl → ML training metrics.

{asset}/train_rl.jsonl → RL training metrics.

Responsibility: Training/evaluation logs for model monitoring.

ui/

app.py → FastAPI app exposing REST endpoints.

templates/ → Dashboard HTML templates.

static/

wire.js → Fetch wrapper and UI event handlers.

nethealth.js → Broker/adapter health polling.

Responsibility: User interface & API. Talks only to runtime_api, strategy_manager, train_manager.

logs/

Rotating structured logs (JSONL + console).

Controlled by loggerconfig.py.

Responsibility: Operational logs. Archived daily by scripts/cleanup_logs.py.

artifacts/

logs/YYYY-MM-DD/ → Archived logs.

(future) screenshots, reports, validation outputs.

Responsibility: Historical archives. Safe to purge without affecting state.

⚠️ Notes

state/ and logs/ should be .gitignored (ephemeral).

Only models/ and metrics/ may be versioned selectively if you want reproducibility.

Everything else is source or UI.

