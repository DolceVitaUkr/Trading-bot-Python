🚀 World-Class Self-Learning Multi-Asset Trading Bot
📌 Overview

This project is a self-learning, multi-asset trading bot designed for institutional-grade robustness.
It supports:

Crypto Spot & Futures (Bybit)

Forex & Options (IBKR)

The bot uses a layered AI approach (Machine Learning, Reinforcement Learning, Ensemble Models) and is built for:

Continuous self-learning

Strict risk & compliance controls

Transparent UI & notifications

Modular, extensible design

It always trains and validates strategies before trading live, with a paper-trading sandbox running in parallel for continuous exploration.

📂 File & Naming Guidelines

All files lowercase without underscores: core/datamanager.py

First line of every source file:

# file: core/datamanager.py


Folder structure:

tradingbot/
  core/        # risk, config, runtime, validation, portfolio, execution
  brokers/     # broker adapters (bybit, ibkr)
  learning/    # ML/RL training and feature processing
  ui/          # FastAPI + dashboard
  config/      # all JSON configs
  tests/       # unit & integration tests
  state/       # checkpoints, runtime states
  logs/        # validation, backtest, ops logs

🧠 Learning Logic

Training First

Starts in paper trading mode only.

Uses historical + live-streamed data for:

ML (triple-barrier, meta-labeling, meta-ensembles).

RL (PPO, DQN, SAC, with experience replay and uncertainty penalties).

Data validated & features versioned in a feature store.

Validation

Every new or retrained strategy must pass the Validation Manager:

Event-driven backtesting with fees, slippage, latency, partial fills.

Walk-forward analysis with purge+embargo to prevent look-ahead bias.

Stress tests: Monte Carlo path bootstraps, slippage shocks, parameter perturbations, regime slicing.

Metrics required: ≥ 500 trades, Sharpe ≥ 2.0, MaxDD ≤ 15%, Profit Factor ≥ 1.5, CVaR(5%) within threshold.

Only validated strategies may be promoted to live trading.

Live Trading + Continuous Exploration

When an operator enables live mode via UI/Telegram:

Only validated strategies trade live.

In parallel, paper sandbox keeps testing new ideas.

Results from paper → validation → possible live promotion.

🔐 Risk Management

Trade Size Policy

Until equity ≥ $1000 → fixed $10 per trade.

After ≥ $1000 → dynamic sizing:

size = (equity * risk_fraction * signal_weight) / (sl_distance * price)


Risk fraction: 0.5%–2% (tiered by equity).

Signal weight: weak=0.5, normal=1.0, strong=1.5.

Drawdown penalty: −0.25% per 5% drawdown, floor 0.25%.

Hard Rules

SL always in place (≤ 15%).

Max portfolio exposure % per asset.

Daily realised loss cap.

Correlation caps: don’t overexpose to correlated pairs.

🧪 Validation Manager

Backtesting

Realistic fills (slippage, spread, partials, fees).

Futures funding + borrow.

Walk-Forward

Purged + embargoed splits.

Per-fold metrics aggregated.

Stress Testing

Monte Carlo reordering of trades.

Parameter perturbations ±10–20%.

Latency/slippage shock scenarios.

Validation across high-vol vs low-vol periods.

OPE for RL

Weighted Importance Sampling (WIS).

Doubly Robust estimators.

Promotion Gate

Trades ≥ 500

Sharpe ≥ 2.0

MaxDD ≤ 15%

PF ≥ 1.5

CVaR(5%) within bounds

Reports

JSON + HTML summary under logs/validation/{strategy_id}

UI modal view, Telegram pass/fail notification

🚨 Kill Switch Logic

Manual toggle (UI or Telegram).

Auto trigger if consecutive losses exceed config threshold.

Behaviour:

Stop opening new live trades.

Paper continues learning.

Close profitable trades gracefully.

Keep SL/TP on open positions.

Resume requires explicit operator approval.

🖥️ UI Dashboard

Four Panels (Crypto Spot, Crypto Futures, Forex, Options).

Each panel shows:

Equity, P&L, MaxDD, Sharpe, Sortino, CVaR, Win%, PF.

Live/paper toggle switch.

Validation report access.

/diff view: “what live trades would have been placed” vs actual paper.

Routes

/status – bot health & balances

/live/{asset}/enable|disable – per-asset toggle

/kill/{asset} – kill switch

/diff/{asset} – dry-run diff preview

/validation/{strategy} – latest validation report

📲 Telegram Notifications

Lifecycle

Start/Stop.

Hourly paper recap: trades, P&L, reward points.

Live trade open/close: size, fees, net P/L.

Validation pass/fail.

Kill switch trigger.

📊 Observability & Ops

Experiment tracking via MLflow/W&B.

Drift monitoring (features, labels, model predictions).

CI/CD with lint, type checks, backtest snapshots, integration tests.

Audit logging: all operator actions (toggles, kill switches).

🔒 Security & Compliance

No secrets in repo.

API keys stored in secure vault (keyring/KMS).

IP whitelisting where possible.

Data retention policies for logs/state.

📦 Requirements

Core: pandas, numpy, pyarrow, pydantic

ML/RL: torch, scikit-learn, lightgbm, xgboost, optuna

Indicators: ta or custom

UI: fastapi, uvicorn

Orchestration: mlflow or wandb

Optimizers: deap

Notifications: python-telegram-bot

Tests: pytest, mypy, flake8

✅ Summary Flow

Train → Validate → Promote → Live

Always paper-trains first.

Validation Manager ensures robustness.

Only then eligible for live toggles.

Paper sandbox continues forever → feeding new candidates.

Kill switch prevents blowups.

UI + Telegram = full transparency.
