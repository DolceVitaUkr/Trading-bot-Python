Trading Bot – Multi-Asset, Training-First, UI-First (World-Class Edition)
Overview

Assets: Crypto Spot & Futures (Bybit), Forex & Options (IBKR)

Modes: Training (paper) by default; Live is per-asset toggle with double-confirm + validation gates

UI: FastAPI web dashboard (loads first) with 4 panels (Crypto Spot, Crypto Futures, Forex, Options)

Goal: A robust, self-learning bot (ML + RL), strict risk, and production-grade reliability

Core Principles

Safety first: Paper mode by default; no live orders until explicitly enabled per asset

Transparency: Telegram alerts, hourly paper recap, /diff preview of would-be live orders

Modularity: Clear, testable modules; simple public APIs; cross-platform paths

Consistency: All module filenames lowercase; each file begins with # file: <path>

Architecture (Folders & Responsibilities)
tradingbot/
  core/        # config_manager, datamanager, indicators, pair_manager,
               # strategy_manager, reward_system, risk_manager, trade_executor,
               # portfolio_manager, validation_manager, optimizer,
               # notifier, error_handler, runtime_controller
  brokers/     # exchangebybit.py (spot/futures), exchangeibkr.py (forex/options)
  learning/    # train_ml_model.py, train_rl_model.py, save_ai_update.py
  ui/          # FastAPI app, routers, websockets, routes/diff.py (dry-run preview)
  config/      # config.json, assets.json, strategies.json
  logs/        # json/csv: trades, decisions, errors, telemetry, validation
  state/       # models/, replay_buffers/, checkpoints/, caches/, runtime.json

Naming & Headers

File names: lowercase with underscores (e.g., core/datamanager.py)

Header in every file (first line):
# file: core/datamanager.py

Safety Defaults (Out-of-the-Box)

Start mode: Training (paper)

Live: Disabled for all assets until manually enabled per asset (double-confirm + validation)

Kill switch (Live only): Manual and auto (on consecutive losses); sets close-only; paper continues

Hard SL cap: 15% max per trade (never widen SL)

UI loads first: See everything before any transaction

Features (What You Get)

Unified Dashboard: 4 panels with status, equity, P&L, open positions, logs, controls (live toggle, kill switch, stop options)

Paper & Live Mode: Paper always on; Live only after validation gates pass

/diff Preview: Compare current paper intents vs would-be live orders before enabling Live

Telegram Alerts: Start/Stop, status, hourly paper recap, live opens/closes (P/L%, fees, pure profit, reward points), validation, kill-switch

Dynamic Pair Rotation: Volatility, momentum, liquidity, spread; optional sentiment weights

Indicators: Rich TA set (SMA/EMA, RSI, Stoch, Williams %R, ATR, stdev bands, MFI, OBV, Fib; plus HMA, KAMA, SuperTrend, Donchian, Keltner, Bollinger %B, VWAP bands; Ichimoku optional)

Learning Engine (ML + RL):

RL: Double+Dueling DQN + Prioritized Replay + n-step; PPO+GAE path; LSTM/Transformer encoders

ML: Triple-Barrier labels + Meta-labeling; walk-forward CV with purge/embargo; SHAP pruning

Validation Gates: Trades ≥ 500, Sharpe ≥ 2.0, Max DD ≤ 15% (per asset, before Live)

Persistence & Recovery: Resume models, replay, pair universe, runtime state; reconcile open live positions

Backtesting & OOS: Purged, embargoed walk-forward; stress slippage; Monte Carlo trade bootstraps

Risk Management (Aligned)

Mandatory SL & TP for every trade; SL cap 15%

Learning phase sizing: $10 fixed notional per trade until equity ≥ $1,000

Growth phase sizing (≥ $1,000):

Equity tiers: ~0.5% → 2.0% risk fraction

Drawdown adjuster: −0.25% risk per 5% DD (floor 0.25%)

Signal weighting: weak <0.6 → 0.5×; strong 0.6–0.8 → 1.0×; very strong >0.8 → 1.5×

Size formula: (equity * risk_fraction) / (stop_loss_distance * price)

Daily loss & exposure caps: configurable; enforce per asset

Kill switch: close-only; ensure SL/TP; close profitable where possible; Live paused, Paper continues

Data & Indicators

Data lake: Parquet + metadata; gap detection/backfill; multi-TF joins (5m entry, 15m regime, 1h context)

Regime tagging: Volatility/trend/liquidity regimes; optional HMM (2-state)

Microstructure (optional/where available): spread, imbalance, depth; funding, OI, basis (futures)

Learning Engine (Details)

RL: DQN (Double+Dueling, Prioritized Replay, n-step), PPO (GAE), soft targets, entropy scheduling, obs/reward norm

ML: Gradient boosting / linear / shallow NN; Triple-Barrier & Meta-labeling; purged WFCV; SHAP

Rewards: After-fees P&L − λ_dddrawdown − λ_turnturnover − λ_tail*CVaR_tail + bonus_TP

Curriculum: Phase 1 ($10 fixed, majors), Phase 2 (tiers+more symbols), Phase 3 (futures/options + microstructure)

UI & Endpoints (FastAPI)

Loads first, before any orders

Key endpoints:

GET /status – runtime snapshot

POST /live/{asset}/enable – double-confirm + validation gates

POST /live/{asset}/disable – turn off live for asset

POST /kill/{scope}/{onoff} – scope = asset|global; onoff = true|false

POST /stop/{asset}?mode=close_all|keep_open – stop options

GET /diff/{asset} – dry-run (paper vs would-be live)

POST /diff/confirm/{asset} – confirm and enable live (after gates)

WS /stream – equity, P&L, positions, logs, metrics tiles

Telegram (Examples)

Lifecycle: “🚀 Start (Training). Live disabled for all assets.” / “🛑 Stop.”

Hourly Paper Recap: P/L%, fees, pure profit, reward points, trade count

Paper Intent: “🧪 BUY BTCUSDT ~ $10.00 (dry-run)”

Live Open/Close: includes notional, P/L%, fees, pure profit, reward points

Validation/Kill: pass/fail reports; kill switch on/off; close-only states

Configuration (Samples)

config/config.json

{
  "START_MODE": "training",
  "LIVE_TRADING_ENABLED": false,
  "REQUIRE_MANUAL_LIVE_CONFIRMATION": true,
  "ORDER_ROUTING": "simulation",
  "MAX_STOP_LOSS_PCT": 15.0,
  "KILL_SWITCH_ENABLED": true,
  "CONSECUTIVE_LOSS_KILL": 3,
  "PAPER_EQUITY_START": 1000.0,
  "PAPER_RESET_THRESHOLD": 10.0,
  "TELEGRAM": {
    "ENABLED": true,
    "BOT_TOKEN": "YOUR_TOKEN",
    "CHAT_ID": "YOUR_CHAT_ID",
    "SEND_TRADE_INTENTS_IN_TRAINING": true,
    "SEND_EXECUTIONS_IN_LIVE": true,
    "SEND_ERRORS": true,
    "SEND_HOURLY_PAPER_RECAP": true
  }
}


config/assets.json (example)

{
  "bybit": {
    "spot":    { "symbols": ["BTCUSDT","ETHUSDT"], "fees_bps": 10, "min_notional": 10, "tick_size": 0.01, "qty_step": 0.0001 },
    "futures": { "symbols": ["BTCUSDT","ETHUSDT"], "fees_bps": 10, "tick_size": 0.5,  "qty_step": 0.001,  "leverage_max": 5 }
  },
  "ibkr": {
    "forex":   { "symbols": ["EURUSD","GBPUSD"],   "fees_bps": 2,  "tick_size": 0.00005, "qty_step": 1000 },
    "options": { "underlyings": ["AAPL","SPY"],    "fees_per_contract": 0.65 }
  }
}


config/strategies.json (high-level)

{
  "global": {
    "validation": { "min_trades": 500, "min_sharpe": 2.0, "max_drawdown_pct": 15.0 }
  },
  "state": { "tf": { "entry": "5m", "regime": "15m", "context": "1h" }, "encoder": "lstm" },
  "indicators": {
    "ema": { "short": [5, 8, 13], "long": [21, 50] },
    "atr": { "period": 14, "tp_mult": [2.0, 2.5], "sl_mult": [1.5, 2.0] },
    "rsi": { "period": 14 },
    "mfi": { "period": 14 },
    "williams_r": { "period": 14 },
    "fib": { "lookback_bars": 300 },
    "extras": { "hma": 21, "kama": 10, "supertrend": {"atr":10,"mult":3}, "donchian": 20, "keltner": {"ema":20,"atr":2} }
  },
  "rl": {
    "engine": "dqn_dueling_prioritized",
    "n_step": 3,
    "double": true,
    "prio": { "alpha": 0.6, "beta0": 0.4, "anneal_steps": 100000 }
  },
  "ppo": { "gae_lambda": 0.95, "entropy_coef": 0.01, "clip_ratio": 0.2 },
  "reward": { "fee_model": "after_fees", "dd_penalty": 0.5, "turnover_penalty": 0.1, "cvar_tail_pct": 5, "tp_bonus": 0.2 },
  "sizing": {
    "fixed_until_equity": 1000,
    "fixed_notional": 10,
    "tiers": [[1000,0.01],[5000,0.015],[20000,0.02]],
    "drawdown_step_pct": 5,
    "drawdown_step_delta": 0.0025,
    "signal_weight": {"weak":0.5,"strong":1.0,"very_strong":1.5}
  }
}

Getting Started
git clone https://github.com/DolceVitaUkr/Trading-bot-Python.git
cd Trading-bot-Python
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m tradingbot.run_bot
# Open the Dashboard URL shown in the console (UI loads first; paper mode only)

Minimal Entrypoint (for reference)
# file: tradingbot/run_bot.py
"""
UI-first, training-first launcher. Live routing is disabled at boot.
"""

from tradingbot.core.config_manager import ConfigManager
from tradingbot.core.runtime_controller import RuntimeController
from tradingbot.ui.app import run_dashboard

def main():
    cfg = ConfigManager("tradingbot/config/config.json").load()
    ctl = RuntimeController(cfg)
    ctl.start()  # sends Telegram "Start", initializes paper mode, persists runtime state
    run_dashboard(cfg=cfg, controller=ctl)

if __name__ == "__main__":
    main()

Tests

Unit: risk sizing math ($10 rule + growth tiers), SL cap, kill-switch transitions, broker precision, regime tagging, triple-barrier labeling

Integration: resume/reconcile open live positions; UI /diff → enable live flow; validation gates; Telegram events

Determinism: seeded backtests; logged configs; reproducible reports in logs/validation

Ops & Monitoring

Tracking: MLflow/W&B runs, metrics, artifacts; model registry

Metrics tiles in UI: Sharpe, Sortino, MaxDD, CVaR(5%), PF, Turnover, Slippage error, Latency p95

Alerts: validation pass/fail, kill-switch actions, daily P&L snapshot, OOS breaches

Conventions Recap

All filenames: lowercase with underscores (e.g., core/pair_manager.py)

First line in every source file: # file: <relative-path/filename.py>

Suggestions (optional)

Add offline RL (CQL/TD3+BC) pretraining on historical fills + doubly-robust OPE before paper rollouts

Add correlation/cluster caps across symbols to reduce concentration risk

Add shadow live (paper mirror) for a week before enabling live on a new asset class
