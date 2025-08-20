Trading Bot – Multi-Asset Unified Architecture (Training-First, UI-First)
Overview

A self-learning, multi-asset trading bot with a web dashboard (FastAPI).
Supports:

Crypto (Bybit) – Spot & Futures

Forex & Options (IBKR)

Safety-first defaults

Starts UI first and Training (paper) first

No live orders until you manually enable per-asset Live+Paper toggles (double-confirmation)

Kill Switch only affects live orders (paper keeps running)

Key Features

Unified Dashboard (4 panels): Crypto Spot, Crypto Futures, Forex, Options

Status, wallet/equity, active strategy, open positions, P&L/balance graphs

Controls: Per-asset Live toggle, Kill Switch, Graceful Exit, Close-Only

Paper & Live Modes

Paper: $1,000 virtual equity per asset class; reset if < $10

Live: requires manual enable per asset + passing Validation Gates (≥500 trades, Sharpe ≥ target, Max DD ≤ cap)

Adaptive Strategies

Flagged by asset type, regime (trend/range/vol), wallet size, time-of-day/week

Risk Management

Mandatory SL & TP, trailing TP, max SL = 15% (configurable), position caps

Kill Switch (manual or auto on consecutive losses) → Close-Only for live

Telegram Notifications

Start/Stop, status updates, hourly paper recap (trades, P/L %, fees, pure profit, reward points)

Live trade open (amount used) and close (P/L %, fees, pure profit, reward points)

Errors, validation pass/fail, mode flips, kill switch events

Learning Engine (RL/ML)

PyTorch (DQN/PPO selectable); experience replay, epsilon-greedy (DQN)

Rewards on profit after fees, DD penalties, TP bonuses

Persist models/checkpoints/replay for continuity

Dynamic Pair Selection (PairManager)

Ranks by volatility (ATR/stdev), momentum (ROC/EMA slope), liquidity/volume, spread

Optional sentiment boost/penalty; 15-min refresh with cache fallback

Technical Indicators (baseline)

Trend: SMA/EMA, MACD (opt)

Momentum: RSI, Stoch, Williams %R

Volatility: ATR, stdev bands

Volume/Flow: MFI, OBV (opt)

Fibonacci swing levels for TP/SL context

Parameter Optimization (offline)

Grid, Random, Evolutionary (DEAP); multi-metric objective (Sharpe, max DD, win%, avg return, trades≥500)

Writes best sets to config/strategies.json

Persistence & Restart Protection

Reload models, replay buffers, pair universe, paper equity

Reconcile live open positions from brokers and resume management

Architecture
tradingbot/
  core/        # config_manager, datamanager, strategymanager, riskmanager,
               # tradeexecutor, portfoliomanager, validationmanager, pairmanager, notifier
  brokers/     # exchangebybit.py, exchangeibkr.py (adapters)
  learning/    # train_ml_model.py, train_rl_model.py, save_ai_update.py
  ui/          # FastAPI app, routers, websockets (UI loads FIRST)
  config/      # config.json, assets.json, strategies.json
  logs/        # json/csv: trades, decisions, errors, telemetry
  state/       # models/, replay_buffers/, checkpoints/, caches/, runtime.json

Main Flow

UI loads first (no orders yet)

DataManager streams market data

StrategyManager (ML/RL) proposes actions

RiskManager enforces sizing + SL/TP + caps

TradeExecutor routes to Simulation (paper) by default

PortfolioManager tracks equity, open trades

ValidationManager back/forward tests (gates live)

PairManager rotates “hot pairs”; optional sentiment

Per-asset Live toggle can enable live routing (double-confirm)

Safety & Mode Behavior

Boot: always Training (paper); LIVE disabled for all assets

Per-asset Live toggle:

Requires: user double-confirmation + Validation Gates pass + Kill Switch off

Can be enabled separately for: Crypto Spot, Crypto Futures, Forex, Options

Kill Switch (Live only):

Manual or auto on N consecutive live losses (config)

Sets asset(s) to Close-Only (no new live orders); ensure SL/TP on remaining positions; close profitable where possible

Paper keeps running

Stop Trading Options

Close All Now (market/IOC; respects risk)

Keep Open & Monitor (no new live orders; maintain SL/TP)

Configuration
config/config.json (core safety & telemetry)
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

config/assets.json (symbols & microstructure)
{
  "bybit": {
    "spot": { "symbols": ["BTCUSDT","ETHUSDT"], "fees_bps": 10, "min_notional": 10, "tick_size": 0.01, "qty_step": 0.0001 },
    "futures": { "symbols": ["BTCUSDT","ETHUSDT"], "fees_bps": 10, "tick_size": 0.5, "qty_step": 0.001, "leverage_max": 5 }
  },
  "ibkr": {
    "forex": { "symbols": ["EURUSD","GBPUSD"], "fees_bps": 2, "tick_size": 0.00005, "qty_step": 1000 },
    "options": { "underlyings": ["AAPL","SPY"], "fees_per_contract": 0.65 }
  }
}

config/strategies.json (models, indicators, gates)
{
  "global": {
    "validation": { "min_trades": 500, "min_sharpe": 2.0, "max_drawdown_pct": 15.0 }
  },
  "rl": {
    "engine": "dqn",  // "dqn" | "ppo"
    "window": 128,
    "epsilon": { "start": 1.0, "end": 0.05, "decay_steps": 100000 },
    "reward": { "fee_model": "after_fees", "dd_penalty": 0.5, "tp_bonus": 0.2 }
  },
  "indicators": {
    "ema": { "short": [5, 8, 13], "long": [21, 50] },
    "rsi": { "period": 14, "level": 50 },
    "atr": { "period": 14, "tp_mult": [2.0, 2.5], "sl_mult": [1.5, 2.0] },
    "mfi": { "period": 14 },
    "williams_r": { "period": 14 },
    "fib": { "lookback_bars": 300 }
  },
  "optimizer": {
    "method": "evolutionary",  // "grid" | "random" | "evolutionary"
    "population": 20, "generations": 15, "resume": true
  }
}

Telegram Events (examples)

Lifecycle:

🚀 Start: UI loaded. Mode = TRAINING. Live disabled for all assets.

🛑 Stop: Bot stopped.

Hourly Paper Recap (per config):

⏱️ 60m Paper Recap — P/L %: 1.23 | Fees: 3.21 | Pure: 120.55 | Points: 87 | Trades: 42

Paper Intents (dry-run):

🧪 Paper Intent: BUY BTCUSDT ~ $100.00 (dry-run)

Live Open/Close:

🟢 Live OPEN [crypto_spot]: BUY BTCUSDT ~ $100.00

🔴 Live CLOSE [forex]: EURUSD | P/L 0.84% | Fees $0.12 | Pure $0.88 | Points 0.88

Mode/Kill/Validation:

🔓 Live ENABLED for crypto_futures (SL ≤ 15%).

🛡️ Kill Switch: crypto_spot → CLOSE-ONLY.

✅ Validation Passed: strategy set allowed for live. / ❌ Validation Failed: gate not met.

Getting Started
git clone https://github.com/DolceVitaUkr/Trading-bot-Python.git
cd Trading-bot-Python
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Create configs under tradingbot/config/ (see samples above)
python -m tradingbot.run_bot
# Open the Dashboard URL printed in console (UI loads first; paper mode only)


Module Names
Use lowercase tradingbot:

from tradingbot.brokers import exchangebybit, exchangeibkr

Validation Gates (default)

Trades: ≥ 500 (paper)

Sharpe: ≥ 2.0

Max Drawdown: ≤ 15%

Result: Only validated strategies can be toggled to Live (per asset)

Stop / Exit Semantics

Disable Live (asset): no live routing; paper unaffected

Kill Switch (asset/global): Close-Only for live; ensure SL/TP; close profitable where possible

Graceful Exit: cancels OCOs, parks SL/TP, logs action

Persistence & Restart

Restores models, replay buffers, pair universe, paper equity, runtime state (per-asset live flags, kill switch)

Reconciles open live positions from brokers and resumes management

Notes

Indicators/optimizer sets are examples; tune in strategies.json.

Add sentiment sources when ready; PairManager will apply weighted boosts.

All paths are pathlib-safe (Windows/macOS/Linux).
