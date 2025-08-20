📘 Modules.md
Project Structure
tradingbot/
  core/
  brokers/
  learning/
  ui/
  config/
  logs/
  state/

Core Modules
core/config_manager.py

Role: Loads & validates configuration (config/config.json, assets.json, strategies.json)

Functions:

load_config() – load JSON config

save_config() – persist updates

get_param(key) – access nested settings

core/datamanager.py

Role: Data lake & preprocessing

Functions:

fetch_historical(symbol, tf) – get OHLCV

save_parquet(df, meta) – save with metadata

multi_tf_join(dfs) – join 5m/15m/1h windows

detect_gaps() – find/fill missing bars

core/indicators.py

Role: Apply technical indicators

Functions:

apply_indicators(df, spec) – SMA/EMA/RSI/Stoch/MFI/OBV/ATR/Fib

Includes advanced set: HMA, KAMA, SuperTrend, Donchian, Keltner, Bollinger %B, VWAP, optional Ichimoku

core/pair_manager.py

Role: Dynamic symbol selection & regime tagging

Functions:

rank_pairs(vol, momentum, liquidity)

tag_regimes(df) – volatility/trend/liquidity regime detection

Optional: HMM regime detection

apply_sentiment_hook() – integrate external feeds

core/strategy_manager.py

Role: Stores & applies strategies (ML, RL, rule-based)

Functions:

load_strategy(name)

apply_strategy(df, model)

evaluate_signals(df)

core/reward_system.py

Role: Reinforcement learning reward shaping

Logic: PnL_after_fees – λ_dd*drawdown – λ_turn*turnover – λ_tail*CVaR_tail + bonus_TP

core/risk_manager.py

Role: Position sizing, exposure, safety rules

Logic:

Learning phase: $10 notional until equity ≥ $1,000

Growth phase: equity-tier % (0.5–2%), −0.25% risk per 5% DD, floor 0.25%

Signal weighting: weak=0.5×, strong=1.0×, very_strong=1.5×

Hard SL ≤ 15%, daily loss caps, exposure limits

core/trade_executor.py

Role: Route orders (simulation vs live)

Features:

Slippage model (spread×vol×size), partial fills, latency jitter

Close-only mode (kill switch)

Reconcile open positions on restart

core/portfolio_manager.py

Role: Tracks balances, equity, open trades, P&L

Functions:

update_balance()

get_equity()

report_positions()

core/validation_manager.py

Role: Backtest & validation gates

Functions:

Purged & embargoed walk-forward CV

Monte Carlo trade bootstraps

Stress-test slippage

Enforces: ≥500 trades, Sharpe ≥ 2.0, Max DD ≤ 15%

core/optimizer.py

Role: Hyperparameter tuning

Methods: grid search, random search, evolutionary algorithms

Targets: indicator params, RL hyperparams, sizing thresholds

core/notifier.py

Role: Telegram integration

Events: start, stop, errors, hourly paper recap, paper intents, live opens/closes, validation pass/fail, kill-switch

core/error_handler.py

Role: Exception handling & logging

Functions:

handle_error() – logs + Telegram alert

recover_state() – safe fallback

core/runtime_controller.py

Role: Orchestrates runtime states

Features:

Per-asset live toggles (crypto spot/futures, forex, options)

Kill switch (manual & auto on consecutive losses)

Hourly recap scheduling

Persists runtime state in state/runtime.json

Brokers
brokers/exchange_bybit.py

Role: Bybit Spot & Futures adapter

Functions:

fetch_ohlcv()

place_order()

cancel_order()

Fee model, precision, min notional checks

brokers/exchange_ibkr.py

Role: IBKR Forex & Options adapter

Functions:

fetch_forex_data()

place_fx_order()

place_option_order()

Compliance with IBKR rules

Learning
learning/train_ml_model.py

Role: ML strategy training

Features:

Triple-Barrier labeling, Meta-labeling

Feature pipeline (lags, rolling stats, PCA/AE opt.)

Purged walk-forward CV

SHAP pruning

learning/train_rl_model.py

Role: Reinforcement learning training

Features:

Double+Dueling DQN + Prioritized Replay + n-step

PPO+GAE path

LSTM/Transformer encoders

Composite reward

learning/save_ai_update.py

Role: Persist and version AI models

Features:

Save weights, replay buffers, validation reports

Register in MLflow/W&B

UI
ui/app.py

Role: FastAPI dashboard entrypoint

Features: 4 panels (crypto spot/futures, forex, options), WebSocket metrics, control endpoints

ui/routes/diff.py

Role: Dry-run preview

Function: show paper vs would-be live orders, require confirmation

ui/routes/validation.py

Role: Show validation reports before enabling live

Config

config/config.json – runtime, Telegram, kill switch, equity rules

config/assets.json – symbols, fees, tick sizes, min notional

config/strategies.json – indicator params, RL/ML settings, sizing, validation

Logs

logs/ contains JSON/CSV logs of: trades, intents, decisions, validation, errors, hourly recaps

State

state/ persists:

runtime.json – toggles, kill state, session info

models/ – ML/RL saved weights

replay_buffers/ – RL experiences

checkpoints/ – training snapshots

caches/ – recent data

Conventions

File names: lowercase with underscores

Header in every file:
# file: core/datamanager.py
