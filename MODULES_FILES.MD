Modules Overview
This document provides a detailed breakdown of each module in the trading bot project.
It reflects the current architecture after August 2025 updates, keeping old goals while adding new features.

1. config.py
Purpose: Central configuration file for all modules.
Key Features:

API keys (Bybit, Telegram) — simulation & live modes.

Global bot parameters (loop intervals, trade size, fee model).

Risk management settings (per-pair, portfolio exposure caps).

Optimization parameters for ML/RL.

Paths for saving historical data & logs.

Supports .env overrides.

2. main.py
Purpose: Entry point for starting the bot.
Key Features:

Initializes UI & core modules.

Reads config to determine simulation or live mode.

Launches bot loop or training loop.

Handles button state changes (color change after click).

3. scheduler.py
Purpose: Handles scheduled tasks & recurring updates.
Key Features:

Triggers top-pairs refresh every 60 min.

Schedules reward and statistics updates.

Can run background backtests or RL sessions.

4. data_manager.py
Purpose: Fetches & stores market data.
Key Features:

Gets live klines from Bybit WebSocket & REST API.

Appends new bars to existing CSVs without redownloading full history.

Avoids exceeding Bybit’s 1000-bar limit (keeps under 900 per request).

Supports only 5m and 15m intervals (for setup & entry logic).

Can use locally stored historical datasets for training.

Future-proofed for Forex/Options connectors.

5. exchange.py
Purpose: Unified interface for Bybit (spot, perp) and future exchanges.
Key Features:

Places orders (market, limit) with TP/SL.

Fetches tickers, order book, positions.

Handles testnet/live domain switching.

Graceful fallback if symbol not in testnet.

Can be extended for Binance, IBKR, Forex.

6. trade_executor.py
Purpose: Executes trades in live or simulation mode.
Key Features:

Simulation mode keeps virtual balance & logs PnL without touching wallet.

Live mode sends orders via exchange module.

Avoids opening new positions on same symbol until closed.

TP/SL monitoring cadence:

Positions not re-entered for 60 min unless closed.

Open positions checked every 1–5 min.

Updates reward points on close.

7. trade_simulator.py
Purpose: Fee/slippage-aware backtesting engine.
Key Features:

Runs decision lists against historical bars.

Applies config-defined fee & slippage models.

Returns equity curve & final balance.

Designed for RL/ML model validation.

8. technical_indicators.py
Purpose: Indicator generation for strategies.
Key Features:

EMA/SMA, RSI, ATR, Bollinger Bands, Fibonacci levels.

Generates buy/sell signals for given timeframes.

Can adapt indicator thresholds dynamically.

Output structured for both ML/RL models & rule-based strategies.

9. risk_management.py
Purpose: Prevents overexposure & enforces rules.
Key Features:

Per-pair and portfolio exposure limits.

Loss streak cooldowns.

Position sizing from config (percentage of balance).

Forces close on KPI violations.

10. reward_system.py
Purpose: Tracks reward points for RL.
Key Features:

Points for profit % ranges.

Penalties for losses.

Bonus for high-profit trades.

Syncs with statistics & UI.

11. top_pairs.py
Purpose: Maintains dynamic list of most active pairs.
Key Features:

Pulls top pairs by volume from Bybit.

Refreshes every 60 minutes.

Removes symbols not supported in testnet when in simulation mode.

Limits returned pairs based on config.

12. self_learning.py
Purpose: RL/ML-driven strategy orchestrator.
Key Features:

Uses 15m candles to detect setups, 5m candles for entries.

Avoids symbols with open positions.

Uses top pairs list for diversification.

Refreshes top pairs list every 60 minutes.

Executes trades via TradeExecutor.

Runs entirely in simulation mode if configured.

13. telegram_bot.py
Purpose: Sends notifications to Telegram.
Key Features:

Alerts on trade entries/exits, errors, KPI summaries.

Heartbeat messages every X minutes.

Supports async sending for low-latency notifications.

Configurable verbosity (quiet, normal, verbose).

14. ui.py
Purpose: A Textual-based UI for the Trading Bot.
Key Features:

Buttons for starting/stopping training and trading.
Live-updating status bar with connection, mode, and heartbeat.
Log view for real-time monitoring.
Sparkline charts for wallet balance, portfolio value, and reward points.

15. forex/
forex_exchange.py — API interface for Forex brokers (future IBKR integration).
forex_strategy.py — Forex-specific strategies.

16. options/
options_exchange.py — Interface for options trading.

17. telemetry/
metrics_exporter.py — Exports metrics to files or dashboard.
report_generator.py — Generates KPI and strategy performance reports.

18. state/
runtime_state.py — Saves/restores bot state between runs.
position_reconciler.py — Syncs open trades with exchange to ensure accuracy.

19. rollout_manager.py
Purpose: Manages staged rollout from simulation → full live trading.
Stages:
- Data only.
- Simulated trades.
- Canary small trades.
- Scaled positions.
- Full deployment.

20. modules/parameter_optimization.py
Purpose: Model optimization & parameter tuning.
Key Features:
Supports grid search, random search, and evolutionary algorithms.
Works with RL/ML modules to improve strategies.
