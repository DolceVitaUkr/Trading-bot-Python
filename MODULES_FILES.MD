Modules.md
Overview
This project is an AI-driven simulation-first trading framework that learns from live Bybit market data (5m & 15m), executes paper trades only, tracks wallet/virtual wallet performance, and enforces centralized risk controls. It’s designed to be exchange-agnostic later (Forex supported in roadmap), but optimized for Bybit spot right now.

Module Map
Module	Responsibility	Key Public APIs / Notes
config.py	Central configuration + env vars	Sets environment (simulation vs production), Bybit keys (testnet/live), fee/slippage, paths, UI intervals, risk caps, telemetry.
modules/exchange.py	Thin CCXT adapter + built-in paper engine	get_price(), get_balance(), get_min_cost(), get_amount_precision(), get_price_precision(), _simulate_order(), create_order() (live), fetch_positions() (stub for spot).
modules/trade_executor.py	Normalized order execution	Accepts simulation_mode=True/False. In sim: routes to ExchangeAPI._simulate_order() and applies fees. Emits trade events.
modules/trade_calculator.py	P&L math	calculate_trade_result() (basic), AdvancedTradeCalculator (leverage, interest, slippage, points).
modules/risk_management.py	Centralized risk controls	Position sizing with caps, SL/TP calc (ATR & R:R), trailing stops, equity/drawdown gate, regime-aware tweaks.
modules/self_learning.py	DQN agent (training loop)	Uses TradeExecutor (sim), DataManager states, epsilon-greedy actions, replay buffer, target net soft updates.
modules/data_manager.py	Market data fetch/cache/append	REST backfill of ≤900 bars (5m/15m), incremental updates, file rotation, top-pairs refresh hourly. (Updated to use Bybit + CCXT; WS coming next batch)
modules/technical_indicators.py	Indicators (list + pandas API)	SMA/EMA/RSI/ATR/ADX/CCI/Bollinger/etc.; pandas-aware wrapper for strategy features.
modules/reward_system.py	Points and reward shaping	Converts profit/time/SL events to reward points.
modules/exchange_scanner.py	(Planned) Top pairs selection	Hourly refresh, filters by liquidity/vol/volatility.
modules/rollout_manager.py	Safety gates (staging)	Centralizes “simulation-only” vs “live” permissions + notes.
modules/trade_simulator.py	Simple backtester on OHLCV	Replays historical candles through TradeExecutor(simulation_mode=True).
modules/ui.py	Tkinter UI	Status, logs, 3 performance charts (Wallet, Virtual Wallet, Reward Points), color-feedback buttons, live metric panel.
modules/error_handler.py	Error taxonomy & logging	RiskViolationError, OrderExecutionError, APIError + logger helpers.
modules/telegram_bot.py	Notifications	Trade events and alerts to Telegram (optional).
utils/utilities.py	Utilities	Robust logging config, JSON I/O, retry() decorator, time helpers, small helpers.
main.py	Entry point	Ties together UI, DataManager, SelfLearningBot, scheduling, and simulation loop.

Data Flow & Timing
Top Pairs (hourly): determine which spot symbols to watch (e.g., top volume on Bybit). Default seed is BTC/USDT; list expands as data accrues.

Backfill (on first touch per symbol & timeframe): fetch ≤900 bars (5m & 15m) to stay well under Bybit limits.

Incremental update: on each tick, get only new bars (not the whole history), append to disk.

Training/Acting: DQN consumes features computed from cached OHLCV; actions routed to TradeExecutor(simulation_mode=True).

Stop Management: trailing & breakeven logic in RiskManager.dynamic_stop_management(); polled minute-level for open positions (configurable).

UI: live updates of wallet, virtual wallet (paper), and reward points.

Storage Layout
markdown
Copy
Edit
historical_data/
  bybit_spot/
    5m/
      BTCUSDT.csv
      ETHUSDT.csv
      ...
    15m/
      BTCUSDT.csv
      ETHUSDT.csv
      ...
Append-only, with rotation safeguards (no 1GB pulls).

New symbols create new files automatically.

Gaps handled by backfill on reconnect.

Simulation vs Live
Default: ENVIRONMENT=simulation (no real orders).

Live path is gated by RolloutManager and remains off until explicitly enabled; even then, strict risk caps apply.

UI Guide (Key Widgets)
Controls: Start/Stop Training & Trading (buttons change color when active/recently pressed).

Charts (three):

Wallet Balance (real if live; in sim mirrors paper wallet).

Virtual Wallet Balance (paper) — main focus in simulation.

Reward Points — should broadly track ROI trend (used to detect “withholding wins” behavior).

Metrics Panel: Balance, Portfolio Value, Current Symbol, Timeframe.

Logs: color-coded INFO/WARN/ERROR/SUCCESS.

Known Limits & Notes
Bybit spot positions API is minimal; perp support is planned (enables richer live position insight).

TensorFlow isn’t required; PyTorch (CPU) is fine.

Windows long paths may need enabling if you install very large packages (see README Troubleshooting).

Roadmap (Short)
Websocket live feed (Bybit public) with REST backfill.

Perp support + better live position tracking.

Strategy: 15m setup detection, 5m entry refinement.

Forex adapter (Oanda/FXCM/CCXT-proxy).

Offloading large historical files to object storage (S3/Backblaze B2/Azure Blob), optional SharePoint sync.
