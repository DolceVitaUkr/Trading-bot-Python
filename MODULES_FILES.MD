# Modules Documentation

This document describes the architecture and role of each module in the **Trading Bot** project.  
All modules follow a consistent naming convention and interact through well-defined interfaces.  

---

## Core Managers

### Config_Manager
- **Purpose:** Centralized configuration and environment manager.
- **Responsibilities:**
  - Load API keys, secrets, and global settings (from `.env` or `config/*.json`).
  - Provide per-asset configurations (Crypto, Futures, Forex, Options).
  - Store thresholds: risk %, SL/TP defaults, kill-switch triggers.
- **Dependencies:** Used by all modules for configuration values.

### Data_Manager
- **Purpose:** Unified access layer for historical and live market data.
- **Responsibilities:**
  - Fetch OHLCV and tick data from `Exchange_Bybit` and `Exchange_IBKR`.
  - Normalize data into a unified candle/tick schema.
  - Save and load historical data for backtests (CSV/JSON).
- **Dependencies:** Broker adapters (`Exchange_Bybit`, `Exchange_IBKR`).

### Strategy_Manager
- **Purpose:** Central strategy decision hub.
- **Responsibilities:**
  - Load and run strategies depending on flags:
    - **Asset class:** Crypto, Crypto Futures, Forex, Forex Options.
    - **Market condition:** Trending, ranging, volatile, stable.
    - **Wallet balance:** Enable/disable depending on account size.
    - **Time/day filters.**
  - Pass signals to `Trade_Executor`.
- **Dependencies:** Uses trained models (`Train_RL_Model`, `Train_ML_Model`) and configs.

### Risk_Manager
- **Purpose:** Enforces global risk controls.
- **Responsibilities:**
  - Position sizing (based on equity and % risk).
  - Enforce max SL = 15% (never increase SL further into loss).
  - Kill-switch logic (pause trading after consecutive losses).
  - Trailing stop & TP management.
- **Dependencies:** `Portfolio_Manager`, `Trade_Calculator`.

### Trade_Executor
- **Purpose:** Execution layer for trade orders.
- **Responsibilities:**
  - Translate trade signals into broker API orders.
  - Attach SL/TP immediately on order placement.
  - Monitor open positions and manage graceful exit.
- **Dependencies:** `Exchange_Bybit`, `Exchange_IBKR`, `Risk_Manager`.

### Portfolio_Manager
- **Purpose:** Track live and paper portfolio state.
- **Responsibilities:**
  - Store open trades, balance, margin usage.
  - Reconcile with broker balances periodically.
  - Provide equity curve and stats to `UI_Manager`.
- **Dependencies:** Data from `Trade_Executor` and broker adapters.

### Validation_Manager
- **Purpose:** Gatekeeper for strategy quality.
- **Responsibilities:**
  - Backtest strategies using historical data.
  - Forward-walk validate strategies on paper trading.
  - Only promote strategies that meet thresholds (Sharpe, Win %, Max Drawdown).
- **Dependencies:** `Data_Manager`, `Strategy_Manager`.

### Notification_Manager
- **Purpose:** Event and alert system.
- **Responsibilities:**
  - Push key events to Telegram or log channels.
  - Configurable verbosity (quiet/normal/verbose).
- **Dependencies:** `Config_Manager` for API tokens.

---

## Learning Modules

### Train_RL_Model
- **Purpose:** Train reinforcement learning agents.
- **Responsibilities:**
  - Use historical data + simulated trading environment.
  - Apply reward system:
    - +points for profit after fees.
    - -points for losses.
    - Reset environment on equity < $10.
- **Dependencies:** `Data_Manager`, `Validation_Manager`.

### Train_ML_Model
- **Purpose:** Train machine learning models for predictive signals.
- **Responsibilities:**
  - Feature engineering (indicators, market features).
  - Train ML classifiers/regressors for signal prediction.
- **Dependencies:** `Data_Manager`.

### Save_AI_Update
- **Purpose:** Manage persistence of trained models.
- **Responsibilities:**
  - Save model checkpoints to `models/`.
  - Update strategies with newly trained versions.

---

## Calculation Modules

### Trade_Calculator
- **Purpose:** Financial and technical calculations.
- **Responsibilities:**
  - Position sizing math.
  - P&L calculations.
  - Technical indicators (EMA, RSI, ATR, Fibonacci).
- **Dependencies:** Used by `Strategy_Manager`, `Risk_Manager`.

---

## Exchange Adapters

### Exchange_Bybit
- **Purpose:** Adapter for Bybit API (Crypto + Futures).
- **Responsibilities:**
  - REST/WebSocket connections.
  - Place/modify/cancel orders.
  - Fetch account balances and positions.
- **Dependencies:** `Config_Manager`.

### Exchange_IBKR
- **Purpose:** Adapter for Interactive Brokers API (Forex + Options).
- **Responsibilities:**
  - Connect to IBKR Gateway/TWS.
  - Define contracts (Forex pairs, Options).
  - Place trades and fetch account data.
- **Dependencies:** `Config_Manager`.

---

## User Interface

### UI_Manager
- **Purpose:** Unified dashboard for all asset classes.
- **Screens:**
  - Crypto
  - Crypto Futures
  - Forex
  - Forex Options
- **Features:**
  - Toggle On/Off, Live/Paper.
  - Show wallet stats, open positions, P&L%.
  - Kill switch status.
  - Graceful Exit button.
  - Graph of balance/equity curve.
  - Global footer with total equity across all assets.
- **Style:** Clean, minimal (see [Dribbble example](https://cdn.dribbble.com/userupload/10326065/file/original-99f8889a83078eebe74d91c2da49c13a.jpg?resize=1024x768&vertical=center)).

---
