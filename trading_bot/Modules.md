# Modules Documentation

This document describes the architecture and role of each module in the **Trading Bot** project.
All modules follow a consistent naming convention and interact through well-defined interfaces.

---

## Core Managers

### configmanager
- **File:** `trading_bot/core/configmanager.py`
- **Purpose:** Centralized configuration and environment manager.
- **Responsibilities:**
  - Load API keys, secrets, and global settings (from `.env` or `config/*.json`).
  - Provide per-asset configurations (Crypto, Futures, Forex, Options).
  - Store thresholds: risk %, SL/TP defaults, kill-switch triggers.
- **Dependencies:** Used by all modules for configuration values.

### datamanager
- **File:** `trading_bot/core/datamanager.py`
- **Purpose:** Unified access layer for historical and live market data.
- **Responsibilities:**
  - Fetch OHLCV and tick data from `Exchange_Bybit` and `Exchange_IBKR`.
  - Normalize data into a unified candle/tick schema.
  - Save and load historical data for backtests (CSV/JSON).
- **Dependencies:** Broker adapters (`Exchange_Bybit`, `Exchange_IBKR`).

### strategymanager
- **File:** `trading_bot/core/strategymanager.py`
- **Purpose:** Central strategy decision hub.
- **Responsibilities:**
  - Load and run strategies depending on flags.
  - Pass signals to `tradeexecutor`.
- **Dependencies:** Uses trained models (`Train_RL_Model`, `Train_ML_Model`) and configs.

### riskmanager
- **File:** `trading_bot/core/riskmanager.py`
- **Purpose:** Enforces global risk controls.
- **Responsibilities:**
  - Position sizing (based on equity and % risk).
  - Enforce max SL = 15% (never increase SL further into loss).
  - Kill-switch logic (pause trading after consecutive losses).
  - Trailing stop & TP management.
- **Dependencies:** `portfoliomanager`, `tradecalculator`.

### tradeexecutor
- **File:** `trading_bot/core/tradeexecutor.py`
- **Purpose:** Execution layer for trade orders.
- **Responsibilities:**
  - Translate trade signals into broker API orders.
  - Attach SL/TP immediately on order placement.
  - Monitor open positions and manage graceful exit.
- **Dependencies:** `Exchange_Bybit`, `Exchange_IBKR`, `riskmanager`.

### portfoliomanager
- **File:** `trading_bot/core/portfoliomanager.py`
- **Purpose:** Track live and paper portfolio state.
- **Responsibilities:**
  - Store open trades, balance, margin usage.
  - Reconcile with broker balances periodically.
  - Provide equity curve and stats to `UI_Manager`.
- **Dependencies:** Data from `tradeexecutor` and broker adapters.

### validationmanager
- **File:** `trading_bot/core/validationmanager.py`
- **Purpose:** Gatekeeper for strategy quality.
- **Responsibilities:**
  - Backtest strategies using historical data.
  - Forward-walk validate strategies on paper trading.
  - Only promote strategies that meet thresholds (Sharpe, Win %, Max Drawdown).
- **Dependencies:** `datamanager`, `strategymanager`.

### notificationmanager
- **File:** `trading_bot/core/notificationmanager.py`
- **Purpose:** Event and alert system.
- **Responsibilities:**
  - Push key events to Telegram or log channels.
  - Configurable verbosity (quiet/normal/verbose).
- **Dependencies:** `configmanager` for API tokens.

### branchmanager
- **File:** `trading_bot/core/branchmanager.py`
- **Purpose:** Manages the lifecycle of concurrent trading processes (branches).
- **Responsibilities:**
  - Initialize and manage broker connections (`IBKRConnectionManager`).
  - Create, start, and stop `Branch` processes for each enabled product.
  - Aggregate telemetry data from all branches.

---

## Core Components

### schemas
- **File:** `trading_bot/core/schemas.py`
- **Purpose:** Defines the core data structures and contracts for the application using Pydantic models.
- **Responsibilities:**
  - Provides standardized objects for `Order`, `Position`, `PortfolioState`, etc.
  - Ensures data consistency and validation between different modules.

### interfaces
- **File:** `trading_bot/core/interfaces.py`
- **Purpose:** Defines the abstract interfaces for key components using Python's `Protocol`.
- **Responsibilities:**
  - Specifies the methods required for components like `MarketData`, `Execution`, and `WalletSync`.
  - Enables loose coupling and dependency inversion.

### null_adapters
- **File:** `trading_bot/core/null_adapters.py`
- **Purpose:** Provides "do-nothing" placeholder implementations of the core interfaces.
- **Responsibilities:**
  - Acts as a default or fallback when a real component is not configured.
  - Useful for testing or running the bot in a state with partial components.

### utilities
- **File:** `trading_bot/core/utilities.py`
- **Purpose:** A collection of helper functions and decorators used across the application.
- **Responsibilities:**
  - Provides common functionalities like filesystem operations (`ensure_directory`, `write_json`), time helpers (`utc_now`), and a `@retry` decorator.

---

## Learning Modules

### Train_RL_Model
- **Purpose:** Train reinforcement learning agents.
- **Responsibilities:**
  - Use historical data + simulated trading environment.
  - Apply reward system.
- **Dependencies:** `datamanager`, `validationmanager`.

### Train_ML_Model
- **Purpose:** Train machine learning models for predictive signals.
- **Responsibilities:**
  - Feature engineering (indicators, market features).
  - Train ML classifiers/regressors for signal prediction.
- **Dependencies:** `datamanager`.

### Save_AI_Update
- **Purpose:** Manage persistence of trained models.
- **Responsibilities:**
  - Save model checkpoints to `models/`.
  - Update strategies with newly trained versions.

---

## Calculation Modules

### tradecalculator
- **File:** `trading_bot/core/tradecalculator.py`
- **Purpose:** Financial and technical calculations.
- **Responsibilities:**
  - Position sizing math.
  - P&L calculations.
  - Technical indicators (EMA, RSI, ATR, Fibonacci).
- **Dependencies:** Used by `strategymanager`, `riskmanager`.

---

## Exchange Adapters

### Exchange_Bybit
- **Purpose:** Adapter for Bybit API (Crypto + Futures).
- **Responsibilities:**
  - REST/WebSocket connections.
  - Place/modify/cancel orders.
  - Fetch account balances and positions.
- **Dependencies:** `configmanager`.

### Exchange_IBKR
- **Purpose:** Adapter for Interactive Brokers API (Forex + Options).
- **Responsibilities:**
  - Connect to IBKR Gateway/TWS.
  - Define contracts (Forex pairs, Options).
  - Place trades and fetch account data.
- **Dependencies:** `configmanager`.

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
- **Style:** Clean, minimal.

---
