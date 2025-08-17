# MODULES_FILES.MD
_Last updated: 2025-08-17_

This document outlines the purpose and high-level API of each core module in the trading bot.

## Logging & Utilities
### `modules/Logger_Config.py`
- **What:** Configures the global structured logger using `structlog`. Outputs colored, human-readable logs to the console and structured JSONL logs to `logs/structured_logs.jsonl`.
- **API:** `setup_logging()` is called at startup. `get_logger(name)` provides a logger instance that supports context binding (e.g., `log.bind(product="FOREX_SPOT")`).

### `utils/rate_limiter.py`
- **What:** Provides a `RateLimiter` class to manage API call frequency, respecting broker-specific limits (e.g., requests per second, historical data pacing).
- **API:** A global `ibkr_rate_limiter` instance is created. Methods like `wait_for_historical_slot()` are awaited before making API calls.

## IBKR Broker Modules (`modules/brokers/ibkr/`)
This suite of modules provides an institutional-grade integration with Interactive Brokers.

### `Connect_IBKR_API.py`
- **What:** Manages connections to both the TWS/Gateway socket API and the Client Portal Web API.
- **API:** `IBKRConnectionManager` class provides `connect_tws()`, `connect_web_api()`, and methods to get connected clients. Handles connection validation (paper vs. live) and safety checks.

### `Contracts_IBKR.py`
- **What:** Helper functions to build canonical IBKR contract objects.
- **API:** `build_fx_spot_contract()`, `build_fx_option_contract()`, and `get_conid_by_symbol()` (which uses the Web API to find contract IDs).

### `Fetch_IBKR_Account.py`
- **What:** Fetches read-only account data, including balances, buying power, and portfolio positions.
- **API:** `IBKRAccountFetcher` class with `get_account_summary()` and `get_positions()`.

### `Fetch_IBKR_MarketData.py`
- **What:** Fetches market data with integrated rate limiting and caching. Performs a pre-flight check for market data subscriptions.
- **API:** `IBKRMarketDataFetcher` class with `fetch_historical_bars()`, `fetch_option_chain()`, and `fetch_greeks_for_contract()`.

### `Place_IBKR_Order.py`
- **What:** Handles order placement with critical safety guards.
- **API:** `IBKROrderPlacer` class. All order methods (`place_market_order`, etc.) will raise a `TrainingOnlyError` if `TRAINING_MODE` is True in the config. Also includes a policy guard to prevent fund transfers.

## Training & Storage Modules
### `modules/training/Train_Forex_Spot.py`
- **What:** A training pipeline for Forex spot strategies. Consumes data from the IBKR fetchers, performs feature engineering, and simulates a strategy to generate metrics.
- **API:** `ForexSpotTrainer.train_strategy()`.

### `modules/training/Train_Forex_Options.py`
- **What:** A training/analysis pipeline for Forex options. Fetches option chains and greeks to perform analysis (e.g., on ATM options).
- **API:** `ForexOptionsTrainer.analyze_atm_greeks()`.

### `modules/storage/Save_AI_Update.py`
- **What:** Saves and loads trained model artifacts.
- **API:** `save_model_artifact(product_name, ...)` and `load_model_artifact(product_name, ...)` functions that persist objects to product-specific directories (e.g., `models/forex_spot/`).

## Core Managers
### `managers/Validation_Manager.py`
- **What:** Handles the backtesting and approval/rejection of trading strategies.
- **Update:** The validation logic is now product-centric.
- **API:** `approved(strategy_id, product)` checks if a strategy is approved for a given product line (e.g., `FOREX_SPOT`). It enforces a minimum trade count (e.g., >=500) per product before approval.

### `managers/Sizer.py`
- **What:** Calculates trade size based on risk parameters and portfolio equity.
- **Notes:** No major changes, but now consumes equity data from different products/brokers.

### `managers/Kill_Switch.py`
- **What:** A circuit breaker that monitors portfolio-level risk.
- **Notes:** The scope of the kill switch can be applied per product.
