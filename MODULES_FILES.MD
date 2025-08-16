# MODULES_FILES.MD
_Last updated: 2025-08-16_

This document outlines the purpose and high-level API of each core module in the trading bot.

## Logging & Configuration
### Logger_Config.py
- **What:** Configures the global logger to output human-readable logs to the console and structured JSON logs to `logs/trading_bot.json`.
- **API:** `setup_logging()` is called from `utils/utilities.py` at startup.

## New Core Modules (`modules/`)
### Validation_Manager.py
- **What:** Handles the backtesting, walk-forward testing, and approval/rejection of trading strategies. Ensures only profitable and robust strategies are deployed.
- **API:** `is_strategy_approved(strategy_id, asset_class) -> bool`
- **Notes:** Maintains an in-memory registry of approved strategies. Logs reasons for all rejections.

### News_Agent.py
- **What:** Fetches news sentiment for assets and provides a filter for high-impact macroeconomic events.
- **API:** `get_news_bias(symbol) -> str`, `is_high_impact_event_imminent() -> bool`
- **Notes:** Helps avoid trading against strong market sentiment or during volatile news releases.

### Kill_Switch.py
- **What:** A circuit breaker that monitors portfolio-level risk. It can halt new trading activity for specific asset classes if severe limits are breached.
- **API:** `is_active(asset_class) -> bool`, `activate(asset_class, reason)`
- **Triggers:** Daily/monthly drawdown, excessive slippage, API error escalations.

## Updated Core Modules (`modules/`)
### Portfolio_Manager.py
- **What:** Manages capital allocation across sub-ledgers (e.g., SPOT, PERP).
- **Update:** Added `get_total_equity(asset)` method to provide the Sizer with the correct equity base for phased scaling calculations.

### Sizer.py
- **What:** Calculates trade size based on the new 4-phase scaling logic.
- **Update:** The `propose()` method now implements the phased scaling plan, moving from fixed sizes to percentage-of-equity and percentage-of-risk models as equity grows.

### Strategy_Manager.py
- **What:** The central decision-making module.
- **Update:** Integrates multiple new checks into its `decide()` method:
  1.  **Liquidity Filter**: Checks for minimum daily volume.
  2.  **Correlation Filter**: Prevents over-exposure to similar assets.
  3.  **News Filter**: Checks for high-impact events and sentiment bias.
  4.  **Validation Filter**: Ensures the selected strategy is approved by the `ValidationManager`.

### Risk_Management.py
- **What:** Performs final risk validation on trade proposals.
- **Update:** The `allow()` method now integrates:
  1.  **Kill Switch Check**: Rejects trades if the kill switch for the asset class is active.
  2.  **Funding/Carry Filter**: Rejects trades if funding rates or carry costs are prohibitively expensive.

### Funds_Controller.py & Wallet_Sync.py
- No major changes in this update, but they remain critical for providing the live wallet context to the `PortfolioManager` and `Sizer`.
