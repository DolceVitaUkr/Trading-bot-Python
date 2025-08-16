# Self-Learning Trading Bot (Multi-Asset, Professional Grade)

Institutional-grade, modular framework for **Crypto Spot**, **Perp/Futures**, **Options**, and **Forex**. This bot features advanced capabilities including **phased scaling**, **strategy validation**, **news sentiment analysis**, and a robust **kill switch** mechanism.

---

## What It Does (Plain English)
- **Validates Strategies**: Before use, strategies are backtested and walk-forward tested by the **Validation Manager** to ensure they meet performance criteria (e.g., >500 trades, Sharpe ratio).
- **Analyzes News**: The **News Agent** fetches market sentiment for assets and pauses trading around high-impact events (CPI, FOMC).
- **Sizes Trades Intelligently**: A **Phased Scaling Plan** adjusts trade sizes based on the live equity of each asset-class sub-ledger.
- **Manages Risk Proactively**:
  - A **Kill Switch** halts trading for an asset class if it breaches drawdown limits (e.g., >5% daily), has excessive slippage, or API errors.
  - **Filters** trades based on liquidity (>$5M daily volume), correlation (max 3 correlated positions), and funding/carry costs.
- **Uses Live Wallet Balances**: Ensures that all sizing and risk calculations are based on real, synced wallet equity, not invented balances.
- **Calculates Net-of-Fees P&L**: All performance tracking, equity progression, and reward calculations are based on profit after all fees and slippage.
- **Sends Telegram** updates and exports **structured JSON logs** for detailed analysis.

---

## Phased Scaling Sizing Logic
The bot uses a sophisticated 4-phase scaling model, applied independently to each asset class sub-ledger (Spot, Forex, etc.).

- **Phase 1 (Equity ≤ $1,000):** Fixed **$10** trades only.
- **Phase 2 ($1,000 < Equity ≤ $5,000):** Trade size is **0.5% to 1.0%** of sub-ledger equity, scaled by signal confidence.
- **Phase 3 ($5,000 < Equity ≤ $20,000):** **Percentage-risk sizing** is enabled, with risk per trade scaling up to a maximum of **5%** of equity for high-confidence signals.
- **Phase 4 (Equity > $20,000):** Advanced percentage-risk sizing, with an additional rule allowing "good setups" to have a total position size of up to **10%** of sub-ledger equity.

---

## Risk & Market Filters
- **Liquidity Filter**: Only trades pairs with ≥ $5M in 24h volume.
- **Correlation Filter**: Prevents opening more than 2 additional positions that are highly correlated with an existing one.
- **Funding/Carry Filter**: Avoids entering positions that would pay excessive funding rates (for perps) or carry costs (for FX).
- **Kill Switch**: Automatically halts *new* trades for an asset class if critical risk limits are breached.
  - Daily drawdown > 5%
  - Monthly drawdown > 15%
  - 3+ catastrophic slippage events in 24h
  - Exchange/API error escalation

---

## Architecture
```
main.py
config.py

modules/
  # Core Logic
  Portfolio_Manager.py   # Manages sub-ledgers, P&L, and asset allocation.
  Sizer.py               # Implements the 4-phase scaling logic.
  Strategy_Manager.py    # Integrates all filters and agents to make trade decisions.
  Risk_Management.py     # Host for Kill Switch handoff and funding filter.

  # New Modules
  Validation_Manager.py  # Backtests and approves/rejects strategies.
  News_Agent.py          # Provides news sentiment and a macro calendar filter.
  Kill_Switch.py         # Monitors for and triggers circuit breaker events.

  # Supporting Modules
  Funds_Controller.py
  Wallet_Sync.py
  Trade_Executor.py
  Logger_Config.py       # Configures structured JSON logging.
  ...

policies/
  sizing_policy.json     # Configuration for sizing parameters.
  ...

logs/
  trading_bot.json       # Structured log output.
```

---

## Testing
- Unit tests for the phased scaling logic in `Sizer.py`.
- Mock tests for the `ValidationManager` and `NewsAgent`.
- A test suite for the `KillSwitch` to ensure it triggers correctly on drawdown and slippage events.
- Tests for all new filters.
