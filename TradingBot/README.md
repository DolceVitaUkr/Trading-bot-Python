# Trading Bot – Multi-Asset Unified Architecture

## Overview
This project is a **self-learning, multi-asset trading bot** that supports:
- **Crypto (Bybit)** – Spot & Futures
- **Forex & Options (IBKR)** – via Interactive Brokers API

The bot uses **Machine Learning (ML)** and **Reinforcement Learning (RL)** to optimize strategies, while enforcing strict **risk management**.  
It is controlled through a **unified UI dashboard**.

---

## Features
- **Unified Dashboard:** Control and monitor Crypto, Crypto Futures, Forex, Forex Options in one UI.
- **Paper & Live Mode:** Run in simulation with real market data for training, or switch to live with proven strategies while running paper simulations to test and develop new strategies
- **Adaptive Strategies:** Strategies flagged by:
  - Asset type (Crypto, Futures, Forex, Options)
  - Market conditions (trend, range, volatility)
  - Wallet size (e.g. scalping disabled under $5,000)
  - Time-of-day/week filters
- **Risk Management:**
  - Mandatory SL & TP for every trade.
  - Max stop-loss = **15%** (industry recommended upper bound  ).
  - Trailing stops to lock in profits.
  - Kill-switch after consecutive losses.
  - Graceful Exit button for safe shutdown.
- **Learning Engine:**
  - Starts paper trading with $1,000 virtual equity.
  - Resets if equity < $10.
  - Rewards based on profit after fees.
  - Backtests + forward validation before live use.
- **Logging & Transparency:**
  - JSON/CSV logs for every decision (trade taken or skipped).
  - Historical backtest data stored per asset class.

---

## Architecture
core/ - Config, Data, Strategy, Risk, Execution, Validation
brokers/ - ExchangeBybit, ExchangeIBKR
strategies/ - Scalping, Swing, Momentum, Options
ui/ - Dashboard UI (4 screens + global view)
data/ - Historical data & backtest files
config/ - Config JSONs (global, assets, strategies)
logs/ - JSON/CSV logs of trades, decisions, errors

yaml
Copy
Edit

**Main Flow:**
1. **DataManager** fetches market data (from Bybit or IBKR).
2. **StrategyManager** evaluates strategies (using ML/RL models).
3. **RiskManager** checks position size, SL/TP rules.
4. **TradeExecutor** routes orders to correct broker adapter.
5. **PortfolioManager** tracks equity & open positions.
6. **UIManager** updates real-time dashboard.
7. **ValidationManager** backtests and forward-tests strategies.

---

## Configuration
- **config/config.json** – API keys, global settings.
- **config/assets.json** – Symbols, tick sizes, fees, min balances.
- **config/strategies.json** – Enable/disable strategies, parameters, flags.

---

## UI Dashboard
- 4 Panels: Crypto | Crypto Futures | Forex | Forex Options
- Each shows:
  - On/Off, Live/Paper status
  - Wallet available, used, total
  - Active strategy
  - Open positions with P&L%
  - Balance graph
  - Kill switch status
- **Bottom footer:** Aggregated portfolio value across all assets.
- **Controls:**
  - Kill Switch (auto/manual)
  - Graceful Exit (safe shutdown)

---

## Risk & Stop-Loss Standards
- **Forex:** 0.5%–2% typical stop-loss .
- **Crypto:** 1%–10% for day trades, max 15% for volatile swings  .
- Bot enforces max SL = 15% (configurable).
- Always sets TP; can trail upwards but never widen SL.

---

## Getting Started
1. Clone repo and install requirements:
   ```bash
   git clone https://github.com/DolceVitaUkr/Trading-bot-Python.git
   cd Trading-bot-Python
   pip install -r requirements.txt
Configure API keys in config/config.json.

Run the bot:

bash
Copy
Edit
python RunBot.py
Open the Dashboard (browser or app) to control trading.
