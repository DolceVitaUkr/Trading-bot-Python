# Trading Bot – Multi-Asset Unified Architecture

## Overview
This project is a **self-learning, multi-asset trading bot** that supports:
- **Crypto (Bybit)** – Spot & Futures
- **Forex & Options (IBKR)** – via Interactive Brokers API

The bot uses **Machine Learning (ML)** and **Reinforcement Learning (RL)** to optimise strategies while enforcing strict **risk management**. It is controlled through a unified web dashboard.

## Features
- **Unified Dashboard:** Control and monitor Crypto, Crypto Futures, Forex and Options in one UI.
- **Paper & Live Mode:** Run in simulation with real market data, or switch to live while continuing paper simulations for new strategies.
- **Adaptive Strategies:** Strategies flagged by asset type, market conditions, wallet size and time-of-day/week filters.
- **Risk Management:** Mandatory SL & TP for every trade, max stop-loss of 15%, trailing stops, kill switch and graceful exit.
- **Learning Engine:** Starts paper trading with $1,000 virtual equity, resets if equity < $10, rewards based on profit after fees, backtests + forward validation before live use.
- **Logging & Transparency:** JSON/CSV logs for every decision and historical backtest data per asset class.

## Architecture
```
core/      - Config, data, strategy, risk, execution, validation
brokers/   - exchangebybit, exchangeibkr adapters
learning/  - trainmlmodel, trainrlmodel, saveaiupdate
ui/        - FastAPI dashboard and routers
config/    - JSON configs (global, assets, strategies)
logs/      - JSON/CSV logs of trades, decisions, errors
```

### Main Flow
1. **DataManager** fetches market data.
2. **StrategyManager** evaluates strategies using ML/RL models.
3. **RiskManager** enforces position sizing and SL/TP rules.
4. **TradeExecutor** routes orders to the correct broker adapter.
5. **PortfolioManager** tracks equity and open positions.
6. **UIManager** updates the real-time dashboard.
7. **ValidationManager** backtests and forward-tests strategies.

See `tradingbot/modules.md` for a detailed description of every module.

## Configuration
`tradingbot/config/config.json` is required for API keys and global settings.
Additional files:
- `config/assets.json` – symbols, tick sizes, fees, minimum balances
- `config/strategies.json` – strategy parameters and flags

All paths use `pathlib` for Windows/macOS/Linux compatibility.

## UI Dashboard
Four panels (Crypto, Crypto Futures, Forex, Forex Options) show status, wallets, active strategy, open positions, balance graph and kill switch state. The footer aggregates total portfolio value. Controls include a kill switch and graceful exit.

## Risk & Stop-Loss Standards
- **Forex:** typical stop-loss 0.5%–2%
- **Crypto:** 1%–10% for day trades, max 15% for volatile swings
- The bot enforces max SL = 15% (configurable) and always sets a TP. SL is never widened, but TP can trail.

## Getting Started
1. Clone repo and install requirements:
   ```bash
   git clone https://github.com/DolceVitaUkr/Trading-bot-Python.git
   cd Trading-bot-Python
   python -m venv .venv && source .venv/bin/activate  # optional
   pip install -r requirements.txt
   ```
2. Add your configuration JSONs under `tradingbot/config/` (at least `config.json`).
3. Run the bot:
   ```bash
   python -m tradingbot.run_bot
   ```
4. Open the dashboard (browser or app) to control trading.

## Module Names
All imports use the lowercase `tradingbot` package name, e.g.:
```python
from tradingbot.brokers import exchangebybit, exchangeibkr
```

---

This README reflects the correct module names, setup instructions and project architecture.
