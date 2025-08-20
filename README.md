# Trading Bot – Multi-Asset Unified Architecture

## Overview
Self-learning trading bot supporting multiple asset classes:
- Crypto via Bybit
- Forex & options via Interactive Brokers (IBKR)

Machine Learning (ML) and Reinforcement Learning (RL) help optimise strategies while
risk management enforces strict position sizing and stop-loss rules. A unified web
UI provides control and telemetry.

## Setup
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `tradingbot/config/config.json` (and related JSON files) with your API keys
   and settings. The files live inside the package so relative paths work on all
   platforms.

## Running
Run the bot from the repository root:
```bash
python -m tradingbot.run_bot
```

## Modules
Key modules (all lowercase):
- `tradingbot.brokers.exchangebybit` – Bybit adapter
- `tradingbot.brokers.exchangeibkr` – IBKR adapter
- `tradingbot.core.configmanager` – configuration loader
- `tradingbot.core.riskmanager` – risk controls
- `tradingbot.core.portfoliomanager` – portfolio state
- `tradingbot.core.tradeexecutor` – order execution
- `tradingbot.core.strategymanager` – strategy decisions
- `tradingbot.ui.uimanager` – FastAPI dashboard

See `tradingbot/modules.md` for a detailed description of every module.

## Configuration
`config/config.json` is required and must reside under `tradingbot/config/`.
Paths are handled with `pathlib` for cross-platform compatibility.
*** End of File
