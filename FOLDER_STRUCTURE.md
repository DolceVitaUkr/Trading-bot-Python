# Project Folder Structure

## Main Folders

### `/tradingbot` - Main application code
- `/brokers` - Exchange connectors (Bybit, IBKR)
- `/config` - Configuration files
- `/core` - Core trading logic and managers
- `/learning` - RL models and training
- `/risk` - Risk management and position sizing
- `/rl` - Reinforcement learning components
- `/state` - Persistent state files
- `/ui` - Web dashboard (Flask/FastAPI)
- `/validation` - Strategy validation

### `/docs` - Documentation
- `/setup` - Setup and configuration guides
  - `IBKR_SETUP_GUIDE.md`
  - `CONNECTION_STATUS.md`
  - `PAPER_LIVE_SETUP.md`
- `/development` - Development documentation

### `/tests` - Test files
- `test_all_connections.py` - Connection diagnostics
- `test_ui_and_ibkr.py` - UI testing
- Other test scripts

### `/scripts` - Utility scripts
- Playwright scripts
- Helper utilities
- Batch files

### `/screenshots` - UI screenshots
- Test screenshots
- Dashboard captures

### `/artifacts` - Generated files
- HTML outputs
- Temporary files

### `/results` - Trading results (created at runtime)
- Strategy backtests
- Paper trading results

## Main Entry Points

### `start_bot.py`
Main launcher with two modes:
- `python start_bot.py` - Start web dashboard
- `python start_bot.py validate ...` - Run validation pipeline

### `main.py`
Alternative entry point

### `start_dashboard.py`  
Direct dashboard launcher

## Configuration

### `/tradingbot/config/config.json`
Main configuration file with:
- API keys
- Bot settings
- Safety parameters
- KPI targets

## Quick Commands

Start the bot:
```bash
python start_bot.py
```

Test connections:
```bash
python tests/test_all_connections.py
```

Run UI tests:
```bash
python tests/test_ui_and_ibkr.py
```