🐍 Self-Learning AI Trading Bot
A modular Python trading framework designed for crypto, forex, and options with built-in Machine Learning (ML) and Reinforcement Learning (RL) strategy engines.

Supports Bybit, Binance, and future connectors. Built for low-risk, scalable, and automated trading with live and simulation modes.

🚀 Features
Multi-market support: Spot, Perpetuals, Forex, FX Options (FX/Options toggles default OFF at boot)

RL/ML Brain: Self-learning trading logic with online reinforcement updates

Live & Paper Trading: Unified code for both

Risk Management: Position sizing, exposure caps, cooldowns

Technical Indicators: SMA, EMA, RSI, ATR, ADX, Bollinger, Fibonacci, and more

Backtesting Engine: Fee/slippage-aware simulations

Telegram Bot Integration: Notifications, daily/weekly KPIs

State Persistence: Restart-safe with order reconciliation

Configurable Rollout Stages: 1–5 progressive deployment phases

📂 Project Structure
plaintext
Copy
Edit
.
├── main.py                        # Entry point
├── config.py                      # Central configuration
├── modules/                       # Core modules
│   ├── data_manager.py
│   ├── exchange.py
│   ├── trade_executor.py
│   ├── risk_management.py
│   ├── reward_system.py
│   ├── technical_indicators.py
│   ├── top_pairs.py
│   ├── self_learning.py
│   ├── trade_simulator.py
│   ├── telegram_bot.py
│   ├── ui.py
│
├── forex/                         # Forex adapters
│   ├── forex_exchange.py
│   ├── forex_data.py
│   └── forex_strategy.py
│
├── options/                       # Options adapters
│   └── options_exchange.py
│
├── telemetry/                     # Metrics and reports
│   ├── metrics_exporter.py
│   └── report_generator.py
│
├── state/                         # State persistence and reconciliation
│   ├── runtime_state.py
│   └── position_reconciler.py
│
├── modules.md                     # API surface & module contracts
└── README.md
⚙️ Installation
1. Clone repo

bash
Copy
Edit
git clone https://github.com/youruser/self-learning-bot.git
cd self-learning-bot
2. Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
3. Configure API keys & settings
Edit config.py:

python
Copy
Edit
BYBIT_API_KEY = "your_api_key"
BYBIT_API_SECRET = "your_secret"
ENVIRONMENT = "simulation"  # or "production"
▶️ Usage
Simulation Mode

bash
Copy
Edit
python main.py --mode simulation
Live Trading

bash
Copy
Edit
python main.py --mode production
Backtest

bash
Copy
Edit
python -m modules.trade_simulator
🧠 RL / ML Training
RL Online: Runs continuously during trading if enabled via UI or config.

ML Training: Load historical data via DataManager → train supervised models offline.

📡 Notifications
The bot can send:

Trade entries/exits

Risk violations

Daily & weekly KPI summaries

via Telegram — configure in config.py:

python
Copy
Edit
TELEGRAM_BOT_TOKEN = "your_token"
TELEGRAM_CHAT_ID = "your_chat_id"
📊 Backtesting
Example:

python
Copy
Edit
from modules.trade_simulator import TradeSimulator
from modules.trade_executor import TradeDecision
import pandas as pd

bars = pd.read_csv("data/BTCUSDT_15m.csv")
decisions = [...]  # generate or load
sim = TradeSimulator(fee_model=config.FEE_MODEL["bybit"]["spot"])
results = sim.run_backtest(decisions, bars)
print(results)
🛡️ Risk Management
Max exposure per pair & portfolio

Position sizing based on balance & stop loss

Loss streak cooldowns

Force exits on rule violations

📌 Rollout Stages
Data only

Simulated trades

Small live trades (canary)

Scaled positions

Full deployment

📅 Roadmap
 Add Binance connector

 Expand options trading module

 Auto strategy selection per market regime

 Web dashboard for live monitoring

📄 License
MIT License — feel free to fork and modify, but contribute back improvements.
