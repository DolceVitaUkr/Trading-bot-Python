# 🐍 Self-Learning AI Trading Bot

A modular Python trading framework designed for **crypto, forex, and options** with built-in **Machine Learning (ML)** and **Reinforcement Learning (RL)** strategy engines.

Supports **Bybit**, **Binance** *(planned)*, and **future connectors**.  
Built for **low-risk, scalable, automated trading** with **live** and **simulation** modes.

---

## 🚀 Features

### ✅ Current Functionality
- **Bybit Spot (Testnet & Live)** with **5m** and **15m** intervals
- **Simulation-first execution** — no real wallet impact in learning mode
- **Live market data** via Bybit WS + REST, incremental backfill to avoid gaps
- **Top Pairs logic** — dynamically selects best-performing symbols every hour
- **Hourly Top Pairs refresh** + minute-level monitoring for open positions
- **Three UI charts**:
  1. Wallet Balance
  2. Virtual Wallet Balance
  3. Reward Points  
- **Incremental data append** — avoids full re-download, keeps history consistent
- **900-bar request cap** (Bybit limit-safe)
- **Position check every 1–5 minutes** while open, with TP & SL rules
- **Telegram integration** for trade, risk, and KPI updates
- **Modular architecture** for easy expansion

### 🛠 In Progress / Planned
- **Binance connector** (spot, futures)
- **Forex & Options adapters**
- **Full RL/ML strategy training from live data**
- **Backtesting engine** with historical market replay
- **Cloud storage sync** (SharePoint / S3) for large datasets
- **Web dashboard** for remote monitoring
- **Multi-market portfolio diversification**
- **Automated strategy selection per market regime**

---

## 📂 Project Structure

.
├── main.py # Entry point
├── config.py # Central configuration
├── modules/ # Core modules
│ ├── data_manager.py
│ ├── exchange.py
│ ├── trade_executor.py
│ ├── risk_management.py
│ ├── reward_system.py
│ ├── technical_indicators.py
│ ├── top_pairs.py
│ ├── self_learning.py
│ ├── trade_simulator.py
│ ├── telegram_bot.py
│ ├── ui.py
│ ├── notification_manager.py
│ └── utils/
│ └── utilities.py
│
├── telemetry/ # Metrics and reports
│ ├── metrics_exporter.py
│ └── report_generator.py
│
├── state/ # State persistence
│ ├── runtime_state.py
│ └── position_reconciler.py
│
├── historical_data/ # Stored market data
├── modules.md # API surface & module contracts
└── README.md

yaml
Copy
Edit

---

## ⚙️ Installation

```bash
# 1. Clone repo
git clone https://github.com/youruser/self-learning-bot.git
cd self-learning-bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys & settings
# Edit config.py or set environment variables
▶️ Usage
Simulation Mode (default)
bash
Copy
Edit
python main.py --mode simulation
Live Trading (production)
bash
Copy
Edit
python main.py --mode production
Backtest (planned)
bash
Copy
Edit
python -m modules.trade_simulator
🧠 RL / ML Training
RL Online — runs continuously during trading if enabled

ML Training — offline supervised learning on historical datasets

Market Replay Mode (planned) — simulate years of trading in hours

📡 Notifications
Trade entries/exits

Risk violations

Daily/weekly KPI summaries

Position monitoring alerts

Configured in config.py:

python
Copy
Edit
TELEGRAM_BOT_TOKEN = "your_token"
TELEGRAM_CHAT_ID = "your_chat_id"
📊 Backtesting
(Planned feature)

Will allow:

Fee/slippage-aware simulations

Strategy parameter optimization

Market regime performance review

🛡️ Risk Management
Max exposure per pair & portfolio

Position sizing based on balance & stop loss

Loss streak cooldowns

Force exits on violations

📌 Rollout Stages
Data only

Simulated trades

Small live trades (canary)

Scaled positions

Full deployment

📅 Roadmap
Add Binance & Forex connectors

Expand options trading module

Auto strategy selection per market regime

Cloud dataset storage integration

Web dashboard

📄 License
MIT License — feel free to fork and modify, but contribute back improvements.
