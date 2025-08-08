# Trading Bot – Python (Crypto, Perps, Forex, Forex Options)

## Overview
A modular, self-learning trading bot capable of paper and live trading across multiple asset classes:
- **Crypto Spot**
- **Crypto Perpetuals (Perps)**
- **Forex**
- **Forex Options** (planned)

Supports both **swing** and **scalp** strategies with dynamic parameter adaptation, multi-timeframe confirmation, capital segregation, and stage-gated rollout.

---

## Key Features

### Multi-Asset Support
- Crypto Spot, Perps, Forex, and Forex Options
- Per-exchange profiles (`spot`, `perp`, `spot+perp`)
- Independent accounts for Forex/Options (no cross-funding)
- Dynamic timeframe mapping per asset class
  - Crypto: **15m** primary, **5m** confirmation
  - Forex: adjustable (default 1h/15m), auto-detect for options expiry

### Data & Execution
- WebSocket + REST market data adapters
- Partitioned parquet historical storage (by day) with deduplication
- Live order execution with TP/SL attachment for **all** trades
- SL/TP can move in favourable conditions, never widened to increase loss
- Reconciliation after restart (reattach missing TP/SL, monitor existing positions)
- Position sizing:
  - Base: 5% of wallet per trade
  - Canary: 2% risk for first 50 trades / 7 days in new domain
  - Exposure caps: 15% per pair, 30% portfolio concurrent

### Risk & Guardrails
- Max drawdown ≤ 15%
- Consecutive-loss cooldown
- Regime awareness:
  - Tight SL in chop
  - Wider SL in trend (still capped)
  - Time-based max hold
  - Avoid weekend carry (configurable)
- Fee/slippage model per pair with periodic calibration
- Global API throttle guard + backfill queue
- Disaster mode: one-click flatten across domains (Telegram confirm)

### Learning & Exploration
- Online reinforcement learning (ON/OFF toggle)
- Minimum exploration rate = 10%, dynamic to 25% in chop
- Strategy adapters per asset class
- Optional sentiment score hook
- Drift detection + scheduled retraining
- Bayesian optimization via Optuna (or grid/evolutionary search)

### Telemetry & Reporting
- Prometheus metrics exporter
- KPI snapshots per rollout stage
- Daily & weekly PnL reports
- Export KPIs/plots to CSV/PNG
- Telegram notifications for:
  - Stage changes
  - KPI threshold results
  - Disaster mode activation
  - Trade entries/exits

---

## Rollout Plan (State Machine)

### Stage 1 – Crypto Paper (Learn)
- Paper trade Crypto Spot (Perps/Forex OFF)
- Train on live data (15m + 5m confirm)
- Paper wallet: `$1,000` (Crypto_Paper)
- Gate to Stage 2: hit KPI thresholds (win rate, Sharpe, DD)

### Stage 2 – Crypto Live + Exploration
- Live trade Crypto with real wallet
- Continue Crypto paper exploration trades (≥10% dynamic)
- Canary sizing for first 50 trades or 7 days
- Perps/Forex still paper-only

### Stage 3 – Crypto Live + Perps/Forex Paper
- Crypto Live + Exploration
- Start Perps_Paper = `$1,000`
- Start Forex_Paper = `$1,000`
- Independent paper wallets per domain

### Stage 4 – Crypto & Forex Live + Options Paper
- Promote Forex to live (separate broker account)
- Start ForexOptions_Paper = `$1,000`
- Exploration for Crypto & Forex live

### Stage 5 – Full Rollout
- Crypto Live + Forex Live
- Continue learning & exploration
- Optional promotion of Options to live

---

## Capital Segregation
- **Paper wallets**: `Crypto_Paper`, `Perps_Paper`, `Forex_Paper`, `ForexOptions_Paper` – start with `$1,000` each when activated
- **Live accounts**:  
  - Crypto/Perps → Bybit (Spot & Perp profiles)  
  - Forex/Options → Separate broker(s)
- No cross-funding between domains
- Real-time wallet balance checks before order placement

---

## UI & Controls
- Stage selection buttons (Stage 1–5)
- Asset toggles:
  - Spot / Perp / Spot+Perp (Crypto)
  - Forex ON/OFF (default OFF every boot)
  - Options ON/OFF (default OFF every boot)
- Exploration % slider (min 10%)
- Online Learning toggle
- Disaster Mode button (flatten all)

---

## Config & State
- `config.py` holds rollout defaults, exposure caps, fee models, and timeframes
- `runtime_state.json` stores:
  - Last stage and toggles
  - Open positions per domain
  - Paper wallet balances
  - Canary/trade stats

---

## Repo Structure
See [`Modules.md`](Modules.md) for full breakdown.

Key directories:
- `modules/` – core bot logic (data, exchange, strategies, risk, UI)
- `self_test/` – unit/integration tests
- `utils/` – common helpers
- `forex/` – forex & options adapters (placeholders until enabled)
- `marketdata/` – WS/REST adapters
- `dataio/` – parquet/historical data handling
- `state/` – runtime persistence
- `risk/` – sizing & guardrails
- `strategy/` – regime classifiers & exploration manager
- `telemetry/` – Prometheus metrics & KPI reporting

---

## Getting Started
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export BYBIT_API_KEY="..."
export BYBIT_API_SECRET="..."
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."

# Start in Stage 1 (Crypto Paper)
python main.py --stage 1
KPI Targets (per domain)
Win rate: ≥ 70% swing / ≥ 60% scalp

Sharpe ratio: ≥ 2.0 swing / ≥ 1.5 scalp

Avg profit per trade: realistic per strategy type

Swing: 10–30%

Scalp: 0.3–2%

Max drawdown: ≤ 15%

Error budget & latency OK
