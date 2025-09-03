# Self-Learning Trading Bot (Bybit + IBKR)

A modular, multi-asset trading system with **paper + live** modes, wallet-aware budgets, ML & RL training, and strict verification so trades cannot be invented.

- **Assets**: crypto_spot, crypto_futures (Bybit), forex, options (IBKR)
- **Modes**: paper and live (kept separate in state, logs, UI)
- **UI**: manual start/stop for training and live trading (live requires double-confirmation), performance & history per asset/mode

---

## What’s new in this version (additive, no breaking changes)

This release **adds implementation logic** while keeping your original behavior and docs:

- **Paper vs Live Routing (Bybit + IBKR)**  
  - **Paper** uses **mainnet quotes (read-only)** and executes **strictly in the local simulator** — no broker order endpoints are called.  
  - **Live** is the only mode that places **real broker orders** (Bybit, IBKR).  
  - Central **routers** ensure the correct sink; they also apply a **final exchange-conformance clamp** (tick/step/minNotional/multipliers) before submit/sim.
- **Memory & Restart Safety**  
  - Atomic state files + snapshots; write-ahead log; startup **reconciler** to heal state against broker truth (for live).  
  - Idempotent `client_order_id` to prevent duplicate orders on restarts.
- **Harsher Paper Execution**  
  - Configurable slippage (bps), taker-only fills, partial-fill probability, and small latency—paper is **more conservative** than live by design.
- **Futures & Options Lifecycle**  
  - Contract **catalog** (expiry, tick, step, multiplier, settlement, exercise style, funding schedule, session calendar).  
  - **DTE gates** (block opening near expiry), **auto-roll** (TWAP), **paper settlement**, **options expiry** with **exercise/assignment simulation**.  
  - **Sessions/Holidays** respected for IBKR assets; **funding accrual** for perps.
- **Strict TP/SL + Risk**  
  - Brackets required for every order; per-asset budgets, position caps, concurrency caps, daily loss and drawdown tripwires.
- **Promotion Gates (Safety for ML/RL)**  
  - The model cannot go live until pre-agreed **paper → shadow → canary → live** gates are passed; automatic **degrade-to-paper** if violated.

These upgrades are **orthogonal** to your strategy logic—no behavior is removed, only guarded.

---

## Quick Start

### Brokers

- **IBKR TWS/Gateway**: `127.0.0.1:7497` (Paper commonly 7497; Live commonly 7496)  
- **Bybit** mainnet (set `testnet=False`). Keys go in `.env`.

> Paper uses **mainnet quotes only** for both Bybit and IBKR and routes orders to the **local simulator**. Live routes to broker endpoints.

### Budgets

Edit `tradingbot/state/budgets.json` (see full schema below).  
Budgets are **per asset**, and the bot cannot exceed either the asset’s budget or the broker’s **available** wallet cash.

### Run

```bash
python run_trading_bot.py
Open the dashboard (e.g., http://localhost:8000).
Use the Paper / Live sliders per asset. Live shows a confirmation modal.

Configuration (budgets.json)
json
Copy code
{
  "alloc_mode": "absolute",
  "alloc": {
    "crypto_spot": 1000.0,
    "crypto_futures": 1500.0,
    "forex": 5000.0,
    "options": 3000.0
  },
  "per_trade_risk_pct": {
    "crypto_spot": 0.02,
    "crypto_futures": 0.01,
    "forex": 0.005,
    "options": 0.01
  },
  "percent_sizing_threshold_usd": 1000,
  "percent_sizing_above_threshold": {
    "crypto_spot": 0.0075,
    "crypto_futures": 0.005,
    "forex": 0.0035,
    "options": 0.005
  },
  "profit_rollover": true,
  "enforce_wallet_available": true,
  "max_concurrent": 5,
  "cooldown_seconds": 900,
  "scale": {
    "enabled": true,
    "equity_curve_window_days": 7,
    "ladder": [
      { "profit_usd": 200,  "risk_multiplier": 1.20 },
      { "profit_usd": 500,  "risk_multiplier": 1.50 },
      { "profit_usd": 1000, "risk_multiplier": 2.00 }
    ]
  },
  "leverage_caps": {
    "crypto_spot": 1,
    "crypto_futures": 3,
    "forex": 10,
    "options": 1
  }
}
Notes

Budgets never cross-bleed. Profit from Forex increases Forex budget (if profit_rollover=true).

Options sizing counts premium notional.

Router also respects per-venue leverage and margin rules.

Position Sizing (with numbers)
We compute a per-order NotionalCap from the asset’s budget and risk settings, and switch to % sizing above a threshold:

Let:
B_A = asset budget (USD), r_A = per_trade_risk_pct[A], m_A = ladder multiplier,
P = latest price from a ≤2s quote snapshot, T = threshold (default 1000),
q_A = percent_sizing_above_threshold[A].

BaseCapUSD = B_A × r_A × m_A. If BaseCapUSD ≤ T, then NotionalCap = BaseCapUSD.

Otherwise PercentCapUSD = max(T, B_A × q_A × m_A) and NotionalCap = PercentCapUSD.

QtyCap = NotionalCap / P (rounded to venue min/step; leverage caps apply).

Wallet stop: if broker available < NotionalCap, clamp or reject per venue rules.

A worked Forex example is included (7k → 50k → 150k progression). Tune q_A per asset to control compounding speed.

UI (Dashboard)
Wallet: Budget (B_A), Wallet Available, Usable now = min(remaining budget, wallet available)

History: 8 tables = asset × mode (sortable by symbol/strategy/date; totals PnL, fees, win-rate, expectancy)

Strategies: read-only cards with params, trades_used, Sharpe, win-rate, avg trade, approval status, allowed sessions/regimes

Training: start/stop ML & RL

Live slider: modal confirmation; banner “LIVE ENABLED” per asset

Charts: equity curve per asset/mode; budget overlay; trade markers

Sizing hint near order controls

Defaults (tunable)

Asset	Base r_A	% Above $1k (q_A)	Max Lev	Max Pos	Notes
crypto_spot	2.00%	0.75%	1×	5	No shorting; tick/step per symbol
crypto_futures	1.00%	0.50%	3×	5	Start low lev; expand after live proof
forex (IBKR)	0.50%	0.35%	10×	5	Respect IBKR margin & lot sizes
options (IBKR)	1.00%	0.50%	1×	3–5	Premium-based sizing

Equity ladder multipliers (7-day PnL): +$200 → ×1.2, +$500 → ×1.5, +$1000 → ×2.0 (clamped 0.5–5.0)

Sessions & Regimes (when strategies can act)
Sessions (UTC): Asian 23:00–07:00, EU 07:00–15:00, US 13:00–21:00 (overlaps OK).
Regimes: TREND_UP/DOWN (ADX≥20 & EMA slope), RANGE (ADX<18 & tight bands), HIGH_VOL (top-20% RV/ATR).
Each strategy carries an allowlist (sessions/regimes/assets). At runtime, router checks: session OK? regime OK? asset OK? Otherwise do not trade (or route to paper).

Training
ML (indicator-supervised)
Models: XGBoost/LightGBM (probabilities), Logistic baseline

Features: multi-TF EMA/RSI/ATR/ADX/MACD/BB/OBV, volume deltas, spread/imbalance, regime flags

Labels: sign(next return) or return quantiles

Validation: walk-forward (train 4w → test 1w rolling)

Routing: enable only in the sessions & regimes where it passed validation

Promotion gates (paper → candidate)
≥100 closed paper trades; Sharpe ≥ 1.0; Max DD ≤ 8%; Win-rate ≥ 52% (within approved sessions/regimes)

RL (PPO baseline; add second policy later)
State: recent OHLCV + indicators + exposure & uPnL + budget usage + session/regime flags + venue limits

Actions: {flat, long, short} × {0, 0.5×, 1.0× size} with action masking for illegal moves

Reward: ΔPnL − α·DD − β·fees − γ·budget_violation − δ·flip_cost (e.g., α=2.0, β=1.0, γ=5.0, δ=0.2)

Safety: paper-only until promotion gates are met; shadow vs baseline; canary live with micro size and tripwires.

Live gates (first month)
Sharpe ≥ 0.8; Max DD ≤ 12%; Win-rate ≥ 50%. If violated → degrade to paper, reduce m_A, or pause asset.

Verification & Validation Manager (no invented trades)
Every order stores a quote snapshot ID (bid/ask/last, ts ≤2s).

Paper fills use this snapshot; live fills come from broker.

This prevents “invented” trades, prices, or timestamps.

Backtests are session/regime aware: approval is contextual (where it actually worked).

Trading Flow
Paper
Strategy signal (ML prob or RL action) → session/regime check

Sizing: compute NotionalCap; above $1k switch to % sizing

Quote snapshot (≤2s), attach snapshot ID

Guardrails: budget, wallet available, positions, leverage, cool-down

Paper execution: conservative fill (slippage/partials/latency); log state/paper/trades_{asset}.jsonl

On close: update history/metrics; if profit_rollover=true, add realized PnL to the same asset budget

Live
1–4 same → router forwards to Bybit/IBKR with idempotent client_order_id
5) If broker unreachable: no backfill; retry only if signal still holds on reconnect
6) On fill: write state/live/{broker}/trades_{asset}.jsonl
7) On close: apply profit_rollover if enabled

Paper vs Live Routing (details)
PaperRouter → clamps → calls local simulator only (never broker order endpoints)

LiveRouter → clamps → forwards to configured broker submitter (Bybit/IBKR)

Exchange-Conformance Clamp: snaps price/qty to tick/step, enforces minNotional & multipliers, logs a CLAMP event

Rate limits: token buckets at ~90% headroom with separate read vs trade budgets so paper training can’t starve live trading.

Memory & Restart Safety
Atomic state writes + hourly/daily snapshots

Write-Ahead Log (WAL) so unfinished actions replay safely

Reconciler (startup + periodic) heals open orders/fills from broker truth (live)

Idempotency: client_order_id = hash(strategy_id, symbol, ts, nonce) prevents duplicate orders

Futures & Options Lifecycle
Contract Catalog (state/contracts.json): id, type (future/option/perp), underlying, expiry, multiplier, tick_size, lot_size, min_notional, settlement, exercise_style, funding_schedule, session_calendar

Pre-trade gates: block opening if days_to_expiry < N_open (e.g., 5 futures / 3 options)

Auto-roll (futures): TWAP slices when DTE ≤ N_roll or next contract leads; log ROLL (slippage/fees)

Paper settlement: at expiry, compute PnL vs settlement and log SETTLEMENT

Options expiry: OTM → expire; ITM → auto-exercise (threshold configurable) and assignment simulation (adjust underlying & cash)

Sessions/Holidays (IBKR): orders outside session hours are queued/blocked with SESSION_CLOSED

Perps funding: accrue pay/receive to PnL (live & paper)

Strict TP/SL & Risk
Mandatory brackets on every order (paper & live)

Max loss per order, max portfolio drawdown, max concurrent positions, daily loss cap

Kill-switch and close-only mode respected everywhere

Files & Directories
tradingbot/state/budgets.json — budgets, risk %, % sizing threshold, ladder, caps

tradingbot/state/paper/ — paper positions & trades

tradingbot/state/live/{broker}/ — live trades per broker

tradingbot/models/{asset}/ml|rl/ — checkpoints

tradingbot/metrics/{asset}/ — training metrics (JSONL)

tradingbot/logs/ — rotating JSON logs

tradingbot/ui/ — dashboard, JS, templates

Environment & Flags (added for clarity)
Copy .env.example → .env, then set:

RATE_LIMIT_PCT=0.9 — headroom for rate limits

HARSH_PAPER_EXECUTION=1, PAPER_SLIPPAGE_BPS=8, PAPER_PARTIAL_PROB=0.2

STRICT_BRACKETS=1

FUTURES_OPEN_DTE_MIN=5, FUTURES_ROLL_DTE=2

OPTIONS_OPEN_DTE_MIN=3, OPTIONS_ITM_EXERCISE_THRESHOLD=0.01

PERP_FUNDING_ENABLED=1

Broker creds/hosts: BYBIT_*, IBKR_HOST/PORT/CLIENT_ID

Quick Start (full)
Prerequisites
Python 3.11+

(Optional) Node.js 18+ for rebuilding UI

IBKR TWS or Gateway installed (Paper mode on port 7497 recommended)

Bybit account (testnet or live)

Setup
bash
Copy code
python -m venv .venv && source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
# (Optional) If UI requires a build:
# npm ci
cp .env.example .env  # fill in your keys
Run
bash
Copy code
# Start the UI
python -m tradingbot.ui.app

# Paper trading loop (example)
python -m tradingbot.runtime.paper_loop --asset crypto_spot

# Training (example)
python -m tradingbot.training.ml_trainer --asset crypto_spot
Tests
bash
Copy code
pytest -q
Troubleshooting
IBKR: Paper uses port 7497. If you can’t run two sockets, keep training on 7497 and switch to live later.

Bybit: Use testnet mode if BYBIT_USE_TESTNET=1.

Order rejects: check clamp logs (tick/step/minNotional) and size caps.

Kill Switch: If triggered, system goes close-only and blocks new orders.

