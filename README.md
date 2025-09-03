Self-Learning Trading Bot (Bybit + IBKR)

A modular, multi-asset trading system with paper + live modes, wallet-aware budgets, ML & RL training, and strict verification so trades cannot be invented.

Assets: crypto_spot, crypto_futures (Bybit), forex, options (IBKR)

Modes: paper and live (kept separate in state, logs, UI)

UI: manual start/stop for training and live trading (live requires double-confirmation), performance & history per asset/mode

Quick Start

Brokers

IBKR TWS/Gateway: 127.0.0.1:7497

Bybit mainnet (testnet=False). Put keys in .env.

Budgets

Edit tradingbot/state/budgets.json (see examples below).

Budgets are per asset, and the bot cannot exceed either the asset’s budget or the broker’s available wallet cash.

Run

python run_trading_bot.py


Open the dashboard (e.g., http://localhost:8000).
Use the Paper / Live sliders per asset. Live shows a confirmation modal.

Configuration (budgets.json)
{
  "alloc_mode": "absolute",                 // "absolute" or "percent" of live wallet
  "alloc": {                                // per-asset budget allocation (USD if absolute, fraction 0-1 if percent)
    "crypto_spot": 1000.0,
    "crypto_futures": 1500.0,
    "forex": 5000.0,
    "options": 3000.0
  },

  "per_trade_risk_pct": {                   // base risk % (below % sizing threshold)
    "crypto_spot": 0.02,                    // 2% of asset budget
    "crypto_futures": 0.01,                 // 1%
    "forex": 0.005,                         // 0.5%
    "options": 0.01                         // 1% (premium-based)
  },

  "percent_sizing_threshold_usd": 1000,     // NEW: above this, switch to percentage sizing
  "percent_sizing_above_threshold": {       // fraction of asset equity/budget per trade once >$1k
    "crypto_spot": 0.0075,                  // 0.75% (compounding)
    "crypto_futures": 0.005,                // 0.50%
    "forex": 0.0035,                        // 0.35%
    "options": 0.005                        // 0.50% (premium)
  },

  "profit_rollover": true,                  // realized PnL adds to the SAME asset budget on close
  "enforce_wallet_available": true,         // block orders if broker 'available' < notional
  "max_concurrent": 5,                      // per asset/mode
  "cooldown_seconds": 900,                  // optional cool-down after risk events

  "scale": {                                // equity-curve based risk multiplier (compounding)
    "enabled": true,
    "equity_curve_window_days": 7,
    "ladder": [
      { "profit_usd":  200, "risk_multiplier": 1.20 },
      { "profit_usd":  500, "risk_multiplier": 1.50 },
      { "profit_usd": 1000, "risk_multiplier": 2.00 }
    ]
  },

  "leverage_caps": {                        // soft caps; router also respects venue limits
    "crypto_spot": 1,
    "crypto_futures": 3,
    "forex": 10,
    "options": 1                             // measured by premium notional
  }
}


Budgets never “cross-bleed.” Profit from Forex increases Forex budget (if profit_rollover=true). Options budget is unaffected unless you later choose to add a “shared pool.”

Position Sizing (with numbers)

We compute a per-order notional cap. Let:

A = asset (e.g., forex)

B_A = asset’s current budget (USD; includes rollover if enabled)

r_A = per_trade_risk_pct[A]

m_A = risk multiplier from equity ladder (clamped 0.5–5.0)

P = current price from quote snapshot (≤2s old)

T = percent_sizing_threshold_usd (default 1000)

q_A = percent_sizing_above_threshold[A] (fraction of B_A)

Base cap (below threshold)

BaseCapUSD = B_A × r_A × m_A
if BaseCapUSD <= T:
    NotionalCap = BaseCapUSD


% sizing (above threshold)

PercentCapUSD = max(T, B_A × q_A × m_A)
if BaseCapUSD > T:
    NotionalCap = PercentCapUSD


Quantity cap

QtyCap = NotionalCap / P
(round to venue min/step; respect leverage caps)


Wallet hard stop
If broker available < NotionalCap, clamp to available or reject (per venue rules and your enforce_wallet_available).

Example (Forex)

Budget B_A = $7,000, r_A=0.5%, q_A=0.35%, profit ladder → m_A=1.5

BaseCapUSD = 7000×0.005×1.5 = $52.5 → below $1,000, so use BaseCapUSD early in growth

Later, as budget grows: B_A = $50,000

BaseCapUSD = 50,000×0.005×1.5 = $375 (still < 1000)

But you want compounding: once a trade would exceed $1,000, switch:

PercentCapUSD = max(1000, 50,000×0.0035×1.5) = max(1000, 262.5) = 1000

As budget keeps growing, PercentCapUSD scales proportionally (e.g., at $150k → 150,000×0.0035×1.5= $787.5 ⇒ cap = $1000 still; adjust q_A upwards if you want faster compounding).

Tune q_A by asset to control compounding speed. Keep m_A conservative until live metrics prove stability.

Risk Management & Guardrails

Max concurrent positions: max_concurrent per asset/mode.

Leverage caps: as per leverage_caps (router also respects venue rules).

Cool-down: optional; bot pauses fresh entries for cooldown_seconds after a risk event (e.g., daily loss cap breach—add this rule if you want).

Drawdown-aware RL reward: penalize actions that increase drawdown; inertia penalty for rapid flips.

Venue constraints: no short on spot; min qty/tick; leverage & margin rules.

Sessions & Regimes (when strategies are allowed to trade)

We characterize sessions (UTC):

Asian: 23:00–07:00

EU: 07:00–15:00

US: 13:00–21:00
(Overlaps are expected; rules evaluate in priority order or “any that match.”)

We classify regimes:

Trending up/down: ADX ≥ 20 and 50-EMA slope magnitude above threshold (e.g., |slope| ≥ 0.02% per bar)

Ranging: ADX < 18 and |slope| small; Bollinger band width below percentile threshold

High-vol: Realized Vol or ATR in top 20% of rolling window

Per-strategy allowlist (saved in strategies.json by Validation Manager):

{
  "strategy_id": "sma_cross_01",
  "allowed_assets": ["forex", "crypto_spot"],
  "allowed_sessions": ["ASIA", "EU"],          // disallow "US" if backtests show degradation
  "allowed_regimes": ["TREND_UP", "TREND_DOWN"],
  "disallowed_regimes": ["RANGE", "HIGH_VOL"],
  "notes": "Works best on Asian/EU trend; avoid US high-vol chop."
}


At runtime, the router asks: session OK? regime OK? asset OK?
If not, do not trade (or route to paper).

Training
ML (indicator-supervised)

Models: XGBoost/LightGBM (probabilities), Logistic baseline

Features: multi-TF EMA/RSI/ATR/ADX/MACD/BB/OBV, volume deltas, spread/imbalance, regime flags

Labels: sign(next return) or return quantiles

Validation: walk-forward (train 4w → test 1w rolling)

Routing: only enable in sessions & regimes where it passed validation

Promotion gates (paper → candidate)

≥ 100 closed paper trades

Sharpe ≥ 1.0, Max DD ≤ 8%, Win-rate ≥ 52%

Meets gates in the allowed sessions/regimes specifically

RL (PPO default; add 2nd model later)

State: recent OHLCV + indicators + exposure & unrealized PnL + budget usage + session/regime flags + venue limits

Actions: {flat, long, short} × {0, 0.5×, 1.0× of suggested size}

Reward:
ΔPnL − α·DD − β·fees − γ·budget_violation − δ·flip_cost
Suggested starting: α=2.0, β=1.0, γ=5.0, δ=0.2 (tune per asset)

Action masking: disallow budget/venue-violating actions at the environment level

Sessions/regimes: policy only acts when allowed (else environment returns “no-op”)

Live evaluation & refinement

Online metrics on rolling windows; if Sharpe < 0 or DD > 2× spec over N trades, automatically:

halve risk multiplier m_A (or switch to paper),

flag re-train task with the latest data slice,

keep trading only in the sessions/regimes where it still passes gates.

Verification & Validation Manager

Quote snapshots: every order captures a snapshot ID (bid/ask/last, ts ≤ 2s).
→ Paper fills must use this snapshot; live fills come from broker.
→ Bot cannot invent trades/time/price.

Backtests are session/regime-aware: a strategy is only approved for the contexts where it passed gates.

Promotion gates (live)

First month: Sharpe ≥ 0.8, Max DD ≤ 12%, Win-rate ≥ 50%

Must hold within approved sessions/regimes

If violated: degrade to paper, reduce m_A, or pause asset

Trading Flow

Paper

Strategy signal (ML prob > threshold or RL action) → session/regime check

Sizing: compute NotionalCap; switch to % sizing above $1,000

Quote snapshot (≤2s), attach snapshot ID

Guardrails: budget, wallet available, positions, leverage, cool-down

Paper fill with snapshot price; log to state/paper/trades_{asset}.jsonl

On close: update history, metrics; if profit_rollover=true, add realized PnL to same asset budget

Live
1–4 as above → broker order with idempotent client_order_id
5. If broker unreachable: no backfill; retry only if signal still holds on reconnect
6. Live fills ingested to state/live/{broker}/trades_{asset}.jsonl
7. Close → update budgets if profit_rollover=true

UI

Wallet tab: Budget (B_A), Wallet available, Usable now = min(remaining budget, wallet available)

History tab: 8 tables (asset × mode), sortable/filterable (symbol/strategy/date); totals row (PnL, fees, win-rate, expectancy)

Strategies tab: read-only cards with params, trades_used, Sharpe, win-rate, avg trade, approval/review status, allowed sessions/regimes

Training tab: start/stop ML & RL

Live slider: modal confirmation; visible “LIVE ENABLED” banner per asset

Charts: equity curve per asset/mode; budget overlay; trade markers

Sizing hint: show suggested position size near order controls

Defaults (you can tune)
Asset	Base Risk (r_A)	% Above $1k (q_A)	Max Leverage	Max Positions	Notes
crypto_spot	2.00%	0.75%	1×	5	No shorting; tick/step per symbol
crypto_futures	1.00%	0.50%	3×	5	Start low leverage; expand only after live success
forex (IBKR)	0.50%	0.35%	10×	5	Respect IBKR margin & lot sizes
options (IBKR)	1.00%	0.50%	1× (premium)	3–5	Count premium notional for sizing

Equity ladder multipliers (7-day PnL): +$200 → ×1.2, +$500 → ×1.5, +$1000 → ×2.0 (clamped 0.5–5.0)

Promotion gates (paper → candidate): ≥100 trades, Sharpe ≥ 1.0, Max DD ≤ 8%, Win-rate ≥ 52% (within approved sessions/regimes)
Live gates (first month): Sharpe ≥ 0.8, Max DD ≤ 12%, Win-rate ≥ 50%

Files & Directories

tradingbot/state/budgets.json — budgets, risk %, % sizing threshold, ladder, caps

tradingbot/state/paper/ — paper positions & trades

tradingbot/state/live/{broker}/ — live trades per broker

tradingbot/models/{asset}/ml|rl/ — checkpoints

tradingbot/metrics/{asset}/ — training metrics (JSONL)

tradingbot/logs/ — rotating JSON logs

tradingbot/ui/ — dashboard, JS, templates

Roadmap (next changes to code)

History ingestion (live) with quote snapshots → UI 8 tables fully populated

Idempotency + revalidation wired everywhere in order flow

UI: live confirm modal, sizing hint, session/regime badges on strategy cards

Training v1: ML XGBoost baseline + PPO with action masking, session/regime gating

(Optional) Policy Registry for a second RL model in shadow/canary (e.g., SAC)

## Quick Start

### Prerequisites
- Python 3.11+
- (Optional) Node.js 18+ if you rebuild the UI
- IBKR TWS or Gateway installed (Paper mode on port 7497 recommended for training)
- Bybit account (testnet or live)

### Setup
```bash
python -m venv .venv && source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
# (Optional) If UI requires a build: npm ci
cp .env.example .env  # fill in your keys
```

### Run
```bash
# Start the UI
python -m tradingbot.ui.app

# Paper trading loop (example)
python -m tradingbot.runtime.paper_loop --asset crypto_spot

# Training (example)
python -m tradingbot.training.ml_trainer --asset crypto_spot
```

### Tests
```bash
pytest -q
```

### Troubleshooting
- **IBKR**: Paper mode uses port **7497**. If you can't run two sockets, keep training on 7497 and switch to live later.
- **Bybit**: Use testnet mode if `BYBIT_USE_TESTNET=1`.
- **Kill Switch**: If triggered, the system goes to close-only and prevents new orders.