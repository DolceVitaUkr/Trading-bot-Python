# Self-Learning Trading Bot (Multi-Asset, Session-Aware)

An institutional-grade, modular trading framework for **Crypto Spot**, **Perp/Futures**, **Options**, and **Forex**, with **regime detection**, **session awareness (Asia/EU/US)**, **mode-based strategies (Scalp, Trend, Swing, Mean-Reversion, Options)**, **sub-ledger portfolio allocations**, **risk-adjusted sizing/leverage**, and **net-of-fees reward tracking**.

Supports Bybit Spot (testnet & live); architecture allows easy expansion to other assets and venues.

---

## What It Does (Plain English)
- Detects **market “weather”** (trend, range, volatile) via indicators.
- Determines **current session** (Asia, EU, US, overlaps).
- Chooses the appropriate **mode** (Scalp, Trend, Swing, Mean-Reversion, Options).
- Evaluates **asset-class-specific KPI gates** (Spot, Perp, Options, Forex).
- Sizes trades using:
  - **Fixed amount** initially until equity threshold.
  - **% risk sizing** beyond threshold, with **leverage tiers**.
- Allocates capital across sub-ledgers (e.g., $14k Spot / $1k Options).
- Executes trades in **simulation or live**, routing to proper exchange per asset class.
- Applies **risk management rails** (drawdown, max leverage, liquidation buffer, etc.).
- Computes **net profit** (after fees/slippage) for ledger tracking and reward calculation.
- Sends **Telegram notifications** and exports KPI telemetry.

---

## Trade Modes & Asset-Class KPIs

Execution allowed if **Expectancy > 0 after fees** and **Sharpe / Sortino / Calmar** meet thresholds. **Win % is advisory**.

| Asset Class  | Modes Available                  | KPIs (Sharpe, Sortino, Calmar, Max DD)           |
|--------------|----------------------------------|--------------------------------------------------|
| Spot Crypto  | Scalp, Trend, Swing, Mean-Reversion | Sharpe ≥ 2.0, Sortino ≥ 3.0, Calmar ≥ 1.2, DD ≤ 15% |
| Perp/Futures | Scalp, Trend, Swing              | Sharpe ≥ 1.8, Sortino ≥ 2.5, Calmar ≥ 1.2, DD ≤ 15% (liq buffer enforced) |
| Options      | Options-style (spreads, hedges)  | Sharpe ≥ 1.5, Sortino ≥ 2.0, Calmar ≥ 1.2, DD ≤ 12%; plus Theta/day, IV-RV edge, Δ-cap |
| Forex        | Scalp, Trend, Mean-Reversion     | Sharpe ≥ 2.0, Sortino ≥ 3.0, Calmar ≥ 1.5, DD ≤ 12% |

---

## Sessions & Mode Selection

**UTC session windows:**
- **ASIA:** 00:00–08:00 → Tend to use Mean-Reversion or Trend with small sizes.
- **EU:** 07:00–15:00 → Trend / Scalping on majors.
- **US:** 13:30–20:00 → Scalping / Breakout, tighter SL, higher size allowed.
- **Overlaps (ASIA-EU, EU-US):** Stronger liquidity, enlarged size, mode mixes allowed.

---

## Portfolio Allocation (Sub-Ledgers)

You can configure by USD or by percentage:

```python
ASSET_ALLOCATION_USD = {"SPOT": 14000, "OPTIONS": 1000, "PERP": 0, "FOREX": 0}
# or percentage:
ASSET_ALLOCATION_PCT = {"SPOT": 0.85, "OPTIONS": 0.05, "PERP": 0.10, "FOREX": 0.0}
The bot manages:

available_budget(asset), reserve(), book_trade(), release()

Prevents overspend and respects caps per asset class.

Sizing & Leverage Logic
Fixed-USD sizing until equity < equity_threshold_usd (e.g., $20k).

After threshold: % risk sizing, e.g., 0.5% of equity risked via ATR-based SL.

Leverage tiers scale up with equity & mode; capped by risk.

Reward and wallet updates use net P&L (after fees & slippage).

Policies (policies/ folder)
kpi_policy.json — mode KPIs.

asset_kpi_policy.json — asset-class KPIs.

session_policy.json — session profiles.

sizing_policy.json — equity thresholds, fixed € trade size, risk %, ATR multipliers, leverage tiers, asset caps.

Architecture
perl
Copy
Edit
main.py  
config.py  

modules/
  Portfolio_Manager.py        # Sub-ledgers & PnL
  Sizer.py                    # Trade sizing & leverage
  Regime_Detector.py          # Market regime detection
  Market_Sessions.py          # Session detection
  Strategy_Manager.py         # Mode + signal logic
  Risk_Management.py          # Rails per asset/mode/session
  Trade_Executor.py           # Routing to exchanges with net PnL
  Reward_System.py            # Net-PnL → rewards/points
  Exchange_* (Spot, Perp, Options, Forex)
  Technical_Indicators.py
  Data_Manager.py
  Top_Pairs.py
  Telegram_Bot.py
state/
telemetry/
utils/
tests/
Quick Start
Clone & install:

bash
Copy
Edit
git clone https://github.com/DolceVitaUkr/Trading-bot-Python.git
cd Trading-bot-Python
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
Configure in config.py:

Asset allocation, fee/slippage bps, equity thresholds, modes enabled, API keys.

Add policy JSONs to policies/.

Run:

Simulation:
python main.py --mode simulation --symbols "BTCUSDT,ETHUSDT" --interval 15m --asset-classes spot,options

Live: similar with --mode production.

Testing & Development
Unit tests for sizing logic, portfolio reservations, KPI gate enforcement.

Mocks for exchanges; no live calls in CI.

Fixtures for net-PnL correctness, per-asset-mode behavior, session boundaries.
