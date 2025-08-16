# Self-Learning Trading Bot (Multi-Asset, Session-Aware)

Institutional-grade, modular framework for **Crypto Spot**, **Perp/Futures**, **Options**, and **Forex**, with **regime detection**, **session awareness (Asia/EU/US)**, **mode-based strategies (Scalp, Trend, Swing, Mean-Reversion, Options)**, **sub-ledger portfolio allocations**, **risk-adjusted sizing/leverage**, and **net-of-fees rewards**.

Supports Bybit Spot (testnet & live). Designed to extend to futures/options/forex connectors.

---

## What It Does (Plain English)
- Detects **market weather** (trend/range/volatile) and **active session** (Asia/EU/US/overlaps).
- Chooses **mode** (Scalp/Trend/Swing/MR/Options) + asset-class KPI gates.
- Sizes trades using **best-practice** rules:
  - **$10 fixed** until equity ≥ **$20,000** (per sub-ledger).
  - Then **%-risk sizing** with **ATR-based stops** and **leverage tiers**.
  - If a **Good Setup**: can size **up to 10% of that sub-ledger’s equity** in a single pair, but **never** exceed risk/leverage rails.
- Uses **live wallet** balances in production (no invented equity), and sub-ledgers in simulation.
- Allocates & tracks capital across **SPOT / PERP / OPTIONS / FOREX** sub-ledgers.
- Executes with **risk rails** (max DD, daily loss, leverage caps, liq-buffer).
- Computes **net P&L (after fees/slippage)** for ledger & rewards.
- Sends **Telegram** updates and exports **telemetry**.

---

## Manual Funds Control (YOU stay in charge)
- **Allow Bot To Use Funds** (global toggle): Off = paper/sim only; On = live orders allowed.
- **Per-Asset Class Toggle**: enable/disable SPOT, PERP, OPTIONS, FOREX independently.
- **Max Pair Allocation %**: default **10%** (cap per symbol from its sub-ledger).
- **UI**: switch + dropdowns (Strategy Mode, Allowed Asset Classes, Max Pair %).
- **Source of truth** for equity:
  - **LIVE**: exchange wallet balances (synced periodically).
  - **SIM**: Portfolio sub-ledger equity.

> Changes are logged (who/when/what), and enforced before each order.

---

## Trade Modes & Asset-Class KPIs

Execute only if **Expectancy > 0 after fees** and **Sharpe / Sortino / Calmar** and **Max Drawdown** meet **asset-class** thresholds. **Win % is advisory**.

| Asset Class  | Modes Available                    | KPIs (Sharpe, Sortino, Calmar, Max DD)                               |
|--------------|------------------------------------|------------------------------------------------------------------------|
| **Spot**     | Scalp, Trend, Swing, Mean-Reversion| ≥ 2.0, ≥ 3.0, ≥ 1.2, **≤ 15%**                                        |
| **Perp/Fut** | Scalp, Trend, Swing                | ≥ 1.8, ≥ 2.5, ≥ 1.2, **≤ 15%** (+ **liq buffer** ≥ 20% to stop)       |
| **Options**  | Options packages (spreads/hedges)  | ≥ 1.5, ≥ 2.0, ≥ 1.2, **≤ 12%** + Theta/day, IV−RV edge, |Δ| cap       |
| **Forex**    | Scalp, Trend, Mean-Reversion       | ≥ 2.0, ≥ 3.0, ≥ 1.5, **≤ 12%**                                        |

---

## Sessions (UTC guidance)
- **ASIA** 00:00–08:00 → MR or patient Trend; smaller size.
- **EU**   07:00–15:00 → Intraday Trend / Scalp; normal size.
- **US**   13:30–20:00 → Scalp/Breakout; tighter SL; higher size allowed.
- **Overlaps** (ASIA↔EU 07:00–08:00; EU↔US 13:00–16:00) → best liquidity; size bias up.

---

## Portfolio Allocations (Sub-Ledgers)
Example:
```python
ASSET_ALLOCATION_USD = {"SPOT": 14000, "OPTIONS": 1000, "PERP": 0, "FOREX": 0}
# or %
ASSET_ALLOCATION_PCT = {"SPOT": 0.85, "OPTIONS": 0.05, "PERP": 0.10, "FOREX": 0.0}
In LIVE, sub-ledgers are hard-capped by actual wallet balance (synced).

In SIM, sub-ledgers behave like virtual wallets.

Sizing & Leverage (Best-Practice)
Fixed phase: if sub-ledger equity < $20,000 → use $10 fixed_trade_usd per order.

%-risk phase (equity ≥ $20k): risk max_risk_pct of equity to stop:

ini
Copy
Edit
risk_per_trade = equity * max_risk_pct
sl_distance    = max(ATR_mult * ATR, min_stop_distance_pct * price)
qty            = risk_per_trade / (sl_distance * price)    # spot/fx math
size_usd       = qty * price
Good Setup boost:

If signal_score ≥ good_setup_score_min AND all KPI rails pass,

Allow up to 10% of sub-ledger equity in one pair, but still clamp by:

risk limits (risk_per_trade), session profile, pair cap, and leverage cap.

Leverage tiers: rise with equity/mode, but clamped by asset caps and risk manager.

Net P&L:

pnl_net = pnl_gross − fees − slippage_cost

Rewards/points use net %, not gross.

Policies (/policies)
kpi_policy.json (modes)
(unchanged; thresholds per Scalp/Trend/Swing/MR)

asset_kpi_policy.json (asset classes)
(unchanged; Spot/Perp/Options/Forex thresholds; leverage caps)

session_policy.json
Session windows; allowed modes; size/TP/SL multipliers; optional per-asset factors.

sizing_policy.json (updated)
json
Copy
Edit
{
  "global": {
    "equity_threshold_usd": 20000,
    "fixed_trade_usd": 10,
    "max_risk_pct": 0.005,
    "max_risk_pct_good_setup": 0.0075,
    "atr_mult_sl": 1.2,
    "min_stop_distance_pct": 0.0015,
    "slippage_bps": 2,
    "fee_bps": 10,
    "good_setup_score_min": 0.80,
    "pair_allocation_cap_pct": 0.10,          // 10% per pair hard cap
    "equity_source": "LIVE_OR_SIM"            // LIVE wallet in prod, sub-ledger in sim
  },
  "leverage_tiers": [
    {"equity_max": 10000,  "SCALP": 1.0, "INTRADAY_TREND": 1.0, "SWING": 1.0, "MEAN_REVERSION": 1.0},
    {"equity_max": 25000,  "SCALP": 2.0, "INTRADAY_TREND": 1.5, "SWING": 1.2, "MEAN_REVERSION": 1.2},
    {"equity_max": 50000,  "SCALP": 3.0, "INTRADAY_TREND": 2.0, "SWING": 1.5, "MEAN_REVERSION": 1.5},
    {"equity_max": 999999, "SCALP": 4.0, "INTRADAY_TREND": 3.0, "SWING": 2.0, "MEAN_REVERSION": 2.0}
  ],
  "asset_caps": {
    "SPOT":   {"max_leverage": 1.0},
    "PERP":   {"max_leverage": 3.0, "liq_buffer_pct": 0.20},
    "OPTIONS":{"max_leverage": 1.0},
    "FOREX":  {"max_leverage": 5.0}
  }
}
Architecture
graphql
Copy
Edit
main.py  
config.py  

modules/
  Portfolio_Manager.py        # Sub-ledgers & PnL (LIVE uses wallet sync caps)
  Funds_Controller.py         # NEW: Allow Bot toggle, per-asset enable, max pair %
  Wallet_Sync.py              # NEW: reads LIVE wallet balances; updates caps
  Sizer.py                    # Sizing & leverage (fixed→% risk; good-setup boost; 10% pair cap)
  Regime_Detector.py          # Market regime detection
  Market_Sessions.py          # Session detection
  Strategy_Manager.py         # Mode + signal + good_setup score
  Risk_Management.py          # Rails per asset/mode/session; leverage & liq buffer
  Trade_Executor.py           # Routing; computes net P&L (fees/slippage)
  Reward_System.py            # Rewards/points from net P&L
  Exchange_*                  # Spot/Perp/Options/Forex adapters
  Technical_Indicators.py
  Data_Manager.py
  Top_Pairs.py
  Telegram_Bot.py
state/
telemetry/
utils/
tests/
Quick Start
Install deps; set keys in config.py.

Add /policies/*.json (see above).

In UI, keep Allow Bot To Use Funds = OFF while testing.

Run simulation first; verify KPIs; then enable LIVE & toggles.

Testing
Unit tests: Wallet_Sync (no negative drift), Funds_Controller gates, Sizer boost logic, pair cap (10%), net-P&L path.

Mocks for exchanges; no live calls in CI.
