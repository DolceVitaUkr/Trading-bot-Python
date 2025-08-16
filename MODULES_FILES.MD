## ✅ Updated `MODULES_FILES.MD` (paste into repo root)

```markdown
# MODULES_FILES.MD
_Last updated: 2025-08-16 (Europe/London)_

## Naming & Logging
- Module names: `Train_RL_Model.py`, `Portfolio_Manager.py`, `Funds_Controller.py`, `Wallet_Sync.py`, `Sizer.py`, etc.
- Log fields: `module, action, asset_class, session, symbol, mode, size_usd, leverage, fees_usd, pnl_net_usd, allowed_funds, pair_cap_pct`.

## Entry
### main.py
- Initializes: config → Wallet_Sync → Portfolio_Manager → Funds_Controller → Sizer → Sessions/Regime → Strategy loop → Risk → Executor → Reward → Telegram.
- CLI: `--mode`, `--symbols`, `--interval`, `--asset-classes=spot,perp,options,forex`.

## Core Modules (`modules/`)
### Portfolio_Manager.py
- **What:** Sub-ledger allocations (USD/%), reservations, P&L, realized fees.
- **LIVE:** Caps available budget by **live wallet** free balance (from Wallet_Sync).
- **API:** `available_budget(asset)`, `reserve(asset, usd)`, `release(id)`, `book_trade(asset, pnl_net, fees)`.

### Funds_Controller.py  **NEW**
- **What:** Runtime permissions & caps.
- **State:** `allow_bot_funds` (bool), `asset_enabled[asset]` (bool), `max_pair_allocation_pct` (default 0.10).
- **API:** `is_allowed(asset, symbol)`, `pair_cap_pct()`, `snapshot()`.
- **UI hooks:** toggle/switch & dropdown handlers; all changes logged.

### Wallet_Sync.py  **NEW**
- **What:** Pulls **live** wallet balances from active exchange(s); maps to sub-ledgers.
- **API:** `sync() -> balances`, `get_equity(asset)`.
- **Notes:** Frequency configurable; handles multiple wallets; tolerates API errors gracefully.

### Sizer.py
- **What:** Best-practice sizing with **fixed→% risk** and **Good Setup boost**.
- **Inputs:** equity (from Wallet_Sync or Portfolio), asset, mode, ATR, price, risk %, good_setup score, pair_cap_pct.
- **Outputs:** `{size_usd, leverage, sl_distance}`; always clamps to pair cap, risk cap, leverage cap.
- **Policy:** from `/policies/sizing_policy.json`.

### Regime_Detector.py
- **What:** Labels `trend | range | volatile | mixed` via ADX/EMA slope/BB-bandwidth/ATR-pct.

### Market_Sessions.py
- **What:** Session detection (ASIA/EU/US/overlaps) + per-session profile.

### Strategy_Manager.py
- **What:** Chooses mode (session + regime + asset-KPI); computes `signal_score` & `good_setup` flag.
- **API:** `select_mode(regime, session, asset)`, `decide(df, context)->{signal, sl, tp, meta{signal_score,good_setup}}`.

### Risk_Management.py
- **What:** Daily loss, max DD, loss-streak cooldown, per-symbol caps, **max_leverage**, **liq_buffer** for perps; validates final proposal.

### Trade_Executor.py
- **What:** Route per asset class; place/cancel; compute **net P&L = gross − fees − slippage**; return receipt.

### Reward_System.py
- **What:** Wallet & points tracked on **net** P&L; series for charts.

### Exchange_* (Spot/Perp/Options/Forex)
- **What:** Venue adapters; live balances API for Wallet_Sync; funding/greeks where applicable.

### Technical_Indicators.py, Data_Manager.py, Top_Pairs.py, Telegram_Bot.py
- As documented previously.

## Policies (`/policies`)
- `kpi_policy.json`, `asset_kpi_policy.json`, `session_policy.json`, `sizing_policy.json` (with `good_setup` fields & `pair_allocation_cap_pct=0.10`).

## Tests
- Wallet_Sync mock balances; Funds_Controller enforcement; Sizer boost + cap; Risk leverage/liq buffer; Executor net-P&L; end-to-end sim run.
