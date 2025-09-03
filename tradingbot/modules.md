# Modules.md (augmented, full)

```md
# Modules
Version 1.0

This document maps the bot’s modules, key functions, important variables, and exact connections so that every data producer has a consumer and every consumer lists its source. It also includes a high-level flow and a folder structure diagram.

---

## 0) Global Conventions

- **Assets (UI keys)**: crypto_spot, crypto_futures, forex, options
- **Modes**: paper, live
- **Quote snapshot**: captured at submit; each trade row includes `quote_snapshot_id`
- **Idempotency**: `client_order_id = hash(strategy_id, symbol, ts)`
- **Budget terms**: `B_A` (asset budget), `r_A` (per_trade_risk_pct), `m_A` (risk multiplier), `q_A` (% sizing above threshold)

---

## 1) Core

### 1.1 tradingbot/core/runtime_api.py
**Purpose**: Aggregates wallets/positions/history for UI & services. Health & retry wrappers.

**Imports**  
`paper_state`, `BybitAdapter` (lazy), `IBKRAdapter` (lazy), `retry_call`, `history_store`

**Singletons (lazy)**  
`_BYBIT`, `_IBKR`

**Functions**  
- `aggregate_status(asset) -> dict` → `{"asset","paper":{...},"live":{...}}`  
- `aggregate_positions(asset) -> dict` → `{"paper":[...], "live":[...]}`  
- `read_trade_history(asset, mode?, limit, since?) -> list[dict]`  
- `health() -> {"bybit":{ok,...}, "ibkr":{ok,...}}`  
- `live_enabled(asset) -> bool` (from `state/runtime.json`)

**Consumers**: `ui/app.py`, `order_router.py`, UI JS.

---

### 1.2 tradingbot/core/order_router.py
**Purpose**: Central order flow (validations, guardrails, idempotency, submit, retry).

**Important types**  
`OrderContext {asset_type, symbol, side, quantity, price?, client_order_id, strategy_id, mode}`  
`OrderResult {success, reason?, broker_order_id?}`

**Core**  
- Attach **fresh quote snapshot** (≤2s) and `quote_snapshot_id`.  
- **Budget guardrails** via `budget_manager.can_place_order(...)`.  
- **Position size cap** via `budget_manager.position_size_cap(...)`.  
- **Venue checks** (min qty/step, long-only on spot, leverage caps).  
- **Idempotency**: reject duplicates by `client_order_id`.  
- **Routing** (see 1.8/1.9 below).  
- **Retry queue** for broker failures: **revalidate** signal on reconnect; never backfill.

---

### 1.3 tradingbot/core/budget_manager.py
**Purpose**: Budget enforcement & position sizing.

**State/Files**: `state/budgets.json`, `state/risk_state.json`  
**Imports**: `runtime_api.aggregate_status`, `runtime_api.aggregate_positions`, optionally `history_store` for realized PnL ladder.

**Key config (`budgets.json`)**  
`alloc_mode`, `alloc{...}`, `per_trade_risk_pct{...}`, `percent_sizing_threshold_usd`, `percent_sizing_above_threshold{...}`, `profit_rollover`, `enforce_wallet_available`, `max_concurrent`, `scale{...}`, `leverage_caps{...}`

**Functions**  
- `get_alloc(asset_ui_key) -> float`  
- `count_open_positions(asset_ui_key, mode) -> int`  
- `set_cooldown(...) / in_cooldown(...) -> bool`  
- `position_size_cap(asset_ui_key, price) -> (cap_notional, cap_qty)`  
- `suggest_position_size(...) -> float`  
- `can_place_order(...) -> (ok, reason)`

**Consumers**: `order_router.py`, UI sizing hints.

---

### 1.4 tradingbot/core/history_store.py
**Purpose**: Normalized trade history across paper/live and assets.

**Files**  
- Paper: `state/paper/trades_{asset}.jsonl`  
- Live: `state/live/bybit/trades_{asset}.jsonl`, `state/live/ibkr/trades_{asset}.jsonl`

**Schema (per row)**  
`trade_id, order_id, strategy_id, mode, asset, symbol, side, qty, avg_price, fees, slippage, realized_pnl, realized_pnl_pct, reward, opened_at, closed_at, duration_s, venue, account_id, quote_snapshot_id, status`

**Functions**  
`read_history(asset, mode?, since?, limit) -> list[dict]`

**Consumers**: `runtime_api.read_trade_history`, UI history.

---

### 1.5 tradingbot/core/retry.py
`retry_call(func, *args, retries=3, backoff=0.5, max_backoff=4.0, exceptions=(Exception,), jitter=True, **kwargs)`  
**Consumers**: runtime_api, adapters, health checks.

---

### 1.6 tradingbot/core/loggerconfig.py
`setup_logging(level: str|int = None) -> None`  
**Consumers**: ui/app.py at import time.

---

### 1.7 **NEW** tradingbot/core/exchange_conformance.py
**Purpose**: **Final clamp** before any submit/sim: tick size, step size, min notional, contract multiplier.  
**Functions**:  
- `clamp_order_if_needed(order_ctx) -> order_ctx'` (logs `CLAMP` before/after)

**Consumers**: routers (paper & live).

---

### 1.8 **NEW** tradingbot/core/routing.py
**Purpose**: Paper/Live split with uniform `OrderContext`.

- `PaperRouter(simulate_order)` → clamps → **local simulator** only (no broker orders)  
- `LiveRouter({venue: submitter})` → clamps → forwards to `bybit_submit_wrapper` or `ibkr_submit_wrapper`

**Note**: Maintains **rate-limit headroom** using token buckets (~90%), with separate **read** vs **trade** budgets.

---

### 1.9 **NEW** tradingbot/core/paper_execution.py
**Purpose**: Conservative paper fills: slippage bps, taker-only, partials, latency; fail if non-conforming after clamp.  
**Consumers**: `paper_trader.py`.

---

### 1.10 **NEW** tradingbot/core/persistence.py
**Purpose**: Persistence helpers: **atomic writes**, **snapshots**, and a simple **Write-Ahead Log**.  
**Consumers**: runtime & strategy state writers.

---

### 1.11 **NEW** tradingbot/core/reconciler.py
**Purpose**: Startup + periodic **reconciliation** of local state to broker truth (live only).  
**Consumers**: engine bootstrap / scheduler.

---

### 1.12 **NEW** tradingbot/core/contract_catalog.py
**Purpose**: Single source of truth for **futures/options/perps** (expiry, tick, step, multiplier, settlement, exercise style, funding schedule, session calendar).  
**Artifacts**: `state/contracts.json` (startup + daily refresh).  
**Consumers**: lifecycle gates, clamp (for multipliers), UI.

---

### 1.13 **NEW** tradingbot/core/market_calendars.py
**Purpose**: Session/holiday windows (e.g., CME, IBKR option sessions, perpetual venues).  
**Consumers**: pre-submit guards; scheduler.

---

### 1.14 **NEW** tradingbot/core/futures_lifecycle.py
**Purpose**: DTE **open gate**, **auto-roll** (TWAP), and **paper settlement**.  
**Events**: `ROLL`, `SETTLEMENT`.

---

### 1.15 **NEW** tradingbot/core/options_lifecycle.py
**Purpose**: **Open gate** near expiry, **close/roll** before cutoff, **paper expiry** (ITM exercise / assignment; OTM expire).  
**Events**: `EXERCISE`, `ASSIGNMENT`, `EXPIRY`.

---

### 1.16 **NEW** tradingbot/core/funding_accrual.py
**Purpose**: Perps **funding** accrual (schedule); posts pay/receive to PnL (paper mirrors live).  
**Consumers**: scheduler.

---

### 1.17 **NEW** tradingbot/core/scheduler.py
**Purpose**: Tick jobs (contract refresh, funding accrual, reconciler).

---

### 1.18 **UPDATED** tradingbot/core/risk_manager.py
**Purpose**: **Mandatory TP/SL brackets**; per-asset caps (loss/order, drawdown, concurrency); trailing stop & TP ladder utilities.

---

## 2) Broker Adapters

### 2.1 tradingbot/brokers/bybit_adapter.py
**Purpose**: Live Bybit via ccxt.  
**Ctor**: `BybitAdapter(api_key=None, api_secret=None)`  
**Functions**: `wallet()`, `positions()`, (planned) `fetch_my_trades()` normalization hook.  
**Consumers**: runtime_api, (future) live history ingestion.

### 2.2 tradingbot/brokers/ibkr_adapter.py
**Purpose**: Live IBKR via ib_insync.  
**Ctor**: `IBKRAdapter(host="127.0.0.1", port=7497, client_id=2)`  
**Functions**: `wallet()`, `positions()`, (planned) executions → normalized live history.  
**Consumers**: runtime_api, (future) live history ingestion.

> **Paper parity for IBKR**: paper orders **never** hit IBKR; paper uses IBKR quotes (read-only) and the **local simulator** via `PaperRouter`.

---

## 3) Strategy & Training

### 3.1 tradingbot/strategies/manager.py
**Purpose**: Registry & toggles.  
**File**: `state/strategies/strategies.json`  
**Functions**: `list()`, `start()`, `stop()`  
**Consumers**: `ui/app.py` endpoints.

### 3.2 tradingbot/training/train_manager.py
**Purpose**: Orchestrates ML/RL runners, persistence & auto-resume.  
**Artifacts**:  
**Models**: `models/{asset}/ml|rl/.../checkpoints/*.json`  
**Metrics**: `metrics/{asset}/train_ml.jsonl`, `metrics/{asset}/train_rl.jsonl`  
**Classes**: `_BaseRunner`

### 3.3 **NEW** training/action_masking.py  
Blocks illegal actions (size/leverage/session/regime).  

### 3.4 **NEW** training/reward_shaping.py  
PnL minus penalties (drawdown, SL hits, fees, exposure).  

### 3.5 **NEW** training/shadow_canary.py  
Shadow vs baseline; **canary** live micro-size with **tripwires**; promotion/degrade integration.

---

## 4) UI & API

`ui/app.py` exposes REST endpoints used by the dashboard.

- **History**: `GET /history/{asset}?mode=paper|live` → 8 tables (asset×mode)  
- **Positions/Asset**: backing runtime API  
- **Strategy controls**: list/start/stop  
- **Health**: adapters up/down

ASCII diagrams show flow from History Store (paper/live JSONL) to endpoints and tables.

---

## 5) End-to-End Sequences (concise)

### 5.1 Paper Order
Strategy → build `OrderContext` (+snapshot) → Router: **guardrails → clamp → paper simulate** → write `state/paper/trades_{asset}.jsonl` → close & budget rollover (optional) → UI pulls `/history`, `/positions`, `/asset`.

### 5.2 Live Order
Same start → send to adapter (Bybit/IBKR) → on failure enqueue **retry with revalidation only** → on fill write `state/live/{broker}/trades_{asset}.jsonl` → close & rollover (optional) → UI updates.

---

## 6) Explicit Connection Map (who calls whom)

UI → `ui/app.py` → `runtime_api.*`, `strategy_manager.*`, `train_manager.*`  
Router → `budget_manager.*` (pre-checks & sizing) → **broker/paper** → `history_store` (writes)  
runtime_api → adapters (BybitAdapter, IBKRAdapter) + paper_state + history_store  
budget_manager → runtime_api (status/positions) (+ history_store for PnL ladder)  
train_manager → `live_enabled()` guard  
adapters ↔ external brokers (Bybit/IBKR)

No module is orphaned; every dependency has a reverse consumer via UI endpoints & router utilization.

---

## 7) Folder Structure (ASCII)

tradingbot/
├─ core/
│ ├─ runtime_api.py
│ ├─ order_router.py
│ ├─ budget_manager.py
│ ├─ history_store.py
│ ├─ paper_state.py
│ ├─ retry.py
│ ├─ loggerconfig.py
│ ├─ exchange_conformance.py # NEW
│ ├─ routing.py # NEW
│ ├─ paper_execution.py # NEW
│ ├─ persistence.py # NEW
│ ├─ reconciler.py # NEW
│ ├─ contract_catalog.py # NEW
│ ├─ market_calendars.py # NEW
│ ├─ futures_lifecycle.py # NEW
│ ├─ options_lifecycle.py # NEW
│ ├─ funding_accrual.py # NEW
│ └─ scheduler.py # NEW
├─ brokers/
│ ├─ bybit_adapter.py
│ └─ ibkr_adapter.py
├─ strategies/
│ ├─ manager.py
│ └─ strategy_.py
├─ training/
│ ├─ train_manager.py
│ ├─ ml_trainer.py (planned)
│ ├─ rl_trainer.py (planned)
│ ├─ action_masking.py # NEW
│ ├─ reward_shaping.py # NEW
│ └─ shadow_canary.py # NEW
├─ state/
│ ├─ budgets.json
│ ├─ runtime.json
│ ├─ contracts.json # NEW
│ ├─ paper/
│ │ ├─ trades_crypto_spot.jsonl
│ │ ├─ trades_crypto_futures.jsonl
│ │ ├─ trades_forex.jsonl
│ │ └─ trades_options.jsonl
│ └─ live/
│ ├─ bybit/
│ │ ├─ trades_crypto_spot.jsonl
│ │ └─ trades_crypto_futures.jsonl
│ └─ ibkr/
│ ├─ trades_forex.jsonl
│ └─ trades_options.jsonl
├─ models/
│ └─ {asset}/ml|rl/.
├─ metrics/
│ └─ {asset}/train_ml.jsonl / train_rl.jsonl
├─ ui/
│ ├─ app.py
│ ├─ templates/
│ │ └─ dashboard.html
│ └─ static/
│ ├─ wire.js
│ └─ nethealth.js
└─ logs/

markdown
Copy code

---

## 8) Key Variables & JSON fields (quick reference)

- **OrderContext**: `asset_type, symbol, side, quantity, price, client_order_id, strategy_id, mode`  
- **Trade row**: `trade_id, order_id, strategy_id, mode, asset, symbol, side, qty, avg_price, fees, slippage, realized_pnl, realized_pnl_pct, reward, opened_at, closed_at, duration_s, venue, account_id, quote_snapshot_id, status`  
- **Budget config**: `alloc_mode`, `alloc{...}`, `per_trade_risk_pct{...}`, `percent_sizing_threshold_usd`, `percent_sizing_above_threshold{...}`, `profit_rollover`, `enforce_wallet_available`, `max_concurrent`, `scale{...}`, `leverage_caps{...}`

---

## 9) Notes

- `state/` and `logs/` are **.gitignored**.  
- Only `models/` and `metrics/` may be versioned selectively if you want reproducibility.  
- Everything else is source or UI.

Anchors to your original Modules content preserved above: Core map (runtime_api, router, budget, history), End-to-End flow & connection map, folder structure & quick-ref variabl