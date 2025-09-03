Modules.md

Version 1.0

This document maps the bot’s modules, key functions, important variables, and exact connections so that every data producer has a consumer and every consumer lists its source. It also includes a high-level flow chart and a folder structure diagram.

0) Global Conventions

Assets (UI keys): crypto_spot, crypto_futures, forex, options

Modes: paper, live

Quote snapshot: captured at submit; each trade row includes quote_snapshot_id

Idempotency: client_order_id = hash(strategy_id, symbol, ts)

Budget terms: B_A (asset budget), r_A (per_trade_risk_pct), m_A (risk multiplier), q_A (% sizing above threshold)

1) Core
1.1 tradingbot/core/runtime_api.py

Purpose: Aggregates wallets/positions/history for UI & services. Health & retry wrappers.
Imports:

from . import paper_state (internal paper wallet/positions)

from tradingbot.brokers.bybit_adapter import BybitAdapter (lazy)

from tradingbot.brokers.ibkr_adapter import IBKRAdapter (lazy)

from tradingbot.core.retry import retry_call

from tradingbot.core import history_store

Key module-level singletons (lazy):

_BYBIT: Optional[BybitAdapter] via _bybit()

_IBKR: Optional[IBKRAdapter] via _ibkr()

Functions (inputs → outputs → calls):

aggregate_status(asset: str) -> dict

→ {"asset", "paper": {total, available, used, unrealized_pnl}, "live": {...}}

Calls: paper_state.get_wallet(asset), retry_call(by.wallet) or retry_call(ib.wallet)

aggregate_positions(asset: str) -> dict

→ {"paper": [..], "live": [..]}

Calls: paper_state.list_positions(asset), retry_call(by.positions) or retry_call(ib.positions)

read_trade_history(asset: str, mode: Optional[str], limit: int, since: Optional[str]) -> list[dict]

→ Normalized list of trades (paper/live)

Calls: history_store.read_history(...)

health() -> dict

→ { "bybit": {ok, error?}, "ibkr": {ok, error?} }

Calls: retry_call(by.wallet), retry_call(ib.wallet)

live_enabled(asset: str) -> bool

→ Checks tradingbot/state/runtime.json for live flag

Consumers: ui/app.py, core/order_router.py, UI JS (via HTTP).

1.2 tradingbot/core/order_router.py

Purpose: Central order flow (validations, guardrails, idempotency, submit, retry).
Imports:

from .budget_manager import can_place_order as _can_place_order

try: from .budget_manager import position_size_cap as _pos_cap

from tradingbot.core.runtime_api import aggregate_status, aggregate_positions (indirect via budget)

Important types / fields (typical):

OrderContext:

asset_type: Literal["spot","futures","forex","options"]

symbol: str

side: Literal["BUY","SELL"]

quantity: float

price: Optional[float] (from quote snapshot)

client_order_id: str

strategy_id: str

mode: Literal["paper","live"]

OrderResult:

success: bool

reason?: str

broker_order_id?: str

Core function:

place_order(context: OrderContext) -> OrderResult

Fresh quote snapshot attached in the calling layer → context.price must be ≤2s old; store quote_snapshot_id in trade log.

Guardrails via _can_place_order(...) (budget, wallet, positions, concurrency).

Clamp sizing via _pos_cap(...) → shrink context.quantity if over cap.

Venue checks (min qty/step, long-only on spot, leverage caps).

Idempotency check (reject duplicates by client_order_id).

Submit to paper engine or broker adapter.

Retry queue (if broker failure): revalidate strategy on reconnect; only submit if conditions still hold; never backfill.

Producers: Broker adapters / paper engine.
Consumers: History writer, UI (through history API).

1.3 tradingbot/core/budget_manager.py

Purpose: Budget enforcement & position sizing.
State/Files: tradingbot/state/budgets.json, tradingbot/state/risk_state.json
Imports:

from tradingbot.core.runtime_api import aggregate_status, aggregate_positions

try: from tradingbot.core import history_store (for realized PnL ladder)

Key configuration fields (budgets.json):

alloc_mode: "absolute" | "percent"

alloc: {crypto_spot, crypto_futures, forex, options} (USD if absolute; [0..1] if percent)

per_trade_risk_pct: {...}

percent_sizing_threshold_usd: number

percent_sizing_above_threshold: {...}

profit_rollover: bool

enforce_wallet_available: bool

max_concurrent: int

scale: {enabled, equity_curve_window_days, ladder:[{profit_usd, risk_multiplier}]}

leverage_caps: {...}

Functions:

get_alloc(asset_ui_key: str) -> float (USD; percent mode derives from live wallet)

count_open_positions(asset_ui_key: str, mode: str) -> int

set_cooldown(asset_ui_key: str, seconds: int) / in_cooldown(asset_ui_key: str) -> bool

position_size_cap(asset_ui_key: str, price: float) -> tuple[cap_notional, cap_qty]

Computes BaseCap (B_A * r_A * m_A) and switches to % sizing above $1,000:

PercentCapUSD = max(1000, B_A * q_A * m_A)

suggest_position_size(asset_ui_key: str, price: float, min_qty: float|None, step: float|None) -> float

can_place_order(asset_ui_key, mode, symbol, side, quantity, price, asset_type, broker) -> (ok: bool, reason: str)

Checks cooldown, budget, concurrency, wallet available (if enabled)

Consumers: core/order_router.py, UI sizing hint.

1.4 tradingbot/core/history_store.py

Purpose: Normalized trade history across paper/live and assets.
Files:

Paper: tradingbot/state/paper/trades_{asset}.jsonl

Live: tradingbot/state/live/bybit/trades_{asset}.jsonl, tradingbot/state/live/ibkr/trades_{asset}.jsonl

Schema (per row):

trade_id, order_id, strategy_id, mode, asset, symbol, side,
qty, avg_price, fees, slippage,
realized_pnl, realized_pnl_pct,
reward, opened_at, closed_at, duration_s,
venue, account_id, quote_snapshot_id, status


Functions:

_read_jsonl(path) -> list[dict]

_norm_row(row, mode_hint) -> dict

read_history(asset: str, mode: Optional[str], since: Optional[str], limit: int) -> list[dict]

Consumers: runtime_api.read_trade_history, UI history.

1.5 tradingbot/core/retry.py

retry_call(func, *args, retries=3, backoff=0.5, max_backoff=4.0, exceptions=(Exception,), jitter=True, **kwargs)
Consumers: runtime_api, adapters, health checks.

1.6 tradingbot/core/loggerconfig.py

setup_logging(level: str|int = None) -> None
Consumers: ui/app.py at import time.

2) Broker Adapters
2.1 tradingbot/brokers/bybit_adapter.py

Purpose: Live Bybit via ccxt.
Constructor: BybitAdapter(api_key=None, api_secret=None) → ccxt.bybit with sandbox=False
Functions:

wallet() -> dict{total, available, used, unrealized_pnl}

positions() -> dict{"spot":[...], "futures":[...]}

(TODO) fetch_my_trades() normalization hook for live history

Consumers: runtime_api, (future) history ingestion.

2.2 tradingbot/brokers/ibkr_adapter.py

Purpose: Live IBKR via ib_insync.
Constructor: IBKRAdapter(host="127.0.0.1", port=7497, client_id=2)
Functions:

wallet() -> dict{total, available, used, unrealized_pnl} (via accountSummary)

positions() -> list[dict]

(TODO) executions → normalized live history

Consumers: runtime_api, (future) history ingestion.

3) Strategy & Training
3.1 tradingbot/strategies/manager.py

Purpose: Registry & toggles.
File: tradingbot/state/strategies/strategies.json
Functions:

list(asset: str) -> list[dict] (filters by asset; includes params, performance)

start(asset: str, sid: str) -> bool

stop(asset: str, sid: str) -> bool

Consumers: ui/app.py endpoints /strategies/{asset}, /strategy/{asset}/{sid}/start|stop.

3.2 tradingbot/training/train_manager.py

Purpose: Orchestrates ML/RL runners with persistence & auto-resume.
Artifacts:

Models: tradingbot/models/{asset}/ml|rl/.../checkpoints/*.json

Metrics: tradingbot/metrics/{asset}/train_ml.jsonl / train_rl.jsonl

Classes: _BaseRunner, _MLRunner, _RLRunner
Functions:

start(asset: str, mode: str["ml"|"rl"]) -> dict (blocks if live_enabled(asset) is true)

stop(asset, mode) -> dict

status(asset) -> dict{"ml":..., "rl":...}

Consumers: ui/app.py endpoints /train/{asset}/{mode}/start|stop|status.

4) UI / API
4.1 tradingbot/ui/app.py (FastAPI)

Purpose: HTTP interface for the dashboard.
Imports: runtime_api, strategy_manager, train_manager, setup_logging

REST endpoints (→ calls):

GET /status → runtime_api.health()

GET /asset/{asset} → runtime_api.aggregate_status(asset)

GET /positions/{asset} → runtime_api.aggregate_positions(asset)

GET /history/{asset}?mode=&since=&limit= → runtime_api.read_trade_history(...)

POST /paper/{asset}/enable|disable → paper state toggle (internal)

POST /live/{asset}/enable|disable → sets live flag (double-confirm UI)

GET /strategies/{asset} → strategy_manager.list(asset)

POST /strategy/{asset}/{sid}/start|stop → strategy_manager.start/stop

POST /train/{asset}/{mode}/start|stop → train_manager.start/stop

GET /train/{asset}/status → train_manager.status(asset)

Consumers: Frontend JS (wire.js / nethealth.js), browser.

4.2 tradingbot/ui/static/wire.js & nethealth.js

wire.js: fetch wrappers jget/jpost, UI actions → endpoints above

nethealth.js: status polling → banner if Bybit/IBKR unhealthy
Consumers: HTML templates (dashboard).

5) Logging, Scripts
5.1 tradingbot/logs/

Rotating JSON logs via loggerconfig.setup_logging().

5.2 scripts/cleanup_logs.py

Archives logs to artifacts/logs/YYYY-MM-DD/ (keep N days).

6) Data Flow (ASCII Flow Chart)
[Strategy (ML/RL)]
   |  (signal: side + conf)                     [UI toggle Paper/Live]
   v
[OrderContext build] -- fetch quote (<=2s) --> [Quote Snapshot Store]
   |        (attach quote_snapshot_id)
   v
[order_router.place_order(context)]
   |---> [budget_manager.can_place_order] -----> checks: budget, wallet available, positions, cooldown
   |---> [budget_manager.position_size_cap] ---> clamp size (BaseCap or %Cap above $1k)
   |---> venue constraints & leverage caps
   |---> idempotency check (client_order_id)
   |
   +--> if context.mode == "paper":
   |        simulate fill @ snapshot -> [state/paper/trades_{asset}.jsonl]
   |        update paper positions
   |
   +--> if context.mode == "live":
            broker adapter (Bybit/IBKR)
            |--> success: [state/live/{broker}/trades_{asset}.jsonl]
            |--> failure: enqueue retry (only if signal revalidates on reconnect)

   (both)
   |--> if trade closed: update realized PnL; if profit_rollover: increase SAME asset budget
   v
[runtime_api.aggregate_*]  -->  [UI /history, /positions, /asset]
                                 charts (equity & budget), 8 tables (asset×mode)

7) Folder Structure (ASCII Tree)
tradingbot/
├─ core/
│  ├─ runtime_api.py
│  ├─ order_router.py
│  ├─ budget_manager.py
│  ├─ history_store.py
│  ├─ paper_state.py
│  ├─ retry.py
│  └─ loggerconfig.py
├─ brokers/
│  ├─ bybit_adapter.py
│  └─ ibkr_adapter.py
├─ strategies/
│  ├─ manager.py
│  └─ (strategy_*.py)
├─ training/
│  ├─ train_manager.py
│  ├─ ml_trainer.py            (planned)
│  ├─ rl_trainer.py            (planned)
│  └─ validation_manager.py    (planned)
├─ state/
│  ├─ budgets.json
│  ├─ runtime.json
│  ├─ paper/
│  │  ├─ trades_crypto_spot.jsonl
│  │  ├─ trades_crypto_futures.jsonl
│  │  ├─ trades_forex.jsonl
│  │  └─ trades_options.jsonl
│  └─ live/
│     ├─ bybit/
│     │  ├─ trades_crypto_spot.jsonl
│     │  └─ trades_crypto_futures.jsonl
│     └─ ibkr/
│        ├─ trades_forex.jsonl
│        └─ trades_options.jsonl
├─ models/
│  └─ {asset}/ml|rl/...
├─ metrics/
│  └─ {asset}/train_ml.jsonl / train_rl.jsonl
├─ ui/
│  ├─ app.py
│  ├─ templates/
│  │  └─ dashboard*.html
│  └─ static/
│     ├─ wire.js
│     └─ nethealth.js
└─ logs/

8) End-to-End Sequences (concise)
8.1 Paper Order

Strategy emits signal → build OrderContext (+snapshot)

Router: guardrails → clamp → paper simulate → write state/paper/trades_{asset}.jsonl

Close → update realized PnL → optional budget rollover

UI pulls /history/{asset}, /positions/{asset}, /asset/{asset}

8.2 Live Order

1–2 same → send to adapter (Bybit/IBKR)
3. If fail: enqueue retry (only if signal revalidates later)
4. On fill: write state/live/{broker}/trades_{asset}.jsonl
5. Close → realized PnL → optional budget rollover
6. UI updates same endpoints

9) Explicit Connection Map (who calls whom)

UI → ui/app.py → runtime_api.*, strategy_manager.*, train_manager.*

Router → budget_manager.* (pre-checks & sizing) → broker/paper → history_store (writes)

runtime_api → adapters (BybitAdapter, IBKRAdapter) + paper_state + history_store

budget_manager → runtime_api.aggregate_status, runtime_api.aggregate_positions (+ history_store for PnL ladder)

train_manager → live_enabled() guard (from runtime_api)

adapters ↔ external brokers (Bybit/IBKR)

No module is orphaned; every dependency has a reverse consumer path via UI endpoints & router utilization.

10) Key Variables & JSON fields (quick reference)

OrderContext: asset_type, symbol, side, quantity, price, client_order_id, strategy_id, mode

Trade row: trade_id, order_id, strategy_id, mode, asset, symbol, side, qty, avg_price, fees, slippage, realized_pnl, realized_pnl_pct, reward, opened_at, closed_at, duration_s, venue, account_id, quote_snapshot_id, status

Budget config: alloc_mode, alloc{...}, per_trade_risk_pct{...}, percent_sizing_threshold_usd, percent_sizing_above_threshold{...}, profit_rollover, enforce_wallet_available, max_concurrent, scale{...}, leverage_caps{...}

11) ASCII Diagram – History Tables (Asset × Mode)
                 ┌─────────────────────────────────────┐
                 │          History Store              │
                 │  (tradingbot/core/history_store.py) │
                 └─────────────────────────────────────┘
                                │
          ┌─────────────────────┼──────────────────────────┐
          │                     │                          │
   ┌───────────────┐     ┌───────────────┐          ┌───────────────┐
   │ state/paper/  │     │ state/live/   │          │ API Endpoints │
   │  (Paper JSONL)│     │  (Live JSONL) │          │   (ui/app.py) │
   └───────────────┘     └───────────────┘          └───────────────┘
          │                     │                          │
   ┌──────┴───────┐     ┌───────┴─────────┐       ┌────────┴─────────┐
   │ Trades by    │     │ Trades by       │       │ GET /history/{asset}  │
   │ Asset        │     │ Broker & Asset  │       │   ?mode=paper|live    │
   └──────┬───────┘     └────────┬────────┘       └────────┬─────────┘
          │                      │                          │
   ┌──────┴───────────────┐ ┌────┴────────────────┐ ┌────────┴───────────┐
   │ crypto_spot_paper    │ │ crypto_spot_live    │ │   UI History Tab   │
   ├──────────────────────┤ ├─────────────────────┤ ├────────────────────┤
   │ crypto_futures_paper │ │ crypto_futures_live │ │ 8 tables: asset×mode│
   ├──────────────────────┤ ├─────────────────────┤ ├────────────────────┤
   │ forex_paper          │ │ forex_live          │ │ sortable/filterable │
   ├──────────────────────┤ ├─────────────────────┤ ├────────────────────┤
   │ options_paper        │ │ options_live        │ │ totals row (PnL,etc)│
   └──────────────────────┘ └─────────────────────┘ └────────────────────┘

12) End-to-End Flow (Condensed View)
[Strategy ML/RL]
   │ signal (side+conf)
   v
[Order Router]
   │ calls budget_manager.can_place_order()
   │ calls budget_manager.position_size_cap()
   v
[Broker Adapter] or [Paper Engine]
   │ returns execution or error
   v
[History Store] writes JSONL
   │
   v
[runtime_api.read_trade_history()]
   │
   v
[UI /history/{asset}?mode=...]
   │
   v
[Dashboard History Tab] (8 tables)

13) Folder Structure (ASCII)
tradingbot/
├─ core/
│  ├─ runtime_api.py
│  ├─ order_router.py
│  ├─ budget_manager.py
│  ├─ history_store.py
│  ├─ paper_state.py
│  ├─ retry.py
│  └─ loggerconfig.py
├─ brokers/
│  ├─ bybit_adapter.py
│  └─ ibkr_adapter.py
├─ strategies/
│  ├─ manager.py
│  └─ strategy_*.py
├─ training/
│  ├─ train_manager.py
│  ├─ ml_trainer.py (planned)
│  ├─ rl_trainer.py (planned)
│  └─ validation_manager.py (planned)
├─ state/
│  ├─ budgets.json
│  ├─ runtime.json
│  ├─ paper/
│  │  ├─ trades_crypto_spot.jsonl
│  │  ├─ trades_crypto_futures.jsonl
│  │  ├─ trades_forex.jsonl
│  │  └─ trades_options.jsonl
│  └─ live/
│     ├─ bybit/
│     │  ├─ trades_crypto_spot.jsonl
│     │  └─ trades_crypto_futures.jsonl
│     └─ ibkr/
│        ├─ trades_forex.jsonl
│        └─ trades_options.jsonl
├─ models/{asset}/ml|rl/...
├─ metrics/{asset}/train_ml.jsonl / train_rl.jsonl
├─ ui/
│  ├─ app.py
│  ├─ templates/dashboard.html
│  └─ static/{wire.js, nethealth.js}
└─ logs/