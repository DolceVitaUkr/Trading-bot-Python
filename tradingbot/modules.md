MODULES.md — Responsibility Map & Extension Rules

## Module Versions (authoritative)
### Core Architecture Modules
- core/order_router.py ........ v1.00
- core/sl_tp_manager.py ...... v1.00  
- core/risk_manager.py ....... v1.00
- core/budget_manager.py ..... v1.00
- core/exposure_manager.py ... v1.00
- core/pnl_reconciler.py ..... v1.00
- core/app_runtime.py ........ v1.00
- core/strategy_registry.py .. v1.00
- core/exploration_manager.py  v1.00
- core/strategy_scheduler.py . v1.00
- core/session_manager.py .... v1.00
- core/bankroll_manager.py ... v1.00
- core/symbol_universe.py .... v1.00
- core/diff_engine.py ........ v1.00
- core/trade_executor.py ..... v1.01

### Existing Core Modules (Legacy Naming)
- core/configmanager.py ...... v1.00
- core/datamanager.py ........ v1.00
- core/paper_trader.py ....... v1.00
- core/validation_manager.py . v1.00
- core/strategy_manager.py ... v1.00
- core/pair_manager.py ....... v1.00
- core/runtime_controller.py . v1.00
- core/telegrambot.py ......... v1.00

### Validation Modules
- validation/online_validator.py v1.00
- validation/promotion_gate.py v1.00

### v1.02 — 2025-08-29 (File Cleanup)
- REMOVED: core/tradeexecutor.py (deprecated wrapper)
- REMOVED: core/pairmanager.py (compatibility wrapper)
- REMOVED: core/runtimecontroller.py (compatibility wrapper) 
- REMOVED: core/validationmanager.py (compatibility wrapper)
- REMOVED: core/errorhandler.py (broken import, unused)
- File naming standardized: Removed all duplicate wrapper files

### v1.01 — 2025-08-29
- core/trade_executor.py: REFACTOR - removed direct broker calls, now routes all orders through order_router only
- core/session_manager.py: NEW - paper session control ($1,000 start), reward reset, session resume after restart
- core/bankroll_manager.py: NEW - auto top-ups (+$1000 when ≤$10), epoch logging, session bankroll tracking
- core/symbol_universe.py: NEW - crypto universe selection (20-40 pairs by 24h volume), refresh timer, spike detector
- core/diff_engine.py: NEW - dry-run diff engine for paper vs hypothetical live comparison audits
- core/interfaces.py: ENHANCED - added Asset, OrderSide, OrderType enums and SessionState dataclass

### v1.00 — 2025-08-28
- core/order_router.py: NEW - centralized order placement with pretrade pipeline (risk→budget→exposure), idempotent client_order_id, broker routing
- core/sl_tp_manager.py: NEW - server-side SL/TP management, Bybit trading-stop, IBKR brackets, reconnect recovery
- core/risk_manager.py: NEW - position sizing, risk checks, drawdown throttling, per-asset caps
- core/budget_manager.py: NEW - asset allocation enforcement, software wallet splits, P&L compounding within buckets
- core/exposure_manager.py: NEW - correlation limits, cluster caps, portfolio exposure control
- core/pnl_reconciler.py: NEW - broker truth reconciliation, position/fill tracking, desync detection
- core/app_runtime.py: NEW - lifecycle orchestration, paper/live toggle, graceful shutdown, main loop
- core/strategy_registry.py: NEW - strategy states/flags/counters, lifecycle management, SQLite persistence
- core/exploration_manager.py: NEW - candidate rotation, fairness scheduling, continuous exploration enforcement
- core/strategy_scheduler.py: NEW - opportunity→strategy mapping with filtering
- validation/online_validator.py: NEW - ≥100 trades validation gate on live feed
- validation/promotion_gate.py: NEW - final promotion criteria (PF≥1.5, Sharpe≥2.0, DD≤15%, etc)

🔎 TL;DR (Quick Recap)

Single Source of Truth (SSOT)

Data → core/data_manager.py (feeds, caching, resampling); core/symbol_universe.py (what to watch).

Capital → core/budget_manager.py (per-asset allocations on Bybit UTA & IBKR).

Risk → core/risk_manager.py (per-trade checks) + core/exposure_manager.py (portfolio/correlation).

Orders → core/order_router.py (the only placer/canceller) + core/sl_tp_manager.py (server-side TP/SL attach).

P&L → core/pnl_reconciler.py (broker-truth final numbers).

Lifecycle → core/app_runtime.py (modes, orchestration).

Paper sessions → core/session_manager.py (start/reset/resume) + core/bankroll_manager.py (auto top-ups).

Exploration & Scheduling → core/exploration_manager.py, core/strategy_registry.py, core/strategy_scheduler.py.

Validation → validation/* only.

Learning → learning/* only.

UI / API → ui/* only.

Broker I/O → brokers/* only.

Who can place orders?

Only order_router.py. Everyone else requests via its API.

Where do we attach SL/TP?

Only sl_tp_manager.py (server-side when possible). It is called by order_router.py.

Where are per-asset budgets enforced?

Only budget_manager.py. order_router must ask it before any order.

Where are risk/exposure checks done?

Only risk_manager.py + exposure_manager.py. order_router must get a green light first.

Where is the universe decided (20–40 crypto pairs)?

Only symbol_universe.py.

Where is paper bankroll top-up logic?

Only bankroll_manager.py.

Where are strategy states & flags?

Only strategy_registry.py (states/flags) + exploration_manager.py (rotation) + strategy_scheduler.py (who gets next opportunity).

When adding new logic

Add within one owner module (listed below).

Others call that owner via its public API.

If you can’t place it in a single existing owner → create a new owner module, then register its API & call rules here.

1) Core Orchestration
1.1 core/app_runtime.py — Lifecycle Orchestrator

Owns: global lifecycle (boot, paper/live toggle), engine startup/stop, reconciliation cycles, graceful shutdowns.
Public API (examples):

start_all(), stop_all()

enable_live(asset), disable_live(asset)

tick() (main loop; schedules exploration, polling, reconciliations)
Calls: config_manager, data_manager, symbol_universe, exploration_manager, strategy_scheduler, order_router, pnl_reconciler, session_manager, budget_manager, risk_manager, ui/api_server.
Never does: strategy math, order placement, SL/TP.

Deep-dive:

Coordinates paper/live engines per asset.

Ensures continuous exploration (keeps active candidate count per asset).

Ensures online reconciliation with brokers after reconnect/restart.

Triggers validation transitions when a candidate hits thresholds.

1.2 core/config_manager.py — Config SSOT

Owns: loading, Pydantic validation, hot-reload of JSON/YAML configs.
Public API: get(section), typed getters (get_assets(), get_risk(asset), etc.).
Calls: filesystem only.
Never does: business logic, math, I/O with brokers.

Deep-dive:

Normalizes and version-tags configs.

Rejects invalid configs early (types, ranges).

2) Data & Universe
2.1 core/data_manager.py — Live Feeds, Cache, Resample

Owns: market data ingestion (WebSocket/REST), normalized ticks/candles, local cache, resampling, rate-limit safety.
Public API:

subscribe(symbols, streams)

get_last_quote(symbol), get_ohlc(symbol, timeframe, lookback)

on_tick(callback) (event bus)
Calls: brokers/* market data endpoints only.
Never does: strategy signals, order placement.

Deep-dive:

WebSockets preferred; fallback to polling with throttling.

Consistent timestamps, timezone normalization.

Provides identical features to training & live (via feature_store).

2.2 core/symbol_universe.py — What to Watch (20–40)

Owns: crypto universe selection & refresh rules (rank by 24h turnover, filters).
Public API:

get_symbols(asset)

refresh_crypto_universe()
Calls: brokers/bybit_client for tickers/instruments.
Never does: trading, risk.

Deep-dive:

Keeps a stable core + rotating tail.

Spike detector outputs candidates to strategy_scheduler.

3) Capital & Risk
3.1 core/budget_manager.py — Allocations (Bybit UTA + IBKR)

Owns: per-asset virtual sub-wallets, dynamic reallocation (+$100 steps), available equity queries.
Public API:

get_alloc(asset) -> usd

can_afford(asset, order_spec) -> bool, reason

apply_pnl(asset, realized_pnl) (paper)
Calls: none (pulls balances via pnl_reconciler updates).
Never does: order placement.

Deep-dive:

Software-enforces futures vs spot separation.

Rejects orders exceeding allocation.

Handles paper equity initialization & updates (with bankroll_manager).

3.2 core/risk_manager.py — Per-Trade Risk Checks

Owns: sizing constraints, SL distance validation, daily loss caps, per-strategy caps, leverage caps.
Public API:

pretrade_check(order_context) -> pass|fail, reason

compute_size(order_context) -> qty
Calls: exposure_manager (for portfolio checks).
Never does: price fetching, order placement.

Deep-dive:

Implements the equity-at-risk / SL-distance formula.

Drawdown throttling of risk_fraction.

Asset-specific bands (spot > futures > options are tightest).

3.3 core/exposure_manager.py — Portfolio/Correlation

Owns: aggregate exposure, symbol clusters, correlation caps, concurrency limits.
Public API:

can_open(symbol, side, size) -> pass|fail, reason

current_exposure() -> dict
Calls: pnl_reconciler for live positions.
Never does: size math, order placement.

Deep-dive:

Cluster definitions (e.g., BTC/L1s; USD majors).

Caps like “max 2 concurrent in cluster” or “max 30% alloc in cluster”.

4) Orders, SL/TP & Reconciliation
4.1 core/order_router.py — The Only Order Placer

Owns: idempotent order submission, modify/cancel, route to broker adapters, attach SL/TP via sl_tp_manager.
Public API:

place_order(order_context) -> order_id

cancel_order(order_id)

amend_order(order_id, fields)
Calls: budget_manager, risk_manager, exposure_manager, sl_tp_manager, brokers/*.
Never does: strategy logic, sizing rules.

Deep-dive (call sequence):

risk_manager.pretrade_check → fail fast.

budget_manager.can_afford

exposure_manager.can_open

Route to brokers/bybit_client or brokers/ibkr_client.

On fill → sl_tp_manager.attach() (server-side).

Persist intents & results; emit events.

4.2 core/sl_tp_manager.py — Server-Side Protections

Owns: TP/SL creation/attachment on broker, correctness per asset (e.g., mark price triggers on futures).
Public API:

attach(broker_order, sl_tp_spec)

sync(position_state) (re-attach/recover after reconnect)
Calls: brokers/*.
Never does: order entry.

Deep-dive:

Bybit: trading-stop API for linear/inverse; spot semantics respected.

IBKR: bracket orders; mirrors OCO lifecycle.

4.3 core/pnl_reconciler.py — Broker-Truth P&L

Owns: P&L/fee reconciliation, position truth, divergence detection (broker_desync flag).
Public API:

pull_positions()

pull_fills()

reconcile() → emits authoritative balances & PnL deltas
Calls: brokers/*.
Never does: strategy logic, order entry.

Deep-dive:

If divergence > tolerance → raise registry flag; halt new entries for that asset.

5) Sessions, Bankroll & Diff
5.1 core/session_manager.py — Paper Session Control

Owns: paper start ($1,000), reward reset (0), resume open paper trades after restarts.
Public API:

start_session()

resume_session()
Calls: bankroll_manager, pnl_reconciler.
Never does: scheduling, order placement.

5.2 core/bankroll_manager.py — Auto Top-Ups

Owns: paper equity top-ups (+$1000 when ≤$10), epoch logging.
Public API:

ensure_min_equity()

record_epoch()
Calls: session_manager.
Never does: trading.

5.3 core/diff_engine.py — Dry-Run Diff

Owns: compute “what live would have done vs paper” for audits.
Public API:

compare(asset, since_ts) → diff report
Calls: data_manager, order_router (simulation mode), pnl_reconciler.
Never does: live order placement.

6) Exploration, Registry & Scheduling
6.1 core/strategy_registry.py — States, Flags, Counters

Owns: the registry of strategies (id, asset, params hash, state, flags, counters, reasons).
Public API:

create(strategy_descriptor)

set_state(id, state)

add_flag(id, flag), remove_flag(id, flag)

inc_counter(id, name)

get_for(asset, state|flags)
Calls: storage only (SQLite/JSON).
Never does: trading or validation logic.

6.2 core/exploration_manager.py — Candidate Rotation

Owns: per-asset active candidate count, fairness, quotas, continuous backfill of new candidates.
Public API:

ensure_active_candidates(asset)

promote_candidate(id) (to validating)

retire_candidate(id, reason)
Calls: strategy_registry, strategy_generator (via learning/ml/...).
Never does: order placement.

Deep-dive:

Implements round_robin_with_boost rotation; max_trades_per_hour_per_candidate; cooldown after trade.

Keeps exploration perpetual.

6.3 core/strategy_scheduler.py — Who Gets the Next Trade?

Owns: mapping of opportunities → candidate strategies, respecting flags, counters, cooling, and fairness.
Public API:

next_candidate(opportunity) -> strategy_id | None
Calls: strategy_registry, exploration_manager.
Never does: signal generation or order placement.

Deep-dive:

An “opportunity” = (asset, symbol, timestamp, features, signal_strength).

Rejects candidates with any blocking flag.

7) Learning (ML & RL)

Only learning modules generate/fit strategies. They do not place orders.

7.1 learning/features/indicator_pipeline.py

Owns: compute RSI/MFI/EMA/SMA/ATR/BB/Fib, derive features; same code for train & serve.
API: build_features(df) -> feature_df

7.2 learning/features/feature_store.py

Owns: schema/versioning; retrieval by symbol/timeframe; consistency between train & live.
API: save(symbol, tf, df), load(symbol, tf, span)

7.3 learning/ml/strategy_generator.py

Owns: emit parameterized strategy specs (simple→complex); param ranges from config/ml_search.json.
API: generate(asset) -> strategy_descriptor

7.4 learning/ml/train_ml_model.py

Owns: fit models (sklearn, xgb, lgbm, torch) using triple-barrier labels; export artifacts.
API: train(spec) -> model_artifact_id

7.5 learning/ml/predict_ml_model.py

Owns: inference; returns signal_strength, side, optional confidence.
API: predict(model_id, features) -> decision

7.6 learning/ml/hyperopt_ml.py

Owns: optuna/DEAP searches over spec param space; produces top-K candidate specs.
API: search(asset) -> [strategy_descriptor]

7.7 learning/rl/env_*.py (spot/futures/forex/options)

Owns: per-asset RL environment dynamics (fees, latency, leverage, SL/TP presets).
API: Gym-like.

7.8 learning/rl/train_rl_agent.py

Owns: train PPO/SAC/DQN per asset; checkpoints; replay.
API: train(policy_spec) -> policy_id

7.9 learning/rl/policy_manager.py

Owns: load/save policies; epsilon/exploration schedule; per-asset policy selection.
API: select_policy(asset), act(policy_id, obs) -> action

7.10 learning/rl/ope_evaluator.py

Owns: off-policy evaluation (WIS/DR) for RL candidates pre-validation.
API: evaluate(policy_id, dataset) -> metrics

8) Validation

Only validation modules decide pass/fail for promotion. They do not place orders.

8.1 validation/online_validator.py

Owns: phase-1 online check: ≥100 closed paper trades per candidate on live feed (SL/TP enforced).
API: start(id), status(id), done(id) -> pass|fail

8.2 validation/backtester.py

Owns: event-driven sim: realistic fills, fees, slippage, latency, partial fills.
API: run(spec, data) -> trades, metrics

8.3 validation/walkforward.py

Owns: purged k-fold + embargo; aggregate fold metrics.
API: run(spec, data) -> metrics

8.4 validation/stress_tests.py

Owns: MC reorder, ±param shocks, slippage/latency shocks, regime slicing.
API: stress(spec, trades) -> stress_metrics

8.5 validation/promotion_gate.py

Owns: final criteria (PF, Sharpe, MaxDD, CVaR), broker-truth reconciliation tolerance.
API: decide(metrics) -> pass|fail, reasons

8.6 validation/report_builder.py

Owns: JSON + HTML reports under logs/validation/{strategy_id}.
API: build(id, metrics, trades) -> paths

9) UI & Notifications
9.1 ui/api_server.py

Owns: FastAPI endpoints (status, toggles, flags, reallocations, reports).
API routes (examples):

GET /status, POST /live/{asset}/enable, POST /kill/{asset},

GET /validation/{id}, POST /strategy/{id}/flag/{flag},

POST /alloc/{asset}/add/100.

9.2 ui/dashboard.py, ui/components.py

Owns: two horizontal charts (Paper & Live, 4 lines each + totals); asset panels with balances, reward, Open & History tables (sortable/collapsible), live quotes (rate-safe).
API: server-render helpers; websocket push.

9.3 notifications/telegram_bot.py

Owns: lifecycle/trade alerts, validation pass/fail, kill switch, flags set/cleared; minimal controls mirroring UI.

10) Brokers
10.1 brokers/bybit_client.py

Owns: Bybit V5 REST/WS; instrument/tickers; order ops; trading-stop for SL/TP; positions & fills.
API: place_order(), cancel(), set_trading_stop(), get_positions(), ws_subscribe()

10.2 brokers/ibkr_client.py

Owns: IBKR TWS/Web API; market data; bracket orders; pacing limits; reconnect; positions & fills.
API: analogous to above.

10.3 brokers/broker_common.py

Owns: retries, backoff, idempotency (client order IDs), clock sync, error normalization.

11) Tests

Unit: risk sizing, exposure caps, budget enforcement, SL/TP attach requests, registry transitions, exploration fairness.

Integration: mock brokers/testnet; order life cycle; reconciliation; UI routes.

Regression: fixed backtest snapshots; performance guardrails.

12) Anti-Duplication Rules (Golden Rules)

Only order_router talks to brokers for orders.

If you need to place/modify/cancel: call order_router.

If you need SL/TP: order_router → sl_tp_manager (never do it elsewhere).

Only risk_manager does per-trade risk math.

Do not re-implement sizing or SL distance checks elsewhere.

Only exposure_manager governs portfolio/correlation caps.

Do not compute exposure rules in strategies.

Only budget_manager decides if you can afford a trade.

Do not check alloc/balance anywhere else.

Only pnl_reconciler is P&L truth.

Any P&L/UI must reconcile to its numbers. If mismatch → raise broker_desync.

Only symbol_universe picks crypto universe.

Do not select symbols inside strategies.

Only exploration_manager/strategy_scheduler assign opportunities.

Strategies produce signals; schedulers decide who trades.

Only validation/* decides promotion.

Do not auto-promote from exploration.

Only session_manager/bankroll_manager control paper equity & top-ups.

Do not mutate paper equity elsewhere.

UI reads; Core decides.

UI never makes business decisions—only triggers APIs.

13) Extension Checklist (When adding “extra” logic)

Pick the owner from this file (or create a new one and add it here).

Define a public API (function names/args).

Wire call flow: callers → owner → (sub-calls).

Write tests (unit + integration).

Document the API and update this MODULES.md.

No duplicate checks: if your logic smells like risk/budget/exposure/orders, you’re in the wrong module—route it.

14) Example Call Flow (Entry → Filled with SL/TP)

strategy_scheduler selects strategy_id for an opportunity.

Strategy (ML/RL) proposes side, signal_weight, preliminary sl/tp.

order_router builds order_context and calls:

risk_manager.pretrade_check(context)

budget_manager.can_afford(context)

exposure_manager.can_open(symbol, side, size)

If all pass → order_router.place_order() → brokers/*

On fill → sl_tp_manager.attach(broker_order, sl_tp_spec)

pnl_reconciler later reconciles fills/positions → updates UI & budgets.
