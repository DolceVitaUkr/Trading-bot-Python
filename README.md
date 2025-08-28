🚀 World-Class Self-Learning Multi-Asset Trading Bot (Final Spec)
🔎 TL;DR (bullet recap)

Per-strategy exploration ≥100 trades (never stops exploring globally)

Each candidate strategy must complete ≥100 live-sim (paper) trades per strategy instance before validation.

Global exploration never stops: the system continuously spawns/rotates new candidates; no global “100 trade” ceiling.

Fair scheduling: exploration controller rotates trade opportunities across candidates so one strategy can’t hog them.

Paper bankroll policy (unbounded exploration)

Paper sessions start at $1,000 and reward points = 0.

If paper equity hits ≤$10, auto top-up +$1,000 and continue; SL/TP/risk still enforced.

This enables 10,000+ trades over time while respecting risk & limits.

Strategy Registry, lifecycle & flags

Registry tracks every strategy with: id, asset, params hash, counters, metrics, state, flags, reasons.

States: draft → exploring → validating → approved_live → suspended/hold → deprecated/rejected.

Flags: manual_hold, manual_kill, risk_alert, data_issue, rate_limit, broker_desync.

You can flip flags any time (UI/Telegram/code) to pause/stop/hold a specific strategy.

Asset-specific engines (distinct logic & limits)

Crypto Spot | Crypto Futures | Forex | Options each has unique sizing, SL/TP, leverage/fees, sessions, validation gates.

Universe coverage (crypto)

Track 20–40 symbols, ranked by 24h volume/turnover; refresh list on a schedule; spike detector to catch 20–70% movers.

Always-online mainnet data

Paper = current live feed only (no back-dated fills).

SL/TP server-side (Bybit TP/SL; IBKR brackets) so fills survive restarts/disconnects.

Lifecycle

Train & explore ≥100 trades (per candidate) → Validate → Approve → Live, while continuous exploration runs in parallel.

Bybit unified wallet budget splits

Software-enforced sub-wallets (e.g., Spot $900, Futures $100).

Futures never exceeds its allocation; P&L stays within its bucket (compounds internally).

Reconcile all P&L/fees to broker truth.

RL per asset; ML explores simple→complex

RL: PPO/SAC/DQN per-asset, indicators in observation; explore/exploit scheduler.

ML: rule-based → ensembles → GBM/DL; massive param search; meta-labeling.

Risk & limits

SL/TP mandatory. Sizing = equity-at-risk / SL-distance.

Per-asset caps, daily loss caps, leverage caps (futures/options), correlation caps.

UI (FastAPI)

Two horizontal graphs (Paper & Live), each with 4 lines (one per asset) + totals.

Per asset & mode: Totals, Available, Used, Reward, Open & History tables, live prices (rate-limit safe).

Side buttons: Total / Paper / Live per asset; open validation report; toggle flags.

1) Per-Strategy Exploration ≥100 Trades (and continuing exploration)

What this means (no ambiguity):

Unit of exploration = one strategy instance (unique strategy_id identified by code + params hash).

Requirement: each candidate must accumulate at least 100 closed paper trades (with SL/TP) before it’s eligible for validation.

Rolling, not blocking: This does not pause exploration after 100 trades.

When a candidate reaches 100, it moves to validation, and the exploration controller immediately keeps feeding other candidates.

The controller always keeps N candidates active per asset (configurable), replenishing from the generator when any candidate advances or is retired.

Fairness & scheduling:

The controller uses round-robin + priority queues to allocate trade opportunities across candidates so that:

No single candidate monopolizes signals.

Candidates with fewer trades get a slight priority boost to reach 100 in reasonable time.

Candidates that violate risk or hit temporary flags are skipped until cleared.

Config (excerpt config/exploration.json):

{
  "per_asset": {
    "spot":   { "active_candidates": 6, "min_trades_to_validate": 100, "priority_boost_under_trades": 40 },
    "futures":{ "active_candidates": 6, "min_trades_to_validate": 100, "priority_boost_under_trades": 40 },
    "forex":  { "active_candidates": 4, "min_trades_to_validate": 100, "priority_boost_under_trades": 40 },
    "options":{ "active_candidates": 4, "min_trades_to_validate": 100, "priority_boost_under_trades": 40 }
  },
  "opportunity_allocation": {
    "rotation": "round_robin_with_boost",
    "max_trades_per_hour_per_candidate": 12,
    "cooldown_seconds_after_trade": 30
  }
}

2) Paper Bankroll Policy (unbounded exploration with strict risk)

Behavior (exact):

On paper session start: set paper_equity = 1000.00 and reward_points = 0.

During operation: if paper_equity <= 10.00 → auto top-up by +$1000 and continue.

Top-ups don’t reset counters/flags/strategy states; they only replenish equity to allow further exploration.

All SL/TP/risk logic applies identically before and after top-ups; there’s no bypass.

Each top-up is logged as a new “paper_epoch” with sequence number for analytics.

Config (excerpt config/session.json):

{
  "paper": {
    "start_equity_usd": 1000,
    "auto_topup_threshold_usd": 10,
    "auto_topup_amount_usd": 1000,
    "log_epoch": true
  }
}

3) Strategy Registry, Lifecycle & Flags (manual intervention ready)

Registry fields (stored in state/strategy_registry.db / JSON / SQLite):

strategy_id (UUID), asset, version, params_hash

state: one of

draft (generated, not trading yet)

exploring (paper live-sim accumulating trades)

validating (validation suite running)

approved_live (eligible & ready; may be currently live)

suspended (auto/ manual pause; not trading)

hold (manual hold; remains listed, no trading)

rejected (failed validation or policy)

deprecated (older superseded strategy)

flags (set of):

manual_hold — human pause

manual_kill — human immediate stop/flat

risk_alert — risk manager triggered (dd cap, per-trade breach)

data_issue — feed gaps/outliers

rate_limit — broker/API pacing hit

broker_desync — P&L/position mismatch with broker

counters: paper_trades_closed, paper_trades_open, live_trades_closed, epochs_used

metrics_last_24h: PF, Sharpe, MaxDD, Win%, CVaR(5), Avg Trade, etc.

reasons: free-text history of decisions (why held/rejected/promoted)

timestamps: created, first_trade, last_trade, last_validation

Transitions (deterministic):

draft → exploring (when scheduled)

exploring → validating (once paper_trades_closed ≥ min_trades_to_validate)

validating → approved_live (if gates pass) or → rejected (else)

approved_live → suspended/hold (manual or kill) → optionally back to approved_live

Any → deprecated (when superseded by a newer strategy).

Flag effects:

manual_hold: strategy remains visible but receives no new trade allocations.

manual_kill: immediately close paper trades, cancel pending paper orders; mark as suspended.

risk_alert: exploration controller skips until risk manager clears; log reason.

Clearing a flag re-enables scheduling (except rejected/deprecated).

APIs/UI:

POST /strategy/{id}/flag/{flag}; DELETE /strategy/{id}/flag/{flag}

POST /strategy/{id}/state/{state} (restricted to allowed transitions)

Filters in UI: show only exploring, validating, approved_live, etc.

4) Asset-Specific Engines (distinct logic)
4.1 Crypto Spot (Bybit, 20–40 symbols)

Universe: rank by 24h turnover; refresh every 10–30 min; keep a core list + rotating tail.

Signals: RSI/EMA/SMA crossovers, volatility breakouts (ATR/Bollinger width), MFI/volume confirmation, Fib confluence.

Sizing:

qty = (spot_alloc * risk_fraction * signal_weight) / (sl_distance * price)


risk_fraction: 0.25–0.75% of spot_alloc (drawdown-tiered).

signal_weight: weak 0.5 / base 1.0 / strong 1.5.

Exits: TP% or ATR×; optional trailing; time-stop bars.

Risk: no leverage; max_concurrent_symbols; correlation cap across correlated L1s/L2s.

Validation: online paper ≥100 trades → offline WFA + stress (then gates).

4.2 Crypto Futures (Bybit)

Budget: futures_alloc (e.g., $100) from unified wallet; cannot exceed alloc.

Leverage: capped per symbol; prefer low (1–5×) initially.

Triggers: use MarkPrice for SL/TP; funding & basis filters.

Sizing: same formula but risk_fraction tighter (0.15–0.5% of futures_alloc).

Risk: cap concurrent positions; daily realized loss cap (e.g. 2–4% of futures_alloc).

Validation: same 100-trade paper rule, plus stress on latency/slippage.

4.3 Forex (IBKR)

Pairs: majors + selected liquid crosses; session-aware (24/5).

Orders: bracket orders with TP/SL at broker; spread/rollover aware.

Sizing: convert pip SL to notional; risk_fraction 0.1–0.4% of forex_alloc.

Risk: blackout windows around illiquid opens; correlation limits among USD pairs.

Validation: online paper (sim feed over live quotes) ≥100 trades, then WFA.

4.4 Options (IBKR, long-premium first)

Selection: delta 0.35–0.55, DTE 7–30, IV term structure sanity.

Exits: TP +30–60%, SL −40–50% of premium, time-stop near DTE/7.

Risk: max premium/trade, max net theta/day; stagger entries over days.

Validation: event-aware slices (macro prints) + stress on IV crush/gap.

5) Universe & Spikes (Crypto 20–40)

Selection: Bybit tickers (spot/linear) sorted by turnover; require min days listed, min turnover, and quality spread.

Spike detector: rolling % change and z-score on 5m/15m; queue candidates; allocate exploration slots to avoid missing 20–70% moves.

Rate-safe data: use WebSockets for prices; REST for metadata/universe refresh.

6) Learning: ML & RL
ML (Strategy Generator → Candidate Factory)

Space: from simple (RSI/EMA/SMA/ATR/BB/Fib) to complex (stacked ensembles, gradient boosting, light NN).

Labeling: triple-barrier with SL/TP/time; meta-labeling for filter.

Search: grid/random/Bayesian + DEAP (genetic) over: RSI window/levels, EMA fast/slow, ATR mult, TP ladders, time-stops, entry filters, session filters.

Shortlisting: top-K by online paper metrics (PF, Sharpe, MaxDD, win% over last N trades).

RL (Per-asset Policies)

Observations: price returns, RSI/MFI/EMA/ATR/BB width, spread, depth proxies, regime flags.

Actions: {hold/buy/sell/flat, size bucket, SL/TP preset bucket, (futures) leverage bucket}.

Rewards: +k × risk-adjusted P&L; penalties for MaxDD/large adverse excursion; bonus on TP hits; small exploration bonus with hard budget guard.

Explore/exploit: enforced 2:1 conservative:exploratory trade mix per candidate (configurable).

Persistence: replay buffers & checkpoints per asset; OPE used in validation.

7) Validation (Online-preferred) & Gates

Phase-1 (online): Each candidate must close ≥100 paper trades on live feed with SL/TP.

Phase-2 (offline): Event-driven backtest; walk-forward with purge+embargo; stress: MC reorder, ±param shocks, slippage/latency shocks; OPE for RL.

Promotion gate (defaults; per-asset overrideable):

PF ≥ 1.5, Sharpe ≥ 2.0, MaxDD ≤ 15%, CVaR(5%) within bound.

Broker-truth reconciliation difference ≤ tolerance (fees/slippage sanity).

No open risk alerts/flags.

8) Risk, Sizing, Exposure

Sizing (all assets):

position_size = (equity_alloc * risk_fraction * signal_weight) / (sl_distance * price)


risk_fraction tiered by drawdown/equity; futures/options use lower bands.

signal_weight: weak 0.5 / base 1.0 / strong 1.5 (from model confidence/ensemble vote).

Mandatory SL/TP: server-side where possible (Bybit trading-stop; IBKR brackets).

Caps: per-strategy, per-asset, per-portfolio; correlation; max concurrent; daily realized loss cap; leverage caps (futures/options).

Drawdown throttling: auto reduce risk_fraction as DD grows; restore slowly on recovery.

9) Bybit Unified Wallet Splits (software-enforced)

Allocations: e.g., { spot: 900, futures: 100, forex: 0, options: 0 } USD.

Enforcement: engines read only their equity_alloc; reject/resize orders that exceed alloc or caps.

PnL domains: PnL for futures stays in futures alloc; spot PnL stays in spot alloc; unified wallet is the broker source of truth.

Reconciliation: periodic and on events; if discrepancy > tolerance, raise broker_desync flag, pause new entries for that engine until resolved.

Dynamic reallocation: UI/Telegram increments of $100; immediate effect.

10) UI (FastAPI)

Top row: two horizontal charts (Paper, Live). Each chart: 4 lines (Spot/Futures/Forex/Options) + mode total in header.

Per asset & mode panel:

Balances: Total / Available / Used; Reward Points.

Open Positions: Time Open, Side, Amount, Entry, Live Price, P&L $, P&L %.

History: Time Open/Close, Side, Amount, Entry/Exit, P&L $, P&L %, New Balance After Trade.

Sortable, collapsible (accordion), pagination; prices update via WS/polled throttling.

Side buttons: Total / Paper / Live per asset; Open Validation Report; Set Flag (hold/kill) per strategy; Toggle Live; Reallocate +$100.

11) Broker Limits & Reliability

IBKR: throttle to ≤50 req/sec; queue historical requests; reconnect/re-subscribe on nightly resets; all exits via brackets.

Bybit: prefer WebSocket for ticks; REST for order ops; SL/TP via trading-stop; universe via tickers/instruments.

Order idempotency: client_order_id & dedup to avoid double placements on retries.

12) Session Rules (Paper)

Start: $1,000, 0 reward points.

Auto top-up +$1,000 when equity ≤ $10 (log new paper_epoch).

Resume open paper trades on restart; do not re-enter on duplicate signals.

In-progress strategies (e.g., 20/100) continue across restarts.

13) What Qualifies as a “Real” Strategy

Must specify: universe/timeframes; explicit entry rules; explicit exit rules (SL/TP/time-stop/trailing); position sizing; risk caps; parameter ranges (for ML search); validation plan & metrics.

Not allowed: trivial fixed-price triggers without SL/TP/sizing.

Must be deterministic given inputs & params (RL policy determinism at inference is acceptable with fixed seeds/epsilon).

14) Module Names (final)
tradingbot/
  main.py
  core/
    app_runtime.py
    config_manager.py
    data_manager.py
    symbol_universe.py
    budget_manager.py
    risk_manager.py
    exposure_manager.py
    order_router.py
    sl_tp_manager.py
    pnl_reconciler.py
    session_manager.py
    diff_engine.py
    audit_logger.py
    exploration_manager.py        # NEW: candidate rotation, fairness, quotas
    strategy_registry.py          # NEW: lifecycle, states, flags, reasons
    strategy_scheduler.py         # NEW: per-asset scheduling of entries
    bankroll_manager.py           # NEW: paper top-ups, epoch logging
  brokers/
    bybit_client.py
    ibkr_client.py
    broker_common.py
  learning/
    features/
      indicator_pipeline.py
      feature_store.py
    ml/
      strategy_generator.py
      train_ml_model.py
      predict_ml_model.py
      hyperopt_ml.py
    rl/
      env_common.py
      env_spot.py
      env_futures.py
      env_forex.py
      env_options.py
      train_rl_agent.py
      policy_manager.py
      ope_evaluator.py
  validation/
    online_validator.py
    backtester.py
    walkforward.py
    stress_tests.py
    promotion_gate.py
    report_builder.py
  ui/
    api_server.py
    dashboard.py
    components.py
  notifications/
    telegram_bot.py
  config/
    assets.json
    risk.json
    ml_search.json
    rl_config.json
    exploration.json              # NEW
    session.json                  # NEW
  tests/
    ...

15) Example Config Snippets

config/assets.json

{
  "allocations_usd": { "spot": 900, "futures": 100, "forex": 0, "options": 0 },
  "crypto_universe": { "target_count": 30, "refresh_minutes": 15, "category": "spot" }
}


config/risk.json

{
  "spot":   { "risk_fraction_min": 0.0025, "risk_fraction_max": 0.0075, "daily_loss_cap_pct": 0.03, "max_concurrent": 6 },
  "futures":{ "risk_fraction_min": 0.0015, "risk_fraction_max": 0.0050, "daily_loss_cap_pct": 0.02, "leverage_cap": 5, "use_mark_for_sl_tp": true, "max_concurrent": 3 },
  "forex":  { "risk_fraction_min": 0.0010, "risk_fraction_max": 0.0040, "daily_loss_cap_pct": 0.02, "max_concurrent": 4 },
  "options":{ "max_premium_per_trade_usd": 50, "theta_cap_usd_per_day": 20, "daily_loss_cap_pct": 0.02, "max_concurrent": 3 }
}


config/exploration.json

{
  "per_asset": {
    "spot":    { "active_candidates": 6, "min_trades_to_validate": 100, "priority_boost_under_trades": 40 },
    "futures": { "active_candidates": 6, "min_trades_to_validate": 100, "priority_boost_under_trades": 40 },
    "forex":   { "active_candidates": 4, "min_trades_to_validate": 100, "priority_boost_under_trades": 40 },
    "options": { "active_candidates": 4, "min_trades_to_validate": 100, "priority_boost_under_trades": 40 }
  },
  "rotation": "round_robin_with_boost",
  "max_trades_per_hour_per_candidate": 12,
  "cooldown_seconds_after_trade": 30
}


config/session.json

{
  "paper": {
    "start_equity_usd": 1000,
    "auto_topup_threshold_usd": 10,
    "auto_topup_amount_usd": 1000,
    "log_epoch": true
  }
}

16) Deterministic P&L and Broker Truth

Reconciliation loop (per fill/interval): pull fills/positions from Bybit/IBKR; recompute P&L & fees; if divergence > tolerance → raise broker_desync flag; halt new entries for that asset until reconciled.

No synthetic fills: paper trades only at current timestamps with real tick/quote data; no back-dated execution.

17) Continuous Improvement (never idle)

Exploration Manager always keeps target active candidates per asset.

On validate/promote/reject, it immediately backfills the slot with a new candidate from strategy_generator (ML) or a mutated RL/ML variant.

No global stop after some count; exploration is perpetual.
