Modules Specification – Self-Learning Multi-Asset Trading Bot

This document defines all modules, their responsibilities, and interfaces.
Every file must begin with:

# file: <path>/<filename>.py


All filenames must be lowercase_with_underscores.py.

📂 Core Modules
core/config_manager.py

Purpose: Centralized configuration loader/validator.

Functions:

load_config(section: str) -> dict

validate_schema(config: dict, schema: dict)

save_config(section: str, config: dict)

Notes: JSON configs only. Strict schema validation.

core/runtime_controller.py

Purpose: Orchestrates runtime state, live/paper toggles, and kill switch.

Functions:

set_mode(asset: str, mode: str) – toggle live/paper/off.

trigger_kill_switch(asset: str) – graceful shutdown of live trading.

resume_asset(asset: str) – resume trading after kill.

hourly_recap() – schedule recap to Telegram & UI.

Notes: Persists runtime state in state/runtime.json.

core/datamanager.py

Purpose: Unified data pipeline.

Functions:

fetch_historical(symbol, interval, start, end)

stream_live(symbol, interval)

clean_and_store(dataframe, symbol, interval)

resample_multi_tf(data, intervals)

Notes: Uses Parquet. Ensures data completeness and gap checks.

core/feature_store.py

Purpose: Versioned feature sets.

Functions:

save_features(symbol, features, version)

load_features(symbol, version)

validate_features(features, schema)

Notes: Online/offline parity.

core/indicators.py

Purpose: Compute technical indicators.

Supported: SMA, EMA, RSI, ATR, Bollinger, MACD, OBV, VWAP, SuperTrend, Donchian, Keltner, Ichimoku.

Functions:

add_indicators(df) – returns enriched DataFrame.

core/pair_manager.py

Purpose: Dynamic asset selection.

Logic:

Volatility/momentum/liquidity scans.

Regime tagging (trending, ranging, volatile).

Functions:

rank_pairs(market_data) -> list

tag_regime(symbol, data) -> str

core/risk_manager.py

Purpose: Position sizing & portfolio protection.

Logic:

$10 per trade until balance ≥ $1000.

Tiered % risk sizing with drawdown penalties.

Hard SL ≤ 15%.

Correlation caps & max exposure.

Functions:

calculate_size(account_equity, signal_strength, sl_distance, price)

apply_portfolio_limits(positions)

core/trade_executor.py

Purpose: Order placement & reconciliation.

Logic:

Spread + slippage models.

Supports market, limit, post-only, TIF.

Fees, funding, borrow costs.

Reconcile positions on restart.

Functions:

submit_order(symbol, side, size, price, order_type)

cancel_order(order_id)

reconcile_positions()

core/portfolio_manager.py

Purpose: Track balances, P&L, exposure.

Functions:

update_portfolio(trade)

get_equity(asset_class)

calculate_metrics() – Sharpe, MaxDD, PF, CVaR.

core/validation_manager.py

Purpose: Industrial-grade validation of strategies.

Logic:

Event-driven backtest with realistic fills.

Purged+embargo walk-forward.

Stress tests (Monte Carlo, perturbations, shocks).

RL off-policy evaluation.

Functions:

run_backtest(strategy, data)

walk_forward(strategy, data, folds)

stress_test(strategy, scenarios)

generate_report(strategy_id)

Outputs: JSON/HTML reports → UI + Telegram.

core/notifier.py

Purpose: Telegram & log notifications.

Functions:

send_message(event_type, payload)

send_recap()

Events: Start/Stop, hourly paper recap, live trade open/close, validation pass/fail, kill switch.

core/drift_monitor.py

Purpose: Detect feature/label drift.

Functions:

check_drift(features, baseline)

alert_if_drift()

📂 Learning Modules
learning/state_featurizer.py

Purpose: Convert market data into model inputs.

Functions:

build_state(df) – include indicators, regimes, embeddings.

Supports: Normalization, sequence stacking, embeddings.

learning/train_ml_model.py

Purpose: ML strategy training.

Logic:

Triple-Barrier + Meta-Labeling.

Lag stacks, PCA, SHAP pruning.

Functions:

train_classifier(features, labels)

validate_model(model, data)

learning/train_rl_model.py

Purpose: RL training.

Algorithms:

DQN (Double+Dueling, PER, n-step).

PPO (GAE, entropy bonus).

SAC/TD3 for continuous sizing.

Functions:

train(env, algo)

save_checkpoint()

load_checkpoint()

📂 Broker Adapters
brokers/exchange_bybit.py

Purpose: Bybit Spot/Futures API integration.

Functions:

fetch_klines(symbol, interval)

place_order()

get_positions()

get_balances()

brokers/exchange_ibkr.py

Purpose: IBKR Forex & Options integration.

Functions:

fetch_quotes(symbol)

place_order()

get_positions()

get_balances()

Notes: Includes options chain support.

📂 UI Modules
ui/app.py

Purpose: FastAPI server for dashboard + API.

Routes:

/status – bot status, balances.

/live/{asset}/enable|disable – toggle live.

/kill/{asset} – kill switch.

/diff/{asset} – dry-run diff.

/validation/{strategy} – validation report.

ui/routes/validation.py

Purpose: Serve validation reports.

Functions:

get_latest_report(strategy_id)

📂 Ops & Tests
tests/

Unit tests:

Risk sizing math.

SL enforcement.

Indicator calculations.

Validation metrics (Sharpe, MaxDD, CVaR).

Integration tests:

Resume after crash.

UI /diff flow.

Walk-forward deterministic snapshots.

📂 State & Logs

state/ – runtime state, RL buffers, model weights.

logs/ – validation results, backtests, ops logs.

🔑 Key Cross-Cutting Concerns

Consistency: All modules lowercase, clear header comment.

Persistence: Resume trades, positions, runtime state across restarts.

Security: Secrets in vault, never in code.

Observability: Every trade, state, and decision logged.

✅ This modules.md + README now serve as a single source of truth.
Any future developer can align code to them without ambiguity.
