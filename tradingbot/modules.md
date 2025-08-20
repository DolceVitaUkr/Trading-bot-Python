Modules.md — Trading Bot (Multi-Asset, Training-First, UI-First)
Conventions

Package root: tradingbot

Boot order: UI first → Training (paper) first. No live orders until manual per-asset enable (double-confirm + validation gates).

Assets (per-asset live toggles): crypto_spot, crypto_futures, forex, options

File naming: snake_case for files; public APIs use concise, consistent Title_Case_With_Underscores.

Paths: cross-platform via pathlib.

Directory Map
tradingbot/
  core/        # config_manager, datamanager, strategymanager, indicators,
               # reward_system, riskmanager, tradeexecutor, portfoliomanager,
               # validationmanager, pairmanager, notifier, error_handler,
               # runtime_controller   (optional but recommended)
  brokers/     # exchangebybit.py, exchangeibkr.py
  learning/    # train_ml_model.py, train_rl_model.py, save_ai_update.py
  ui/          # FastAPI app, routers, websockets, routes/diff.py (dry-run)
  config/      # config.json, assets.json, strategies.json
  logs/        # json/csv: trades, decisions, errors, telemetry, validation
  state/       # models/, replay_buffers/, checkpoints/, caches/, runtime.json

Core Modules
1) core/config_manager.py — Config Manager

Purpose

Central load/validate/merge of JSON configs and env overrides.

Enforce training-first defaults at boot.

Public API

Load_Config() -> dict

Get_Key(path: str, default=None) -> Any

Set_Key(path: str, value: Any) -> None

Persist_Runtime(state: dict, path="state/runtime.json") -> None

Key Settings (min)

Safety: START_MODE="training", LIVE_TRADING_ENABLED=false, REQUIRE_MANUAL_LIVE_CONFIRMATION=true, ORDER_ROUTING="simulation", MAX_STOP_LOSS_PCT, KILL_SWITCH_ENABLED, CONSECUTIVE_LOSS_KILL

Paper: PAPER_EQUITY_START=1000.0, PAPER_RESET_THRESHOLD=10.0

Validation gates: min_trades, min_sharpe, max_drawdown_pct

Telegram block (token, chat_id, flags)

2) core/datamanager.py — Data Manager

Purpose

Fetch historical/live data; cache; normalize across Bybit/IBKR; timeframe alignment (e.g., 5m entries, 15m regime).

Public API

Fetch_Klines(symbol, timeframe, since=None, limit=None) -> DataFrame

Subscribe_Live(symbol, timeframe, callback) -> None

Save_To_Cache(df, symbol, timeframe) / Load_From_Cache(...)

Compute_Indicators(df, spec) -> DataFrame (delegates to Indicators)

Data Contract

DataFrame columns: ts, open, high, low, close, volume

3) core/indicators.py — Technical Indicators

Purpose

Indicator computations for pipeline/features.

Indicators

Trend: SMA, EMA, (opt) MACD

Momentum: RSI, Stochastic, Williams %R

Volatility: ATR, StdBands

Flow: MFI, (opt) OBV

Fibonacci: auto swing levels for TP/SL context

Public API

Apply_Indicators(df, spec: dict) -> df_with_features

4) core/pairmanager.py — Pair Manager (Rotation)

Purpose

Build & rank “hot” pairs by volatility (ATR/stdev), momentum (ROC/EMA slope), liquidity/volume, spread; optional sentiment boost/penalty.

Cache/fallback on API issues.

Public API

Refresh_Universe() -> dict[asset, list[str]]

Get_Top(symbol_count=20, asset="crypto_spot") -> list[str]

Set_Sentiment_Provider(provider_fn) -> None

Config

Refresh cadence (e.g., 15m), min volume, excluded symbols.

5) core/strategymanager.py — Strategy Manager (ML/RL + Rules)

Purpose

Policy inference from RL/ML models + optional rule filters.

Training-first: generate paper intents until live enabled.

Public API

Score_Action(context: dict) -> dict{action,size,sl,tp,meta}

Load_Strategy_Set(name: str) -> None

Select_Strategy(asset: str) -> str

Set_Model_Handle(model) -> None

Set_Indicators(spec: dict) -> None

Get_Validation_Status() -> dict

Notes

State = OHLCV windows + indicators + position context + regime flags.

Actions: hold / open / scale / close (respect asset constraints).

6) core/reward_system.py — Reward System

Purpose

After-fee rewards, DD penalties, TP bonuses, Reward Points.

Public API

Compute_Reward(trade_ctx: dict) -> {"reward": float, "points": float}

Set_Params(dd_penalty: float, tp_bonus: float, fee_model="after_fees") -> None

7) core/riskmanager.py — Risk Manager

Purpose

Enforce mandatory SL/TP, max SL ≤ 15%, sizing, exposure caps, daily loss guards.

Public API

Validate_Order(symbol, side, notional, sl_pct, tp_schema) -> (ok: bool, reason: str)

Compute_Size(balance, risk_pct, min_notional) -> float

Apply_Trailing_TP(position, atr_or_pct) -> None

Risk_Breach_Check() -> list[str]

Asset Standards

Forex: typical SL 0.5–2%

Crypto: 1–10% intraday, cap 15%

8) core/tradeexecutor.py — Trade Executor

Purpose

Route simulation (paper) vs broker (live) orders; precision/min-notional; OCO TP/SL; Close-Only handling; reconcile on restart.

Public API

Route_Order(order_intent: dict, mode="simulation"|"broker") -> dict

Enable_Live(asset: str) / Disable_Live(asset: str, close_only: bool=False) -> None

Close_All_Open(asset: str) -> None

Ensure_SL_TP(asset: str) -> None

Reconcile_Open_Positions(asset: str) -> None

Behavior

Training: dry-run intents + paper fills (optionally Telegram “intent”).

Live: respect per-asset live_enabled and close_only.

9) core/portfoliomanager.py — Portfolio Manager

Purpose

Track balances/equity, open positions, P&L (U/R), fees, Reward Points.

Public API

Update_On_Fill(fill_event: dict) -> None

Valuation(asset: str) -> dict{equity,pnl_u,pnl_r}

Get_Open_Positions(asset: str) -> list[dict]

Flatten(asset: str) -> None

10) core/validationmanager.py — Validation Gates

Purpose

Backtest + forward test; gate before enabling live.

Public API

Validate_Strategy(strategy_id, data_spec, gates) -> dict{pass:bool,report:dict}

Backtest(strategy_id, period) -> dict

Forward_Test(strategy_id, period) -> dict

Writes to logs/validation/*.json

Default Gates

Trades ≥ 500, Sharpe ≥ 2.0, Max DD ≤ 15%

11) core/notifier.py — Telegram Notifier

Purpose

Lifecycle, intents, executions, hourly paper recap, errors, mode flips, validation.

Public API

Send_Start() / Send_Stop()

Send_Status(msg: str)

Send_Error(err: str)

Send_Paper_Intent(symbol, side, notional)

Send_Paper_Hourly_Recap(stats)

Send_Live_Open(asset, symbol, side, notional)

Send_Live_Close(asset, symbol, pnl_pct, fees, pure_profit, points)

Send_Mode_Change(asset, live_enabled: bool, close_only: bool)

Send_Kill_Switch(scope: str) (asset|global)

Config (config.json)

"TELEGRAM": {
  "ENABLED": true,
  "BOT_TOKEN": "xxx",
  "CHAT_ID": "yyy",
  "SEND_TRADE_INTENTS_IN_TRAINING": true,
  "SEND_EXECUTIONS_IN_LIVE": true,
  "SEND_ERRORS": true,
  "SEND_HOURLY_PAPER_RECAP": true
}

12) core/error_handler.py — Error Handling & Recovery

Purpose

Classify exceptions (Network/API/Order/Strategy), error-rate circuit breaker, safe shutdown, restart hooks.

Public API

Handle(exc: Exception, context: dict) -> str (action)

Error_Rate_Circuit_Breaker(window_s: int, threshold: int) -> bool

Restart Protection

Reload models/replay, pair universe, runtime state; reconcile live open positions.

Learning
13) learning/train_rl_model.py — RL Training

Purpose

PyTorch DQN/PPO with replay, target nets (DQN), entropy reg (PPO).

Public API

Train_RL_Model(data_stream, reward_fn, config) -> checkpoint_path

Load_RL_Model(path) -> model

Evaluate_RL_Model(data_stream) -> metrics

Notes

Epsilon-greedy schedule for DQN; checkpoints under state/models/, replay under state/replay_buffers/.

14) learning/train_ml_model.py — ML Training

Public API

Train_ML_Model(features_df, labels, config) -> path

Load_ML_Model(path) -> model

Predict(features_df) -> np.ndarray

15) learning/save_ai_update.py — Save AI State

Public API

Save_AI_Update(models, replay, meta) -> dict{paths}

Brokers
16) brokers/exchangebybit.py — Bybit Adapter

Scope

Spot & USDT-M Futures (symbols, filters, klines, order ops, positions, balances).

Public API

Get_Precision(symbol), Get_Min_Notional(symbol)

Get_Klines(symbol, timeframe, ...)

Create_Order(...), Cancel_Order(...)

Fetch_Positions(), Fetch_Balances()

Notes

Testnet flag; rate-limit backoff.

17) brokers/exchangeibkr.py — IBKR Adapter

Scope

Forex quotes/orders; Options chains, place/cancel, balances/positions.

Public API

Get_Tick(contract), Get_History(contract, ...)

Place_Order(contract, side, qty, type, price=None)

Cancel_Order(order_id)

Fetch_Balances(), Fetch_Positions()

UI
18) ui/ — FastAPI Dashboard

Purpose

Load first; 4 panels; websockets stream; controls:

Per-asset Live toggle (double confirm + validation)

Kill Switch (asset/global → Close-Only)

Stop Trading: Close All Now / Keep Open & Monitor

Training runs regardless of live state

Suggested Endpoints

GET /status — summary per asset

POST /live/{asset}/enable — enable after confirmation + gates pass

POST /live/{asset}/disable

POST /kill/{scope}/{onoff} — scope=asset|global

POST /stop/{asset}?mode=close_all|keep_open

WS /stream — equity, P&L, open positions, logs

Optional Extensions (Recommended)
19) core/runtime_controller.py — Runtime Controller

Purpose

Thin façade orchestrating Config, Executor, Portfolio, Risk, Notifier.

Centralizes per-asset live toggles, kill switch, auto-kill on consecutive losses, and hourly paper recap scheduling.

Responsibilities

Persist state to state/runtime.json (restart-safe).

Apply training-first guard at boot.

Validate before enabling live (min trades, Sharpe, DD).

Manage Close-Only behavior for kill switch (manual/auto).

Public API

Start() / Stop()

Enable_Live(asset: str) / Disable_Live(asset: str, close_only: bool=False)

Set_Global_Kill(active: bool)

Record_Trade_Result(asset: str, is_live: bool, pnl_after_fees: float)

Hourly_Paper_Recap(stats: dict)

Get_State() -> dict

20) ui/routes/diff.py — Dry-Run Diff API

Purpose

Preview would-be live orders (paper vs live) for trust before enabling live.

Endpoints

GET /diff/{asset} →

{
  "asset": "crypto_spot",
  "intents": [
    {
      "symbol": "BTCUSDT",
      "side": "buy",
      "notional": 100.0,
      "sl_pct": 0.01,
      "tp_schema": {"type":"atr","mult":2.0},
      "score": 0.73,
      "mode_if_live": "broker"
    }
  ],
  "as_of": "ISO-8601"
}


POST /diff/confirm/{asset} → flips to live (after validation + confirmation).

UI

Button “Preview Live Orders” in each panel; side-by-side paper intents vs would-be live.

Logging & Telemetry
21) logs/ — Files & Formats

Format

JSON Lines (machine-readable) + optional CSV summaries.

Files

logs/trades/{asset}_trades_YYYYMMDD.jsonl

logs/decisions/{asset}_decisions_YYYYMMDD.jsonl (intent, scores, features hash)

logs/errors/errors_YYYYMMDD.jsonl

logs/validation/*.json

Common Fields

ts, asset, symbol, side, notional, price, sl, tp, fees, pnl_u, pnl_r, points, mode, txid

State & Persistence
22) state/ — Contents

models/ (RL/ML checkpoints)

replay_buffers/ (RL experience)

checkpoints/ (bundle snapshots)

caches/ (pair universe, symbol meta)

runtime.json (per-asset live_enabled, close_only, consecutive losses; global kill flag; last recap)

Startup Sequence

Load configs → enforce training-first guards.

Load runtime.json → restore per-asset live/close-only.

Reconcile broker open positions; ensure SL/TP on live.

Resume data streams, models, pair universe, paper equity.

Data Contracts
23) OrderIntent (Strategy → Executor)
{
  "asset": "crypto_spot",
  "symbol": "BTCUSDT",
  "side": "buy|sell",
  "notional": 100.0,
  "sl_pct": 0.01,
  "tp_schema": {"type":"atr","mult":2.0},
  "meta": {"source":"RL","score":0.73}
}

24) Position
{
  "id":"uuid",
  "asset":"crypto_spot",
  "symbol":"BTCUSDT",
  "qty": 1.234,
  "entry": 30000.0,
  "sl": 29700.0,
  "tp": 30600.0,
  "mode":"paper|live",
  "opened_at":"ISO-8601"
}

25) TradeFill
{
  "position_id":"uuid",
  "price": 29990.0,
  "qty": 1.234,
  "fees": 0.12,
  "pnl_realized": 3.45,
  "closed_at":"ISO-8601"
}

Mode & Safety Logic (Summary)

Boot: UI first → Training (paper) only. Live disabled for all assets.

Enable Live (per asset): UI double-confirmation + Validation gates pass (Trades ≥ 500, Sharpe ≥ 2.0, Max DD ≤ 15%) + Kill off.

Kill Switch (asset/global): Manual or auto on N consecutive live losses → Close-Only (no new live orders); ensure SL/TP; close profitable where possible; paper continues.

Stop Trading:

Close All Now (market/IOC; risk caps)

Keep Open & Monitor (no new live; maintain SL/TP)

Tests (skeleton)

tests/test_config_manager.py — loads/overrides/guards

tests/test_riskmanager.py — SL cap, sizing, breaches

tests/test_trade_routing.py — paper vs live, close-only logic

tests/test_validation_gates.py — gates & reports

tests/test_persistence.py — runtime resume + reconcile

Traceability

This Modules.md matches the README architecture and your training-first, UI-first, per-asset-toggle, Telegram-enabled design.
