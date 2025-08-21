modules.md — Full Spec (renamed modules + no-underscore function APIs)

All files and functions must follow: lowercase, no underscores.
Every file begins with # file: <path>/<filename>.py.

core/configmanager.py

Purpose: central config load/validate/save.
Public API (functions):

loadconfig(section: str | None = None) -> dict

validateschema(config: dict, schema: dict) -> None

saveconfig(section: str, config: dict) -> None
Notes: JSON only; strict ranges/types; env overrides supported.

core/runtimecontroller.py

Purpose: runtime modes, per-asset live toggles, kill switch, hourly recap, state persistence.
Public API:

start(), stop()

enablelive(asset: str) -> bool (runs validation gate, double-confirm)

disablelive(asset: str, closeonly: bool = False) -> None

setglobalkill(active: bool) -> None

recordtraderesult(asset: str, islive: bool, pnlafterfees: float) -> None

hourlypaperrecap() -> None

getstate() -> dict
State: persisted in state/runtime.json.

core/datamanager.py

Purpose: historical/live data, parquet cache, multi-TF joins, gap checks.
Public API:

fetchklines(symbol: str, timeframe: str, start: str | None, end: str | None) -> DataFrame

savelocal(df, symbol: str, timeframe: str) -> None

loadlocal(symbol: str, timeframe: str) -> DataFrame

jointimeframes(spec: dict) -> DataFrame (e.g., entry=5m, regime=15m, context=1h)

subscribelive(symbol: str, timeframe: str, callback) -> None

core/featurestore.py

Purpose: versioned features with schema.
Public API: savefeatures, loadfeatures, validatefeatures.

core/indicators.py

Purpose: TA calculations.
Coverage: SMA, EMA, MACD, RSI, Stoch, W%R, ATR, stdev bands, MFI, OBV, Fib; HMA, KAMA, SuperTrend, Donchian, Keltner, Bollinger%B, VWAP bands, (optional) Ichimoku.
Public API: applyindicators(df: DataFrame, spec: dict) -> DataFrame.

core/pairmanager.py

Purpose: universe selection & regimes.
Public API:

refreshuniverse() -> dict[asset, list[str]]

gettop(count: int, asset: str) -> list[str]

setsentiment(providerfn) -> None

tagregimes(df: DataFrame) -> DataFrame
Logic: rank by volatility, momentum, liquidity, spread; optional sentiment boost.

core/riskmanager.py

Purpose: sizing & hard risk.
Public API:

validorder(symbol, side, notional, slpct, tpschema) -> tuple[bool, str]

computesize(equity: float, riskpct: float, minnotional: float) -> float

applytrailingtp(position, atrorpct) -> None

riskbreachcheck() -> list[str]

positionsize(equity, drawdownpct, signalscore, sldistance, price) -> float
Rules:

Equity < $1000 → fixed $10 notional per trade.

≥ $1000 → tiered risk (0.5–2%), −0.25% per 5% drawdown (floor 0.25%), signal-weighted (0.5×/1.0×/1.5×).

SL ≤ 15%, exposure/daily-loss/correlation caps.

core/tradeexecutor.py

Purpose: route paper vs live; precision/min-notional; slippage/latency; OCO SL/TP; reconcile.
Public API:

routeorder(orderintent: dict, mode: str = "simulation") -> dict

enablelive(asset: str) / disablelive(asset: str, closeonly: bool = False)

closeallopen(asset: str) -> None

ensuresltp(asset: str) -> None

reconcilepositions(asset: str) -> None
Execution realism: spread & volatility-based slippage, partial fills, funding (futures), maker/taker fees.

core/portfoliomanager.py

Purpose: balances, equity, U/R P&L, reward points, exposures.
Public API:

updateonfill(fill: dict) -> None

valuation(asset: str) -> dict (equity, pnl_u, pnl_r)

getopenpositions(asset: str) -> list[dict]

flatten(asset: str) -> None

core/validationmanager.py

Purpose: backtest + walk-forward + stress + (optional) OPE + gate.
Public API:

validatestrategy(strategyid: str, dataspec: dict, gates: dict) -> dict

backtest(strategyid: str, period: dict) -> dict

walkforward(strategyid: str, dataspec: dict, splits: int, embargobars: int) -> dict

stress(strategyid: str, dataspec: dict, specs: dict) -> dict

ope(strategyid: str, dataspec: dict, behaviorstats: dict | None) -> dict

savereport(report: dict, path: Path) -> None

latestreport(strategyid: str) -> dict

eligibleforlive(strategyid: str, gates: dict | None = None) -> bool
Gate defaults: trades≥500, sharpe≥2.0, maxdd≤15%, pf≥1.5, cvar within bounds.
Artifacts: logs/validation/{strategyid}/....

core/optimizer.py

Purpose: grid/random/evolutionary tuning.
Public API: optimize(paramspec: dict, objectivefn) -> dict.

core/notifier.py

Purpose: Telegram/ops notifications.
Public API:

sendstart, sendstop, sendstatus, senderror,

sendpaperintent, sendpaperhourlyrecap,

sendliveopen, sendliveclose,

sendmodechange, sendkillswitch.

core/errorhandler.py

Purpose: classify/log exceptions, circuit breaker, restart hooks.
Public API: handle(exc, context) -> str, errorratecircuitbreaker(window, threshold) -> bool.

core/driftmonitor.py

Purpose: feature/label/model drift; alerts & auto de-risk hooks.
Public API: checkdrift(features, baseline) -> dict, alertifdrift(report) -> None.

brokers/exchangebybit.py

Purpose: Bybit adapter (spot + USDT-M futures).
Public API:

getprecision, getminnotional

getklines, createorder, cancelorder

fetchpositions, fetchbalances

brokers/exchangeibkr.py

Purpose: IBKR adapter (forex + options).
Public API:

gettick, gethistory

placeorder, cancelorder

fetchbalances, fetchpositions

learning/statefeaturizer.py

Purpose: build state tensors (multi-TF, indicators, regimes, embeddings).
Public API: buildstate(df: DataFrame, spec: dict) -> Any.

learning/trainmlmodel.py

Purpose: triple-barrier labels, meta-labeling, WFCV, SHAP pruning.
Public API:

trainml(features: DataFrame, labels: Array, config: dict) -> str (returns path)

loadml(path: str) -> Any

predictml(model: Any, features: DataFrame) -> Array

learning/trainrlmodel.py

Purpose: DQN (double/dueling/PER/n-step) & PPO (GAE); optional SAC/TD3; replay & checkpoints.
Public API:

trainrl(datastream, rewardfn, config) -> str

loadrl(path: str) -> Any

evaluaterl(datastream) -> dict

learning/saveaiupdate.py

Purpose: persist models, replay, metadata.
Public API: saveaiupdate(models: dict, replay, meta: dict) -> dict.

ui/app.py

Purpose: FastAPI app + WebSocket dashboard.
Routes:

GET /status

POST /live/{asset}/enable, POST /live/{asset}/disable

POST /kill/{scope}/{onoff}

POST /stop/{asset}?mode=close_all|keep_open

GET /diff/{asset}, POST /diff/confirm/{asset}

GET /validation/{strategy}

WS /stream

ui/routes/diff.py

Purpose: dry-run diff (paper vs would-be live).

ui/routes/validation.py

Purpose: serve latest validation report JSON.

tests/

Unit: risksizing math ($10 rule + tiers + drawdown + signal weight), slcap, precision/minnotional, indicators, splits (purge+embargo), metrics (sharpe/maxdd/cvar).
Integration: resume/reconcile, /diff enable-live flow, validation e2e snapshot (seeded), killswitch transitions.

state/ & logs/

state/: models/, replay_buffers/, checkpoints/, caches/, runtime.json

logs/: trades/, decisions/, errors/, validation/

Final consistency checklist

Files & functions: all lowercase, no underscores.

Classes: CamelCase.

Header line in every file:

# file: <path>/<filename>.py


Training-first → Validation → per-asset Live; Kill switch semantics unchanged.

If you want, I can generate a search-and-replace refactor plan (old→new filenames & function names) so your agent can auto-rename everything safely.
