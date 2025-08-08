# Modules Overview & API Surface (v0.3)

This file documents **interfaces you must not break** (functions, classes, and events), plus naming rules and a change-map so multiple devs stay aligned.

> ✅ Target timeframes: **15m primary**, **5m confirm** for entries/scalps  
> ✅ Domains: **Crypto Spot**, **Perps**, **Forex**, **FX Options** (FX/Options toggles default OFF at boot but positions are reconciled/managed)

---

## 0) Global Conventions

- **Naming:** `snake_case` for functions/vars, `CamelCase` for classes, module names lowercase.
- **Type hints** required. Return `None` explicitly if nothing is returned.
- **Async:** network I/O functions are `async` and end with `_async` (e.g., `fetch_klines_async`).
- **Time:** all timestamps UTC, ms since epoch (`int`); expose `pd.Timestamp` only inside data modules.
- **Money/Qty:** use `Decimal` internally; accept `float` at API edge and cast.
- **Symbols:** canonical: `"BTCUSDT"` (no slash) inside the engine; adapters may accept `"BTC/USDT"`.
- **Errors:** raise from `modules.exceptions`; never raise bare `Exception`.
- **Config vs State:** constants in `config.py`; mutable runtime in `state/runtime_state.py`.

---

## 1) Core Modules (`/modules`)

### `config.py`
Purpose: central knobs, rollout flags, fee models.

**Key attributes (read-only at runtime):**
```py
ENVIRONMENT: Literal["simulation","production"]
ROLLOUT_STAGE: int                      # 1..5
EXCHANGE_PROFILE: Literal["spot","perp","spot+perp"]
FOREX_ENABLED: int                      # 0/1 boot default OFF
OPTIONS_ENABLED: int
HISTORICAL_DATA_PATH: str
LIVE_LOOP_INTERVAL: float
SIMULATION_ORDER_DELAY: float
TIMEFRAMES: dict = {"primary":"15m","confirm":"5m"}
EXPOSURE_CAPS = {"per_pair_pct": 0.15, "portfolio_concurrent_pct": 0.30}
RISK = {"canary_pct":0.02, "base_pct":0.05, "scale_up_max_pct":0.10}
EXPLORATION = {"min":0.10, "target":0.25, "dynamic":1}
FEE_MODEL = {"bybit":{"spot":{"taker":0.001,"maker":0.0007},"perp":{"taker":0.00055,"maker":0.0002}}}
CRITICAL_ERROR_CODES: set[int]
data_manager.py
Purpose: unified data access (WS + REST backfill), dedupe, parquet partitioned by day.

Classes

py
Copy
Edit
class DataManager:
    def __init__(self, venue: str, domain: Literal["spot","perp","forex","options"])
    async def fetch_klines_async(self, symbol: str, timeframe: str, since_ms: int | None, limit: int = 1000) -> list[list]
    def load_bars(self, symbol: str, timeframe: str, start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame
    def upsert_bars(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> int
    def last_bar_time(self, symbol: str, timeframe: str) -> int | None
    def gap_backfill_plan(self, symbol: str, timeframe: str) -> list[tuple[int,int]]
Adapter hooks

py
Copy
Edit
class MarketDataAdapter:
    async def ws_subscribe_async(self, symbols: list[str], timeframe: str) -> None
    async def ws_close_async(self) -> None
    async def rest_klines_async(self, symbol: str, timeframe: str, start_ms: int, limit: int) -> list[list]
exchange.py
Purpose: place/cancel orders; reconcile & reattach TP/SL on restart; throttle.

Interfaces

py
Copy
Edit
class Exchange:
    def __init__(self, venue: str, profile: Literal["spot","perp"], testnet: bool = False)

    async def get_price_async(self, symbol: str) -> Decimal
    async def get_balance_async(self, asset: str) -> Decimal
    async def create_order_async(self, *,
        symbol: str,
        side: Literal["buy","sell"],
        qty: Decimal,
        order_type: Literal["market","limit"]="market",
        price: Decimal | None = None,
        tp: Decimal | None = None,
        sl: Decimal | None = None,
        reduce_only: bool = False,
        client_id: str | None = None
    ) -> dict

    async def cancel_order_async(self, symbol: str, order_id: str) -> dict
    async def get_open_positions_async(self, symbol: str | None = None) -> list[dict]
    async def reconcile_async(self) -> None  # re-attach missing TP/SL, sync state
trade_executor.py
Purpose: domain-aware execution with risk/fees attached; always attach TP/SL.

py
Copy
Edit
class TradeExecutor:
    def __init__(self, exchange: Exchange, domain: Literal["crypto_spot","perp","forex","options"])

    async def submit_trade_async(self, decision: "TradeDecision") -> "ExecResult"
    async def close_position_async(self, symbol: str, reason: str) -> "ExecResult"
    async def disaster_flatten_all_async(self, reason: str) -> list["ExecResult"]
Data models (shared)

py
Copy
Edit
@dataclass
class TradeDecision:
    symbol: str
    side: Literal["long","short"]
    confidence: float
    entry: Decimal | None
    ttl_s: int
    attach_tp: Decimal | None
    attach_sl: Decimal | None
    is_exploration: bool = False
risk_management.py
Purpose: position sizing, caps, time-based exits, cooldowns.

py
Copy
Edit
def position_size(balance: Decimal, risk_pct: float, sl_pct: float, contract_value: Decimal = Decimal("1")) -> Decimal
def enforce_exposure(symbol: str, qty_usd: Decimal, caps: dict) -> None           # raise RiskViolationError
def compute_tp_sl(entry: Decimal, regime: str, atr: Decimal) -> tuple[Decimal,Decimal]
def should_force_exit(open_ms: int, max_hold_s: int, regime: str) -> bool
def cooldown_required(loss_streak: int, max_streak: int) -> bool
reward_system.py
Purpose: RL reward + human KPIs.

py
Copy
Edit
def trade_points(pnl_usd: Decimal, duration_s: int, stop_hit: bool) -> float
def rl_reward(pnl_usd: Decimal, max_dd_usd: Decimal, vol_proxy: float, is_exploration: bool) -> float
technical_indicators.py
Purpose: fast TA features.

py
Copy
Edit
def sma(arr: Sequence[float], n: int) -> float | None
def ema(arr: Sequence[float], n: int) -> float | None
def rsi(arr: Sequence[float], n: int = 14) -> float | None
def atr(high: Sequence[float], low: Sequence[float], close: Sequence[float], n: int = 14) -> float | None
def adx(high: Sequence[float], low: Sequence[float], close: Sequence[float], n: int = 14) -> float | None
top_pairs.py
Purpose: priority queue of symbols, sentiment hook, 2-hour watchlist refresh.

py
Copy
Edit
class PairManager:
    def __init__(self, venue: str, domain: Literal["spot","perp"])
    async def refresh_async(self) -> list[str]
    def rank(self, features: dict[str, dict]) -> list[str]
    def monitored_today(self) -> list[str]  # 5–10 extra symbols by sentiment
self_learning.py
Purpose: strategy brain (supervised + online RL), exploration cadence ~≥10%.

py
Copy
Edit
class SelfLearningBot:
    def __init__(self, data: DataManager, exec: TradeExecutor)

    async def on_bar_async(self, symbol: str, timeframe: str, bar: dict) -> TradeDecision | None
    async def learn_online_async(self, experience: dict) -> None
    def regime(self, symbol: str) -> Literal["trend","range","unknown"]
trade_simulator.py
Purpose: paper engine/backtests with fee/slippage model.

py
Copy
Edit
class TradeSimulator:
    def __init__(self, fee_model: dict, slippage_bps: int = 5)
    def run_backtest(self, decisions: list[TradeDecision], bars: pd.DataFrame) -> dict  # returns KPIs
telegram_bot.py
Purpose: notifications + inline controls.

py
Copy
Edit
class TelegramNotifier:
    def send(self, text: str, parse_mode: str = "Markdown") -> None
    async def send_async(self, text: str, parse_mode: str = "Markdown") -> None
    def kpi_daily(self, snapshot: dict) -> None
    def kpi_weekly(self, snapshot: dict, report_paths: list[str]) -> None
ui.py
Purpose: headless web UI / CLI controls.

py
Copy
Edit
def serve_ui() -> None  # does not block trading loop
# Exposes buttons:
#   Stage 1..5, Profile Spot/Perp/Spot+Perp, Toggle Forex/Options, Exploration slider,
#   Online Learning ON/OFF, Disaster Flatten
2) New Domains (placeholders)
/forex/forex_exchange.py
py
Copy
Edit
class ForexExchange:
    def __init__(self, broker: str)
    async def get_price_async(self, symbol: str) -> Decimal
    async def create_order_async(... same signature as Exchange.create_order_async ...)
    async def reconcile_async() -> None
/forex/forex_data.py
py
Copy
Edit
class ForexDataAdapter(MarketDataAdapter):
    async def ws_subscribe_async(self, symbols: list[str], timeframe: str) -> None
    async def rest_klines_async(self, symbol: str, timeframe: str, start_ms: int, limit: int) -> list[list]
/forex/forex_strategy.py
py
Copy
Edit
class ForexStrategyAdapter:
    def decide(self, features: dict) -> TradeDecision | None
/options/options_exchange.py (FX options later)
py
Copy
Edit
class OptionsExchange:
    async def get_chain_async(self, symbol: str, expiry: str) -> list[dict]
    async def create_order_async(..., instrument_id: str, ...) -> dict
3) Telemetry & Reports
/telemetry/metrics_exporter.py
py
Copy
Edit
def start_prometheus(port: int = 9103) -> None
def incr_error(code: int) -> None
def observe_latency(endpoint: str, ms: float) -> None
def set_gauge(name: str, value: float, labels: dict | None = None) -> None
/telemetry/report_generator.py
py
Copy
Edit
def daily_snapshot(portfolios: dict) -> dict
def weekly_report(portfolios: dict, out_dir: str = "reports") -> list[str]  # returns file paths (CSV/PNG)
4) State & Reconciliation
/state/runtime_state.py
py
Copy
Edit
def load_state() -> dict
def save_state(state: dict) -> None
def remember_toggle(name: str, value: bool) -> None  # Forex/Options OFF at boot, but monitored positions persist
/state/position_reconciler.py
py
Copy
Edit
async def reconcile_all_async(exchanges: list[Exchange]) -> None
5) Change-Control / Update Map
Add new function: append below related section, bump API Surface minor (v0.x → v0.(x+1)).

Rename: keep old name for 2 releases; mark with # @deprecated since v0.x, use new_name() and forward call.

Remove: only after 2 releases of deprecation; list in “BREAKING CHANGES”.

Signature change: add **kwargs shim to old entry, log warning, deprecate.

Docs rule: every public function/class must have a docstring with:

Args/Returns types

Raises (exceptions from modules.exceptions)

Threading/async notes if applicable

Lint/Type: enforce ruff + mypy in CI; fail on missing type hints.

6) Missing Components Checklist (blocking rollout)
 /state/runtime_state.py

 /state/position_reconciler.py

 /marketdata/ws_adapter.py + /marketdata/rest_backfill.py

 /dataio/parquet_writer.py + /dataio/parquet_reader.py

 /telemetry/metrics_exporter.py + /telemetry/report_generator.py

 /forex/* and /options/* adapters (stubs acceptable Stage 1–3)

7) Shared Types (add modules/types.py)
py
Copy
Edit
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

Side = Literal["long","short"]
OrderSide = Literal["buy","sell"]
OrderType = Literal["market","limit"]

@dataclass
class ExecResult:
    ok: bool
    order_id: str | None
    symbol: str
    message: str
    pnl_usd: Decimal | None = None
8) Timeframes & Bars (canonical)
Primary: 15m

Confirm: 5m

Backfill must create daily partitions: historical_data/{domain}/{symbol}/{timeframe}/dt=YYYY-MM-DD/*.parquet
