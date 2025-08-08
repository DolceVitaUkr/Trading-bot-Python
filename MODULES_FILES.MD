Modules Overview & API Surface (v0.4)
This file documents interfaces you must not break (functions, classes, and events), plus naming rules and a change-map so multiple devs stay aligned.

✅ Target timeframes: 15m primary, 5m confirm for entries/scalps
✅ Domains: Crypto Spot, Perps, Forex, FX Options (FX/Options toggles default OFF at boot but positions are reconciled/managed)

0) Global Conventions
Naming: snake_case for functions/vars, CamelCase for classes, module names lowercase.

Type hints required. Return None explicitly if nothing is returned.

Async: network I/O functions are async and end with _async (e.g., fetch_klines_async).

Time: all timestamps UTC, ms since epoch (int); expose pd.Timestamp only inside data modules.

Money/Qty: use Decimal internally; accept float at API edge and cast.

Symbols: canonical: "BTCUSDT" (no slash) inside the engine; adapters may accept "BTC/USDT".

Errors: raise from modules.exceptions; never raise bare Exception.

Config vs State: constants in config.py; mutable runtime in state/runtime_state.py.

1) Core Modules (/modules)
config.py
Purpose: central knobs, rollout flags, fee models.

Key attributes (read-only at runtime):

py
Copy
Edit
ENVIRONMENT: Literal["simulation","production"]
ROLLOUT_STAGE: int
EXCHANGE_PROFILE: Literal["spot","perp","spot+perp"]
FOREX_ENABLED: int
OPTIONS_ENABLED: int
HISTORICAL_DATA_PATH: str
LIVE_LOOP_INTERVAL: float
SIMULATION_ORDER_DELAY: float
TIMEFRAMES: dict
EXPOSURE_CAPS: dict
RISK: dict
EXPLORATION: dict
FEE_MODEL: dict
CRITICAL_ERROR_CODES: set[int]
data_manager.py
Purpose: unified data access (WS + REST backfill), dedupe, parquet partitioned by day.

py
Copy
Edit
class DataManager:
    def __init__(self, venue: str, domain: Literal["spot","perp","forex","options"])
    async def fetch_klines_async(...)
    def load_bars(...)
    def upsert_bars(...)
    def last_bar_time(...)
    def gap_backfill_plan(...)
exchange.py
Purpose: place/cancel orders; reconcile & reattach TP/SL on restart; throttle.

py
Copy
Edit
class Exchange:
    def __init__(...)
    async def get_price_async(...)
    async def get_balance_async(...)
    async def create_order_async(...)
    async def cancel_order_async(...)
    async def get_open_positions_async(...)
    async def reconcile_async()
trade_executor.py
Purpose: execution with risk/fees; always attach TP/SL.

py
Copy
Edit
class TradeExecutor:
    def __init__(...)
    async def submit_trade_async(...)
    async def close_position_async(...)
    async def disaster_flatten_all_async(...)
risk_management.py
Purpose: position sizing, caps, time-based exits, cooldowns.

py
Copy
Edit
def position_size(...)
def enforce_exposure(...)
def compute_tp_sl(...)
def should_force_exit(...)
def cooldown_required(...)
reward_system.py
Purpose: RL reward + human KPIs.

py
Copy
Edit
def trade_points(...)
def rl_reward(...)
technical_indicators.py
Purpose: TA feature calculations for strategy and ML pipelines.

py
Copy
Edit
# Moving averages
def sma(data: list[float], window: int) -> float | None
def ema(data: list[float], window: int) -> float | None

# Oscillators
def rsi(close: list[float], window: int = 14) -> float | None
def stochastic_oscillator(...)
def williams_r(...)
def obv(close: list[float], volume: list[float]) -> list[float] | None

# Trend / volatility
def atr(...)
def adx(...)
def cci(...)
def bollinger_bands(...)
def entropy_volatility(...)

# Other
def fibonacci_retracement(...)
def market_regime(...)

# Wrapper
class TechnicalIndicators:
    sma(...)
    ema(...)
    rsi(...)
    atr(...)
    adx(...)
    cci(...)
    williams_r(...)
    obv(...)
    stochastic_oscillator(...)
    bollinger_bands(...)
    fibonacci_retracement(...)
    market_regime(...)
    entropy_volatility(...)
top_pairs.py
Purpose: priority queue of symbols, sentiment hook, watchlist refresh.

self_learning.py
Purpose: supervised + online RL strategy loop.

trade_simulator.py
Purpose: paper engine/backtests with fee/slippage.

telegram_bot.py
Purpose: notifications + inline controls.

ui.py
Purpose: headless web UI / CLI controls.

2) New Domains (placeholders)
/forex/forex_exchange.py

/forex/forex_data.py

/forex/forex_strategy.py

/options/options_exchange.py

3) Telemetry & Reports
/telemetry/metrics_exporter.py

/telemetry/report_generator.py

4) State & Reconciliation
/state/runtime_state.py

/state/position_reconciler.py

5) Change-Control / Update Map
Keep old names for 2 releases when renaming

Deprecate before removal

✅ Updated in v0.4:

technical_indicators.py section aligned with actual codebase (removed duplicate class def, clarified function signatures)

Global conventions unchanged but reconfirmed

Added clarity on wrappers vs functions

