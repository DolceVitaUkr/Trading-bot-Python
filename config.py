import os
from typing import Literal
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ───────────────────────────────────────────────
# Environment & Mode
# ───────────────────────────────────────────────
ENVIRONMENT = os.getenv("ENVIRONMENT", "simulation").lower()
USE_SIMULATION = ENVIRONMENT == "simulation"
ROLLOUT_STAGE = int(os.getenv("ROLLOUT_STAGE", "1"))
EXCHANGE_PROFILE: Literal["spot", "perp", "spot+perp"] = os.getenv(
    "EXCHANGE_PROFILE", "spot"
).lower()

FOREX_ENABLED_DEFAULT = False
OPTIONS_ENABLED_DEFAULT = False

# ───────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "bot.log")

# ───────────────────────────────────────────────
# API Keys
# ───────────────────────────────────────────────
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "TujcvO0Luboj8ET0tp")
BYBIT_API_SECRET = os.getenv(
    "BYBIT_API_SECRET", "0oT2ENYllETn3sKXx0m42LvWG8zWnEqhN72k")

SIMULATION_BYBIT_API_KEY = os.getenv(
    "SIMULATION_BYBIT_API_KEY", "TujcvO0Luboj8ET0tp")
SIMULATION_BYBIT_API_SECRET = os.getenv(
    "SIMULATION_BYBIT_API_SECRET", "0oT2ENYllETn3sKXx0m42LvWG8zWnEqhN72k")

# ───────────────────────────────────────────────
# Telegram
# ───────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", "7611295732:AAHazYz46ynfueYthvQXvQRA9bYlxihEf1c")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "380533508")
ASYNC_TELEGRAM = os.getenv(
    "ASYNC_TELEGRAM", "true").lower() in {"1", "true", "yes", "on"}
TELEGRAM_PAPER_RECAP_MIN = int(os.getenv("TELEGRAM_PAPER_RECAP_MIN", "60"))
TELEGRAM_LIVE_ALERT_LEVEL = os.getenv("TELEGRAM_LIVE_ALERT_LEVEL", "normal")
TELEGRAM_HEARTBEAT_MIN = int(os.getenv("TELEGRAM_HEARTBEAT_MIN", "10"))

# ───────────────────────────────────────────────
# Paths & Intervals
# ───────────────────────────────────────────────
HISTORICAL_DATA_PATH = os.getenv("HISTORICAL_DATA_PATH", "historical_data")
UI_REFRESH_INTERVAL = int(os.getenv("UI_REFRESH_INTERVAL", "1000"))
LIVE_LOOP_INTERVAL = float(os.getenv("LIVE_LOOP_INTERVAL", "5"))
SIMULATION_ORDER_DELAY = float(os.getenv("SIMULATION_ORDER_DELAY", "0.5"))

# ───────────────────────────────────────────────
# Trading Defaults
# ───────────────────────────────────────────────
DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "BTCUSDT")
SIMULATION_START_BALANCE = float(
    os.getenv("SIMULATION_START_BALANCE", "1000.0"))
TRADE_SIZE_PERCENT = float(os.getenv("TRADE_SIZE_PERCENT", "0.05"))
MIN_TRADE_AMOUNT_USD = float(os.getenv("MIN_TRADE_AMOUNT_USD", "10.0"))
FEE_PERCENTAGE = float(os.getenv("FEE_PERCENTAGE", "0.002"))

# Intervals for learning/trading
PRIMARY_INTERVALS = ["5m", "15m"]

# Max klines per request — keep well under Bybit's 1000 limit
MAX_KLINES_REQUEST = 900
KLINE_BACKFILL_HOURS = 48  # initial backfill when missing data

# ───────────────────────────────────────────────
# Risk & Exposure
# ───────────────────────────────────────────────
RISK_CAPS = {
    "crypto_spot": {"per_pair_pct": 0.15, "portfolio_concurrent_pct": 0.30},
    "perp": {"per_pair_pct": 0.15, "portfolio_concurrent_pct": 0.30},
    "forex": {"per_pair_pct": 0.10, "portfolio_concurrent_pct": 0.20},
    "options": {"per_pair_pct": 0.05, "portfolio_concurrent_pct": 0.10}
}

CANARY_RAMP_SCHEDULE = [0.02, 0.03, 0.05]

KPI_TARGETS = {
    "win_rate": 0.70,
    "sharpe_ratio": 1.8,
    "avg_profit_swing": 0.10,
    "avg_profit_scalp": 0.002,
    "max_drawdown": 0.15,
    "consec_loss_cooldown": 3
}

# ───────────────────────────────────────────────
# Fee/Slippage
# ───────────────────────────────────────────────
FEE_MODEL = {
    "bybit": {
        "spot": {"taker": 0.001, "maker": 0.0007},
        "perp": {"taker": 0.00055, "maker": 0.0002}
    },
    "forex": {"spread_bps": 0.8},
    "options": {"commission_per_contract": 0.65}
}

SLIPPAGE_BPS = {
    "crypto_spot": 5,
    "perp": 3,
    "forex": 1,
    "options": 5
}

# ───────────────────────────────────────────────
# Backtester
# ───────────────────────────────────────────────
MAX_SIMULATION_PAIRS = int(os.getenv("MAX_SIMULATION_PAIRS", "5"))

# ───────────────────────────────────────────────
# Optimization
# ───────────────────────────────────────────────
OPTIMIZATION_METHOD = os.getenv("OPTIMIZATION_METHOD", "grid_search")
OPTIMIZATION_PARAMETERS = {
    "ema_long_period": [21, 50],
    "ema_short_period_1": [5, 8],
    "ema_short_period_2": [13, 21],
    "rsi_period": [14],
    "rsi_level": [50],
    "atr_period": [14],
    "tp_atr_multiplier": [2.0, 2.5],
    "sl_atr_multiplier": [1.5]
}
EA_POPULATION_SIZE = int(os.getenv("EA_POPULATION_SIZE", "20"))
EA_CROSSOVER_PROB = float(os.getenv("EA_CROSSOVER_PROB", "0.5"))
EA_MUTATION_PROB = float(os.getenv("EA_MUTATION_PROB", "0.2"))
EA_GENERATIONS = int(os.getenv("EA_GENERATIONS", "10"))

# ───────────────────────────────────────────────
# Misc
# ───────────────────────────────────────────────
USE_TESTNET = USE_SIMULATION
USE_MAINNET_IN_SIMULATION = os.getenv(
    "USE_MAINNET_IN_SIMULATION", "false"
).lower() in {"1", "true", "yes", "on"}
