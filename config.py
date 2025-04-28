# modules/config.py
import os

# ────────────────────────────────────────────────────────────────────────────────
# Environment & Mode
# ────────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT: "simulation" or "production"
ENVIRONMENT = os.getenv("ENVIRONMENT", "simulation").lower()
USE_SIMULATION = ENVIRONMENT == "simulation"

# ────────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()          # e.g. DEBUG, INFO, WARNING
LOG_FILE  = os.getenv("LOG_FILE", "bot.log")                # path to write log file

# ────────────────────────────────────────────────────────────────────────────────
# Bybit API Credentials
# ────────────────────────────────────────────────────────────────────────────────
# For real trading (MAINNET)
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "I2CQH3mXhPLIuWtS2W")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "YmV6jkihwZDTFyIaiOarho1tzRr2QZ9xuNBv")

# For simulation trading (TESTNET)
SIMULATION_BYBIT_API_KEY = "ELdo9O5wuzDwyoVYDY"
SIMULATION_BYBIT_API_SECRET = "g2MHBRpVyjAw7ITdmz2RT6cg8rRFIZGqtDrG"

# ────────────────────────────────────────────────────────────────────────────────
# Telegram Bot for Notifications
# ────────────────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7611295732:AAHazYz46ynfueYthvQXvQRA9bYlxihEf1c")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "380533508")

# ────────────────────────────────────────────────────────────────────────────────
# Paths & Intervals
# ────────────────────────────────────────────────────────────────────────────────
HISTORICAL_DATA_PATH = os.getenv("HISTORICAL_DATA_PATH", "historical_data")
UI_REFRESH_INTERVAL  = int(os.getenv("UI_REFRESH_INTERVAL", "1000"))  # ms between UI updates
LIVE_LOOP_INTERVAL   = float(os.getenv("LIVE_LOOP_INTERVAL", "5"))    # seconds between live ticks
SIMULATION_ORDER_DELAY = float(os.getenv("SIMULATION_ORDER_DELAY", "0.5"))  # sec delay in sim order

# ────────────────────────────────────────────────────────────────────────────────
# Trading Defaults
# ────────────────────────────────────────────────────────────────────────────────
DEFAULT_SYMBOL         = os.getenv("DEFAULT_SYMBOL", "BTC/USDT")
SIMULATION_START_BALANCE = float(os.getenv("SIMULATION_START_BALANCE", "1000.0"))

# When live trading, risk at most this fraction per trade, but at least this USD
TRADE_SIZE_PERCENT     = float(os.getenv("TRADE_SIZE_PERCENT", "0.05"))  # 5% of balance
MIN_TRADE_AMOUNT_USD   = float(os.getenv("MIN_TRADE_AMOUNT_USD", "10.0"))

# Exchange fees (as fraction of trade value)
FEE_PERCENTAGE         = float(os.getenv("FEE_PERCENTAGE", "0.002"))     # 0.2%

# ────────────────────────────────────────────────────────────────────────────────
# Error Handling
# ────────────────────────────────────────────────────────────────────────────────
# Which error codes are considered "critical" and should trigger immediate alerts
CRITICAL_ERROR_CODES   = {5000, 6000, 7000, 8000}   # e.g. RiskViolationError, OrderExecutionError, ConfigurationError, NotificationError
# If an error code occurs more than this many times in a row, circuit breaker trips
ERROR_RATE_THRESHOLD   = int(os.getenv("ERROR_RATE_THRESHOLD", "5"))

# ────────────────────────────────────────────────────────────────────────────────
# Backtester / Simulator
# ────────────────────────────────────────────────────────────────────────────────
# How many top pairs to test in backtests by default (if you incorporate pair rotation)
MAX_SIMULATION_PAIRS   = int(os.getenv("MAX_SIMULATION_PAIRS", "5"))

# ────────────────────────────────────────────────────────────────────────────────
# Genetic / Evolutionary Optimization
# ────────────────────────────────────────────────────────────────────────────────
OPTIMIZATION_METHOD    = os.getenv("OPTIMIZATION_METHOD", "grid_search")  # grid_search | random_search | evolutionary

# Define which strategy parameters to tune, and their ranges or lists
# Example params for EMA crossover + RSI threshold strategies:
OPTIMIZATION_PARAMETERS = {
    "ema_short": { "min": 5,   "max": 20   },  # short EMA window
    "ema_long":  { "min": 20,  "max": 100  },  # long EMA window
    "rsi_period":{ "min": 5,   "max": 30   },  # RSI lookback
    "rsi_overbought": [70, 75, 80],            # list of discrete choices
    "rsi_oversold":   [20, 25, 30]
}

# DEAP evolutionary algorithm defaults (only if OPTIMIZATION_METHOD == "evolutionary")
EA_POPULATION_SIZE    = int(os.getenv("EA_POPULATION_SIZE", "20"))
EA_CROSSOVER_PROB     = float(os.getenv("EA_CROSSOVER_PROB", "0.5"))
EA_MUTATION_PROB      = float(os.getenv("EA_MUTATION_PROB", "0.2"))
EA_GENERATIONS        = int(os.getenv("EA_GENERATIONS", "10"))

# ────────────────────────────────────────────────────────────────────────────────
# Misc / Backwards Compat
# ────────────────────────────────────────────────────────────────────────────────
USE_TESTNET      = USE_SIMULATION   # legacy alias