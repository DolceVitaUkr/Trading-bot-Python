import os
from typing import Literal
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ────────────────────────────────────────────────────────────────────────────────
# Environment & Mode
# ────────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT: "simulation" or "production"
ENVIRONMENT = os.getenv("ENVIRONMENT", "simulation").lower()
USE_SIMULATION = ENVIRONMENT == "simulation"

# Rollout Stage: 1..5 (Stage logic handled in rollout_manager)
ROLLOUT_STAGE = int(os.getenv("ROLLOUT_STAGE", "1"))

# Exchange profile: "spot", "perp", "spot+perp"
EXCHANGE_PROFILE: Literal["spot", "perp", "spot+perp"] = os.getenv(
    "EXCHANGE_PROFILE", "spot"
).lower()

# Domain toggles (Forex / Options start OFF by default)
FOREX_ENABLED_DEFAULT = False
OPTIONS_ENABLED_DEFAULT = False

# ────────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()          # e.g. DEBUG, INFO, WARNING
LOG_FILE  = os.getenv("LOG_FILE", "bot.log")                # path to write log file

# ────────────────────────────────────────────────────────────────────────────────
# Bybit API Credentials
# ────────────────────────────────────────────────────────────────────────────────
# Env var names must be exactly these:
#   BYBIT_API_KEY, BYBIT_API_SECRET
#   SIMULATION_BYBIT_API_KEY, SIMULATION_BYBIT_API_SECRET
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "I2CQH3mXhPLIuWtS2W")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "YmV6jkihwZDTFyIaiOarho1tzRr2QZ9xuNBv")

# For simulation trading (TESTNET)
SIMULATION_BYBIT_API_KEY = os.getenv("SIMULATION_BYBIT_API_KEY", "ELdo9O5wuzDwyoVYDY")
SIMULATION_BYBIT_API_SECRET = os.getenv("SIMULATION_BYBIT_API_SECRET", "g2MHBRpVyjAw7ITdmz2RT6cg8rRFIZGqtDrG")


# ────────────────────────────────────────────────────────────────────────────────
# Telegram Bot for Notifications
# ────────────────────────────────────────────────────────────────────────────────
# Env var names:
#   TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7611295732:AAHazYz46ynfueYthvQXvQRA9bYlxihEf1c")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "380533508")

# Async sending for Telegram notifier
ASYNC_TELEGRAM = os.getenv("ASYNC_TELEGRAM", "true").lower() in {"1", "true", "yes", "on"}

# Notification policy
TELEGRAM_PAPER_RECAP_MIN = int(os.getenv("TELEGRAM_PAPER_RECAP_MIN", "60"))
TELEGRAM_LIVE_ALERT_LEVEL = os.getenv("TELEGRAM_LIVE_ALERT_LEVEL", "normal")  # quiet|normal|verbose
TELEGRAM_HEARTBEAT_MIN = int(os.getenv("TELEGRAM_HEARTBEAT_MIN", "10"))

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
DEFAULT_SYMBOL           = os.getenv("DEFAULT_SYMBOL", "BTC/USDT")
SIMULATION_START_BALANCE = float(os.getenv("SIMULATION_START_BALANCE", "1000.0"))

# Risk sizing per trade (domain-specific handled in RISK_CAPS below)
TRADE_SIZE_PERCENT     = float(os.getenv("TRADE_SIZE_PERCENT", "0.05"))  # 5% of balance
MIN_TRADE_AMOUNT_USD   = float(os.getenv("MIN_TRADE_AMOUNT_USD", "10.0"))

# Exchange fees (as fraction of trade value)
FEE_PERCENTAGE         = float(os.getenv("FEE_PERCENTAGE", "0.002"))     # 0.2%

# ────────────────────────────────────────────────────────────────────────────────
# Risk & Exposure Caps
# ────────────────────────────────────────────────────────────────────────────────
RISK_CAPS = {
    "crypto_spot":  {"per_pair_pct": 0.15, "portfolio_concurrent_pct": 0.30},
    "perp":         {"per_pair_pct": 0.15, "portfolio_concurrent_pct": 0.30},
    "forex":        {"per_pair_pct": 0.10, "portfolio_concurrent_pct": 0.20},
    "options":      {"per_pair_pct": 0.05, "portfolio_concurrent_pct": 0.10}
}

# Canary → Ramp schedule (percent of balance)
CANARY_RAMP_SCHEDULE = [0.02, 0.03, 0.05]

# KPI guardrails
KPI_TARGETS = {
    "win_rate": 0.70,           # per domain
    "sharpe_ratio": 1.8,
    "avg_profit_swing": 0.10,   # 10%+ for swing trades
    "avg_profit_scalp": 0.002,  # 0.2%+ for scalps
    "max_drawdown": 0.15,
    "consec_loss_cooldown": 3
}

# ────────────────────────────────────────────────────────────────────────────────
# Fee/Slippage Models
# ────────────────────────────────────────────────────────────────────────────────
FEE_MODEL = {
    "bybit": {
        "spot": {"taker": 0.001, "maker": 0.0007},
        "perp": {"taker": 0.00055, "maker": 0.0002}
    },
    "forex": {"spread_bps": 0.8},
    "options": {"commission_per_contract": 0.65}
}

# Slippage assumptions in simulation/backtest (basis points)
SLIPPAGE_BPS = {
    "crypto_spot": 5,
    "perp": 3,
    "forex": 1,
    "options": 5
}

# ────────────────────────────────────────────────────────────────────────────────
# Error Handling
# ────────────────────────────────────────────────────────────────────────────────
CRITICAL_ERROR_CODES   = {5000, 6000, 7000, 8000}
ERROR_RATE_THRESHOLD   = int(os.getenv("ERROR_RATE_THRESHOLD", "5"))

# ────────────────────────────────────────────────────────────────────────────────
# Backtester / Simulator
# ────────────────────────────────────────────────────────────────────────────────
MAX_SIMULATION_PAIRS   = int(os.getenv("MAX_SIMULATION_PAIRS", "5"))

# ────────────────────────────────────────────────────────────────────────────────
# Genetic / Evolutionary Optimization
# ────────────────────────────────────────────────────────────────────────────────
OPTIMIZATION_METHOD    = os.getenv("OPTIMIZATION_METHOD", "grid_search")  # grid_search | random_search | evolutionary
OPTIMIZATION_PARAMETERS = {
    "ema_short": { "min": 5,   "max": 20   },
    "ema_long":  { "min": 20,  "max": 100  },
    "rsi_period":{ "min": 5,   "max": 30   },
    "rsi_overbought": [70, 75, 80],
    "rsi_oversold":   [20, 25, 30]
}
EA_POPULATION_SIZE    = int(os.getenv("EA_POPULATION_SIZE", "20"))
EA_CROSSOVER_PROB     = float(os.getenv("EA_CROSSOVER_PROB", "0.5"))
EA_MUTATION_PROB      = float(os.getenv("EA_MUTATION_PROB", "0.2"))
EA_GENERATIONS        = int(os.getenv("EA_GENERATIONS", "10"))

# ────────────────────────────────────────────────────────────────────────────────
# Misc / Backwards Compat
# ────────────────────────────────────────────────────────────────────────────────
USE_TESTNET = USE_SIMULATION  # legacy alias






