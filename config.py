# modules/config.py
import os
from utils.utilities import configure_logging

# ========================
# Environment Configuration
# ========================
ENVIRONMENT = os.getenv("TRADING_ENV", "simulation")  # "production" or "simulation"
USE_SIMULATION = True if ENVIRONMENT == "simulation" else False  # Added missing parameter


SIMULATION_BYBIT_API_KEY = "ELdo9O5wuzDwyoVYDY"
SIMULATION_BYBIT_API_SECRET = "g2MHBRpVyjAw7ITdmz2RT6cg8rRFIZGqtDrG"

# ========================
# Exchange API Configuration
# ========================
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "I2CQH3mXhPLIuWtS2W")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "YmV6jkihwZDTFyIaiOarho1tzRr2QZ9xuNBv")
API_REQUEST_TIMEOUT = 10  # Seconds
API_RETRY_ATTEMPTS = 3

# ========================
# Trading Parameters
# ========================
DEFAULT_TRADE_AMOUNT = 100.0  # USDT per trade
MAX_OPEN_POSITIONS = 5         # Maximum concurrent trades
FEE_PERCENTAGE = 0.002        # Exchange fee percentage
SLIPPAGE_PERCENTAGE = 0.05    # Estimated price slippage

# ========================
# Risk Management
# ========================
MAX_DAILY_DRAWDOWN = 5.0      # Max daily loss percentage
STOP_LOSS_PERCENT = 10        # Initial stop loss percentage
TAKE_PROFIT_PERCENT = 20      # Initial take profit percentage
RISK_PER_TRADE = 2.0          # Percentage of capital per trade

# ========================
# Simulation Parameters
# ========================
SIMULATION_ORDER_DELAY = 0.5  # Order execution delay in seconds
SIMULATION_START_BALANCE = 1000.0  # Initial simulation balance
MAX_SIMULATION_PAIRS = 50     # Max pairs to include in simulation
HISTORICAL_DATA_PATH = "data/historical"  # Path to historical data

# ========================
# Notification Settings
# ========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7611295732:AAHazYz46ynfueYthvQXvQRA9bYlxihEf1c")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "380533508")
NOTIFY_ON_SL_TP = True        # Send alerts for stop loss/take profit
NOTIFY_EVERY_TRADE = True     # Send trade execution notifications

# ========================
# System Configuration
# ========================
LOG_LEVEL = "INFO"            # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "logs/bot.log"
DATA_CACHE_TTL = 3600         # Seconds to keep market data cached
MAX_LOG_SIZE = 50             # MB per log file
LOG_BACKUP_COUNT = 3

# ========================
# Development Flags
# ========================
DEBUG_MODE = False            # Enable debug features
DRY_RUN = True               # Prevent real trades even in production

# ========================
# Optimization Parameters
# ========================
OPTIMIZATION_METHOD = "random_search"
OPTIMIZATION_PARAMETERS = {
    'ma_fast': {'min': 10, 'max': 20},
    'ma_slow': {'min': 20, 'max': 30}
}

# ========================
# Evolutionary Algorithm Settings
# ========================
EA_POPULATION_SIZE = 100
EA_GENERATIONS = 50
EA_CROSSOVER_PROB = 0.7
EA_MUTATION_PROB = 0.2



ERROR_RATE_THRESHOLD = 5
CRITICAL_ERROR_CODES = [5000, 6000]  # Includes RiskViolation and OrderExecution errors
LOG_LEVEL = "INFO"
LOG_FILE = "logs/bot.log"