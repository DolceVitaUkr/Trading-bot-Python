# modules/config.py
import os

# ========================
# Environment Configuration
# ========================
ENVIRONMENT = os.getenv("TRADING_ENV", "simulation")  # "production" or "simulation"
USE_SIMULATION = False  # Added missing parameter


SIMULATION_BYBIT_API_KEY = "ELdo9O5wuzDwyoVYDY"
SIMULATION_BYBIT_API_SECRET = "g2MHBRpVyjAw7ITdmz2RT6cg8rRFIZGqtDrG"

# ========================
# Exchange API Configuration
# ========================
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "I2CQH3mXhPLIuWtS2W")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "YmV6jkihwZDTFyIaiOarho1tzRr2QZ9xuNBv")
API_REQUEST_TIMEOUT = 10  # Seconds
API_RETRY_ATTEMPTS = 3

# --- Simulation Settings ---
SIMULATION_INITIAL_BALANCE = 1000.0  # Starting virtual balance in USD for paper trading simulation

# --- Trading Parameters ---
MIN_TRADE_SIZE_USD = 10.0  # Minimum order size in USD for live trades
MIN_TRADE_SIZE_PERCENT = 0.05  # Minimum order size as a percentage of account balance for live trades (e.g., 0.05 = 5%)
# (The actual minimum trade value can be determined as max(MIN_TRADE_SIZE_USD, account_balance * MIN_TRADE_SIZE_PERCENT).)

# --- Risk Management ---
DEFAULT_STOP_LOSS_PERCENT = 0.15  # Default stop-loss level as a fraction of entry price (15% loss)
MAX_STOP_LOSS_PERCENT = 0.20  # Maximum allowable stop-loss as a fraction of entry price (20% loss)

# --- Reward System Settings ---
REWARD_MULTIPLIER = 1.0  # Multiplier for reward (profit) points in the reward system
PENALTY_MULTIPLIER = 3.0  # Multiplier for penalty (loss) points in the reward system (e.g., 3.0 means penalties are triple weighted relative to rewards)

# --- Data and Storage ---
HISTORICAL_DATA_PATH = "data"  # Directory for storing historical market data files (will be created if it doesn't exist)

# --- Logging Configuration ---
LOG_DIR = "logs"  # Directory for log files
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the log directory exists
LOG_FILE = os.path.join(LOG_DIR, "trading_bot.log")  # Log file path for recording runtime logs
LOG_LEVEL = "INFO"  # Logging level for the application ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# --- Trading Behavior Settings ---
UI_REFRESH_INTERVAL = 1000  # UI update interval in milliseconds for refreshing data in the interface
TOP_PAIRS_REFRESH_INTERVAL = 3600  # Interval in seconds to refresh the list of top trading pairs (e.g., 3600 = 1 hour)
MARKET_SENTIMENT_ENABLED = True  # Whether to incorporate market sentiment analysis in trading decisions
MARKET_SENTIMENT_THRESHOLD = 0.0  # Threshold for market sentiment indicator to influence trades (e.g., >0 for bullish sentiment, <0 for bearish)

# --- Error Handling Defaults ---
NETWORK_MAX_RETRIES = 3  # Maximum number of retry attempts for network/API calls on failure
NETWORK_TIMEOUT_SECONDS = 10  # Timeout in seconds for network requests (e.g., API calls)
# (If a network call fails or times out, it can be retried up to NETWORK_MAX_RETRIES times with appropriate delays.)

# --- Optimization Settings ---
OPTIMIZATION_METHOD = "grid_search"  # Default parameter optimization method ("grid_search", "random_search", "evolutionary")
OPTIMIZATION_PARAMETERS = {
    # Define parameter ranges or values for optimization if using parameter optimization module.
    # e.g., "threshold": [0.1, 0.2, 0.3],
    #       "window_size": [10, 20, 50]
}

# --- Notification Settings (Telegram) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7611295732:AAHazYz46ynfueYthvQXvQRA9bYlxihEf1c")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "380533508")
NOTIFY_ON_SL_TP = True        # Send alerts for stop loss/take profit
NOTIFY_EVERY_TRADE = True     # Send trade execution notifications

#