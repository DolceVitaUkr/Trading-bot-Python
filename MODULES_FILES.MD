Module Files Overview
Below is a detailed breakdown of each module in the project, describing the purpose and structure of each file. The project follows a modular architecture, with each module encapsulating specific functionality (data handling, trading logic, UI, etc.). This section serves as a reference for developers to understand and navigate the codebase.
config.py
Role: Central configuration file for the bot. It defines various constants and settings used across modules, such as API keys, file paths, and tunable parameters. Key Contents:
API Credentials: BYBIT_API_KEY, BYBIT_API_SECRET for live trading on Bybit; SIMULATION_BYBIT_API_KEY, SIMULATION_BYBIT_API_SECRET if using a separate key for simulation or testnet (optional). These are used by the exchange module when placing orders or fetching private data.
Telegram Config: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID for sending Telegram notifications.
Modes and Toggles: Flags like USE_SIMULATION (True/False to indicate if the bot should run in simulation mode), and possibly USE_TESTNET if applicable.
Paths: HISTORICAL_DATA_PATH (directory where historical market data and training datasets are stored, e.g. "historical_data/"), and LOG_FILE (path to the log file, e.g. "bot.log").
Logging Level: LOG_LEVEL (e.g. "INFO" or "DEBUG" to control verbosity).
Error Handling Settings: CRITICAL_ERROR_CODES (list of error codes that should be treated as critical, triggering special handling or alerts), ERROR_RATE_THRESHOLD (if the same error occurs this many times, activate circuit breaker).
Trading Parameters: e.g. INITIAL_TRAINING_BALANCE = 1000.0 (starting paper trading capital), REAL_TRADE_PERCENT = 0.05 (5% of balance per trade in live mode), MIN_TRADE_AMOUNT = 10.0 (minimum USD value per trade).
Other Tunables: UI_REFRESH_INTERVAL (GUI update frequency in milliseconds), SIMULATION_ORDER_DELAY (artificial delay to simulate network latency when executing simulated orders), OPTIMIZATION_METHOD (e.g. "evolutionary" or "grid_search"), OPTIMIZATION_PARAMETERS (dictionary of parameters to tune and their ranges for the optimizer).
Structure: This is not a class or function but a simple Python module with variables. Other modules import config and use config.VAR_NAME to access these settings. By changing values here, one can configure the bot’s behavior without modifying the logic code.
modules/data_manager.py
Role: Manages retrieval and storage of market data (historical price candles). This module abstracts the details of how data is fetched from the exchange and stored locally, providing an interface for the strategy to get historical price series easily. Classes and Functions:
Class DataManager:
Attributes:
data_folder: Directory path where data files are stored (initialized from config.HISTORICAL_DATA_PATH). Each trading pair’s data is stored in a separate Parquet file for each timeframe.
test_mode: Boolean indicating if the DataManager should avoid real API calls (used in unit tests or simulation contexts).
exchange: An instance of Exchange (from modules.exchange) used to fetch data when not in test mode.
cache: An in-memory cache (dictionary) for storing recently loaded DataFrames to speed up repeated access.
Methods:
__init__(self, test_mode=False): Initializes the DataManager. If not in test_mode, it creates an Exchange instance to allow real API calls. It also ensures the data directory exists and prepares a cache.
_ensure_data_folder(): Internal helper to create the data folder if it doesn’t exist.
_get_filename(symbol, timeframe): Internal helper that constructs a standardized filename for a given symbol and timeframe. In test_mode, it prefixes filenames with "test_" to avoid clobbering real data.
_generate_mock_klines(periods, timeframe): Internal helper used in test mode to create synthetic OHLCV data for testing. It generates periods number of fake candles with incremental prices (useful for deterministic test cases).
update_klines(symbol, timeframe, klines=None) -> bool: Core function to fetch and append new market data. If klines are provided (as a list of raw OHLCV records), it will merge them into the existing dataset for the symbol/timeframe. If klines is None and test_mode is True, it generates mock data; if klines is None and not test_mode, it will call the exchange API to get fresh data (not shown in snippet, but intended). It then appends new data, removes duplicates, sorts by time, and saves the updated dataset to disk in Parquet format. Returns True if successful.
load_historical_data(symbol, timeframe) -> pd.DataFrame: Loads the full historical dataset for the given symbol and timeframe from the local Parquet file. It uses an internal cache to avoid reading from disk multiple times for the same data. If the data isn’t cached, it reads the Parquet file (which contains OHLCV columns with a DateTime index) and caches it for quick reuse. If the file isn’t found, it raises FileNotFoundError (the calling code should handle this, possibly by invoking an update to fetch data).
_process_data(raw) -> pd.DataFrame: Helper that takes raw OHLCV data (list of lists format) and converts it into a pandas DataFrame with proper columns (["timestamp", "open", "high", "low", "close", "volume"]) and a DateTime index (in UTC). It ensures no duplicate timestamps and that data is sorted.
_save_data(df, path): Saves a DataFrame to a Parquet file at the given path. It localizes the DateTime index to naive (no timezone) for storage.
_load_data(path) -> pd.DataFrame: Reads a Parquet file and returns a DataFrame with a UTC DateTime index.
Usage: The DataManager is used by the strategy loop to get historical prices. For example, before making a trading decision, the bot might call data_manager.load_historical_data("BTC/USDT", "15m") to get recent 15-minute candle data for BTC/USDT. Or it might call update_klines periodically to fetch the latest candle that closed. This ensures the bot always has up-to-date market context. In tests, DataManager is instantiated in test_mode=True so it doesn’t attempt real API calls; instead it will use the _generate_mock_klines to simulate data.
modules/exchange.py
Role: Wraps interaction with the Bybit exchange (or a simulation thereof). This module provides methods to get current market prices and to execute trades, while keeping track of open positions in simulation mode. It serves as a single interface for both paper trading and live trading depending on context. Classes and Functions:
Class Exchange:
Attributes:
base_url: Base URL for API requests (default set to Bybit’s base endpoint for real trading). Note: The code ignores any use_testnet flag to always use the real market data endpoint, as the objective is to always train on real market data even if using testnet for orders.
positions: A dictionary to store open position data in simulation mode. Keys are symbols, values are details of the position (side, quantity, entry price, etc.).
Methods:
__init__(self, use_testnet=False): Initializes the Exchange object. If one wanted, this could toggle base_url to testnet, but in this implementation use_testnet is not utilized (we always use real data source for consistency in training).
get_price(self, symbol: str) -> float: Fetches the latest market price for the given symbol from Bybit’s API. Internally, it calls the /v5/market/tickers endpoint with the symbol and parses the JSON to extract the last traded price. If the response structure isn’t as expected, it raises an error. This method is critical for getting up-to-the-moment price, used in both making trading decisions and in filling orders at current prices in simulation.
create_order(self, symbol: str, side: str, quantity: float = 1.0, price: float = None): Simulates placing an order on the exchange. In the current implementation, this method does not actually send orders to Bybit but instead updates the positions dictionary to reflect the effect of an order:
It determines an execution price: if a price is given it may use that (for a limit order simulation), otherwise it calls get_price to simulate a market order at current price.
It checks if there’s an existing position in the symbol:
If no position exists and this is a buy/long order, it opens a new long position with the given quantity at the exec_price.
If no position exists and this is a sell/short order, it opens a new short position.
If a position already exists:
If the new order is in the opposite direction (e.g. currently long and new order is sell/short), it will close or reduce the position. If the quantity of the new order is greater or equal to the current position, it closes it out (and if there’s excess order size, it flips to a position in the opposite direction with the remainder). If the new order quantity is smaller, it reduces the position size accordingly.
If the new order is in the same direction as the existing position (adding to a long or short), it will increase the position’s quantity and recalculate the average entry price.
It returns a dictionary representing the new or updated position (or None if a position was closed fully).
Note: This is a simplified fill logic for simulation. It assumes orders execute immediately at current price. In a real scenario, one would need to handle order book simulation, partial fills, etc., but that’s beyond scope here.
get_position(self, symbol: str): Returns the current open position for the given symbol from the positions dict, or None if no position is open.
close_position(self, symbol: str): Simulates closing an open position at the current market price. It will fetch the current price, calculate P&L (profit and loss) based on the difference between entry price and current price for the position quantity, remove the position from the internal tracking, and return a summary of the closed position (including P&L). This is useful for the bot to call when it wants to explicitly exit a position (for example, on Stop Trading if user chose to close positions, the TradeExecutor could call this for each open position in simulation mode to log the P&L).
Alias: At the bottom, ExchangeAPI = Exchange. This alias is used elsewhere to refer to the exchange class in a generic way.
Live Trading Note: In the current code, the real API integration is only partial (only get_price uses a real HTTP call). create_order is not actually sending requests to Bybit’s trading endpoints. In a production scenario, we would extend this class with methods to truly execute orders when simulation_mode is False, possibly using an official Bybit SDK or sending signed requests. The TradeExecutor (next module) actually handles deciding whether to simulate or send real orders. For our bot, this Exchange class suffices for simulation. For live trading, one could integrate something like ccxt or the Bybit HTTP API for actual order placement using the same interface.
modules/trade_executor.py
Role: Handles the final step of placing trades and ensures that each order conforms to requirements. The TradeExecutor acts as an intermediary between the strategy’s trade decision and the Exchange module’s actions. It also integrates messaging for trade events. Classes and Functions:
Function send_telegram_sync(bot, message): A utility function to send a Telegram message using a TelegramNotifier bot instance. It checks if there is an active asyncio event loop and either schedules the send or runs it to completion. This ensures that even if called from synchronous code, the message will be delivered. The TradeExecutor uses this to send out messages for each order executed or error encountered.
Class TradeExecutor:
Attributes:
simulation_mode: Boolean indicating whether to simulate trades (True for paper trading) or execute real trades (False for live).
exchange: An instance of ExchangeAPI (which is the Exchange class) to interact with for price and orders. This is created on initialization.
telegram_bot: An instance of TelegramNotifier for sending notifications. It’s created with disable_async=True to allow simple synchronous use.
Methods:
__init__(self, simulation_mode=True): Initializes the TradeExecutor. It saves the mode, instantiates an ExchangeAPI (which will use real price fetching but simulated order logic if needed), and sets up a TelegramNotifier for immediate messaging.
execute_order(self, symbol, side, amount, price, order_type="limit"): This is the main method to execute a trade order as directed by the strategy.
It first sanitizes the symbol (if symbol is "BTC/USDT", it might remove the slash to match exchange API format like "BTCUSDT" if needed).
It obtains the current market price via exchange.get_price (here referred to as get_current_price in code – which should map to the same functionality).
It checks the amount against a minimum order size. Typically, exchanges have a minimum trade size; this code uses either a hardcoded minimum (like 0.002 in terms of coin amount) or calls exchange.get_min_order_size(symbol) if available. If the requested amount is below the minimum, it raises a ValueError to prevent an order that would be rejected.
It formats the price to the correct precision by calling exchange.get_price_precision(symbol). This ensures the price is rounded to what the exchange expects (e.g., 2 decimal places for price).
It logs an informational message indicating whether it is simulating or executing a real order, including details (side, symbol, amount, price). This log is both printed to console and sent via Telegram for transparency.
It then checks simulation_mode:
If True (simulation), it simply waits a short delay (config.SIMULATION_ORDER_DELAY seconds) to mimic network latency, then returns a dictionary like {"status": "simulated", "symbol": ..., "side": ..., "amount": ..., "price": ...}. No actual position is recorded here because in simulation mode, the Exchange object itself already tracks positions when we call its methods (though currently our Exchange.create_order was simulation logic; in this stub, TradeExecutor isn’t calling Exchange.create_order in simulation path, it’s just mocking the result – one might adjust to actually call self.exchange.create_order even in simulation to update positions).
If False (live trading mode), it attempts to place a real order by calling self.exchange.create_order(...). In our implementation, this still calls the Exchange’s create_order (which doesn’t communicate with Bybit in provided code), but the idea is that in a fully implemented version, this would hit the real API. If the create_order call succeeds, it returns whatever that function returns (likely an order confirmation dict). If an exception is raised during order placement, it catches it, logs an error message, sends a Telegram alert with the failure, and re-raises the exception to be handled by higher-level logic (like the error handler).
The TradeExecutor essentially ensures the mechanics of order execution are handled in one place – checking minimums, rounding prices, logging, and notifications. This keeps the strategy code cleaner and focused on what to trade rather than how to place the trade.
Additional Context: The TradeExecutor can be expanded with more features, like:
Handling different order types (market vs limit vs stop orders) in more detail. Currently the parameter is accepted but in simulation all orders act like market.
Incorporating results of trade into the bot’s portfolio tracking (in simulation, we rely on Exchange to track positions; in live, after an order we might want to update an internal view of positions immediately or trust the exchange’s state).
Ensuring thread-safety if called from multiple strategy threads (currently, the design uses a single strategy thread so it’s fine).
By having TelegramNotifier integrated here, any order a user places (either automatic or manual via UI) will immediately be reported to them, which is very useful for live monitoring.
modules/technical_indicators.py
Role: Provides functions for calculating common technical indicators on price data. This module is a self-contained library of indicator computations that can be used by the strategy or elsewhere to interpret market data. Functions:
moving_average(data: List[float], period: int) -> Optional[float]: Computes the Simple Moving Average (SMA) of the last period values in the list data. Returns None if not enough data points are present.
exponential_moving_average(data: List[float], period: int) -> Optional[float]: Computes the Exponential Moving Average. It starts by calculating a simple average of the first period points, then applies the EMA formula to the rest of the data. Returns the final EMA value.
adx(high: List[float], low: List[float], close: List[float], period: int = 14) -> Optional[float]: Calculates the Average Directional Index, a measure of trend strength. It uses the classic formula: compute True Range (TR), +DI and -DI, then DX and smooth into ADX. It returns the latest ADX value for the provided data arrays. If there’s not enough data (need 2*period points to start), returns None.
cci(high: List[float], low: List[float], close: List[float], period: int = 20) -> Optional[float]: Computes the Commodity Channel Index. It calculates the Typical Price (average of high, low, close) and compares it to its moving average over period, scaled by mean deviation. Returns the latest CCI value.
williams_r(high: List[float], low: List[float], close: List[float], period: int = 14) -> Optional[float]: Computes Williams %R (percent range), which is a momentum indicator showing the level of the close relative to the recent high-low range. It returns a value between 0 and -100 (0 means the close is at the highest high of the lookback, -100 means it’s at the lowest low).
Structure: All these are top-level functions using pure Python or numpy for calculations. The module imports numpy for any needed numeric operations and possibly torch if any advanced indicators needed it (though current functions don’t use torch). In some versions, a class TechnicalIndicators could be defined to group these, but in the provided content we only see functions. If a class TechnicalIndicators exists, it might simply wrap these functions for OO access. Usage: The strategy (SelfLearningBot) might use these to compute features. For example, it could maintain a rolling list of closes and call moving_average(closes, 50) to get a 50-period MA as part of state. Or compute adx(highs, lows, closes, 14) each time a new candle closes to decide if market is trending. The simplicity of this module means it has no side effects or state – it just computes and returns values. Note: Logging is set up at the top (with basicConfig to INFO). If needed, these functions will log errors if something goes wrong (like divide by zero or bad input), but typically they return None when data is insufficient. This is by design so the strategy can handle a None (meaning indicator not available yet) gracefully. Extending this module is straightforward: you can add any other indicator calculation (RSI, Bollinger Bands, etc.) following the same pattern. Since it doesn’t depend on internal state, it won’t interfere with other components.
modules/risk_management.py
Role: Defines risk management calculations and checks to ensure trading strategies remain within safe bounds. This includes stop-loss calculations, dynamic risk adjustments, and custom exceptions for risk breaches. Functions and Classes:
Risk Limit Calculations:
calculate_stop_loss(entry_price: float, risk_factor: float) -> float (assumed): Likely computes a stop-loss price given an entry price and a risk tolerance. For example, it might return a price that is 2% below the entry for longs (if risk_factor=0.02). This would be used by the TradeSimulator or strategy to set stop-loss orders on open positions.
dynamic_adjustment(...): This function (referenced in TradeSimulator) might adjust parameters like trade size or risk factor depending on current conditions. For instance, if the bot is on a losing streak, it might reduce trade_size_percent to prevent further drawdown, or if volatility spikes, it might widen stop losses. The exact signature isn’t shown, but we can infer it returns updated values or simply applies changes to some global risk settings.
Exception Classes: While the error_handler module defines the exception classes, the risk_management module likely either re-raises those or provides convenience in raising them:
RiskViolationError: an alias or subclass of error_handler.RiskViolationError to signal a breach of risk rules (like exceeding max allowed loss).
Possibly other exceptions like NetworkError if risk management monitors network issues (though that fits better in error_handler).
Risk Checks: There might be methods to enforce:
Maximum daily drawdown: If cumulative losses exceed a threshold, raise RiskViolationError.
Maximum single trade loss or risk/reward: If a calculated position size or stop distance would risk more than allowed, prevent the trade.
Leverage limits: If using leverage, ensure it’s within config limits.
Usage: Risk management functions are called at critical points:
Before placing a trade, the strategy or TradeExecutor could call a function to verify the trade meets all risk criteria.
When simulating trades in TradeSimulator, after each action, it might adjust some parameters via dynamic_adjustment.
The error handler uses RiskViolationError as a cue to take drastic action (stop trading) because a risk limit was hit.
Enhancements: This module can be expanded to incorporate a comprehensive risk framework:
Keep track of performance metrics like current drawdown, consecutive losses, win rate, etc., and intervene if things go out of bounds.
Implement a circuit breaker at the strategy level: e.g., if 3 losing trades happen in a row beyond a certain size, pause trading for an hour.
Integrate with the strategy to possibly reduce position sizes dynamically if volatility increases (this could be part of dynamic_adjustment).
By centralizing these calculations, you ensure the strategy always consults the same source of truth about what is safe or not, which makes the behavior predictable and easy to configure (via config thresholds).
modules/reward_system.py
Role: Calculates reward points and trading rewards for the reinforcement learning algorithm. This module translates trading outcomes into a numerical reward signal that the RL agent uses to learn. It can also provide simple point calculations for gamified performance tracking. Functions and Classes:
calculate_points(profit: float, entry_time: datetime, exit_time: datetime, stop_loss_triggered: bool, risk_adjusted: bool = True) -> float: This function provides a basic way to score a trade (perhaps for a leaderboard or just debugging). It takes the profit (in base currency or USD) of a trade, the entry/exit timestamps, and whether a stop loss was triggered. It computes a points value considering:
Base points proportional to profit * 100.
A time bonus that rewards shorter trades (the formula given subtracts some value as duration increases, giving up to 50 extra points for very quick trades).
A penalty of 100 points if a stop loss was hit (meaning the trade went wrong).
If risk_adjusted is True, it divides the total points by (risk_penalty+1) as a form of normalization. This returns a float score rounded to 2 decimals. The idea is to encourage profitable, short, risk-avoiding trades if one were to use this points system.
Class RewardSystem:
Methods:
calculate_reward(self, entry_price, exit_price, position_size, entry_time, exit_time, max_drawdown, volatility, stop_loss_triggered) -> float: This method is used by the RL training loop to compute the reward for a trade or time step. Currently, the implementation shown is a placeholder that simply calculates profit = (exit_price - entry_price) * position_size (which is basically the P&L of the trade). In a more developed form, this function could incorporate:
Profit or loss of the trade.
Duration of the trade (to possibly penalize holding too long or reward quick flips).
max_drawdown during the trade (to penalize trades that went very negative before closing positive, for example).
volatility experienced (maybe penalize taking excessive risk in volatile conditions).
Whether stop_loss_triggered (penalty for hitting stop loss).
The result is a single reward score that is then used in the reinforcement learning update step. A positive reward encourages the actions taken, a negative reward discourages them.
The class can hold internal configurations for reward shaping if needed (not shown, but one could imagine weighting factors for different components of the reward).
Usage: When the SelfLearningBot executes a trade (especially a “main” trade in its act_and_learn loop), it will call RewardSystem.calculate_reward(...) to get the reward. That reward is then fed into the learning algorithm (e.g., used to compute loss for a Q-network or to policy gradient feedback). The separate calculate_points function might be used for tracking performance in a more human-readable way or for testing that the reward logic is consistent. Note: Because reward shaping is crucial to RL success, this module is likely to be tuned frequently. The modular approach means one can adjust the reward function here without touching other parts of the code, and see how it affects the learning behavior. In unit tests, a simple sanity check of calculate_reward ensures it returns expected values (as seen in tests where they just check the type and range of outputs).
modules/top_pairs.py
Role: Dynamically retrieves and filters trading pairs from the exchange and manages a cache of active pairs. This module is key for the Dynamic Pair Rotation feature, allowing the bot to know which symbols are available and active to trade, and to prefer certain ones. Class PairManager:
Attributes:
cache_file: Filename for storing cached pair data (default "pair_cache.json").
cache_duration: A timedelta specifying how long the cache is valid (default 1 hour). If cached data is newer than this, it can be re-used without hitting the API.
fallback_file: Filename for a static fallback list of pairs (default "fallback_pairs.json"). This can be a user-provided list of important pairs to use if the API is unavailable and no cache exists.
api_endpoint: The URL for the API request to get symbols. In this case, "https://api.bybit.com/spot/v1/symbols" which returns a list of available trading pairs on Bybit Spot.
headers: HTTP headers to use for the API request (includes a User-Agent string identifying the bot).
Methods:
get_active_pairs(self) -> List[str]: Main method to get a list of currently active trading pairs. It attempts multiple strategies:
API Fetch: Calls _fetch_api_pairs(). If that returns a non-empty list of pairs, it updates the cache (writes the list and a timestamp to pair_cache.json) and returns the list.
Cache Fallback: If the API fetch fails or returns nothing, it tries _get_cached_pairs(). This will load pair_cache.json if present, check the timestamp, and if it’s still within cache_duration, it returns the cached list (assuming it to be still valid).
File Fallback: If no valid cache, it tries _get_file_fallback(), which would load a predefined list from fallback_file (perhaps a file maintained with some known good pairs).
Hardcoded Fallback: If all else fails, _get_hardcoded_fallback() could return a hardcoded list of major pairs (like ["BTCUSDT","ETHUSDT",...]) as a last resort. This layered approach ensures that even with network issues, the bot will have some pairs to work with.
_fetch_api_pairs(self) -> Optional[List[str]]: Handles the HTTP GET request to the Bybit symbols API. It sets a timeout (3.05s to connect, 10s to read as per requests tuple). After getting a response, it raises for bad status codes. It then parses JSON and validates the structure using _validate_api_response. If valid, it processes the data through _process_pairs to filter and format the pair list. If anything goes wrong, it logs an error and propagates the exception.
_validate_api_response(self, data: Dict) -> bool: Checks that the JSON has expected keys and that ret_code == 0 (Bybit’s indication of success). This prevents us from assuming structure on error or changed API.
_process_pairs(self, symbols: List[Dict]) -> List[str]: Goes through the list of symbol info dictionaries from the API and filters them. In this implementation, it selects those where quoteCurrency == "USDT" (we only trade USDT pairs) and status == "Trading" (ignoring suspended pairs). It then returns a list of the name field of each (which is the symbol string like "BTCUSDT"). It also logs how many pairs were processed. This yields the raw list of active USDT trading pairs.
_update_cache(self, pairs: List[str]): Stores the given list of pairs to the cache file, along with the current timestamp. It catches any exception during file write and logs an error if, for example, the disk is not writeable.
_get_cached_pairs(self) -> Optional[List[str]]: Reads the cache file if it exists, verifies the timestamp in it against cache_duration. If the cache is fresh, returns the list of pairs from it. If the cache is missing or stale, returns None.
_get_file_fallback(self) -> Optional[List[str]]: Tries to load pairs from the fallback_file (if one is provided by the user with a static list of pairs they care about). If available, returns that list.
_get_hardcoded_fallback(self) -> List[str]: Returns a hardcoded default list (e.g. a small list of major coins like ["BTCUSDT","ETHUSDT","XRPUSDT","SOLUSDT"]) to ensure there's always at least something. This is used only if all other methods fail.
Usage: The PairManager would typically be used at startup and possibly at intervals. For example, when the bot initializes, it might do:
python
Copy
Edit
pair_manager = PairManager()
tradeable_pairs = pair_manager.get_active_pairs()
The returned tradeable_pairs list could then be fed into the SelfLearningBot or TradeSimulator to set which symbols to trade. If the strategy trades one at a time, it might pick one from the list (perhaps the first, or apply further selection criteria as described earlier). If it trades multiple, it might initialize data streams for each of the top pairs. The test code shows usage of setting pm.cache_file and pm.fallback_file to test paths, and then presumably calling get_active_pairs to ensure caching works. Sentiment Extension: While not in code, one could extend PairManager with a method to fetch sentiment (e.g., _fetch_sentiment(symbol) that scrapes a website or API). The results could influence which pairs are returned or perhaps PairManager could return not just pairs but a scored list. To keep it simple, any extension could be done after obtaining active pairs – e.g., another module could take the active_pairs list and sort it by some external factor. Caching and Performance: By caching the pair list, we avoid hitting the exchange for pair info too often. The pair list doesn’t change frequently intraday, so an hour cache is reasonable. Also, by saving to a file, if the bot restarts within an hour, it doesn’t even need to call the API at startup – it can load the cache and proceed, which makes for a faster, more reliable start.
modules/telegram_bot.py
Role: Integrates with the Telegram messaging platform to send notifications and alerts. This module manages the Telegram Bot API communication, handling asynchronous message sending and rate limiting. It’s used by the ErrorHandler and TradeExecutor to notify the user of important events. Classes and Functions:
Class TelegramNotifier:
Attributes:
_bot: An instance of telegram.Bot from the python-telegram-bot library, which is the interface to Telegram’s API.
_chat_id: The target chat ID where messages will be sent (usually your Telegram user ID or a group ID).
_message_queue: An asyncio Queue that holds messages to be sent (used when not in disable_async mode).
_worker_task: The asyncio task running the message consumer loop.
_rate_limit_delay: A delay in seconds to wait between messages (initialized to 0, but can be set if Telegram asks to slow down via RetryAfter exceptions).
_max_retries: How many times to retry sending a message if it fails due to rate limiting or transient errors (default 3).
_disable_async: If True, the notifier will send messages synchronously (blocking calls). If False, it spins up an async loop on a separate thread to handle messages in the background. By default, we use async sending so the main program isn’t delayed by network calls.
_shutting_down: Flag to indicate the loop should stop.
_loop_thread: The Thread running the asyncio event loop (for async mode).
_loop: The asyncio event loop instance.
Methods:
__init__(self, disable_async: bool = False): Initializes the notifier. It validates the configuration (makes sure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are provided in config and are of correct type). It then creates the Bot object with the token and sets the chat ID. If not disabling async, it starts the background event loop thread by calling _start_event_loop().
_start_event_loop(self): Starts a dedicated thread for the asyncio loop that will send messages. It creates a thread targeting _run_event_loop and starts it (daemon thread so it won’t block program exit).
_run_event_loop(self): The target function for the thread. It sets up the asyncio loop and kicks off the _message_consumer coroutine as a task, then runs the loop forever. This essentially waits for messages to appear in the queue and sends them out.
_validate_config(self): Ensures that the Telegram token and chat ID are present and of correct format. It attempts to instantiate the Bot and convert chat ID to int to confirm everything is okay. If not, it logs critical errors and raises exceptions to fail fast.
_message_consumer(self): Coroutine that runs in the event loop, constantly checking the _message_queue. Whenever a message is enqueued (via send_message method, described below), this consumer will dequeue it and call _safe_send_message to actually deliver it. It catches exceptions so it doesn’t crash on one failure, and it respects _shutting_down to break the loop if needed.
_safe_send_message(self, message: str, parse_mode: Optional[str] = None): Coroutine that tries to send a message to Telegram with retry logic. It uses _bot.send_message(chat_id=self._chat_id, text=message, parse_mode=parse_mode) to actually send. If it encounters a RetryAfter exception (meaning Telegram is rate limiting because too many messages), it will wait for the specified delay Telegram gives and retry. If any other TelegramError occurs, it logs it. It will try up to _max_retries. It also implements a simple circuit breaker: if messages consistently fail, it could set a longer _rate_limit_delay.
send_message(self, content: str, parse_mode: str = 'Markdown'): Public method to send a message. If disable_async is True, this will call _bot.send_message directly in the current thread (and handle exceptions accordingly). If async is enabled, it puts the message and parse_mode into the _message_queue for the background thread to handle and returns immediately.
send_message_sync(self, content: str): Synchronous helper used in some modules to quickly send a message without dealing with async (it basically wraps the logic to ensure even in an async loop, message sending happens, as implemented in TradeExecutor’s send_telegram_sync).
Usage:
The ErrorHandler uses TelegramNotifier to send critical error alerts. In ErrorHandler.__init__, it creates a TelegramNotifier(disable_async=True) for simplicity (synchronous mode for immediate sends in the same thread as errors, to not spawn new loops during an exception).
The TradeExecutor uses TelegramNotifier similarly to send a message whenever an order is executed or fails.
If desired, the bot could also use this for periodic status updates or performance reports to Telegram (not implemented yet, but easily done by calling send_message in the main loop with a summary).
Threading and Async: By running an async loop in a separate thread, we ensure that sending a Telegram message (which involves network I/O) does not block the main trading logic or UI thread. This is important because delays in sending notifications shouldn’t slow down trading decisions. The notifier will buffer messages if needed. If a user doesn't want to deal with async, they can enable disable_async (like in error handler) and just send directly, which is fine for infrequent messages (e.g., critical alerts).
modules/error_handler.py
Role: Centralized error logging and recovery logic. This module defines a hierarchy of exceptions for different error types and the ErrorHandler class that manages error events, logs them, notifies the user, and triggers recovery actions or a shutdown if necessary. Exception Classes:
TradingBotError(Exception): Base class for custom exceptions. It stores a code (an error code number), a timestamp, and an optional context dict describing the circumstances. It overrides __str__ to include the code, module, and time for easier identification.
Subclasses for specific error categories, each assigning a unique error code range:
NetworkError(TradingBotError): code 1000 – issues with network connectivity or timeouts (often recoverable).
APIError(TradingBotError): code 2000 – errors communicating with exchange APIs (could be bad responses, authentication issues, etc.).
DataIntegrityError(TradingBotError): code 3000 – any data validation failure or unexpected data format.
StrategyError(TradingBotError): code 4000 – logical errors in the trading strategy (like impossible conditions or algorithm failures).
RiskViolationError(TradingBotError): code 5000 – triggered when a risk management rule is violated (e.g., a trade that breaks limits was attempted).
OrderExecutionError(TradingBotError): code 6000 – failures during order placement or execution (e.g., order rejected, not filled when expected in simulation).
ConfigurationError(TradingBotError): code 7000 – misconfiguration issues (like missing API keys, invalid parameters).
NotificationError(TradingBotError): code 8000 – failures in the alert/notification system (e.g., Telegram message failed).
These categories help the ErrorHandler decide what to do (some are recoverable, some critical). Class ErrorHandler:
Attributes:
telegram_bot: An instance of TelegramNotifier to send alerts. Created with disable_async=True to send alerts immediately in the same thread.
error_counts: A dictionary counting occurrences of error codes.
circuit_breakers: A dict to record when a circuit breaker was activated for a given error code (stores timestamp).
error_log: A list that accumulates detailed info of errors that have occurred (could be used for further analysis or debug).
logger: A logging.Logger specifically for ErrorHandler (writes to the log file with a specified format).
debug_mode: If True, the ErrorHandler will print detailed error info to stdout as well (useful during development or troubleshooting).
Methods:
__init__(self): Sets up logging (configuring format and level from config), initializes a TelegramNotifier for alerts, and prepares counters.
log_error(self, error: Exception, context: Optional[Dict] = None, extra: Optional[Dict] = None): Logs an error with full details. It assembles an error_info dict containing type, code, message, module, context, extra data, and the traceback. It appends this to error_log. If debug_mode is True, it prints a human-readable summary to console (with emoji indicators). It then uses the logger to output a structured error message (including the entire traceback). The logger is configured to write to a file (as set by config.LOG_FILE). This ensures all error details are saved for later analysis.
_get_traceback(self, error): Utility to get traceback string from an Exception (not often called directly since we gather traceback in log_error).
handle(self, error: Exception, context: Dict[str, Any] = None): Main entry for handling an error. This is typically called in except blocks. It will:
Set context default if none.
Get the error’s code (if it’s a TradingBotError subclass, it has one; if not, code 0000 is used).
Update the error occurrence count via _update_error_stats.
Call log_error to record it.
Call _execute_recovery to attempt recovery actions specific to the error type.
If the error code is listed in config.CRITICAL_ERROR_CODES, it triggers _send_critical_alert to notify the user immediately that a critical failure happened.
_update_error_stats(self, error_code: int): Increments the count for this code in error_counts. If the count exceeds config.ERROR_RATE_THRESHOLD, it activates a circuit breaker via _activate_circuit_breaker.
_activate_circuit_breaker(self, error_code: int): Flags that error_code as having a circuit breaker tripped by recording the current time in circuit_breakers. It logs a warning about this. If the error_code is critical (also in CRITICAL_ERROR_CODES), it calls _disable_trading to stop trading operations (perhaps switching a global flag or instructing the strategy to halt). This is where integration with the rest of the system occurs: e.g., the strategy loop might periodically check with ErrorHandler if any circuit breaker is active and if so, break out of the loop safely.
_execute_recovery(self, error: Exception, context: Dict[str, Any]): Routes the error to a specific handler based on its type:
If error is NetworkError: calls _handle_network_error.
If APIError: calls _handle_api_error.
If RiskViolationError: calls _handle_risk_violation.
If OrderExecutionError: calls _handle_order_error.
(Other types could be added similarly.) Each of these _handle_* methods implements a strategy:
_handle_network_error: perhaps retry the failed operation. The code in the snippet shows it checking a retry count in context and suggesting a retry up to 3 times, logging info each attempt. After max retries, it logs failure and would let it propagate (so higher level might stop trying).
_handle_api_error: likely logs and could refresh API keys or reset connections if needed (here just an error log).
_handle_risk_violation: logs a critical message, and importantly, calls _disable_trading() and _trigger_manual_review(). This means if a risk violation happens, the bot stops trading immediately and flags for a human to intervene.
_handle_order_error: similar to risk, if an order fails in a critical way repeatedly, it might disable trading (since something is wrong with execution). These recovery methods are intended to encapsulate what the bot should do next after an error. For example, network errors might be transient so it’s ok to retry, but a strategy error might indicate a bug so perhaps stop trading and require a code fix.
_send_critical_alert(self, error: Exception, context: Dict[str, Any]): Formats a big alert message with the error details and uses the telegram_bot.send_message_sync to immediately push it to Telegram. This is used for codes deemed critical where user attention is needed ASAP.
_disable_trading(self): Takes action to stop all trading activity. In practice, this could set a global flag or instruct the main strategy loop to terminate. The handler logs a critical message "Trading operations disabled". In a live system, this might also close positions or cancel orders depending on how severe the situation is.
_trigger_manual_review(self): Placeholder to mark that human intervention is required. It just logs that manual review is required. In a GUI, this could also pop up an alert box. Essentially it’s telling the user "the bot stopped because of a serious issue, please check before restarting".
Usage: The ErrorHandler is typically instantiated once (e.g. in the UI or main program) and then its handle() method is invoked wherever exceptions are caught. For example:
python
Copy
Edit
try:
    # ... trading logic ...
except Exception as e:
    error_handler.handle(e, context={"stage": "order_execution"})
This would capture the exception, log it, notify if needed, and possibly stop trading. In tests, they directly create specific errors (like RiskViolationError("Exceeded", context={'x':1})) and call handle to ensure it sends a critical alert, etc. Integration with restart mechanism: If a truly critical error occurs such that the process must restart, the ErrorHandler will have disabled trading. If the bot is run via a script that monitors for exit or exceptions, it could exit and rely on an external systemd or supervisor to restart it. Alternatively, the error handler could itself call something like os.execv to relaunch the program. Implementation may vary; what’s important is the documentation acknowledges that critical errors cause a restart, and the restart protection logic (in UI/Exchange as described) handles continuity. By centralizing error logic here, the rest of the code can remain clean (no large try/except blocks everywhere; just raise exceptions and let ErrorHandler manage them). This module is also a good place to add any new error categories or recovery behaviors as the bot evolves (for example, handling WebSocket disconnections gracefully could be added as a NetworkError recovery that attempts reconnection).
modules/trade_simulator.py
Role: A utility module for simulating trades on historical data (backtesting) and for testing the strategy logic in a controlled environment. This can be used to verify how the bot would perform on past data or to debug the trading logic without live interaction. Class TradeSimulator:
Attributes:
wallet_balance: The simulated cash balance the bot has (initialized to a given amount, default $10,000.00).
positions: A dictionary of current open positions {symbol: quantity} in the simulation (simplified: positive quantity might mean long position, zero or absence means no position; short positions could be represented as negative quantity in some designs, but here it seems they only go long for simplicity).
trading_pairs: A list of symbols that the simulator will iterate over. In this implementation, if multiple pairs are provided, it appears to only simulate the first one (symbol = self.trading_pairs[0] in code) – it can be extended to loop through all.
trade_history: A list to record each trade order executed (storing perhaps order details and resulting balance after trade).
current_prices: A dict tracking the most recent price for each symbol.
Simulation parameters:
trade_size_percent: Fraction of wallet to use per trade (default 0.1, i.e. 10% per trade).
slippage: Assumed slippage rate for trades (0.0005 = 0.05% default).
trade_fee: Trading fee rate per trade (0.0002 = 0.02% default).
Components:
executor: An instance of TradeExecutor(simulation_mode=True). This is used to simulate executing orders (though in the code provided, they also manually adjust balance and positions, doubling up with Exchange logic somewhat).
(It also imports needed functions like calculate_trade_result for profit calculation and calculate_points for scoring, but those aren’t explicitly used in the snippet shown.)
Methods:
__init__(self, initial_wallet=10000.0): Sets starting balance and initializes data structures. It also configures logging to INFO level (so that simulation logs are visible).
run(self, market_data: List[list]): The main loop to run the simulation on a provided sequence of market data (which could be historical OHLCV data).
It expects market_data as a list of OHLCV records (timestamp, open, high, low, close, volume).
It requires self.trading_pairs to be set with at least one symbol; otherwise it raises an error.
It then iterates over each record in market_data:
Extracts timestamp and price (using the close price as current market price).
Updates current_prices[symbol] = price.
Calls _generate_trading_signals(kline) to determine if a buy or sell signal occurs at this time. (In the example, this method simply triggers a buy if price is a bit below 50000+some threshold and sell if above another threshold – a dummy strategy meant to simulate some activity.)
If signals indicate a buy, it calls _execute_buy_order(symbol, price).
If signals indicate a sell, it calls _execute_sell_order(symbol, price).
After any trade actions, it updates the portfolio value by calling _update_portfolio_value(symbol, price) – this calculates the total of cash plus position value at current price.
Logs the current status via _log_status(timestamp, symbol, price) – which prints an info line showing timestamp, price, current balance, current position (quantity), and total portfolio value.
If any exception arises in the loop, it logs an error and re-raises (so that could be caught by an external test or cause a test to fail).
_generate_trading_signals(self, kline) -> Dict[str, bool]: Very simple logic for demonstration. It uses the price and compares to fixed thresholds to decide buy or sell signals. In a real scenario, this would incorporate technical indicator logic or an AI model’s output. For example, one could plug in: if some_indicator > X: signals['buy']=True etc. Here it’s just placeholder logic (always returns True at some initial steps to cause a couple of trades).
_execute_buy_order(self, symbol, price):
Checks if already in a position for that symbol; if yes (position > 0), it doesn’t buy again (avoids doubling up – could be strategy that only holds one position at a time).
Calculates max_trade_value as wallet_balance * trade_size_percent (so how much money to use for this trade).
Applies slippage and fees to price to get fee_adjusted_price (price a bit higher for a buy, assuming we pay the spread/fees on entry).
Determines quantity = max_trade_value / fee_adjusted_price – how many units of the asset we can buy with that amount of money.
Tries to execute the order via self.executor.execute_order(...). This will simulate the order and return an order dict. (In the provided code, the TradeExecutor in simulation returns a dict with status "simulated"; it doesn’t actually alter TradeSimulator’s own records, so the TradeSimulator itself then updates positions and balance.)
On success, it updates self.positions[symbol] = quantity and deducts the used amount from wallet_balance (quantity * fee_adjusted_price).
It appends the order dict to trade_history.
If an exception occurs, logs a warning (e.g., order could fail if below minimum size – though in simulation that likely won’t happen with these numbers).
_execute_sell_order(self, symbol, price): This essentially mirrors the buy logic for selling:
Checks if there’s a position to sell; if not, it returns (nothing to sell).
Uses the entire position quantity for that symbol.
Adjusts price with slippage and fee assumption (fee_adjusted_price = price * (1 - slippage - trade_fee) assuming we get slightly less on sell due to fees).
Calls self.executor.execute_order(symbol, "sell", amount=quantity, price=price, order_type="market"). The TradeExecutor simulation will return a dict (and possibly clear the position in its Exchange, but we also manage it here).
On success, increases wallet_balance by quantity * fee_adjusted_price (proceeds from sale), sets positions[symbol] = 0 (no position now), and records the order in trade_history.
Logs warning if any issue happens.
_update_portfolio_value(self, symbol, price): Recalculates the total portfolio value (cash + value of open positions). If holding a position, it multiplies current price by quantity and adds to wallet balance for total.
_log_status(self, timestamp, symbol, price): Logs an info line with key metrics (time, price, wallet balance, position size, total value). It converts timestamp (ms since epoch) to human datetime for readability.
Property: get_performance_report(self) -> Dict[str, float]: Iterates over the trade history to compute overall performance metrics such as:
initial_balance (starting wallet, or first recorded balance in history),
final_balance (ending wallet_balance after simulation),
return_pct (percentage return over the period),
total_trades (count of trades executed),
max_drawdown (calls _calculate_max_drawdown() to compute the largest peak-to-trough drop in portfolio value during the run).
_calculate_max_drawdown(self) -> float: A placeholder for calculating max drawdown. In a full implementation, it would iterate through the equity curve and find the maximum percentage drop from any peak. Currently returns 0.0 as a stub.
Usage: TradeSimulator can be used in two ways:
For internal strategy testing: As part of unit tests or dev experiments, one can feed a predefined price series and see how the strategy signals behave. In the test file, there's an example where they create a TradeSimulator, provide a small DataFrame of price data (open, high, low, close series) and call trade_simulator.run(df), then perhaps check the final balance or trade count to ensure it matches expectations.
For backtesting strategy changes: If a developer tweaks the strategy or parameters, they can run TradeSimulator on historical data to get a rough idea of performance (like computing total return, number of trades, etc., using get_performance_report).
TradeSimulator uses a very simplified strategy logic and doesn’t incorporate the actual SelfLearningBot decisions. However, one could modify TradeSimulator to plug in the SelfLearningBot’s decision-making for each step, effectively creating a backtest harness for the RL strategy. That would be an advanced usage: feed historical data, but instead of _generate_trading_signals, call the neural network to decide trades, and train it over one pass of data (which would be off-policy learning; true RL would require multiple epochs or a different approach like training in simulation of environment). Summary: This module is mainly for testing and demonstration. It confirms that core pieces like TradeExecutor, risk adjustments, etc., work as expected in a controlled scenario. It prints out logs that can be observed to ensure the trading loop logic is correct (buy low, sell high in the dummy logic). For a production bot, one might rely more on live/paper trading with the actual strategy, but having a simulator is valuable for quick iterations or if you want to run what-if analyses on past data without risking money.
modules/parameter_optimization.py
Role: Provides tools to automatically tune the bot’s parameters (either strategy hyperparameters or technical settings) using various search algorithms. This module is separate from the trading runtime and would typically be executed offline or on historical data to improve the strategy. Class ParameterOptimizer:
Attributes:
method: The optimization method to use (loaded from config.OPTIMIZATION_METHOD, e.g. "grid_search", "random_search", or "evolutionary").
params: A dictionary of parameters to optimize, loaded from config.OPTIMIZATION_PARAMETERS. This dictionary might look like:
python
Copy
Edit
{
   "learning_rate": [0.001, 0.0005, 0.0001],
   "gamma": [0.95, 0.99],
   "epsilon_decay": {"min": 0.990, "max": 0.999}
}
It supports either explicit lists of values or ranges (min/max for continuous ranges).
results: A list to store results of parameter evaluations (tuples of (params, score)).
checkpoint_file: Filename for saving progress ("optimization_checkpoint.pkl" by default).
toolbox: In evolutionary mode, this is a DEAP Toolbox object configured with how to initialize, mutate, and mate individuals (set up in _setup_evolutionary_parameters).
Methods:
__init__(self): Initializes the ParameterOptimizer. It sets up the DEAP environment if evolutionary algorithms are to be used (creating Fitness and Individual classes and registering genetic operators). It also prepares the toolbox with population initialization, evaluation function (to be set later), selection method (e.g. tournament or best), mating (crossover) and mutation functions. If the method is not evolutionary, this still runs but might not be used.
optimize(self, objective_function: Callable) -> Dict: The main method to perform optimization. It expects an objective_function that, given a dictionary of parameter values, will return a numeric "score" (higher meaning better performance). This could be, for example, a function that runs a backtest with those parameters and returns total profit or Sharpe ratio.
It first checks if a checkpoint file exists, and if so, loads it (to resume previous optimization state).
Depending on self.method, it calls one of:
_grid_search(objective_function)
_random_search(objective_function)
_evolutionary_optimization(objective_function)
If an unknown method is specified, it raises a ValueError.
After completion, it ensures to save a checkpoint (in case we want to examine results later or resume).
It returns the best found parameters (likely as a dict).
_grid_search(self, objective_function) -> Dict: Iterates over all combinations of parameters:
It generates combinations via _generate_parameter_combinations() (which likely creates a list of dicts for each combo from the params lists).
It uses a ProcessPoolExecutor to evaluate combinations in parallel, to speed up the search. Each combination is submitted as a separate task calling _evaluate_parameters(objective_function, combination).
As results come in, it collects them. If any evaluation fails, it logs a warning but continues.
After trying all combinations, it uses _select_best_result(results) to pick the best performing parameter set and returns it.
_random_search(self, objective_function, n_iter=1000) -> Dict: Randomly samples the parameter space:
For n_iter times, it picks a random value for each parameter (if the param’s value in self.params is a list, it picks a random element; if it’s a dict with 'min'/'max', it picks a uniform random number in that range).
It evaluates each set with _evaluate_parameters. It collects results and may log debug info on progress.
Every 100 iterations or so, it saves a checkpoint (to not lose progress if it’s long-running).
Finally, selects and returns the best set found.
_evolutionary_optimization(self, objective_function) -> Dict: Uses a genetic algorithm approach:
It likely initializes a population of random individuals (each individual is a set of parameters).
Then across multiple generations, it evaluates the fitness (using the objective function) of each individual, selects the fittest, and produces a new generation through crossover and mutation.
It uses DEAP’s algorithms like eaSimple or custom loops while saving checkpoints occasionally.
The result is the best individual’s parameters.
_evaluate_parameters(self, objective_function, params: Dict) -> Tuple[Dict, float]: Calls the given objective function with a set of parameters and returns a tuple of (params, score). It may also handle exceptions, ensuring that a failure returns some default low score or raises an error handled by the caller.
_generate_parameter_combinations(self) -> List[Dict]: Prepares a list of all possible combinations of parameters (cartesian product of all parameter options). This is used by grid search. If any parameter has a range specified (min, max) instead of a list, the implementation might decide on some discretization or skip ranges in grid search.
_select_best_result(self, results: List[Tuple[Dict, float]]) -> Dict: Scans through the collected (params, score) pairs and picks the one with the highest score, returning just the params dict. It might also log or store the best result.
_save_checkpoint(self): Saves the current optimization state (self.results or population state for GA) to checkpoint_file using pickle. Called periodically and at end.
_load_checkpoint(self): Loads state from checkpoint_file if it exists, so the search can resume or at least start from previously found results.
_setup_evolutionary_parameters(self): A helper called in init to register genetic operations with the DEAP toolbox (like how to create random individuals, how to mate two individuals, how to mutate an individual, and how to select survivors). For example, it might:
Register initRepeat for population generation given param ranges.
Register mate as tools.cxTwoPoint (two-point crossover).
Register mutate as something like tools.mutUniformInt or a custom mutation that perturbs parameter values.
Register select as tools.selTournament (tournament selection). This sets the stage for the evolutionary optimization.
Usage: This module would be used offline. For instance, a developer might create a separate script or notebook:
python
Copy
Edit
from modules import parameter_optimization
optimizer = parameter_optimization.ParameterOptimizer()
def objective(params):
    # integrate with TradeSimulator or a backtest for given params
    profit = run_backtest(strategy_params=params)
    return profit
best_params = optimizer.optimize(objective)
print("Best found parameters:", best_params)
This will try many combinations and eventually output the best set. Those can then be applied in config.OPTIMIZATION_PARAMETERS or directly into the strategy. Important: The ParameterOptimizer does not alter the running bot by itself; it’s a support tool. We maintain it as a module to keep the optimization logic separate from trading logic. This also means the optimization can use multiple CPUs (via ProcessPool) without interfering with a live trading process. Structure: It uses Python standard libraries and the DEAP library for GA. Logging is used for error reporting and maybe debug. The presence of checkpointing is very useful for long runs or if using a large grid search, since it might take a long time. Conclusion for modules: Each module in the project has a clear role:
Some act as service providers (exchange, data manager, telegram notifier).
Some implement core logic (self_learning strategy, reward system, risk management).
Some handle interface (UI, error handler, pair manager).
Others are for testing and optimization (trade simulator, parameter optimizer). This modular breakdown not only makes the system easier to navigate, but also safer to modify. For example, one can improve the neural network in self_learning.py without touching how data is fetched or how the UI works. Or one can refine the UI and add new buttons without risking the trading algorithm, as long as it calls the appropriate start/stop functions.
When adding new features, one should determine which module it belongs in or if a new module is warranted. For instance, adding a SentimentAnalyzer module for scraping and evaluating sentiment would be wise rather than mixing that code into top_pairs.py (though top_pairs could call it). Similarly, if implementing a new strategy (say a different RL model or a completely different approach like a rule-based strategy), one might create a new module (e.g. strategy_ma_crossover.py) and ensure it adheres to the same interfaces so it can plug into the TradeExecutor and DataManager. The documentation above can guide future contributors (or AI modifications) to the correct file for the area they want to enhance, and provides insight into how those files work and interact. This aligns both human developers and AI agents on the structure, facilitating collaborative development. Each module’s responsibilities and how they connect has been described to prevent misunderstandings and to maintain the integrity of the system as it evolves.