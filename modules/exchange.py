# modules/exchange.py

import ccxt
import config
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class ExchangeAPI:
    """
    Wrapper around ccxt.bybit for both market data and order execution.
    Honors config.USE_SIMULATION to toggle sandbox mode.
    """

    def __init__(self):
        # Select credentials based on simulation flag
        if config.USE_SIMULATION:
            api_key = config.SIMULATION_BYBIT_API_KEY
            api_secret = config.SIMULATION_BYBIT_API_SECRET
        else:
            api_key = config.BYBIT_API_KEY
            api_secret = config.BYBIT_API_SECRET

        params = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',        # use spot by default
            },
            'timeout': getattr(config, 'API_REQUEST_TIMEOUT', 30) * 1000
        }

        # Instantiate the ccxt client
        self.client = ccxt.bybit(params)
        # If in simulation, flip on sandbox mode
        if config.USE_SIMULATION and hasattr(self.client, 'set_sandbox_mode'):
            try:
                self.client.set_sandbox_mode(True)
            except Exception:
                logger.warning("Sandbox mode not supported by this ccxt version.")
        # Load markets metadata
        self.client.load_markets()

    def fetch_market_data(self, symbol: str, timeframe: str,
                          since: int = None, limit: int = 1000) -> list:
        """
        Fetch OHLCV [timestamp, open, high, low, close, volume] from exchange.
        """
        return self.client.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

    def get_current_price(self, symbol: str) -> float:
        """
        Fetch the latest ticker price.
        """
        ticker = self.client.fetch_ticker(symbol)
        # ccxt ticker may use 'last' or 'close'
        price = ticker.get('last') or ticker.get('close')
        return float(price)

    def get_min_order_size(self, symbol: str) -> float:
        """
        Return the minimum order size for this market.
        Falls back to 0 if market metadata is missing.
        """
        market = self.client.markets.get(symbol)
        if not market:
            return 0.0
        return float(market.get('limits', {}).get('amount', {}).get('min', 0.0))

    def get_price_precision(self, symbol: str) -> int:
        """
        Return the number of decimal places allowed for prices on this market.
        Falls back to 8.
        """
        market = self.client.markets.get(symbol)
        if not market:
            return 8
        return int(market.get('precision', {}).get('price', 8))

    def create_order(self, symbol: str, order_type: str, side: str,
                     amount: float, price: float = None) -> dict:
        """
        Place an order.
        - order_type: 'limit' or 'market'
        - side: 'buy' or 'sell'
        """
        order_type = order_type.lower()
        side = side.lower()
        if order_type == 'market':
            return self.client.create_order(symbol, 'market', side, amount)
        # limit
        return self.client.create_order(symbol, 'limit', side, amount, price)

    # Backwards compatibility alias
    # Some parts of code refer to ExchangeAPI via ExchangeAPI
    # so no extra alias needed here.

