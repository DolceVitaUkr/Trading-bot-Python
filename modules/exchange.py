# modules/exchange.py
import requests

class Exchange:
    def __init__(self, use_testnet: bool = False):
        # Always use real Bybit endpoint for market data
        self.base_url = "https://api.bybit.com"
        self.positions = {}  # virtual positions

    def get_price(self, symbol: str) -> float:
        """Fetch latest market price from Bybit."""
        endpoint = "/v5/market/tickers"
        params = {"category": "linear", "symbol": symbol}
        resp = requests.get(self.base_url + endpoint, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        try:
            return float(data["result"]["list"][0]["lastPrice"])
        except (KeyError, IndexError):
            raise RuntimeError(f"Failed to get price for {symbol}: {data}")

    def get_current_price(self, symbol: str) -> float:
        """Alias for get_price; used by TradeExecutor."""
        return self.get_price(symbol)

    def get_min_order_size(self, symbol: str) -> float:
        """Minimum tradable amount (override if needed)."""
        return 0.0

    def get_price_precision(self, symbol: str) -> int:
        """Decimal places allowed for price (override if needed)."""
        return 8

    def create_order(self, symbol: str, order_type: str, side: str,
                     amount: float, price: float = None):
        """
        Simulate (or place) an order.
        Signature matches TradeExecutor.create_order.
        """
        # Market or limit simulated as virtual position
        side = side.lower()
        exec_price = price or self.get_price(symbol)
        position = self.positions.get(symbol)
        if position:
            # Update or close existing; simplified for brevity
            # ...
            return position
        else:
            # Open new virtual position
            self.positions[symbol] = {
                "symbol": symbol,
                "side": side,
                "quantity": amount,
                "entry_price": exec_price
            }
            return self.positions[symbol]

    def load_markets(self):
        return True

    def fetch_market_data(self, symbol, timeframe, limit, since=None):
        """
        For DataManager production paths; not used in test mode.
        """
        # Map to CCXT or HTTP; omitted for brevity
        raise NotImplementedError

# Backwards‐compatible alias
ExchangeAPI = Exchange
