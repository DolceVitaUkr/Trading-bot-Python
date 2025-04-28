# modules/exchange.py

import time
import logging
from typing import List, Optional, Union, Dict

import ccxt
import config
from modules.error_handler import APIError, OrderExecutionError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ExchangeAPI:
    """
    Unified exchange interface. Uses CCXT under the hood for live or testnet
    trading, and falls back to built-in simulation logic when USE_SIMULATION=True.
    """

    def __init__(self):
        # Determine mode from config
        self.simulation = bool(getattr(config, "USE_SIMULATION", False))
        self.use_testnet = bool(getattr(config, "USE_TESTNET", False)) or self.simulation

        if not self.simulation:
            # Choose API credentials
            api_key = config.BYBIT_API_KEY
            secret = config.BYBIT_API_SECRET

            # Initialize CCXT Bybit client
            self.client = ccxt.bybit({
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True,
            })
            if self.use_testnet:
                # CCXT sandbox mode for testnet trades
                self.client.set_sandbox_mode(True)

            # Load markets once
            try:
                self.client.load_markets()
            except Exception as e:
                logger.error("Failed to load markets from Bybit", exc_info=True)
                raise APIError("Market load failure", context={"exception": str(e)})
        else:
            # Simple in-memory position simulation
            self.positions: Dict[str, Dict] = {}

    def _retry(fun):
        """Simple retry decorator with exponential backoff."""
        def wrapper(self, *args, **kwargs):
            delay = 1
            for attempt in range(3):
                try:
                    return fun(self, *args, **kwargs)
                except Exception as e:
                    if attempt < 2:
                        time.sleep(delay)
                        delay *= 2
                    else:
                        raise
        return wrapper

    @_retry
    def get_price(self, symbol: str) -> float:
        """Fetch latest market price."""
        if self.simulation:
            # In simulation, return last known or a default
            pos = self.positions.get(symbol)
            if pos:
                return pos.get("entry_price", 0.0)
            return 0.0

        try:
            ticker = self.client.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception as e:
            logger.error(f"get_price failed for {symbol}: {e}", exc_info=True)
            raise APIError(f"Failed to fetch price for {symbol}", context={"symbol": symbol})

    @_retry
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: Optional[int] = None,
        limit: int = 1000
    ) -> List[List[Union[int, float]]]:
        """
        Fetch historical OHLCV data via CCXT.
        Returns a list of [timestamp, open, high, low, close, volume].
        """
        if self.simulation:
            raise APIError("OHLCV fetch not available in simulation mode")

        try:
            return self.client.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except Exception as e:
            logger.error(f"fetch_ohlcv failed for {symbol}/{timeframe}: {e}", exc_info=True)
            raise APIError(
                f"Failed to fetch OHLCV for {symbol}", context={"symbol": symbol, "timeframe": timeframe}
            )

    def fetch_market_data(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        limit: int = 1000
    ) -> List[List[Union[int, float]]]:
        """Alias for compatibility with DataManager."""
        return self.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

    @_retry
    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[dict] = None
    ) -> Dict:
        """
        Place an order. In simulation mode, updates an in-memory position and
        returns a detailed dict including P&L if a position is closed.
        In live mode, proxies to CCXT and returns its order dict.
        """
        params = params or {}

        if self.simulation:
            return self._simulate_order(symbol, side.lower(), amount, price)

        try:
            order = self.client.create_order(symbol, order_type, side, amount, price, params)
            return order  # CCXT order response
        except Exception as e:
            logger.error(f"create_order failed: {e}", exc_info=True)
            raise OrderExecutionError(f"Order execution failed for {symbol}", context={"symbol": symbol, "side": side})

    def _simulate_order(self, symbol, side, amount, price):
        """Internal simulation logic with position tracking."""
        now = int(time.time() * 1000)
        pos = self.positions.get(symbol)

        # Closing logic
        if pos and ((pos["side"] == "long" and side in ("sell", "short")) or
                    (pos["side"] == "short" and side in ("buy", "long"))):
            # Close existing position
            exit_price = price or pos["entry_price"]
            pnl = (exit_price - pos["entry_price"]) * pos["quantity"] \
                  if pos["side"] == "long" \
                  else (pos["entry_price"] - exit_price) * pos["quantity"]

            closed = {
                "symbol": symbol,
                "side": pos["side"],
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "quantity": pos["quantity"],
                "pnl": pnl,
                "entry_time": pos["entry_time"],
                "exit_time": now,
            }
            # Remove position
            del self.positions[symbol]
            return closed

        # Opening or increasing position
        new_side = "long" if side in ("buy", "long") else "short"
        entry_price = price or 0.0
        if pos and pos["side"] == new_side:
            # Increase and recalculate average price
            total_qty = pos["quantity"] + amount
            avg_price = ((pos["entry_price"] * pos["quantity"]) + (entry_price * amount)) / total_qty
            pos.update({
                "quantity": total_qty,
                "entry_price": avg_price
            })
        else:
            # New position
            self.positions[symbol] = {
                "symbol": symbol,
                "side": new_side,
                "quantity": amount,
                "entry_price": entry_price,
                "entry_time": now
            }

        return {
            "symbol": symbol,
            "side": new_side,
            "quantity": self.positions[symbol]["quantity"],
            "entry_price": self.positions[symbol]["entry_price"],
            "status": "open",
            "time": now
        }

    @_retry
    def close_position(self, symbol: str) -> Optional[Dict]:
        """
        Explicitly close a position at market price.
        In live mode, we fetch market price and place a reverse order.
        """
        if self.simulation:
            return self.create_order(symbol, "market",
                                     "sell" if self.positions.get(symbol, {}).get("side") == "long" else "buy",
                                     self.positions.get(symbol, {}).get("quantity", 0),
                                     price=self.get_price(symbol))

        pos = self.client.fetch_positions()[0]  # example fetch; adjust per CCXT structure
        qty = abs(pos.get("contracts", 0))
        if qty == 0:
            return None

        side = "sell" if pos["side"] == "long" else "buy"
        return self.create_order(symbol, "market", side, qty)

