# file: brokers/exchangebybit.py
import asyncio
from typing import Optional, Dict, Any, List

from pybit.unified_trading import HTTP
import ccxt

from tradingbot.core.configmanager import config_manager
from tradingbot.core.loggerconfig import get_logger
from tradingbot.core.ratelimiter import bybit_RateLimiter
from tradingbot.core.schemas import Order, PortfolioState, Position


class ExchangeBybit:
    """
    Adapter for interacting with the Bybit v5 API (Unified Trading).
    """

    def __init__(self, product_name: str, mode: str = "paper"):
        """
        Initializes the Bybit v5 adapter. Always connects to Mainnet.
        The 'mode' parameter determines if trades are simulated or live.

        :param product_name: The name of the product branch (e.g., "CRYPTO_SPOT").
        :param mode: The trading mode ("paper" or "live").
        """
        self.log = get_logger(f"bybit_adapter.{product_name.lower()}")
        self.product_name = product_name
        self.mode = mode

        api_keys = config_manager.get_config().get("api_keys", {}).get("bybit", {})
        api_key = api_keys.get("key")
        api_secret = api_keys.get("secret")

        if not api_key or not api_secret:
            self.log.error("Bybit API key or secret is not configured.")
            raise ValueError("Bybit API credentials are not set.")

        # Always use Mainnet as per user requirements
        self.session = HTTP(
            testnet=False,
            api_key=api_key,
            api_secret=api_secret
        )
        self.client = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
        })
        self.client.set_sandbox_mode(False)
        self.log.info(f"Bybit v5 adapter initialized for {product_name} in {mode} mode (connected to Mainnet).")

    async def test_connection(self) -> bool:
        """Test Bybit API connection and authentication."""
        try:
            self.log.info("Testing Bybit API connection...")
            # Use server time to test connectivity without requiring specific account permissions
            response = self.session.get_server_time()
            if response.get("retCode") == 0:
                self.log.info("Bybit API connection successful")
                return True
            else:
                self.log.error(f"Bybit connection test failed: {response}")
                return False
        except Exception as e:
            self.log.error(f"Bybit connection test exception: {e}")
            return False

    @bybit_RateLimiter.limit
    async def get_wallet_balance(self, account_type: str = "UNIFIED") -> Optional[PortfolioState]:
        """
        Fetches the wallet balance and normalizes it into a PortfolioState object.

        :param account_type: The type of account (e.g., "UNIFIED", "CONTRACT", "SPOT").
        :return: A PortfolioState object or None if an error occurs.
        """
        try:
            self.log.info(f"Fetching wallet balance for account type: {account_type}")
            
            # First test connection
            if not await self.test_connection():
                self.log.error("Connection test failed, skipping wallet balance fetch")
                return None
            
            response = self.session.get_wallet_balance(accountType=account_type)
            
            if response.get("retCode") == 0 and response.get("result"):
                result_list = response['result'].get('list', [])
                if not result_list:
                    self.log.warning("No wallet data found in response")
                    return PortfolioState(
                        total_balance_usd=0.0,
                        available_balance_usd=0.0,
                        margin_used=0.0,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        positions=[]
                    )
                
                data = result_list[0]
                self.log.info(f"Successfully fetched wallet balance: {data}")
                
                # Parse wallet data with better error handling
                total_equity = float(data.get("totalEquity", "0") or "0")
                total_available = float(data.get("totalAvailableBalance", "0") or "0") 
                margin_used = float(data.get("totalInitialMargin", "0") or "0")
                unrealized_pnl = float(data.get("totalUnrealisedPnl", "0") or "0")
                
                return PortfolioState(
                    total_balance_usd=total_equity,
                    available_balance_usd=total_available,
                    margin_used=margin_used,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=0.0,
                    positions=[]
                )
            else:
                self.log.error(f"Error fetching wallet balance - retCode: {response.get('retCode')}, retMsg: {response.get('retMsg')}")
                
                # Handle specific error codes
                ret_code = response.get("retCode")
                if ret_code == 10006:
                    self.log.error("Rate limit exceeded - too many API requests")
                elif ret_code == 10001:
                    self.log.error("Authentication failed - check API credentials")
                elif ret_code == 10003:
                    self.log.error("API key expired or invalid")
                elif ret_code == 10004:
                    self.log.error("Insufficient permissions for this API endpoint")
                    
                return None
        except Exception as e:
            self.log.error(f"An exception occurred while fetching wallet balance: {e}", exc_info=True)
            return None

    @bybit_RateLimiter.limit
    async def get_positions(self, category: str) -> List[Position]:
        """
        Fetches open positions and normalizes them into a list of Position objects.

        :param category: The category of positions (e.g., "spot", "linear", "inverse", "option").
        :return: A list of Position objects.
        """
        positions = []
        try:
            self.log.info(f"Fetching positions for category: {category}")
            response = self.session.get_positions(category=category, settleCoin="USDT")
            if response.get("retCode") == 0 and response.get("result"):
                raw_positions = response['result'].get('list', [])
                self.log.info(f"Successfully fetched {len(raw_positions)} positions.")
                for pos in raw_positions:
                    positions.append(
                        Position(
                            symbol=pos.get("symbol"),
                            side="long" if pos.get("side", "").lower() == "buy" else "short",
                            quantity=float(pos.get("size", 0)),
                            entry_price=float(pos.get("avgPrice", 0)),
                            current_price=float(pos.get("markPrice", 0)),
                            unrealized_pnl=float(pos.get("unrealisedPnl", 0)),
                            leverage=float(pos.get("leverage", 1)),
                            liquidation_price=float(pos.get("liqPrice", 0))
                        )
                    )
            else:
                self.log.error(f"Error fetching positions: {response}")
        except Exception as e:
            self.log.error(f"An exception occurred while fetching positions: {e}", exc_info=True)
        return positions

    @bybit_RateLimiter.limit
    async def get_instrument_info(self, category: str) -> Dict[str, Any]:
        """
        Fetches instrument information (symbols metadata).

        :param category: The category of instruments (e.g., "spot", "linear").
        :return: A dictionary containing the API response.
        """
        try:
            self.log.info(f"Fetching instrument info for category: {category}")
            response = self.session.get_instruments_info(category=category)
            if response.get("retCode") == 0:
                self.log.info("Successfully fetched instrument info.")
                return response['result']
            else:
                self.log.error(f"Error fetching instrument info: {response}")
                return {}
        except Exception as e:
            self.log.error(f"An exception occurred while fetching instrument info: {e}", exc_info=True)
            return {}

    @bybit_RateLimiter.limit
    async def place_order(self, order: Order, category: str = "linear") -> Dict[str, Any]:
        """
        Places an order on the exchange.

        :param order: The standardized Order object.
        :param category: The category of the product (e.g., "linear" for futures, "spot").
        :return: The API response from the exchange.
        """
        try:
            self.log.info(f"Placing order: {order.dict()}")
            response = self.session.place_order(
                category=category,
                symbol=order.symbol,
                side=order.side.capitalize(),
                orderType=order.order_type.capitalize(),
                qty=str(order.quantity),
                price=str(order.limit_price) if order.limit_price else None,
                stopLoss=str(order.stop_loss) if order.stop_loss else None,
                takeProfit=str(order.take_profit) if order.take_profit else None,
            )
            if response.get("retCode") == 0:
                self.log.info(f"Successfully placed order: {response.get('result')}")
            else:
                self.log.error(f"Error placing order: {response}")
            return response
        except Exception as e:
            self.log.error(f"An exception occurred while placing order: {e}", exc_info=True)
            return {"retCode": -1, "retMsg": str(e)}

    @bybit_RateLimiter.limit
    async def fetch_ohlcv(
        self, symbol: str, interval: str = "1m", limit: int = 200
    ) -> List[Dict[str, Any]]:
        """Fetch historical OHLCV data for a symbol."""
        try:
            self.log.info(
                f"Fetching OHLCV for {symbol} interval {interval} limit {limit}"
            )
            ohlcv = await asyncio.to_thread(
                self.client.fetch_ohlcv, symbol, interval, None, limit
            )
            return [
                {
                    "timestamp": o[0],
                    "open": o[1],
                    "high": o[2],
                    "low": o[3],
                    "close": o[4],
                    "volume": o[5],
                }
                for o in ohlcv
            ]
        except Exception as e:
            self.log.error(
                f"An exception occurred while fetching OHLCV: {e}", exc_info=True
            )
            return []

    @bybit_RateLimiter.limit
    async def cancel_order(
        self, order_id: str, symbol: str, category: str = "linear"
    ) -> Dict[str, Any]:
        """Cancel an existing order."""
        try:
            self.log.info(f"Canceling order {order_id} for {symbol}")
            response = self.session.cancel_order(
                category=category, symbol=symbol, orderId=order_id
            )
            if response.get("retCode") == 0:
                self.log.info(f"Successfully canceled order: {order_id}")
            else:
                self.log.error(f"Error canceling order: {response}")
            return response
        except Exception as e:
            self.log.error(
                f"An exception occurred while canceling order: {e}", exc_info=True
            )
            return {"retCode": -1, "retMsg": str(e)}
