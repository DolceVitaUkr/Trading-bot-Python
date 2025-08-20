from typing import List, Optional, Dict, Any, Union

from ib_insync import IB, Forex, Option, MarketOrder, LimitOrder, StopOrder, Trade

from tradingbot.core.configmanager import config_manager
from tradingbot.core.loggerconfig import get_logger
from tradingbot.core.ratelimiter import ibkr_RateLimiter
from tradingbot.core.schemas import MarketData, Order, PortfolioState, Position


class ExchangeIBKR:
    """
    Consolidated adapter for all interactions with Interactive Brokers.
    Manages TWS connection, contract creation, data fetching, and order execution.
    """

    def __init__(self, product_name: str, mode: str = "paper"):
        self.log = get_logger(f"ibkr_adapter.{product_name.lower()}")
        self.product_name = product_name
        self.mode = mode
        self.ib = IB()

        ibkr_config = config_manager.get_config().get("api_keys", {}).get("ibkr", {})
        self.host = ibkr_config.get("host", "127.0.0.1")
        self.port = ibkr_config.get("port", 7497)
        self.client_id = ibkr_config.get("client_id", 1)

        bot_settings = config_manager.get_config().get("bot_settings", {})
        self.training_mode = bot_settings.get("training_mode", True)

    async def connect(self):
        """Connects to the TWS/Gateway."""
        if self.ib.isConnected():
            self.log.info("IBKR client is already connected.")
            return
        try:
            self.log.info(f"Connecting to IBKR TWS/Gateway at {self.host}:{self.port}...")
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=15,
                readonly=self.training_mode
            )
            self.log.info(f"IBKR connection successful. Server version: {self.ib.serverVersion()}")
        except Exception as e:
            self.log.error(f"Failed to connect to IBKR: {e}")
            raise ConnectionError(f"IBKR connection failed: {e}")

    async def disconnect(self):
        """Disconnects from the TWS/Gateway."""
        if self.ib.isConnected():
            self.log.info("Disconnecting from IBKR TWS/Gateway...")
            self.ib.disconnect()

    @ibkr_RateLimiter.limit
    async def get_wallet_balance(self) -> Optional[PortfolioState]:
        """Fetches account summary and normalizes it into a PortfolioState object."""
        self.log.info("Fetching IBKR wallet balance...")
        if not self.ib.isConnected():
            await self.connect()

        summary_tags = "NetLiquidation,TotalCashValue,BuyingPower,GrossPositionValue,MaintMarginReq,InitMarginReq,UnrealizedPnL,RealizedPnL"
        account_values = self.ib.accountSummary()

        summary = {val.tag: val.value for val in account_values if val.currency == 'USD'} # Assuming USD base

        positions = await self.get_positions()

        return PortfolioState(
            total_balance_usd=float(summary.get("NetLiquidation", 0)),
            available_balance_usd=float(summary.get("BuyingPower", 0)),
            margin_used=float(summary.get("InitMarginReq", 0)),
            unrealized_pnl=float(summary.get("UnrealizedPnL", 0)),
            realized_pnl=float(summary.get("RealizedPnL", 0)),
            positions=positions
        )

    @ibkr_RateLimiter.limit
    async def get_positions(self) -> List[Position]:
        """Fetches current open positions and normalizes them."""
        self.log.info("Fetching IBKR positions...")
        if not self.ib.isConnected():
            await self.connect()

        portfolio_items = self.ib.portfolio()
        normalized_positions = []
        for item in portfolio_items:
            normalized_positions.append(
                Position(
                    symbol=item.contract.localSymbol,
                    side="long" if item.position > 0 else "short",
                    quantity=abs(item.position),
                    entry_price=item.averageCost,
                    current_price=item.marketPrice,
                    unrealized_pnl=item.unrealizedPNL,
                    leverage=1, # IBKR portfolio margin is complex, defaulting to 1
                    liquidation_price=None, # Not directly available
                )
            )
        return normalized_positions

    async def get_historical_data(self, symbol: str, timeframe: str, duration: str) -> Optional[MarketData]:
        self.log.info(f"Fetching IBKR historical data for {symbol}...")
        # TODO: Implement logic from FetchIBKRMarketData.py and normalize to MarketData
        return None

    @ibkr_RateLimiter.limit
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """Places an order on IBKR, translating a standard Order object."""
        self.log.info(f"Placing IBKR order: {order.dict()}")
        if self.training_mode:
            self.log.warning("TRAINING MODE is ON. Order placement is disabled.")
            return {"status": "REJECTED", "reason": "Training mode is on."}

        if not self.ib.isConnected():
            await self.connect()

        # Build contract
        # This is simplified; a real implementation would need more robust contract resolution
        contract = self._build_fx_spot_contract(order.symbol)

        # Build order
        ib_order: Union[MarketOrder, LimitOrder]
        if order.order_type == "market":
            ib_order = MarketOrder(action=order.side.upper(), totalQuantity=order.quantity)
        elif order.order_type == "limit":
            if order.limit_price is None:
                raise ValueError("Limit price must be set for a limit order.")
            ib_order = LimitOrder(action=order.side.upper(), totalQuantity=order.quantity, lmtPrice=order.limit_price)
        else:
            raise ValueError(f"Unsupported order type for IBKR: {order.order_type}")

        # Attach SL/TP if present
        if order.stop_loss or order.take_profit:
            if order.order_type != 'limit':
                raise ValueError("Bracket orders (SL/TP) are only supported for limit orders in this implementation.")
            if not order.take_profit or not order.stop_loss:
                raise ValueError("Both take_profit and stop_loss must be specified for a bracket order.")

            if order.limit_price is None:
                raise ValueError("Limit price must be set for a bracket order.")
            bracket_orders = self.ib.bracketOrder(
                action=order.side.upper(),
                quantity=order.quantity,
                limitPrice=order.limit_price,
                takeProfitPrice=order.take_profit,
                stopLossPrice=order.stop_loss
            )
            # Transmit all orders
            trades = [self.ib.placeOrder(contract, o) for o in bracket_orders]
            trade = trades[0] # The parent order
        else:
            trade = self.ib.placeOrder(contract, ib_order)

        # The trade object itself is not returned, but its order status
        self.log.info(f"Placed order with id: {trade.order.orderId}")
        return {"status": "SUBMITTED", "id": trade.order.orderId}

    # --- Private helper methods for contract building ---

    def _build_fx_spot_contract(self, pair: str) -> Forex:
        """Creates an IBKR Forex (CASH) contract."""
        # Logic from ContractsIBKR.py
        pair = pair.replace("/", "").upper()
        return Forex(pair, exchange="IDEALPRO", symbol=pair[:3], currency=pair[3:])

    def _build_option_contract(self, symbol: str, last_trade_date: str, strike: float, right: str) -> Option:
        """Creates an IBKR Option contract."""
        # Logic from ContractsIBKR.py
        return Option(symbol=symbol[:3], lastTradeDateOrContractMonth=last_trade_date, strike=strike, right=right.upper(), exchange="IDEALPRO", tradingClass=symbol[:3], currency=symbol[3:])
