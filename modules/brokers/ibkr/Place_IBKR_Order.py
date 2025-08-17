import logging
from typing import Dict, Any

from ib_insync import IB, Contract, MarketOrder, LimitOrder, StopOrder, Order, Trade

from modules.brokers.ibkr.Connect_IBKR_API import IBKRConnectionManager
from config import TRAINING_MODE, ALLOW_FUNDS_TRANSFER

log = logging.getLogger(__name__)


class TrainingOnlyError(Exception):
    """Custom exception raised when an order is attempted in training mode."""
    pass


class FundsTransferError(NotImplementedError):
    """Custom exception raised when a fund transfer is attempted."""
    pass


class IBKROrderPlacer:
    """
    Handles order placement on Interactive Brokers.
    Includes critical safety checks for training mode and fund transfers.
    """

    def __init__(self, conn_manager: IBKRConnectionManager):
        self.conn_manager = conn_manager
        self.ib: IB = None

    async def _ensure_connected(self):
        """Ensures the IB client is connected before making a request."""
        if not self.ib or not self.ib.isConnected():
            self.ib = await self.conn_manager.get_tws_client()

    def _check_training_mode(self):
        """Raises an error if the bot is in training mode."""
        if TRAINING_MODE:
            raise TrainingOnlyError(
                "Order placement is disabled. The bot is running in TRAINING_MODE."
            )

    def _check_funds_transfer_policy(self):
        """Raises an error if fund transfers are not explicitly allowed."""
        if not ALLOW_FUNDS_TRANSFER:
            raise FundsTransferError(
                "Fund transfers are disabled by policy (ALLOW_FUNDS_TRANSFER=False)."
            )

    async def place_market_order(
        self, contract: Contract, action: str, quantity: float
    ) -> Dict[str, Any]:
        """
        Places a market order.

        Args:
            contract: The contract to trade.
            action: 'BUY' or 'SELL'.
            quantity: The order size.

        Returns:
            A dictionary representing the trade confirmation.
        """
        self._check_training_mode()
        await self._ensure_connected()

        order = MarketOrder(action, quantity)

        log.info(f"STUB: Placing MARKET order: {action} {quantity} {contract.localSymbol}")

        # The actual order placement logic is commented out until live trading is enabled.
        # trade = self.ib.placeOrder(contract, order)
        # log.info(f"Placed order: {trade}")
        # return self._format_trade_status(trade)

        return {"status": "SUBMITTED_STUB", "orderId": -1, "reason": "Live order placement is disabled in TRAINING_MODE."}

    async def place_limit_order(
        self, contract: Contract, action: str, quantity: float, limit_price: float
    ) -> Dict[str, Any]:
        """
        Places a limit order.
        """
        self._check_training_mode()
        await self._ensure_connected()

        order = LimitOrder(action, quantity, limit_price)
        log.info(f"STUB: Placing LIMIT order: {action} {quantity} {contract.localSymbol} @ {limit_price}")

        # trade = self.ib.placeOrder(contract, order)
        # log.info(f"Placed order: {trade}")
        # return self._format_trade_status(trade)

        return {"status": "SUBMITTED_STUB", "orderId": -1, "reason": "Live order placement is disabled in TRAINING_MODE."}

    async def place_bracket_order(
        self, contract: Contract, action: str, quantity: float, limit_price: float, take_profit_price: float, stop_loss_price: float
    ) -> Dict[str, Any]:
        """
        Places a bracket order (an entry order with attached take profit and stop loss orders).
        """
        self._check_training_mode()
        await self._ensure_connected()

        # Create the parent LMT order and the attached TP and SL orders
        bracket_orders = self.ib.bracketOrder(
            action=action,
            quantity=quantity,
            limitPrice=limit_price,
            takeProfitPrice=take_profit_price,
            stopLossPrice=stop_loss_price
        )

        log.info(f"STUB: Placing BRACKET order: {action} {quantity} {contract.localSymbol} @ {limit_price} (TP: {take_profit_price}, SL: {stop_loss_price})")

        # for order in bracket_orders:
        #     trade = self.ib.placeOrder(contract, order)
        #     # The parent trade is the one we care about for the initial status

        return {"status": "SUBMITTED_STUB", "orderId": -1, "reason": "Live order placement is disabled in TRAINING_MODE."}


    async def transfer_funds(self, amount: float, currency: str, destination: str):
        """
        This method is a placeholder and will raise an error as per policy.
        """
        self._check_funds_transfer_policy()

        # This part of the code should never be reached if the policy is enforced.
        log.critical("Fund transfer attempted and not blocked by policy! This is a critical safety failure.")
        raise FundsTransferError("Fund transfer functionality is not implemented.")

    def _format_trade_status(self, trade: Trade) -> Dict[str, Any]:
        """Helper to format the response from ib.placeOrder."""
        # Wait for the order status to be populated
        self.ib.sleep(0.1) # Use a small sleep to allow status to arrive
        return {
            "status": trade.orderStatus.status,
            "orderId": trade.order.orderId,
            "permId": trade.order.permId,
            "filled": trade.orderStatus.filled,
            "avgFillPrice": trade.orderStatus.avgFillPrice,
            "lastFillPrice": trade.orderStatus.lastFillPrice,
            "log": [entry.message for entry in trade.log]
        }
