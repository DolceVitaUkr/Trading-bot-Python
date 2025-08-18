import logging
from typing import Dict, Any, Optional

from trading_bot.core.schemas import Order
from trading_bot.core.Config_Manager import config_manager
from trading_bot.brokers.Exchange_Bybit import Exchange_Bybit
from trading_bot.brokers.Exchange_IBKR import Exchange_IBKR

logger = logging.getLogger(__name__)

class Trade_Executor:
    """
    Routes standardized Order objects to the correct broker adapter for execution.
    """

    def __init__(self, bybit_adapter: Optional[Exchange_Bybit] = None, ibkr_adapter: Optional[Exchange_IBKR] = None):
        """
        Initializes the Trade_Executor with broker adapters.
        """
        self.bybit_adapter = bybit_adapter
        self.ibkr_adapter = ibkr_adapter
        self.bot_settings = config_manager.get_config().get("bot_settings", {})
        self.account_scope = self.bot_settings.get("account_scope", {})

    def _get_broker_for_symbol(self, symbol: str) -> Optional[Any]:
        """Determines which broker to use for a given symbol."""
        # This is a simplified routing logic. A more robust implementation might
        # query an asset master or use more complex rules.
        # For now, we assume crypto is Bybit and forex/options are IBKR.
        if "USD" in symbol or "USDT" in symbol: # Simple crypto check
            return self.bybit_adapter
        else: # Assume Forex
            return self.ibkr_adapter

    async def execute_order(self, order: Order) -> Dict[str, Any]:
        """
        Routes the order to the appropriate broker for execution.

        Args:
            order (Order): The standardized order to be executed.

        Returns:
            A dictionary containing the execution result from the broker.
        """
        broker = self._get_broker_for_symbol(order.symbol)

        if not broker:
            msg = f"No broker found for symbol: {order.symbol}"
            logger.error(msg)
            return {"status": "REJECTED", "reason": msg}

        try:
            logger.info(f"Routing order {order.order_id} for {order.symbol} to {broker.__class__.__name__}")
            result = await broker.place_order(order)
            return result
        except Exception as e:
            msg = f"An error occurred while executing order {order.order_id} on {broker.__class__.__name__}: {e}"
            logger.exception(msg)
            return {"status": "ERROR", "reason": msg}
