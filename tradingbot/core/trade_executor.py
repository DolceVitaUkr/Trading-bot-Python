"""
Order execution routing with slippage, fee reconciliation, and mode switching.
"""

# file: core/trade_executor.py
import logging
from typing import Dict, Any, Optional

from tradingbot.brokers.exchangebybit import ExchangeBybit
from tradingbot.brokers.exchangeibkr import ExchangeIBKR
from tradingbot.core.configmanager import config_manager
from tradingbot.core.schemas import Order

logger = logging.getLogger(__name__)


class TradeExecutor:
    """
    Routes standardized Order objects to the correct broker adapter for execution.
    """

    def __init__(self, bybit_adapter: Optional[ExchangeBybit] = None, ibkr_adapter: Optional[ExchangeIBKR] = None):
        """
        Initializes the Trade_Executor with broker adapters.
        """
        self.bybit_adapter = bybit_adapter
        self.ibkr_adapter = ibkr_adapter
        self.bot_settings = config_manager.get_config().get("bot_settings", {})
        self.account_scope = self.bot_settings.get("account_scope", {})
        safety = config_manager.get_config().get("safety", {})
        self.mode = safety.get("START_MODE", "paper")

    def _get_asset_key(self, broker: Any) -> str:
        """Infer asset config key based on the broker type."""
        if isinstance(broker, ExchangeBybit):
            return "crypto_spot"
        return "forex"

    def _apply_slippage(self, order: Order, slippage_bps: float) -> Order:
        """Apply slippage (in basis points) to the order prices."""
        if slippage_bps <= 0:
            return order

        adjusted = order.copy(deep=True)
        multiplier = 1 + (slippage_bps / 10000) if order.side == "buy" else 1 - (slippage_bps / 10000)
        if adjusted.limit_price:
            adjusted.limit_price *= multiplier
        if adjusted.stop_loss:
            adjusted.stop_loss *= multiplier
        if adjusted.take_profit:
            adjusted.take_profit *= multiplier
        return adjusted

    async def _apply_fees_and_reconcile(
        self,
        broker: Any,
        order: Order,
        result: Dict[str, Any],
        asset_key: str,
    ) -> Dict[str, Any]:
        """Annotate result with fee info and updated positions."""
        asset_cfg = config_manager.get_asset_config(asset_key)
        fees = asset_cfg.get("fees", {})
        fee_rate = fees.get("maker" if order.order_type == "limit" else "taker", 0)
        price = result.get("price") or order.limit_price or 0
        result["fee"] = price * order.quantity * fee_rate

        if hasattr(broker, "get_positions"):
            try:
                if isinstance(broker, ExchangeBybit):
                    result["positions"] = await broker.get_positions("linear")
                else:
                    result["positions"] = await broker.get_positions()
            except Exception as e:
                logger.warning(f"Failed to reconcile positions: {e}")

        return result

    async def _execute_oco_order(self, broker: Any, order: Order) -> Dict[str, Any]:
        """Execute an order with stop-loss and take-profit legs."""
        if hasattr(broker, "place_oco_order"):
            return await broker.place_oco_order(order)
        return await broker.place_order(order)

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
            asset_key = self._get_asset_key(broker)
            asset_cfg = config_manager.get_asset_config(asset_key)
            slippage_bps = asset_cfg.get("slippage_bps", 0)
            adjusted_order = self._apply_slippage(order, slippage_bps)

            logger.info(
                f"Routing order {adjusted_order.order_id} for {adjusted_order.symbol} to {broker.__class__.__name__} in {self.mode} mode"
            )

            if self.mode == "paper":
                result = {"status": "FILLED", "simulated": True, "mode": "paper"}
            else:
                if adjusted_order.stop_loss and adjusted_order.take_profit:
                    result = await self._execute_oco_order(broker, adjusted_order)
                else:
                    result = await broker.place_order(adjusted_order)

            result = await self._apply_fees_and_reconcile(
                broker, adjusted_order, result, asset_key
            )
            return result
        except Exception as e:
            msg = (
                f"An error occurred while executing order {order.order_id} on {broker.__class__.__name__}: {e}"
            )
            logger.exception(msg)
            return {"status": "ERROR", "reason": msg}
