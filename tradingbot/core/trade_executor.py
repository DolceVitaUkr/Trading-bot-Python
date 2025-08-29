# file: tradingbot/core/trade_executor.py
# module_version: v1.01

"""
Trade Executor - Now a CLIENT of order_router.py
Converts legacy Order objects to OrderContext and delegates to order_router.
NO DIRECT ORDER PLACEMENT - all orders route through order_router only.
"""

import logging
from typing import Dict, Any, Optional

from tradingbot.core.configmanager import config_manager
from tradingbot.core.schemas import Order
from tradingbot.core.order_router import order_router, OrderContext

logger = logging.getLogger(__name__)


class TradeExecutor:
    """
    LEGACY COMPATIBILITY: Routes standardized Order objects to order_router.
    NO LONGER PLACES ORDERS DIRECTLY - ALL ROUTING VIA ORDER_ROUTER.
    """

    def __init__(self, bybit_adapter=None, ibkr_adapter=None):
        """
        Initialize trade executor as order_router client.
        Broker adapters are now managed by order_router, not here.
        """
        self.bot_settings = config_manager.get_config().get("bot_settings", {})
        self.account_scope = self.bot_settings.get("account_scope", {})
        safety = config_manager.get_config().get("safety", {})
        self.mode = safety.get("START_MODE", "paper")
        
        logger.info("TradeExecutor initialized as order_router client - NO DIRECT ORDER PLACEMENT")

    def _convert_order_to_context(self, order: Order) -> OrderContext:
        """Convert legacy Order object to OrderContext for order_router."""
        # Determine asset type based on symbol
        asset_type = "spot"  # Default
        if "USD" in order.symbol and "USDT" not in order.symbol:
            asset_type = "forex"
        elif order.symbol.endswith("PERP") or "FUT" in order.symbol:
            asset_type = "futures"
            
        return OrderContext(
            asset_type=asset_type,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.limit_price,
            order_type=order.order_type,
            sl_price=order.stop_loss,
            tp_price=order.take_profit,
            strategy_id=getattr(order, 'strategy_id', None),
            signal_weight=getattr(order, 'signal_weight', 1.0),
            client_order_id=order.order_id,
            metadata={
                "converted_from_legacy": True,
                "original_order_type": order.order_type
            }
        )

    async def execute_order(self, order: Order) -> Dict[str, Any]:
        """
        REFACTORED: Routes the order through order_router.py ONLY.
        NO DIRECT BROKER CALLS - all routing via order_router.

        Args:
            order (Order): The standardized order to be executed.

        Returns:
            A dictionary containing the execution result from order_router.
        """
        try:
            # Convert legacy Order to OrderContext
            context = self._convert_order_to_context(order)
            
            logger.info(
                f"[TRADE_EXECUTOR] Routing order {order.order_id} for {order.symbol} "
                f"through order_router in {self.mode} mode"
            )

            # Route through order_router ONLY
            order_result = await order_router.place_order(context)
            
            # Convert OrderResult back to legacy format
            if order_result.success:
                result = {
                    "status": "FILLED" if order_result.status and order_result.status.value == "filled" else "SUBMITTED",
                    "order_id": order_result.order_id,
                    "client_order_id": order_result.client_order_id,
                    "filled_qty": order_result.filled_qty,
                    "filled_price": order_result.filled_price,
                    "fees": order_result.fees,
                    "timestamp": order_result.timestamp.isoformat() if order_result.timestamp else None,
                    "routed_via": "order_router",
                    "mode": self.mode
                }
            else:
                result = {
                    "status": "REJECTED",
                    "reason": order_result.reason,
                    "client_order_id": order_result.client_order_id,
                    "routed_via": "order_router",
                    "mode": self.mode
                }
                
            return result
            
        except Exception as e:
            msg = f"TradeExecutor failed to route order {order.order_id} through order_router: {e}"
            logger.exception(msg)
            return {"status": "ERROR", "reason": msg}
