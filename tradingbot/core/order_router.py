# file: tradingbot/core/order_router.py
# module_version: v1.00

"""
Order Router - The ONLY module that places orders with brokers.
All order placement requests MUST go through this module.
Enforces risk→budget→exposure pipeline before any order.
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .configmanager import config_manager
from .loggerconfig import get_logger


class OrderStatus(Enum):
    """Order status states"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class OrderContext:
    """Complete order context for pretrade checks"""
    asset_type: str  # spot/futures/forex/options
    symbol: str
    side: str  # buy/sell
    quantity: float
    price: Optional[float] = None  # None for market orders
    order_type: str = "market"  # market/limit
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    strategy_id: Optional[str] = None
    signal_weight: float = 1.0
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    """Result of order placement"""
    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: Optional[OrderStatus] = None
    filled_qty: float = 0.0
    filled_price: float = 0.0
    fees: float = 0.0
    reason: Optional[str] = None
    timestamp: Optional[datetime] = None


class OrderRouter:
    """
    The ONLY module that places orders with brokers.
    Enforces pretrade pipeline: risk → budget → exposure → broker.
    """
    
    def __init__(self):
        self.log = get_logger("order_router")
        self.config = config_manager
        
        # Order tracking
        self.active_orders: Dict[str, OrderContext] = {}
        self.order_history: List[Dict[str, Any]] = []
        
        # Idempotency tracking
        self.client_order_ids: set = set()
        
        # Module dependencies (will be injected)
        self.risk_manager = None
        self.budget_manager = None
        self.exposure_manager = None
        self.sl_tp_manager = None
        self.pnl_reconciler = None
        
        # Broker connections (will be injected)
        self.bybit_client = None
        self.ibkr_client = None
        
        self.log.info("Order Router initialized - ALL orders must route through here")
    
    def set_dependencies(self, 
                        risk_manager=None,
                        budget_manager=None,
                        exposure_manager=None,
                        sl_tp_manager=None,
                        pnl_reconciler=None,
                        bybit_client=None,
                        ibkr_client=None):
        """Inject dependencies"""
        if risk_manager:
            self.risk_manager = risk_manager
        if budget_manager:
            self.budget_manager = budget_manager
        if exposure_manager:
            self.exposure_manager = exposure_manager
        if sl_tp_manager:
            self.sl_tp_manager = sl_tp_manager
        if pnl_reconciler:
            self.pnl_reconciler = pnl_reconciler
        if bybit_client:
            self.bybit_client = bybit_client
        if ibkr_client:
            self.ibkr_client = ibkr_client
    
    async def place_order(self, context: OrderContext) -> OrderResult:
        """
        Place order through pretrade pipeline.
        This is the ONLY function that places orders.
        
        Pipeline:
        1. Idempotency check
        2. Risk manager pretrade check
        3. Budget manager affordability check
        4. Exposure manager correlation check
        5. Route to broker
        6. Attach SL/TP if filled
        7. Update reconciler
        """
        
        # Generate client_order_id if not provided
        if not context.client_order_id:
            context.client_order_id = f"{context.asset_type}_{uuid.uuid4().hex[:8]}"
        
        # Idempotency check
        if context.client_order_id in self.client_order_ids:
            self.log.warning(f"Duplicate order rejected: {context.client_order_id}")
            return OrderResult(
                success=False,
                client_order_id=context.client_order_id,
                reason="Duplicate client_order_id"
            )
        
        self.client_order_ids.add(context.client_order_id)
        self.log.info(f"Processing order: {context.client_order_id} for {context.symbol}")
        
        # 1. Risk Manager Check
        if self.risk_manager:
            risk_check = await self.risk_manager.pretrade_check(context)
            if not risk_check['pass']:
                self.log.warning(f"Risk check failed: {risk_check['reason']}")
                return OrderResult(
                    success=False,
                    client_order_id=context.client_order_id,
                    reason=f"Risk check failed: {risk_check['reason']}"
                )
            
            # Risk manager may adjust quantity
            if 'adjusted_qty' in risk_check:
                context.quantity = risk_check['adjusted_qty']
        
        # 2. Budget Manager Check
        if self.budget_manager:
            budget_check = await self.budget_manager.can_afford(context)
            if not budget_check['pass']:
                self.log.warning(f"Budget check failed: {budget_check['reason']}")
                return OrderResult(
                    success=False,
                    client_order_id=context.client_order_id,
                    reason=f"Budget check failed: {budget_check['reason']}"
                )
        
        # 3. Exposure Manager Check
        if self.exposure_manager:
            exposure_check = await self.exposure_manager.can_open(
                context.symbol, context.side, context.quantity
            )
            if not exposure_check['pass']:
                self.log.warning(f"Exposure check failed: {exposure_check['reason']}")
                return OrderResult(
                    success=False,
                    client_order_id=context.client_order_id,
                    reason=f"Exposure check failed: {exposure_check['reason']}"
                )
        
        # 4. Route to appropriate broker
        try:
            broker_result = await self._route_to_broker(context)
            
            if not broker_result.success:
                self.log.error(f"Broker placement failed: {broker_result.reason}")
                return broker_result
            
            # 5. Attach SL/TP if order filled
            if broker_result.status in [OrderStatus.FILLED, OrderStatus.PARTIAL]:
                if self.sl_tp_manager and (context.sl_price or context.tp_price):
                    sl_tp_spec = {
                        'sl_price': context.sl_price,
                        'tp_price': context.tp_price,
                        'sl_trigger_type': 'mark' if context.asset_type == 'futures' else 'last'
                    }
                    
                    sl_tp_result = await self.sl_tp_manager.attach(
                        broker_result.order_id,
                        sl_tp_spec,
                        context.asset_type
                    )
                    
                    if not sl_tp_result['success']:
                        self.log.error(f"SL/TP attach failed: {sl_tp_result['reason']}")
            
            # 6. Update reconciler
            if self.pnl_reconciler:
                await self.pnl_reconciler.record_order(broker_result, context)
            
            # Track order
            self.active_orders[broker_result.order_id] = context
            self._record_order_history(context, broker_result)
            
            self.log.info(f"Order placed successfully: {broker_result.order_id}")
            return broker_result
            
        except Exception as e:
            self.log.error(f"Order placement exception: {e}", exc_info=True)
            return OrderResult(
                success=False,
                client_order_id=context.client_order_id,
                reason=f"Exception: {str(e)}"
            )
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an active order"""
        if order_id not in self.active_orders:
            return {'success': False, 'reason': 'Order not found'}
        
        context = self.active_orders[order_id]
        
        try:
            # Route cancellation to appropriate broker
            if context.asset_type in ['spot', 'futures']:
                if self.bybit_client:
                    result = await self.bybit_client.cancel_order(order_id, context.symbol)
                else:
                    return {'success': False, 'reason': 'Bybit client not available'}
            else:  # forex/options
                if self.ibkr_client:
                    result = await self.ibkr_client.cancel_order(order_id)
                else:
                    return {'success': False, 'reason': 'IBKR client not available'}
            
            if result.get('success'):
                del self.active_orders[order_id]
                self.log.info(f"Order cancelled: {order_id}")
            
            return result
            
        except Exception as e:
            self.log.error(f"Cancel order exception: {e}", exc_info=True)
            return {'success': False, 'reason': str(e)}
    
    async def amend_order(self, order_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Amend an active order"""
        if order_id not in self.active_orders:
            return {'success': False, 'reason': 'Order not found'}
        
        context = self.active_orders[order_id]
        
        # Validate amendments through risk/budget if size changed
        if 'quantity' in fields:
            # Create amended context
            amended_context = OrderContext(**context.__dict__)
            amended_context.quantity = fields['quantity']
            
            # Re-run checks
            if self.risk_manager:
                risk_check = await self.risk_manager.pretrade_check(amended_context)
                if not risk_check['pass']:
                    return {'success': False, 'reason': f"Amended order failed risk: {risk_check['reason']}"}
        
        try:
            # Route amendment to broker
            if context.asset_type in ['spot', 'futures']:
                if self.bybit_client:
                    result = await self.bybit_client.amend_order(order_id, context.symbol, fields)
                else:
                    return {'success': False, 'reason': 'Bybit client not available'}
            else:
                if self.ibkr_client:
                    result = await self.ibkr_client.modify_order(order_id, fields)
                else:
                    return {'success': False, 'reason': 'IBKR client not available'}
            
            if result.get('success'):
                # Update context
                for key, value in fields.items():
                    if hasattr(context, key):
                        setattr(context, key, value)
                
                self.log.info(f"Order amended: {order_id}")
            
            return result
            
        except Exception as e:
            self.log.error(f"Amend order exception: {e}", exc_info=True)
            return {'success': False, 'reason': str(e)}
    
    async def _route_to_broker(self, context: OrderContext) -> OrderResult:
        """Route order to appropriate broker based on asset type"""
        
        if context.asset_type in ['spot', 'futures']:
            # Route to Bybit
            if not self.bybit_client:
                return OrderResult(success=False, reason="Bybit client not available")
            
            broker_response = await self.bybit_client.place_order(
                symbol=context.symbol,
                side=context.side,
                order_type=context.order_type,
                quantity=context.quantity,
                price=context.price,
                client_order_id=context.client_order_id
            )
            
        elif context.asset_type in ['forex', 'options']:
            # Route to IBKR
            if not self.ibkr_client:
                return OrderResult(success=False, reason="IBKR client not available")
            
            broker_response = await self.ibkr_client.place_order(
                symbol=context.symbol,
                side=context.side,
                order_type=context.order_type,
                quantity=context.quantity,
                price=context.price,
                client_order_id=context.client_order_id
            )
        else:
            return OrderResult(success=False, reason=f"Unknown asset type: {context.asset_type}")
        
        # Parse broker response
        if broker_response.get('success'):
            return OrderResult(
                success=True,
                order_id=broker_response.get('order_id'),
                client_order_id=context.client_order_id,
                status=OrderStatus.SUBMITTED,
                timestamp=datetime.now()
            )
        else:
            return OrderResult(
                success=False,
                client_order_id=context.client_order_id,
                reason=broker_response.get('reason', 'Unknown broker error')
            )
    
    def _record_order_history(self, context: OrderContext, result: OrderResult):
        """Record order in history"""
        self.order_history.append({
            'timestamp': datetime.now().isoformat(),
            'order_id': result.order_id,
            'client_order_id': result.client_order_id,
            'asset_type': context.asset_type,
            'symbol': context.symbol,
            'side': context.side,
            'quantity': context.quantity,
            'price': context.price,
            'status': result.status.value if result.status else None,
            'strategy_id': context.strategy_id,
            'success': result.success,
            'reason': result.reason
        })
    
    def get_active_orders(self, asset_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active orders, optionally filtered by asset type"""
        orders = []
        for order_id, context in self.active_orders.items():
            if asset_type is None or context.asset_type == asset_type:
                orders.append({
                    'order_id': order_id,
                    'asset_type': context.asset_type,
                    'symbol': context.symbol,
                    'side': context.side,
                    'quantity': context.quantity,
                    'price': context.price,
                    'strategy_id': context.strategy_id
                })
        return orders


# Module initialization
order_router = OrderRouter()