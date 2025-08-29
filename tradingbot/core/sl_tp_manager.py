# file: tradingbot/core/sl_tp_manager.py
# module_version: v1.00

"""
Stop Loss & Take Profit Manager - Server-side SL/TP attachment.
Handles Bybit trading-stop and IBKR bracket orders.
This is the ONLY module that manages SL/TP orders.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

from .configmanager import config_manager
from .loggerconfig import get_logger


class SLTPStatus(Enum):
    """SL/TP order status"""
    PENDING = "pending"
    ACTIVE = "active"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"
    FAILED = "failed"


class SLTPManager:
    """
    Manages server-side Stop Loss and Take Profit orders.
    Bybit: Uses trading-stop API for server-side SL/TP.
    IBKR: Uses bracket orders for atomic SL/TP attachment.
    """
    
    def __init__(self):
        self.log = get_logger("sl_tp_manager")
        self.config = config_manager
        
        # Track active SL/TP orders
        self.active_sl_tp: Dict[str, Dict[str, Any]] = {}
        
        # Broker connections (will be injected)
        self.bybit_client = None
        self.ibkr_client = None
        
        self.log.info("SL/TP Manager initialized - Server-side protection enforced")
    
    def set_brokers(self, bybit_client=None, ibkr_client=None):
        """Inject broker dependencies"""
        if bybit_client:
            self.bybit_client = bybit_client
        if ibkr_client:
            self.ibkr_client = ibkr_client
    
    async def attach(self, order_id: str, sl_tp_spec: Dict[str, Any], 
                    asset_type: str) -> Dict[str, Any]:
        """
        Attach SL/TP to a filled order.
        
        Args:
            order_id: The parent order ID
            sl_tp_spec: {
                'sl_price': float,
                'tp_price': float,
                'sl_trigger_type': 'mark'|'last',  # For futures
                'position_size': float  # Required for proper SL/TP sizing
            }
            asset_type: spot/futures/forex/options
        
        Returns:
            {'success': bool, 'sl_id': str, 'tp_id': str, 'reason': str}
        """
        
        self.log.info(f"Attaching SL/TP to order {order_id}: SL={sl_tp_spec.get('sl_price')}, "
                     f"TP={sl_tp_spec.get('tp_price')}")
        
        try:
            if asset_type in ['spot', 'futures']:
                result = await self._attach_bybit_sl_tp(order_id, sl_tp_spec, asset_type)
            elif asset_type in ['forex', 'options']:
                result = await self._attach_ibkr_bracket(order_id, sl_tp_spec, asset_type)
            else:
                result = {'success': False, 'reason': f'Unknown asset type: {asset_type}'}
            
            if result['success']:
                # Track active SL/TP
                self.active_sl_tp[order_id] = {
                    'sl_id': result.get('sl_id'),
                    'tp_id': result.get('tp_id'),
                    'sl_price': sl_tp_spec.get('sl_price'),
                    'tp_price': sl_tp_spec.get('tp_price'),
                    'asset_type': asset_type,
                    'status': SLTPStatus.ACTIVE,
                    'created_at': datetime.now().isoformat()
                }
                self.log.info(f"SL/TP attached successfully for order {order_id}")
            else:
                self.log.error(f"Failed to attach SL/TP: {result['reason']}")
            
            return result
            
        except Exception as e:
            self.log.error(f"Exception attaching SL/TP: {e}", exc_info=True)
            return {'success': False, 'reason': str(e)}
    
    async def _attach_bybit_sl_tp(self, order_id: str, sl_tp_spec: Dict[str, Any], 
                                  asset_type: str) -> Dict[str, Any]:
        """
        Attach SL/TP using Bybit trading-stop API.
        
        Bybit V5 API for server-side SL/TP:
        - Spot: OCO orders for SL/TP
        - Futures: trading-stop with mark/last price triggers
        """
        
        if not self.bybit_client:
            return {'success': False, 'reason': 'Bybit client not available'}
        
        try:
            if asset_type == 'spot':
                # Spot uses OCO (One-Cancels-Other) orders
                # Create stop-loss order
                sl_result = None
                if sl_tp_spec.get('sl_price'):
                    sl_result = await self.bybit_client.place_stop_order(
                        symbol=sl_tp_spec.get('symbol'),
                        side='sell',  # Assuming long position
                        stop_price=sl_tp_spec['sl_price'],
                        quantity=sl_tp_spec.get('position_size'),
                        order_link_id=f"SL_{order_id}"
                    )
                
                # Create take-profit order
                tp_result = None
                if sl_tp_spec.get('tp_price'):
                    tp_result = await self.bybit_client.place_limit_order(
                        symbol=sl_tp_spec.get('symbol'),
                        side='sell',
                        price=sl_tp_spec['tp_price'],
                        quantity=sl_tp_spec.get('position_size'),
                        order_link_id=f"TP_{order_id}"
                    )
                
                return {
                    'success': True,
                    'sl_id': sl_result.get('order_id') if sl_result else None,
                    'tp_id': tp_result.get('order_id') if tp_result else None
                }
                
            elif asset_type == 'futures':
                # Futures uses trading-stop API for position SL/TP
                # This modifies the position's SL/TP, not creates new orders
                
                params = {}
                
                if sl_tp_spec.get('sl_price'):
                    params['stop_loss'] = sl_tp_spec['sl_price']
                    params['sl_trigger_by'] = sl_tp_spec.get('sl_trigger_type', 'mark')
                
                if sl_tp_spec.get('tp_price'):
                    params['take_profit'] = sl_tp_spec['tp_price']
                    params['tp_trigger_by'] = sl_tp_spec.get('tp_trigger_type', 'last')
                
                result = await self.bybit_client.set_trading_stop(
                    symbol=sl_tp_spec.get('symbol'),
                    **params
                )
                
                if result.get('success'):
                    return {
                        'success': True,
                        'sl_id': f"SL_{order_id}",
                        'tp_id': f"TP_{order_id}"
                    }
                else:
                    return {'success': False, 'reason': result.get('msg', 'Trading-stop failed')}
            
        except Exception as e:
            return {'success': False, 'reason': f'Bybit SL/TP exception: {str(e)}'}
    
    async def _attach_ibkr_bracket(self, order_id: str, sl_tp_spec: Dict[str, Any], 
                                   asset_type: str) -> Dict[str, Any]:
        """
        Attach SL/TP using IBKR bracket orders.
        
        IBKR bracket orders:
        - Parent order (already filled)
        - Stop-loss child order
        - Take-profit child order
        - OCO (One-Cancels-Other) relationship
        """
        
        if not self.ibkr_client:
            return {'success': False, 'reason': 'IBKR client not available'}
        
        try:
            # Create bracket orders
            bracket_orders = []
            
            # Stop-loss order
            if sl_tp_spec.get('sl_price'):
                sl_order = {
                    'parent_id': order_id,
                    'order_type': 'stop',
                    'stop_price': sl_tp_spec['sl_price'],
                    'quantity': sl_tp_spec.get('position_size'),
                    'side': 'sell',  # Opposite of parent
                    'oco_group': f"OCO_{order_id}"
                }
                bracket_orders.append(sl_order)
            
            # Take-profit order
            if sl_tp_spec.get('tp_price'):
                tp_order = {
                    'parent_id': order_id,
                    'order_type': 'limit',
                    'limit_price': sl_tp_spec['tp_price'],
                    'quantity': sl_tp_spec.get('position_size'),
                    'side': 'sell',  # Opposite of parent
                    'oco_group': f"OCO_{order_id}"
                }
                bracket_orders.append(tp_order)
            
            # Submit bracket orders
            results = []
            for bracket_order in bracket_orders:
                result = await self.ibkr_client.place_bracket_order(bracket_order)
                results.append(result)
            
            # Check results
            sl_id = results[0].get('order_id') if len(results) > 0 and results[0].get('success') else None
            tp_id = results[1].get('order_id') if len(results) > 1 and results[1].get('success') else None
            
            if sl_id or tp_id:
                return {
                    'success': True,
                    'sl_id': sl_id,
                    'tp_id': tp_id
                }
            else:
                return {'success': False, 'reason': 'Failed to create bracket orders'}
            
        except Exception as e:
            return {'success': False, 'reason': f'IBKR bracket exception: {str(e)}'}
    
    async def sync(self, position_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sync/recover SL/TP after reconnect.
        Re-attaches SL/TP if missing or updates if position changed.
        
        Args:
            position_state: Current position state from broker
        
        Returns:
            {'success': bool, 'synced': int, 'failed': int}
        """
        
        self.log.info("Syncing SL/TP orders with broker positions")
        
        synced = 0
        failed = 0
        
        for position_id, position in position_state.items():
            # Check if we have SL/TP tracked
            if position_id in self.active_sl_tp:
                sl_tp_info = self.active_sl_tp[position_id]
                
                # Verify SL/TP still active on broker
                if position.get('sl_price') != sl_tp_info['sl_price'] or \
                   position.get('tp_price') != sl_tp_info['tp_price']:
                    
                    # Re-attach with correct values
                    result = await self.attach(
                        position_id,
                        {
                            'sl_price': sl_tp_info['sl_price'],
                            'tp_price': sl_tp_info['tp_price'],
                            'symbol': position.get('symbol'),
                            'position_size': position.get('size')
                        },
                        sl_tp_info['asset_type']
                    )
                    
                    if result['success']:
                        synced += 1
                    else:
                        failed += 1
        
        self.log.info(f"SL/TP sync complete: {synced} synced, {failed} failed")
        
        return {
            'success': True,
            'synced': synced,
            'failed': failed
        }
    
    async def cancel_sl_tp(self, order_id: str) -> Dict[str, Any]:
        """Cancel SL/TP orders for a position"""
        
        if order_id not in self.active_sl_tp:
            return {'success': False, 'reason': 'No SL/TP found for order'}
        
        sl_tp_info = self.active_sl_tp[order_id]
        asset_type = sl_tp_info['asset_type']
        
        try:
            if asset_type in ['spot', 'futures']:
                if self.bybit_client:
                    # Cancel individual SL/TP orders
                    if sl_tp_info.get('sl_id'):
                        await self.bybit_client.cancel_order(sl_tp_info['sl_id'])
                    if sl_tp_info.get('tp_id'):
                        await self.bybit_client.cancel_order(sl_tp_info['tp_id'])
                    
                    # For futures, also clear position SL/TP
                    if asset_type == 'futures':
                        await self.bybit_client.set_trading_stop(
                            symbol=sl_tp_info.get('symbol'),
                            stop_loss=0,
                            take_profit=0
                        )
            
            elif asset_type in ['forex', 'options']:
                if self.ibkr_client:
                    # Cancel bracket orders
                    if sl_tp_info.get('sl_id'):
                        await self.ibkr_client.cancel_order(sl_tp_info['sl_id'])
                    if sl_tp_info.get('tp_id'):
                        await self.ibkr_client.cancel_order(sl_tp_info['tp_id'])
            
            # Update tracking
            sl_tp_info['status'] = SLTPStatus.CANCELLED
            del self.active_sl_tp[order_id]
            
            self.log.info(f"SL/TP cancelled for order {order_id}")
            return {'success': True}
            
        except Exception as e:
            self.log.error(f"Error cancelling SL/TP: {e}", exc_info=True)
            return {'success': False, 'reason': str(e)}
    
    def get_active_sl_tp(self, asset_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active SL/TP orders"""
        result = []
        for order_id, info in self.active_sl_tp.items():
            if asset_type is None or info['asset_type'] == asset_type:
                result.append({
                    'order_id': order_id,
                    'sl_price': info['sl_price'],
                    'tp_price': info['tp_price'],
                    'asset_type': info['asset_type'],
                    'status': info['status'].value,
                    'created_at': info['created_at']
                })
        return result


# Module initialization
sl_tp_manager = SLTPManager()