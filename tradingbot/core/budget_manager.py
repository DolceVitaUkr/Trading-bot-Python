# file: tradingbot/core/budget_manager.py  
# module_version: v1.00

"""
Budget Manager - Per-asset allocation enforcement.
This is the ONLY module that manages capital allocations.
Enforces Bybit UTA and IBKR budget splits.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .configmanager import config_manager
from .loggerconfig import get_logger


@dataclass
class AssetAllocation:
    """Asset allocation tracking"""
    asset_type: str
    total_allocated: float
    available: float
    used: float
    reserved: float
    pnl_realized: float
    pnl_unrealized: float
    last_updated: datetime


class BudgetManager:
    """
    Manages per-asset capital allocations.
    Enforces software budget splits for Bybit UTA and IBKR.
    """
    
    def __init__(self):
        self.log = get_logger("budget_manager")
        self.config = config_manager
        
        # Asset allocations
        self.allocations: Dict[str, AssetAllocation] = {}
        
        # Initialize allocations from config
        self._initialize_allocations()
        
        # Track reallocation history
        self.reallocation_history: List[Dict[str, Any]] = []
        
        # Dependencies (will be injected)
        self.pnl_reconciler = None
        
        self.log.info("Budget Manager initialized - Capital allocations enforced")
    
    def _initialize_allocations(self):
        """Initialize allocations from config"""
        assets_config = self.config.config.get('assets', {})
        allocations_usd = assets_config.get('allocations_usd', {
            'spot': 900,
            'futures': 100,
            'forex': 0,
            'options': 0
        })
        
        for asset_type, amount in allocations_usd.items():
            self.allocations[asset_type] = AssetAllocation(
                asset_type=asset_type,
                total_allocated=amount,
                available=amount,
                used=0,
                reserved=0,
                pnl_realized=0,
                pnl_unrealized=0,
                last_updated=datetime.now()
            )
            
            self.log.info(f"Initialized {asset_type} allocation: ${amount}")
    
    def set_dependencies(self, pnl_reconciler=None):
        """Inject dependencies"""
        if pnl_reconciler:
            self.pnl_reconciler = pnl_reconciler
    
    def get_alloc(self, asset_type: str) -> float:
        """
        Get total allocation for an asset type.
        
        Args:
            asset_type: spot/futures/forex/options
        
        Returns:
            Total allocated USD amount
        """
        if asset_type not in self.allocations:
            self.log.error(f"Unknown asset type: {asset_type}")
            return 0
        
        return self.allocations[asset_type].total_allocated
    
    async def can_afford(self, order_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if we can afford an order within budget constraints.
        
        Args:
            order_context: {
                'asset_type': str,
                'symbol': str,
                'quantity': float,
                'price': float,
                'leverage': float (for futures)
            }
        
        Returns:
            {
                'pass': bool,
                'reason': str,
                'available': float,
                'required': float
            }
        """
        
        asset_type = order_context.get('asset_type')
        
        if asset_type not in self.allocations:
            return {
                'pass': False,
                'reason': f'Unknown asset type: {asset_type}',
                'available': 0,
                'required': 0
            }
        
        alloc = self.allocations[asset_type]
        
        # Calculate required capital
        quantity = order_context.get('quantity', 0)
        price = order_context.get('price', 0)
        
        if asset_type == 'futures':
            # Account for leverage
            leverage = order_context.get('leverage', 1)
            required_capital = (quantity * price) / leverage
        elif asset_type == 'options':
            # Options use premium
            option_price = order_context.get('option_price', price)
            required_capital = quantity * option_price * 100  # 100 shares per contract
        else:
            # Spot and forex
            required_capital = quantity * price
        
        # Check availability
        if required_capital > alloc.available:
            self.log.warning(f"Insufficient budget for {asset_type}: "
                           f"required ${required_capital:.2f}, available ${alloc.available:.2f}")
            return {
                'pass': False,
                'reason': f'Insufficient {asset_type} budget: need ${required_capital:.2f}, have ${alloc.available:.2f}',
                'available': alloc.available,
                'required': required_capital
            }
        
        # Reserve the capital (will be committed on fill)
        alloc.reserved += required_capital
        alloc.available -= required_capital
        
        self.log.info(f"Budget check passed for {asset_type}: "
                     f"reserved ${required_capital:.2f}, remaining ${alloc.available:.2f}")
        
        return {
            'pass': True,
            'reason': 'Budget available',
            'available': alloc.available,
            'required': required_capital
        }
    
    async def commit_capital(self, asset_type: str, amount: float):
        """
        Commit reserved capital after order fill.
        
        Args:
            asset_type: Asset type
            amount: Amount to commit from reserved
        """
        if asset_type not in self.allocations:
            self.log.error(f"Unknown asset type: {asset_type}")
            return
        
        alloc = self.allocations[asset_type]
        
        # Move from reserved to used
        commit_amount = min(amount, alloc.reserved)
        alloc.reserved -= commit_amount
        alloc.used += commit_amount
        alloc.last_updated = datetime.now()
        
        self.log.info(f"Committed ${commit_amount:.2f} for {asset_type}, "
                     f"used: ${alloc.used:.2f}, available: ${alloc.available:.2f}")
    
    async def release_capital(self, asset_type: str, amount: float):
        """
        Release capital after position close.
        
        Args:
            asset_type: Asset type
            amount: Amount to release back to available
        """
        if asset_type not in self.allocations:
            self.log.error(f"Unknown asset type: {asset_type}")
            return
        
        alloc = self.allocations[asset_type]
        
        # Move from used to available
        release_amount = min(amount, alloc.used)
        alloc.used -= release_amount
        alloc.available += release_amount
        alloc.last_updated = datetime.now()
        
        self.log.info(f"Released ${release_amount:.2f} for {asset_type}, "
                     f"used: ${alloc.used:.2f}, available: ${alloc.available:.2f}")
    
    def apply_pnl(self, asset_type: str, realized_pnl: float):
        """
        Apply realized P&L to the asset allocation.
        P&L stays within its asset bucket (compounds internally).
        
        Args:
            asset_type: Asset type
            realized_pnl: Realized P&L amount (positive or negative)
        """
        if asset_type not in self.allocations:
            self.log.error(f"Unknown asset type: {asset_type}")
            return
        
        alloc = self.allocations[asset_type]
        
        # Update P&L tracking
        alloc.pnl_realized += realized_pnl
        
        # Adjust available capital
        alloc.available += realized_pnl
        alloc.total_allocated += realized_pnl  # Compound within bucket
        alloc.last_updated = datetime.now()
        
        self.log.info(f"Applied P&L ${realized_pnl:.2f} to {asset_type}, "
                     f"new total: ${alloc.total_allocated:.2f}, "
                     f"available: ${alloc.available:.2f}")
    
    async def reallocate(self, asset_type: str, amount: float) -> Dict[str, Any]:
        """
        Dynamically reallocate capital (+$100 increments).
        
        Args:
            asset_type: Asset to add allocation to
            amount: Amount to add (typically $100)
        
        Returns:
            {'success': bool, 'new_allocation': float, 'reason': str}
        """
        
        if asset_type not in self.allocations:
            return {
                'success': False,
                'new_allocation': 0,
                'reason': f'Unknown asset type: {asset_type}'
            }
        
        # Check if amount is in allowed increments
        if amount % 100 != 0:
            return {
                'success': False,
                'new_allocation': self.allocations[asset_type].total_allocated,
                'reason': 'Reallocation must be in $100 increments'
            }
        
        alloc = self.allocations[asset_type]
        
        # Apply reallocation
        alloc.total_allocated += amount
        alloc.available += amount
        alloc.last_updated = datetime.now()
        
        # Record reallocation
        self.reallocation_history.append({
            'timestamp': datetime.now().isoformat(),
            'asset_type': asset_type,
            'amount': amount,
            'new_total': alloc.total_allocated
        })
        
        self.log.info(f"Reallocated ${amount} to {asset_type}, "
                     f"new total: ${alloc.total_allocated:.2f}")
        
        return {
            'success': True,
            'new_allocation': alloc.total_allocated,
            'reason': f'Successfully added ${amount} to {asset_type}'
        }
    
    def get_budget_summary(self) -> Dict[str, Any]:
        """Get comprehensive budget summary"""
        summary = {
            'total_capital': 0,
            'total_available': 0,
            'total_used': 0,
            'total_pnl': 0,
            'allocations': {}
        }
        
        for asset_type, alloc in self.allocations.items():
            summary['total_capital'] += alloc.total_allocated
            summary['total_available'] += alloc.available
            summary['total_used'] += alloc.used
            summary['total_pnl'] += alloc.pnl_realized
            
            summary['allocations'][asset_type] = {
                'total': alloc.total_allocated,
                'available': alloc.available,
                'used': alloc.used,
                'reserved': alloc.reserved,
                'pnl_realized': alloc.pnl_realized,
                'pnl_unrealized': alloc.pnl_unrealized,
                'utilization_pct': (alloc.used / alloc.total_allocated * 100) if alloc.total_allocated > 0 else 0,
                'last_updated': alloc.last_updated.isoformat()
            }
        
        return summary
    
    async def reconcile_with_broker(self, broker_balances: Dict[str, float]) -> Dict[str, Any]:
        """
        Reconcile internal allocations with broker truth.
        
        Args:
            broker_balances: {'spot': float, 'futures': float, ...}
        
        Returns:
            {
                'success': bool,
                'discrepancies': list,
                'synced': bool
            }
        """
        
        discrepancies = []
        
        for asset_type, broker_balance in broker_balances.items():
            if asset_type in self.allocations:
                alloc = self.allocations[asset_type]
                internal_total = alloc.total_allocated
                
                # Check for discrepancy
                diff = abs(broker_balance - internal_total)
                tolerance = 0.01  # $0.01 tolerance for rounding
                
                if diff > tolerance:
                    discrepancy_pct = (diff / internal_total * 100) if internal_total > 0 else 100
                    discrepancies.append({
                        'asset_type': asset_type,
                        'internal': internal_total,
                        'broker': broker_balance,
                        'difference': broker_balance - internal_total,
                        'difference_pct': discrepancy_pct
                    })
                    
                    # Auto-sync if discrepancy is significant
                    if discrepancy_pct > 1:  # More than 1% difference
                        self.log.warning(f"Significant discrepancy for {asset_type}: "
                                       f"internal=${internal_total:.2f}, broker=${broker_balance:.2f}")
                        
                        # Adjust to broker truth
                        adjustment = broker_balance - internal_total
                        alloc.total_allocated = broker_balance
                        alloc.available += adjustment
                        alloc.last_updated = datetime.now()
                        
                        self.log.info(f"Auto-synced {asset_type} to broker balance: ${broker_balance:.2f}")
        
        if discrepancies:
            self.log.warning(f"Budget discrepancies found: {len(discrepancies)} assets")
        
        return {
            'success': True,
            'discrepancies': discrepancies,
            'synced': len(discrepancies) == 0
        }
    
    def enforce_separation(self) -> bool:
        """
        Enforce strict separation between asset allocations.
        Ensures futures never uses spot budget, etc.
        
        Returns:
            True if separation is intact, False if violations found
        """
        
        # Check that no allocation is negative (would indicate cross-contamination)
        for asset_type, alloc in self.allocations.items():
            if alloc.available < 0:
                self.log.error(f"VIOLATION: {asset_type} has negative available balance: ${alloc.available:.2f}")
                # Force correction
                alloc.available = 0
                return False
            
            if alloc.total_allocated < alloc.used:
                self.log.error(f"VIOLATION: {asset_type} used ${alloc.used:.2f} exceeds allocated ${alloc.total_allocated:.2f}")
                return False
        
        return True


# Module initialization
budget_manager = BudgetManager()