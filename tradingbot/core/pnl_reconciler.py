# file: tradingbot/core/pnl_reconciler.py
# module_version: v1.00

"""
P&L Reconciler - The single source of truth for P&L.
This is the ONLY module that reconciles P&L with broker truth.
All P&L reporting must use this module's numbers.
"""

import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .configmanager import config_manager
from .loggerconfig import get_logger


class ReconciliationStatus(Enum):
    """Reconciliation status"""
    SYNCED = "synced"
    DIVERGED = "diverged"
    PENDING = "pending"
    ERROR = "error"


@dataclass
class Position:
    """Broker position representation"""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    fees: float
    asset_type: str
    broker: str  # bybit/ibkr
    position_id: str
    last_updated: datetime


@dataclass
class Fill:
    """Trade fill record"""
    order_id: str
    symbol: str
    side: str
    size: float
    price: float
    fee: float
    timestamp: datetime
    asset_type: str
    broker: str


@dataclass
class ReconciliationReport:
    """Reconciliation result"""
    status: ReconciliationStatus
    timestamp: datetime
    positions_count: int
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_fees: float
    discrepancies: List[Dict[str, Any]] = field(default_factory=list)
    broker_desync: bool = False


class PnLReconciler:
    """
    The authoritative source for all P&L calculations.
    Reconciles with broker truth and detects divergence.
    """
    
    def __init__(self):
        self.log = get_logger("pnl_reconciler")
        self.config = config_manager
        
        # Current positions from brokers
        self.positions: Dict[str, Position] = {}
        
        # Fill history
        self.fills: List[Fill] = []
        
        # P&L tracking
        self.realized_pnl: Dict[str, float] = {
            'spot': 0.0,
            'futures': 0.0,
            'forex': 0.0,
            'options': 0.0
        }
        
        self.unrealized_pnl: Dict[str, float] = {
            'spot': 0.0,
            'futures': 0.0,
            'forex': 0.0,
            'options': 0.0
        }
        
        self.total_fees: Dict[str, float] = {
            'spot': 0.0,
            'futures': 0.0,
            'forex': 0.0,
            'options': 0.0
        }
        
        # Reconciliation state
        self.last_reconciliation: Optional[ReconciliationReport] = None
        self.reconciliation_tolerance = 0.01  # $0.01 tolerance
        
        # Broker connections (will be injected)
        self.bybit_client = None
        self.ibkr_client = None
        
        self.log.info("P&L Reconciler initialized - Broker truth enforced")
    
    def set_brokers(self, bybit_client=None, ibkr_client=None):
        """Inject broker dependencies"""
        if bybit_client:
            self.bybit_client = bybit_client
        if ibkr_client:
            self.ibkr_client = ibkr_client
    
    async def pull_positions(self) -> Dict[str, Position]:
        """
        Pull current positions from all brokers.
        
        Returns:
            Dictionary of positions by symbol
        """
        
        self.log.info("Pulling positions from brokers")
        
        all_positions = {}
        
        # Pull from Bybit
        if self.bybit_client:
            try:
                bybit_positions = await self.bybit_client.get_positions()
                
                for pos_data in bybit_positions:
                    position = Position(
                        symbol=pos_data['symbol'],
                        side=pos_data['side'].lower(),
                        size=float(pos_data['size']),
                        entry_price=float(pos_data['avgPrice']),
                        current_price=float(pos_data.get('markPrice', pos_data['avgPrice'])),
                        unrealized_pnl=float(pos_data.get('unrealisedPnl', 0)),
                        realized_pnl=float(pos_data.get('realisedPnl', 0)),
                        fees=float(pos_data.get('cumRealisedPnl', 0)) - float(pos_data.get('realisedPnl', 0)),
                        asset_type='futures' if pos_data.get('isLinear') else 'spot',
                        broker='bybit',
                        position_id=pos_data.get('positionIdx', ''),
                        last_updated=datetime.now()
                    )
                    
                    all_positions[position.symbol] = position
                    
            except Exception as e:
                self.log.error(f"Error pulling Bybit positions: {e}", exc_info=True)
        
        # Pull from IBKR
        if self.ibkr_client:
            try:
                ibkr_positions = await self.ibkr_client.get_positions()
                
                for pos_data in ibkr_positions:
                    position = Position(
                        symbol=pos_data['symbol'],
                        side='long' if pos_data['position'] > 0 else 'short',
                        size=abs(float(pos_data['position'])),
                        entry_price=float(pos_data['averageCost']),
                        current_price=float(pos_data['marketPrice']),
                        unrealized_pnl=float(pos_data.get('unrealizedPNL', 0)),
                        realized_pnl=float(pos_data.get('realizedPNL', 0)),
                        fees=0,  # IBKR fees tracked separately
                        asset_type=self._get_ibkr_asset_type(pos_data),
                        broker='ibkr',
                        position_id=str(pos_data.get('conid', '')),
                        last_updated=datetime.now()
                    )
                    
                    all_positions[f"ibkr_{position.symbol}"] = position
                    
            except Exception as e:
                self.log.error(f"Error pulling IBKR positions: {e}", exc_info=True)
        
        # Update internal state
        self.positions = all_positions
        
        # Recalculate P&L
        self._recalculate_pnl()
        
        self.log.info(f"Pulled {len(all_positions)} positions from brokers")
        
        return all_positions
    
    async def pull_fills(self, since: Optional[datetime] = None) -> List[Fill]:
        """
        Pull trade fills from brokers.
        
        Args:
            since: Pull fills since this timestamp (default: last 24h)
        
        Returns:
            List of fills
        """
        
        if since is None:
            since = datetime.now() - timedelta(hours=24)
        
        self.log.info(f"Pulling fills since {since}")
        
        all_fills = []
        
        # Pull from Bybit
        if self.bybit_client:
            try:
                bybit_fills = await self.bybit_client.get_fills(since=since)
                
                for fill_data in bybit_fills:
                    fill = Fill(
                        order_id=fill_data['orderId'],
                        symbol=fill_data['symbol'],
                        side=fill_data['side'].lower(),
                        size=float(fill_data['qty']),
                        price=float(fill_data['price']),
                        fee=float(fill_data.get('execFee', 0)),
                        timestamp=datetime.fromtimestamp(int(fill_data['execTime']) / 1000),
                        asset_type='futures' if fill_data.get('isLinear') else 'spot',
                        broker='bybit'
                    )
                    
                    all_fills.append(fill)
                    
            except Exception as e:
                self.log.error(f"Error pulling Bybit fills: {e}", exc_info=True)
        
        # Pull from IBKR
        if self.ibkr_client:
            try:
                ibkr_fills = await self.ibkr_client.get_executions(since=since)
                
                for fill_data in ibkr_fills:
                    fill = Fill(
                        order_id=str(fill_data['orderId']),
                        symbol=fill_data['symbol'],
                        side=fill_data['side'].lower(),
                        size=float(fill_data['shares']),
                        price=float(fill_data['price']),
                        fee=float(fill_data.get('commission', 0)),
                        timestamp=datetime.fromisoformat(fill_data['time']),
                        asset_type=self._get_ibkr_asset_type(fill_data),
                        broker='ibkr'
                    )
                    
                    all_fills.append(fill)
                    
            except Exception as e:
                self.log.error(f"Error pulling IBKR fills: {e}", exc_info=True)
        
        # Sort by timestamp
        all_fills.sort(key=lambda x: x.timestamp)
        
        # Append to history (deduplicate)
        existing_ids = {f.order_id for f in self.fills}
        new_fills = [f for f in all_fills if f.order_id not in existing_ids]
        self.fills.extend(new_fills)
        
        self.log.info(f"Pulled {len(all_fills)} fills ({len(new_fills)} new)")
        
        return all_fills
    
    async def reconcile(self) -> ReconciliationReport:
        """
        Perform full P&L reconciliation with brokers.
        
        Returns:
            ReconciliationReport with status and any discrepancies
        """
        
        self.log.info("Starting P&L reconciliation")
        
        # Pull latest data
        await self.pull_positions()
        await self.pull_fills()
        
        # Calculate P&L from fills
        calculated_pnl = self._calculate_pnl_from_fills()
        
        # Compare with broker reported P&L
        discrepancies = []
        broker_desync = False
        
        for asset_type in ['spot', 'futures', 'forex', 'options']:
            # Get broker reported P&L
            broker_realized = sum(p.realized_pnl for p in self.positions.values() 
                                if p.asset_type == asset_type)
            broker_unrealized = sum(p.unrealized_pnl for p in self.positions.values() 
                                  if p.asset_type == asset_type)
            broker_fees = sum(p.fees for p in self.positions.values() 
                            if p.asset_type == asset_type)
            
            # Get calculated P&L
            calc_realized = calculated_pnl[asset_type]['realized']
            calc_fees = calculated_pnl[asset_type]['fees']
            
            # Check for discrepancies
            realized_diff = abs(broker_realized - calc_realized)
            fee_diff = abs(broker_fees - calc_fees)
            
            if realized_diff > self.reconciliation_tolerance:
                discrepancies.append({
                    'type': 'realized_pnl',
                    'asset_type': asset_type,
                    'broker_value': broker_realized,
                    'calculated_value': calc_realized,
                    'difference': realized_diff
                })
                
                # Significant discrepancy triggers broker_desync
                if realized_diff > 1.0:  # More than $1 difference
                    broker_desync = True
            
            if fee_diff > self.reconciliation_tolerance:
                discrepancies.append({
                    'type': 'fees',
                    'asset_type': asset_type,
                    'broker_value': broker_fees,
                    'calculated_value': calc_fees,
                    'difference': fee_diff
                })
            
            # Use broker values as truth
            self.realized_pnl[asset_type] = broker_realized
            self.unrealized_pnl[asset_type] = broker_unrealized
            self.total_fees[asset_type] = broker_fees
        
        # Create report
        report = ReconciliationReport(
            status=ReconciliationStatus.DIVERGED if discrepancies else ReconciliationStatus.SYNCED,
            timestamp=datetime.now(),
            positions_count=len(self.positions),
            total_unrealized_pnl=sum(self.unrealized_pnl.values()),
            total_realized_pnl=sum(self.realized_pnl.values()),
            total_fees=sum(self.total_fees.values()),
            discrepancies=discrepancies,
            broker_desync=broker_desync
        )
        
        self.last_reconciliation = report
        
        if broker_desync:
            self.log.error(f"BROKER DESYNC DETECTED: {len(discrepancies)} discrepancies")
        elif discrepancies:
            self.log.warning(f"Minor discrepancies found: {len(discrepancies)}")
        else:
            self.log.info("Reconciliation successful - all values match")
        
        return report
    
    async def record_order(self, order_result: Dict[str, Any], 
                          order_context: Dict[str, Any]):
        """Record an order for P&L tracking"""
        
        # This would typically update internal tracking
        # In production, might also persist to database
        
        self.log.debug(f"Recorded order: {order_result.get('order_id')}")
    
    def _recalculate_pnl(self):
        """Recalculate P&L from current positions"""
        
        for asset_type in ['spot', 'futures', 'forex', 'options']:
            positions = [p for p in self.positions.values() if p.asset_type == asset_type]
            
            self.unrealized_pnl[asset_type] = sum(p.unrealized_pnl for p in positions)
            self.realized_pnl[asset_type] = sum(p.realized_pnl for p in positions)
            self.total_fees[asset_type] = sum(p.fees for p in positions)
    
    def _calculate_pnl_from_fills(self) -> Dict[str, Dict[str, float]]:
        """Calculate P&L from fill history"""
        
        result = {
            'spot': {'realized': 0, 'fees': 0},
            'futures': {'realized': 0, 'fees': 0},
            'forex': {'realized': 0, 'fees': 0},
            'options': {'realized': 0, 'fees': 0}
        }
        
        # Group fills by symbol
        fills_by_symbol: Dict[str, List[Fill]] = {}
        for fill in self.fills:
            if fill.symbol not in fills_by_symbol:
                fills_by_symbol[fill.symbol] = []
            fills_by_symbol[fill.symbol].append(fill)
        
        # Calculate P&L for each symbol
        for symbol, symbol_fills in fills_by_symbol.items():
            # Sort by timestamp
            symbol_fills.sort(key=lambda x: x.timestamp)
            
            # FIFO P&L calculation
            buy_queue = []
            
            for fill in symbol_fills:
                asset_type = fill.asset_type
                result[asset_type]['fees'] += fill.fee
                
                if fill.side in ['buy', 'long']:
                    buy_queue.append((fill.size, fill.price))
                else:  # sell/short
                    remaining_size = fill.size
                    
                    while remaining_size > 0 and buy_queue:
                        buy_size, buy_price = buy_queue[0]
                        
                        match_size = min(remaining_size, buy_size)
                        pnl = match_size * (fill.price - buy_price)
                        result[asset_type]['realized'] += pnl
                        
                        remaining_size -= match_size
                        
                        if match_size >= buy_size:
                            buy_queue.pop(0)
                        else:
                            buy_queue[0] = (buy_size - match_size, buy_price)
        
        return result
    
    def _get_ibkr_asset_type(self, data: Dict[str, Any]) -> str:
        """Determine asset type from IBKR data"""
        
        sec_type = data.get('secType', '').upper()
        
        if sec_type == 'STK':
            return 'spot'
        elif sec_type == 'FUT':
            return 'futures'
        elif sec_type == 'CASH':
            return 'forex'
        elif sec_type == 'OPT':
            return 'options'
        else:
            return 'spot'  # Default
    
    def get_pnl_summary(self) -> Dict[str, Any]:
        """Get comprehensive P&L summary"""
        
        return {
            'realized_pnl': self.realized_pnl.copy(),
            'unrealized_pnl': self.unrealized_pnl.copy(),
            'total_fees': self.total_fees.copy(),
            'total_realized': sum(self.realized_pnl.values()),
            'total_unrealized': sum(self.unrealized_pnl.values()),
            'total_fees_paid': sum(self.total_fees.values()),
            'net_pnl': sum(self.realized_pnl.values()) + sum(self.unrealized_pnl.values()) - sum(self.total_fees.values()),
            'positions_count': len(self.positions),
            'last_reconciliation': self.last_reconciliation.status.value if self.last_reconciliation else 'never',
            'broker_desync': self.last_reconciliation.broker_desync if self.last_reconciliation else False
        }
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get a specific position"""
        return self.positions.get(symbol)
    
    def has_broker_desync(self) -> bool:
        """Check if broker desync flag is set"""
        return self.last_reconciliation.broker_desync if self.last_reconciliation else False


# Module initialization
pnl_reconciler = PnLReconciler()