# file: tradingbot/core/strategy_scheduler.py
# module_version: v1.00

"""
Strategy Scheduler - Maps opportunities to candidate strategies.
This is the ONLY module that decides which strategy gets each opportunity.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .configmanager import config_manager  
from .loggerconfig import get_logger
from .exploration_manager import exploration_manager


@dataclass
class TradingOpportunity:
    """Trading opportunity representation"""
    symbol: str
    asset_type: str
    timestamp: datetime
    signal_strength: float
    side: str  # buy/sell/long/short
    features: Dict[str, float]
    source: str  # indicator/ml/rl/spike_detector
    confidence: float


class StrategyScheduler:
    """
    Maps trading opportunities to candidate strategies.
    Ensures fair distribution and respects strategy flags/counters.
    """
    
    def __init__(self):
        self.log = get_logger("strategy_scheduler")
        self.config = config_manager
        
        # Opportunity queue per asset
        self.opportunity_queue: Dict[str, List[TradingOpportunity]] = {
            'spot': [],
            'futures': [],
            'forex': [],
            'options': []
        }
        
        # Scheduling statistics
        self.stats = {
            'opportunities_processed': 0,
            'opportunities_assigned': 0,
            'opportunities_dropped': 0
        }
        
        self.log.info("Strategy Scheduler initialized")
    
    async def process_opportunity(self, opportunity: TradingOpportunity) -> Optional[str]:
        """
        Process a trading opportunity and assign to a strategy.
        
        Args:
            opportunity: The trading opportunity
            
        Returns:
            strategy_id if assigned, None if dropped
        """
        
        self.stats['opportunities_processed'] += 1
        
        # Get candidate from exploration manager
        selected_strategy = await exploration_manager.get_next_candidate(
            opportunity.asset_type,
            {
                'symbol': opportunity.symbol,
                'timestamp': opportunity.timestamp,
                'signal_strength': opportunity.signal_strength,
                'features': opportunity.features
            }
        )
        
        if selected_strategy:
            self.stats['opportunities_assigned'] += 1
            
            self.log.debug(f"Assigned opportunity {opportunity.symbol} to strategy {selected_strategy}")
            
            return selected_strategy
        else:
            self.stats['opportunities_dropped'] += 1
            
            self.log.debug(f"Dropped opportunity {opportunity.symbol} - no eligible candidates")
            
            return None
    
    def queue_opportunity(self, opportunity: TradingOpportunity):
        """Queue an opportunity for processing"""
        
        if opportunity.asset_type in self.opportunity_queue:
            self.opportunity_queue[opportunity.asset_type].append(opportunity)
            
            # Maintain queue size limit
            max_queue_size = 100
            if len(self.opportunity_queue[opportunity.asset_type]) > max_queue_size:
                self.opportunity_queue[opportunity.asset_type].pop(0)
    
    async def process_queued_opportunities(self, asset_type: str) -> List[Tuple[TradingOpportunity, str]]:
        """
        Process all queued opportunities for an asset type.
        
        Returns:
            List of (opportunity, strategy_id) pairs
        """
        
        assignments = []
        queue = self.opportunity_queue.get(asset_type, [])
        
        for opportunity in queue:
            strategy_id = await self.process_opportunity(opportunity)
            if strategy_id:
                assignments.append((opportunity, strategy_id))
        
        # Clear processed opportunities
        self.opportunity_queue[asset_type] = []
        
        return assignments
    
    def get_scheduling_stats(self) -> Dict[str, Any]:
        """Get scheduling statistics"""
        
        total = self.stats['opportunities_processed']
        
        return {
            'total_processed': total,
            'assigned': self.stats['opportunities_assigned'],
            'dropped': self.stats['opportunities_dropped'],
            'assignment_rate': self.stats['opportunities_assigned'] / total if total > 0 else 0,
            'queue_sizes': {k: len(v) for k, v in self.opportunity_queue.items()}
        }


# Module initialization
strategy_scheduler = StrategyScheduler()