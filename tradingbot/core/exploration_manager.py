# file: tradingbot/core/exploration_manager.py
# module_version: v1.00

"""
Exploration Manager - Candidate rotation and fairness control.
This is the ONLY module that manages exploration quotas and fairness.
Ensures continuous exploration never stops globally.
"""

import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import random

from .configmanager import config_manager
from .loggerconfig import get_logger
from .strategy_registry import strategy_registry, StrategyState


class RotationPolicy(Enum):
    """Strategy rotation policies"""
    ROUND_ROBIN = "round_robin"
    ROUND_ROBIN_WITH_BOOST = "round_robin_with_boost"
    PRIORITY_QUEUE = "priority_queue"
    RANDOM = "random"


@dataclass
class CandidateInfo:
    """Active candidate tracking"""
    strategy_id: str
    asset_type: str
    trade_count: int
    last_trade_time: Optional[datetime]
    priority_boost: float
    cooldown_until: Optional[datetime]
    trades_this_hour: int
    hour_start: datetime


class ExplorationManager:
    """
    Manages per-asset candidate rotation and fairness.
    Ensures continuous exploration with fair opportunity allocation.
    """
    
    def __init__(self):
        self.log = get_logger("exploration_manager")
        self.config = config_manager
        
        # Active candidates per asset
        self.active_candidates: Dict[str, List[CandidateInfo]] = {
            'spot': [],
            'futures': [],
            'forex': [],
            'options': []
        }
        
        # Rotation state
        self.rotation_indices: Dict[str, int] = {
            'spot': 0,
            'futures': 0,
            'forex': 0,
            'options': 0
        }
        
        # Configuration
        self._load_exploration_config()
        
        # Dependencies (will be injected)
        self.strategy_generator = None  # ML/RL strategy generator
        
        self.log.info("Exploration Manager initialized - Continuous exploration enforced")
    
    def _load_exploration_config(self):
        """Load exploration configuration"""
        
        exploration_config = self.config.config.get('exploration', {})
        
        # Per-asset configuration
        self.config_per_asset = exploration_config.get('per_asset', {
            'spot': {
                'active_candidates': 6,
                'min_trades_to_validate': 100,
                'priority_boost_under_trades': 40
            },
            'futures': {
                'active_candidates': 6, 
                'min_trades_to_validate': 100,
                'priority_boost_under_trades': 40
            },
            'forex': {
                'active_candidates': 4,
                'min_trades_to_validate': 100,
                'priority_boost_under_trades': 40
            },
            'options': {
                'active_candidates': 4,
                'min_trades_to_validate': 100,
                'priority_boost_under_trades': 40
            }
        })
        
        # Opportunity allocation
        opportunity_config = exploration_config.get('opportunity_allocation', {})
        
        self.rotation_policy = RotationPolicy(opportunity_config.get('rotation', 'round_robin_with_boost'))
        self.max_trades_per_hour_per_candidate = opportunity_config.get('max_trades_per_hour_per_candidate', 12)
        self.cooldown_seconds = opportunity_config.get('cooldown_seconds_after_trade', 30)
    
    def set_dependencies(self, strategy_generator=None):
        """Inject dependencies"""
        if strategy_generator:
            self.strategy_generator = strategy_generator
    
    async def ensure_active_candidates(self, asset_type: str):
        """
        Ensure target number of active candidates for an asset.
        Backfills slots with new candidates when needed.
        """
        
        if asset_type not in self.active_candidates:
            self.log.error(f"Unknown asset type: {asset_type}")
            return
        
        config = self.config_per_asset[asset_type]
        target_count = config['active_candidates']
        current_candidates = self.active_candidates[asset_type]
        
        # Remove completed candidates (those promoted to validation)
        completed_ids = []
        for candidate in current_candidates:
            strategy = strategy_registry.get_strategy(candidate.strategy_id)
            
            if not strategy or strategy.state != StrategyState.EXPLORING:
                completed_ids.append(candidate.strategy_id)
        
        # Remove completed candidates
        self.active_candidates[asset_type] = [
            c for c in current_candidates 
            if c.strategy_id not in completed_ids
        ]
        
        current_count = len(self.active_candidates[asset_type])
        needed = target_count - current_count
        
        if needed > 0:
            self.log.info(f"Need {needed} new candidates for {asset_type} "
                         f"(current: {current_count}, target: {target_count})")
            
            # Generate new candidates
            for _ in range(needed):
                new_candidate = await self._create_new_candidate(asset_type)
                if new_candidate:
                    self.active_candidates[asset_type].append(new_candidate)
        
        self.log.debug(f"Active candidates for {asset_type}: {len(self.active_candidates[asset_type])}")
    
    async def _create_new_candidate(self, asset_type: str) -> Optional[CandidateInfo]:
        """Create a new exploration candidate"""
        
        if not self.strategy_generator:
            self.log.warning("No strategy generator available")
            return None
        
        try:
            # Generate new strategy
            strategy_descriptor = await self.strategy_generator.generate(asset_type)
            
            # Register with strategy registry
            strategy_id = strategy_registry.create(strategy_descriptor)
            
            # Transition to exploring state
            strategy_registry.set_state(
                strategy_id, 
                StrategyState.EXPLORING,
                "Added to exploration rotation"
            )
            
            # Create candidate info
            candidate = CandidateInfo(
                strategy_id=strategy_id,
                asset_type=asset_type,
                trade_count=0,
                last_trade_time=None,
                priority_boost=0.0,
                cooldown_until=None,
                trades_this_hour=0,
                hour_start=datetime.now()
            )
            
            self.log.info(f"Created new candidate {strategy_id} for {asset_type}")
            return candidate
            
        except Exception as e:
            self.log.error(f"Failed to create new candidate for {asset_type}: {e}", exc_info=True)
            return None
    
    async def get_next_candidate(self, asset_type: str, opportunity: Dict[str, Any]) -> Optional[str]:
        """
        Get the next candidate strategy for an opportunity.
        
        Args:
            asset_type: Asset type for the opportunity
            opportunity: {
                'symbol': str,
                'timestamp': datetime,
                'signal_strength': float,
                'features': dict
            }
        
        Returns:
            strategy_id of selected candidate, or None if none available
        """
        
        # Ensure we have candidates
        await self.ensure_active_candidates(asset_type)
        
        candidates = self.active_candidates[asset_type]
        
        if not candidates:
            self.log.warning(f"No candidates available for {asset_type}")
            return None
        
        # Filter candidates
        eligible = self._filter_eligible_candidates(candidates)
        
        if not eligible:
            self.log.debug(f"No eligible candidates for {asset_type} (all in cooldown or blocked)")
            return None
        
        # Select candidate based on rotation policy
        selected = self._select_candidate(eligible, asset_type)
        
        if selected:
            # Update candidate state
            self._update_candidate_after_selection(selected)
            
            self.log.debug(f"Selected candidate {selected.strategy_id} for {asset_type}")
            
            return selected.strategy_id
        
        return None
    
    def _filter_eligible_candidates(self, candidates: List[CandidateInfo]) -> List[CandidateInfo]:
        """Filter candidates to only eligible ones"""
        
        now = datetime.now()
        eligible = []
        
        for candidate in candidates:
            # Check if strategy has blocking flags
            strategy = strategy_registry.get_strategy(candidate.strategy_id)
            if not strategy:
                continue
            
            if strategy.flags:
                self.log.debug(f"Candidate {candidate.strategy_id} has flags: {[f.value for f in strategy.flags]}")
                continue
            
            # Check cooldown
            if candidate.cooldown_until and now < candidate.cooldown_until:
                continue
            
            # Check hourly trade limit
            if self._is_new_hour(candidate):
                candidate.trades_this_hour = 0
                candidate.hour_start = now
            
            if candidate.trades_this_hour >= self.max_trades_per_hour_per_candidate:
                continue
            
            eligible.append(candidate)
        
        return eligible
    
    def _select_candidate(self, eligible: List[CandidateInfo], asset_type: str) -> Optional[CandidateInfo]:
        """Select candidate based on rotation policy"""
        
        if not eligible:
            return None
        
        if self.rotation_policy == RotationPolicy.ROUND_ROBIN:
            return self._round_robin_select(eligible, asset_type)
        
        elif self.rotation_policy == RotationPolicy.ROUND_ROBIN_WITH_BOOST:
            return self._round_robin_with_boost_select(eligible, asset_type)
        
        elif self.rotation_policy == RotationPolicy.PRIORITY_QUEUE:
            return self._priority_select(eligible)
        
        elif self.rotation_policy == RotationPolicy.RANDOM:
            return random.choice(eligible)
        
        else:
            return eligible[0]  # Default to first
    
    def _round_robin_select(self, eligible: List[CandidateInfo], asset_type: str) -> CandidateInfo:
        """Simple round-robin selection"""
        
        rotation_index = self.rotation_indices[asset_type]
        selected = eligible[rotation_index % len(eligible)]
        
        # Update rotation index
        self.rotation_indices[asset_type] = (rotation_index + 1) % len(eligible)
        
        return selected
    
    def _round_robin_with_boost_select(self, eligible: List[CandidateInfo], asset_type: str) -> CandidateInfo:
        """Round-robin with priority boost for candidates with fewer trades"""
        
        config = self.config_per_asset[asset_type]
        boost_threshold = config['priority_boost_under_trades']
        
        # Calculate priority scores
        candidates_with_scores = []
        
        for candidate in eligible:
            base_score = 1.0
            
            # Priority boost for candidates with fewer trades
            if candidate.trade_count < boost_threshold:
                boost = (boost_threshold - candidate.trade_count) / boost_threshold
                candidate.priority_boost = boost
                base_score += boost
            else:
                candidate.priority_boost = 0.0
            
            candidates_with_scores.append((candidate, base_score))
        
        # Sort by score (higher is better)
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select from top candidates with some randomness
        top_candidates = [c for c, s in candidates_with_scores[:3]]  # Top 3
        return random.choice(top_candidates)
    
    def _priority_select(self, eligible: List[CandidateInfo]) -> CandidateInfo:
        """Select based on priority (fewer trades = higher priority)"""
        
        # Sort by trade count (ascending)
        eligible.sort(key=lambda x: x.trade_count)
        return eligible[0]
    
    def _update_candidate_after_selection(self, candidate: CandidateInfo):
        """Update candidate state after being selected for a trade"""
        
        now = datetime.now()
        
        # Set cooldown
        candidate.cooldown_until = now + timedelta(seconds=self.cooldown_seconds)
        
        # Update hourly counter
        if self._is_new_hour(candidate):
            candidate.trades_this_hour = 0
            candidate.hour_start = now
        
        candidate.trades_this_hour += 1
    
    def _is_new_hour(self, candidate: CandidateInfo) -> bool:
        """Check if we're in a new hour for rate limiting"""
        
        now = datetime.now()
        return (now - candidate.hour_start).total_seconds() >= 3600
    
    async def on_trade_completed(self, strategy_id: str, trade_result: Dict[str, Any]):
        """
        Notify exploration manager of completed trade.
        
        Args:
            strategy_id: Strategy that completed the trade
            trade_result: Trade outcome data
        """
        
        # Find candidate
        candidate = None
        asset_type = None
        
        for at, candidates in self.active_candidates.items():
            for c in candidates:
                if c.strategy_id == strategy_id:
                    candidate = c
                    asset_type = at
                    break
            if candidate:
                break
        
        if not candidate:
            self.log.debug(f"Candidate {strategy_id} not found in active candidates")
            return
        
        # Update candidate
        candidate.trade_count += 1
        candidate.last_trade_time = datetime.now()
        
        # Update strategy registry
        strategy_registry.inc_counter(strategy_id, 'paper_trades_closed')
        
        # Check if ready for validation
        config = self.config_per_asset[asset_type]
        min_trades = config['min_trades_to_validate']
        
        if candidate.trade_count >= min_trades:
            await self.promote_candidate(strategy_id, "Reached minimum trade threshold")
        
        self.log.debug(f"Trade completed for candidate {strategy_id}: "
                      f"{candidate.trade_count} total trades")
    
    async def promote_candidate(self, strategy_id: str, reason: str):
        """
        Promote candidate to validation.
        
        Args:
            strategy_id: Strategy to promote
            reason: Reason for promotion
        """
        
        # Transition to validating state
        success = strategy_registry.set_state(
            strategy_id,
            StrategyState.VALIDATING,
            reason
        )
        
        if success:
            # Remove from active candidates
            for candidates in self.active_candidates.values():
                self.active_candidates[candidates[0].asset_type] = [
                    c for c in candidates if c.strategy_id != strategy_id
                ]
            
            self.log.info(f"Promoted candidate {strategy_id} to validation: {reason}")
            
            # Backfill the slot immediately
            strategy = strategy_registry.get_strategy(strategy_id)
            if strategy:
                await self.ensure_active_candidates(strategy.asset_type)
    
    async def retire_candidate(self, strategy_id: str, reason: str):
        """
        Retire candidate from exploration.
        
        Args:
            strategy_id: Strategy to retire
            reason: Reason for retirement
        """
        
        # Find and remove candidate
        for asset_type, candidates in self.active_candidates.items():
            self.active_candidates[asset_type] = [
                c for c in candidates if c.strategy_id != strategy_id
            ]
        
        # Update strategy registry
        strategy_registry.set_state(strategy_id, StrategyState.SUSPENDED, reason)
        
        self.log.info(f"Retired candidate {strategy_id}: {reason}")
    
    def get_exploration_status(self) -> Dict[str, Any]:
        """Get current exploration status"""
        
        status = {
            'active_candidates_count': {},
            'rotation_policy': self.rotation_policy.value,
            'candidates_detail': {}
        }
        
        for asset_type, candidates in self.active_candidates.items():
            status['active_candidates_count'][asset_type] = len(candidates)
            
            status['candidates_detail'][asset_type] = [
                {
                    'strategy_id': c.strategy_id,
                    'trade_count': c.trade_count,
                    'priority_boost': c.priority_boost,
                    'trades_this_hour': c.trades_this_hour,
                    'in_cooldown': c.cooldown_until and datetime.now() < c.cooldown_until
                }
                for c in candidates
            ]
        
        return status


# Module initialization  
exploration_manager = ExplorationManager()