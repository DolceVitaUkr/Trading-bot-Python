# file: tradingbot/core/strategy_registry.py
# module_version: v1.00

"""
Strategy Registry - States, flags, counters, and lifecycle management.
This is the ONLY module that manages strategy metadata and states.
"""

import json
import sqlite3
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import uuid

from .configmanager import config_manager
from .loggerconfig import get_logger


class StrategyState(Enum):
    """Strategy lifecycle states"""
    DRAFT = "draft"                 # Generated, not trading yet
    EXPLORING = "exploring"         # Paper trading, accumulating trades
    VALIDATING = "validating"       # Validation suite running
    APPROVED_LIVE = "approved_live" # Eligible for live trading
    SUSPENDED = "suspended"         # Auto/manual pause
    HOLD = "hold"                   # Manual hold, remains listed
    REJECTED = "rejected"           # Failed validation
    DEPRECATED = "deprecated"       # Superseded by newer version


class StrategyFlag(Enum):
    """Strategy control flags"""
    MANUAL_HOLD = "manual_hold"         # Human pause
    MANUAL_KILL = "manual_kill"         # Human immediate stop/flat
    RISK_ALERT = "risk_alert"           # Risk manager triggered
    DATA_ISSUE = "data_issue"           # Feed gaps/outliers
    RATE_LIMIT = "rate_limit"           # Broker/API pacing hit
    BROKER_DESYNC = "broker_desync"     # P&L/position mismatch


@dataclass
class StrategyRecord:
    """Complete strategy record"""
    strategy_id: str
    asset_type: str  # spot/futures/forex/options
    version: str
    params_hash: str
    state: StrategyState
    flags: Set[StrategyFlag]
    counters: Dict[str, int]
    metrics_last_24h: Dict[str, float]
    reasons: List[str]  # History of decisions
    timestamps: Dict[str, str]  # ISO format timestamps
    created_at: str
    last_updated: str


class StrategyRegistry:
    """
    Central registry for all strategy lifecycle management.
    Tracks states, flags, counters, and transition history.
    """
    
    def __init__(self):
        self.log = get_logger("strategy_registry")
        self.config = config_manager
        
        # Strategy records
        self.strategies: Dict[str, StrategyRecord] = {}
        
        # Persistence
        self.db_path = Path("tradingbot/state/strategy_registry.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Load existing strategies
        self._load_strategies()
        
        self.log.info("Strategy Registry initialized")
    
    def _init_database(self):
        """Initialize SQLite database for strategy records"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id TEXT PRIMARY KEY,
                    asset_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    state TEXT NOT NULL,
                    flags TEXT NOT NULL,  -- JSON array
                    counters TEXT NOT NULL,  -- JSON object
                    metrics_last_24h TEXT NOT NULL,  -- JSON object
                    reasons TEXT NOT NULL,  -- JSON array
                    timestamps TEXT NOT NULL,  -- JSON object
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_asset_state ON strategies(asset_type, state)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_state ON strategies(state)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_asset_type ON strategies(asset_type)')
            
            conn.commit()
    
    def _load_strategies(self):
        """Load strategies from database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('SELECT * FROM strategies')
            
            for row in cursor.fetchall():
                record = StrategyRecord(
                    strategy_id=row['strategy_id'],
                    asset_type=row['asset_type'],
                    version=row['version'],
                    params_hash=row['params_hash'],
                    state=StrategyState(row['state']),
                    flags=set(StrategyFlag(f) for f in json.loads(row['flags'])),
                    counters=json.loads(row['counters']),
                    metrics_last_24h=json.loads(row['metrics_last_24h']),
                    reasons=json.loads(row['reasons']),
                    timestamps=json.loads(row['timestamps']),
                    created_at=row['created_at'],
                    last_updated=row['last_updated']
                )
                
                self.strategies[record.strategy_id] = record
        
        self.log.info(f"Loaded {len(self.strategies)} strategies from database")
    
    def create(self, strategy_descriptor: Dict[str, Any]) -> str:
        """
        Register a new strategy.
        
        Args:
            strategy_descriptor: {
                'asset_type': str,
                'strategy_type': str,  # ml/rl/rule_based
                'parameters': dict,
                'version': str,
                'generator_info': dict
            }
        
        Returns:
            strategy_id
        """
        
        strategy_id = str(uuid.uuid4())
        params_hash = self._hash_parameters(strategy_descriptor.get('parameters', {}))
        
        now = datetime.now().isoformat()
        
        record = StrategyRecord(
            strategy_id=strategy_id,
            asset_type=strategy_descriptor['asset_type'],
            version=strategy_descriptor.get('version', '1.0'),
            params_hash=params_hash,
            state=StrategyState.DRAFT,
            flags=set(),
            counters={
                'paper_trades_closed': 0,
                'paper_trades_open': 0,
                'live_trades_closed': 0,
                'live_trades_open': 0,
                'epochs_used': 0,
                'validation_attempts': 0
            },
            metrics_last_24h={
                'pf': 0.0,
                'sharpe': 0.0,
                'max_dd': 0.0,
                'win_rate': 0.0,
                'cvar_5': 0.0,
                'avg_trade': 0.0
            },
            reasons=[],
            timestamps={
                'created': now,
                'first_trade': None,
                'last_trade': None,
                'last_validation': None
            },
            created_at=now,
            last_updated=now
        )
        
        self.strategies[strategy_id] = record
        self._save_strategy(record)
        
        self.log.info(f"Created strategy {strategy_id} for {strategy_descriptor['asset_type']}")
        
        return strategy_id
    
    def set_state(self, strategy_id: str, new_state: StrategyState, 
                  reason: Optional[str] = None) -> bool:
        """
        Transition strategy to new state.
        
        Args:
            strategy_id: Strategy ID
            new_state: Target state
            reason: Reason for transition
        
        Returns:
            True if transition allowed and completed
        """
        
        if strategy_id not in self.strategies:
            self.log.error(f"Strategy {strategy_id} not found")
            return False
        
        record = self.strategies[strategy_id]
        old_state = record.state
        
        # Validate transition
        if not self._is_valid_transition(old_state, new_state):
            self.log.warning(f"Invalid transition from {old_state.value} to {new_state.value}")
            return False
        
        # Perform transition
        record.state = new_state
        record.last_updated = datetime.now().isoformat()
        
        if reason:
            record.reasons.append(f"{datetime.now().isoformat()}: {old_state.value} → {new_state.value}: {reason}")
        
        # Update timestamps
        if new_state == StrategyState.EXPLORING and not record.timestamps.get('first_trade'):
            record.timestamps['exploring_started'] = record.last_updated
        elif new_state == StrategyState.VALIDATING:
            record.timestamps['validation_started'] = record.last_updated
        elif new_state == StrategyState.APPROVED_LIVE:
            record.timestamps['approved'] = record.last_updated
        
        self._save_strategy(record)
        
        self.log.info(f"Strategy {strategy_id} transitioned: {old_state.value} → {new_state.value}")
        
        return True
    
    def add_flag(self, strategy_id: str, flag: StrategyFlag, reason: str = "") -> bool:
        """Add a flag to a strategy"""
        
        if strategy_id not in self.strategies:
            return False
        
        record = self.strategies[strategy_id]
        
        if flag not in record.flags:
            record.flags.add(flag)
            record.last_updated = datetime.now().isoformat()
            
            if reason:
                record.reasons.append(f"{record.last_updated}: Flag added {flag.value}: {reason}")
            
            self._save_strategy(record)
            
            self.log.info(f"Added flag {flag.value} to strategy {strategy_id}")
        
        return True
    
    def remove_flag(self, strategy_id: str, flag: StrategyFlag, reason: str = "") -> bool:
        """Remove a flag from a strategy"""
        
        if strategy_id not in self.strategies:
            return False
        
        record = self.strategies[strategy_id]
        
        if flag in record.flags:
            record.flags.remove(flag)
            record.last_updated = datetime.now().isoformat()
            
            if reason:
                record.reasons.append(f"{record.last_updated}: Flag removed {flag.value}: {reason}")
            
            self._save_strategy(record)
            
            self.log.info(f"Removed flag {flag.value} from strategy {strategy_id}")
        
        return True
    
    def inc_counter(self, strategy_id: str, counter_name: str, amount: int = 1) -> bool:
        """Increment a counter for a strategy"""
        
        if strategy_id not in self.strategies:
            return False
        
        record = self.strategies[strategy_id]
        
        if counter_name not in record.counters:
            record.counters[counter_name] = 0
        
        record.counters[counter_name] += amount
        record.last_updated = datetime.now().isoformat()
        
        # Update timestamps for trade counters
        if counter_name in ['paper_trades_closed', 'live_trades_closed']:
            record.timestamps['last_trade'] = record.last_updated
            
            if not record.timestamps.get('first_trade'):
                record.timestamps['first_trade'] = record.last_updated
        
        self._save_strategy(record)
        
        self.log.debug(f"Incremented {counter_name} for strategy {strategy_id}: {record.counters[counter_name]}")
        
        return True
    
    def update_metrics(self, strategy_id: str, metrics: Dict[str, float]) -> bool:
        """Update 24h metrics for a strategy"""
        
        if strategy_id not in self.strategies:
            return False
        
        record = self.strategies[strategy_id]
        record.metrics_last_24h.update(metrics)
        record.last_updated = datetime.now().isoformat()
        
        self._save_strategy(record)
        
        self.log.debug(f"Updated metrics for strategy {strategy_id}")
        
        return True
    
    def get_for(self, asset_type: Optional[str] = None, 
               state: Optional[StrategyState] = None,
               flags: Optional[List[StrategyFlag]] = None) -> List[StrategyRecord]:
        """
        Get strategies matching criteria.
        
        Args:
            asset_type: Filter by asset type
            state: Filter by state
            flags: Filter by flags (must have ALL specified flags)
        
        Returns:
            List of matching strategy records
        """
        
        results = []
        
        for record in self.strategies.values():
            # Asset type filter
            if asset_type and record.asset_type != asset_type:
                continue
            
            # State filter
            if state and record.state != state:
                continue
            
            # Flags filter
            if flags:
                if not all(flag in record.flags for flag in flags):
                    continue
            
            results.append(record)
        
        return results
    
    def get_exploring_candidates(self, asset_type: str, 
                               min_trades_required: int = 100) -> List[StrategyRecord]:
        """Get exploring strategies eligible for validation"""
        
        candidates = []
        
        for record in self.get_for(asset_type=asset_type, state=StrategyState.EXPLORING):
            # Skip flagged strategies
            if record.flags:
                continue
            
            # Check trade count
            if record.counters.get('paper_trades_closed', 0) >= min_trades_required:
                candidates.append(record)
        
        return candidates
    
    def get_active_strategies(self, asset_type: str) -> List[StrategyRecord]:
        """Get strategies actively trading (exploring or live)"""
        
        active_states = [StrategyState.EXPLORING, StrategyState.APPROVED_LIVE]
        
        results = []
        for state in active_states:
            results.extend(self.get_for(asset_type=asset_type, state=state))
        
        # Filter out flagged strategies
        results = [r for r in results if not r.flags]
        
        return results
    
    def get_strategy(self, strategy_id: str) -> Optional[StrategyRecord]:
        """Get a specific strategy"""
        return self.strategies.get(strategy_id)
    
    def _is_valid_transition(self, from_state: StrategyState, 
                           to_state: StrategyState) -> bool:
        """Check if state transition is valid"""
        
        valid_transitions = {
            StrategyState.DRAFT: [StrategyState.EXPLORING],
            StrategyState.EXPLORING: [StrategyState.VALIDATING, StrategyState.SUSPENDED, StrategyState.HOLD],
            StrategyState.VALIDATING: [StrategyState.APPROVED_LIVE, StrategyState.REJECTED],
            StrategyState.APPROVED_LIVE: [StrategyState.SUSPENDED, StrategyState.HOLD, StrategyState.DEPRECATED],
            StrategyState.SUSPENDED: [StrategyState.EXPLORING, StrategyState.APPROVED_LIVE, StrategyState.HOLD],
            StrategyState.HOLD: [StrategyState.EXPLORING, StrategyState.APPROVED_LIVE, StrategyState.DEPRECATED],
            StrategyState.REJECTED: [StrategyState.DEPRECATED],
            StrategyState.DEPRECATED: []  # Terminal state
        }
        
        return to_state in valid_transitions.get(from_state, [])
    
    def _hash_parameters(self, parameters: Dict[str, Any]) -> str:
        """Generate hash for parameters"""
        import hashlib
        
        param_str = json.dumps(parameters, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:16]
    
    def _save_strategy(self, record: StrategyRecord):
        """Save strategy record to database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO strategies (
                    strategy_id, asset_type, version, params_hash, state,
                    flags, counters, metrics_last_24h, reasons, timestamps,
                    created_at, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.strategy_id,
                record.asset_type,
                record.version,
                record.params_hash,
                record.state.value,
                json.dumps([f.value for f in record.flags]),
                json.dumps(record.counters),
                json.dumps(record.metrics_last_24h),
                json.dumps(record.reasons),
                json.dumps(record.timestamps),
                record.created_at,
                record.last_updated
            ))
            
            conn.commit()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary statistics"""
        
        summary = {
            'total_strategies': len(self.strategies),
            'by_asset': {},
            'by_state': {},
            'flagged_count': 0
        }
        
        # Count by asset type
        for asset_type in ['spot', 'futures', 'forex', 'options']:
            count = len(self.get_for(asset_type=asset_type))
            summary['by_asset'][asset_type] = count
        
        # Count by state
        for state in StrategyState:
            count = len(self.get_for(state=state))
            summary['by_state'][state.value] = count
        
        # Count flagged
        summary['flagged_count'] = len([r for r in self.strategies.values() if r.flags])
        
        return summary


# Module initialization
strategy_registry = StrategyRegistry()