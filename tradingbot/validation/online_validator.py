# file: tradingbot/validation/online_validator.py
# module_version: v1.00

"""
Online Validator - Phase-1 validation with live feed paper trading.
Each strategy must complete ≥100 closed trades on live feed before promotion.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

from tradingbot.core.configmanager import config_manager
from tradingbot.core.loggerconfig import get_logger
from tradingbot.core.strategy_registry import strategy_registry, StrategyState


class ValidationStatus(Enum):
    """Online validation status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class OnlineValidationResult:
    """Result of online validation"""
    strategy_id: str
    status: ValidationStatus
    trades_completed: int
    trades_required: int
    start_time: datetime
    end_time: Optional[datetime]
    metrics: Dict[str, float]
    pass_criteria: bool
    failure_reason: Optional[str]


class OnlineValidator:
    """
    Phase-1 online validation manager.
    Monitors strategies accumulating ≥100 live-feed paper trades.
    """
    
    def __init__(self):
        self.log = get_logger("online_validator")
        self.config = config_manager
        
        # Active validations
        self.active_validations: Dict[str, OnlineValidationResult] = {}
        
        # Validation history
        self.validation_history: List[OnlineValidationResult] = []
        
        # Configuration
        self._load_validation_config()
        
        self.log.info("Online Validator initialized")
    
    def _load_validation_config(self):
        """Load validation configuration"""
        
        validation_config = self.config.config.get('validation', {})
        
        self.min_trades_per_asset = validation_config.get('min_trades_per_asset', {
            'spot': 100,
            'futures': 100,
            'forex': 100,
            'options': 100
        })
        
        # Minimum performance thresholds for online validation
        self.min_performance = validation_config.get('online_thresholds', {
            'min_profit_factor': 1.1,
            'max_drawdown_pct': 25.0,
            'min_win_rate': 0.35,
            'min_avg_trade_usd': -50  # Max average loss per trade
        })
        
        # Timeout for validation (if no trades for this long, fail)
        self.validation_timeout_hours = validation_config.get('validation_timeout_hours', 168)  # 1 week
    
    def start(self, strategy_id: str) -> bool:
        """
        Start online validation for a strategy.
        
        Args:
            strategy_id: Strategy to validate
            
        Returns:
            True if validation started, False if already running or invalid
        """
        
        if strategy_id in self.active_validations:
            self.log.warning(f"Validation already running for {strategy_id}")
            return False
        
        strategy = strategy_registry.get_strategy(strategy_id)
        if not strategy:
            self.log.error(f"Strategy {strategy_id} not found")
            return False
        
        if strategy.state != StrategyState.VALIDATING:
            self.log.error(f"Strategy {strategy_id} not in validating state")
            return False
        
        # Get required trades for this asset type
        required_trades = self.min_trades_per_asset.get(strategy.asset_type, 100)
        current_trades = strategy.counters.get('paper_trades_closed', 0)
        
        # Create validation record
        validation = OnlineValidationResult(
            strategy_id=strategy_id,
            status=ValidationStatus.RUNNING,
            trades_completed=current_trades,
            trades_required=required_trades,
            start_time=datetime.now(),
            end_time=None,
            metrics={},
            pass_criteria=False,
            failure_reason=None
        )
        
        self.active_validations[strategy_id] = validation
        
        self.log.info(f"Started online validation for {strategy_id}: "
                     f"{current_trades}/{required_trades} trades completed")
        
        return True
    
    def update_progress(self, strategy_id: str, trade_result: Dict[str, Any]):
        """
        Update validation progress with a completed trade.
        
        Args:
            strategy_id: Strategy that completed a trade
            trade_result: Trade outcome data
        """
        
        if strategy_id not in self.active_validations:
            return
        
        validation = self.active_validations[strategy_id]
        
        if validation.status != ValidationStatus.RUNNING:
            return
        
        # Update trade count
        validation.trades_completed += 1
        
        # Check if validation complete
        if validation.trades_completed >= validation.trades_required:
            asyncio.create_task(self._complete_validation(strategy_id))
        
        self.log.debug(f"Validation progress for {strategy_id}: "
                      f"{validation.trades_completed}/{validation.trades_required}")
    
    async def _complete_validation(self, strategy_id: str):
        """Complete online validation and compute metrics"""
        
        validation = self.active_validations[strategy_id]
        strategy = strategy_registry.get_strategy(strategy_id)
        
        if not strategy:
            validation.status = ValidationStatus.FAILED
            validation.failure_reason = "Strategy not found"
            return
        
        # Compute performance metrics from strategy counters/metrics
        metrics = await self._compute_validation_metrics(strategy_id)
        validation.metrics = metrics
        
        # Check pass criteria
        pass_checks = self._check_pass_criteria(metrics)
        validation.pass_criteria = pass_checks['pass']
        
        if not pass_checks['pass']:
            validation.failure_reason = pass_checks['reason']
            validation.status = ValidationStatus.FAILED
            
            # Transition strategy to rejected
            strategy_registry.set_state(
                strategy_id,
                StrategyState.REJECTED,
                f"Failed online validation: {validation.failure_reason}"
            )
            
            self.log.warning(f"Online validation FAILED for {strategy_id}: {validation.failure_reason}")
            
        else:
            validation.status = ValidationStatus.COMPLETED
            
            self.log.info(f"Online validation PASSED for {strategy_id}")
            
            # Strategy remains in VALIDATING state for offline validation
            # The promotion_gate will handle final promotion
        
        validation.end_time = datetime.now()
        
        # Move to history
        self.validation_history.append(validation)
        del self.active_validations[strategy_id]
    
    async def _compute_validation_metrics(self, strategy_id: str) -> Dict[str, float]:
        """Compute validation metrics for a strategy"""
        
        strategy = strategy_registry.get_strategy(strategy_id)
        if not strategy:
            return {}
        
        # Use 24h metrics from strategy registry as baseline
        metrics = strategy.metrics_last_24h.copy()
        
        # Add additional computed metrics
        trades_closed = strategy.counters.get('paper_trades_closed', 0)
        
        if trades_closed > 0:
            # These would be computed from actual trade history in production
            # For now, use placeholder calculations
            
            metrics.update({
                'trades_completed': trades_closed,
                'validation_duration_hours': (datetime.now() - datetime.fromisoformat(strategy.created_at)).total_seconds() / 3600,
                'avg_trades_per_day': trades_closed / max(1, (datetime.now() - datetime.fromisoformat(strategy.created_at)).days),
                'consistency_score': 0.75  # Placeholder
            })
        
        return metrics
    
    def _check_pass_criteria(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check if metrics meet pass criteria"""
        
        failures = []
        
        # Profit Factor check
        pf = metrics.get('pf', 0)
        if pf < self.min_performance['min_profit_factor']:
            failures.append(f"PF {pf:.2f} < {self.min_performance['min_profit_factor']}")
        
        # Drawdown check  
        max_dd = metrics.get('max_dd', 100)
        if max_dd > self.min_performance['max_drawdown_pct']:
            failures.append(f"MaxDD {max_dd:.1f}% > {self.min_performance['max_drawdown_pct']}%")
        
        # Win rate check
        win_rate = metrics.get('win_rate', 0)
        if win_rate < self.min_performance['min_win_rate']:
            failures.append(f"WinRate {win_rate:.1%} < {self.min_performance['min_win_rate']:.1%}")
        
        # Average trade check
        avg_trade = metrics.get('avg_trade', -1000)
        if avg_trade < self.min_performance['min_avg_trade_usd']:
            failures.append(f"AvgTrade ${avg_trade:.2f} < ${self.min_performance['min_avg_trade_usd']}")
        
        if failures:
            return {
                'pass': False,
                'reason': '; '.join(failures)
            }
        
        return {'pass': True, 'reason': 'All criteria met'}
    
    async def check_timeouts(self):
        """Check for validation timeouts"""
        
        now = datetime.now()
        timeout_threshold = timedelta(hours=self.validation_timeout_hours)
        
        timed_out = []
        
        for strategy_id, validation in self.active_validations.items():
            if now - validation.start_time > timeout_threshold:
                timed_out.append(strategy_id)
        
        # Handle timeouts
        for strategy_id in timed_out:
            validation = self.active_validations[strategy_id]
            validation.status = ValidationStatus.TIMEOUT
            validation.failure_reason = f"No progress for {self.validation_timeout_hours} hours"
            validation.end_time = now
            
            # Transition strategy
            strategy_registry.set_state(
                strategy_id,
                StrategyState.REJECTED,
                f"Online validation timeout: {validation.failure_reason}"
            )
            
            # Move to history
            self.validation_history.append(validation)
            del self.active_validations[strategy_id]
            
            self.log.warning(f"Online validation TIMEOUT for {strategy_id}")
    
    def status(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get validation status for a strategy"""
        
        if strategy_id in self.active_validations:
            validation = self.active_validations[strategy_id]
            
            return {
                'status': validation.status.value,
                'trades_completed': validation.trades_completed,
                'trades_required': validation.trades_required,
                'progress_pct': (validation.trades_completed / validation.trades_required * 100),
                'started': validation.start_time.isoformat(),
                'duration_hours': (datetime.now() - validation.start_time).total_seconds() / 3600
            }
        
        # Check history
        for validation in self.validation_history:
            if validation.strategy_id == strategy_id:
                return {
                    'status': validation.status.value,
                    'trades_completed': validation.trades_completed,
                    'trades_required': validation.trades_required,
                    'progress_pct': 100,
                    'started': validation.start_time.isoformat(),
                    'ended': validation.end_time.isoformat() if validation.end_time else None,
                    'pass_criteria': validation.pass_criteria,
                    'failure_reason': validation.failure_reason,
                    'metrics': validation.metrics
                }
        
        return None
    
    def done(self, strategy_id: str) -> Optional[bool]:
        """
        Check if validation is done and return pass/fail result.
        
        Returns:
            True if passed, False if failed, None if still running
        """
        
        if strategy_id in self.active_validations:
            return None  # Still running
        
        # Check history
        for validation in self.validation_history:
            if validation.strategy_id == strategy_id:
                return validation.pass_criteria
        
        return None  # Not found
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations"""
        
        return {
            'active_count': len(self.active_validations),
            'active_validations': [
                {
                    'strategy_id': v.strategy_id,
                    'progress_pct': v.trades_completed / v.trades_required * 100,
                    'duration_hours': (datetime.now() - v.start_time).total_seconds() / 3600
                }
                for v in self.active_validations.values()
            ],
            'completed_count': len(self.validation_history),
            'pass_rate': len([v for v in self.validation_history if v.pass_criteria]) / len(self.validation_history) if self.validation_history else 0
        }


# Module initialization
online_validator = OnlineValidator()