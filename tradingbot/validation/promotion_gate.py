# file: tradingbot/validation/promotion_gate.py
# module_version: v1.00

"""
Promotion Gate - Final criteria for strategy promotion to live trading.
This is the ONLY module that makes final promotion decisions.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

from tradingbot.core.configmanager import config_manager
from tradingbot.core.loggerconfig import get_logger
from tradingbot.core.strategy_registry import strategy_registry, StrategyState


@dataclass
class PromotionCriteria:
    """Promotion gate criteria"""
    min_profit_factor: float
    min_sharpe_ratio: float
    max_drawdown_pct: float
    min_cvar_pct: float
    min_trades: int
    max_reconciliation_error_pct: float
    required_flags_clear: List[str]


class PromotionGate:
    """
    Final promotion gate for strategy approval to live trading.
    Applies strict criteria before allowing live capital allocation.
    """
    
    def __init__(self):
        self.log = get_logger("promotion_gate")
        self.config = config_manager
        
        # Load promotion criteria per asset
        self._load_promotion_criteria()
        
        # Promotion history
        self.promotion_history: List[Dict[str, Any]] = []
        
        self.log.info("Promotion Gate initialized")
    
    def _load_promotion_criteria(self):
        """Load promotion criteria from config"""
        
        validation_config = self.config.config.get('validation', {})
        promotion_config = validation_config.get('promotion_gates', {})
        
        # Default criteria (can be overridden per asset)
        default_criteria = PromotionCriteria(
            min_profit_factor=1.5,
            min_sharpe_ratio=2.0,
            max_drawdown_pct=15.0,
            min_cvar_pct=-5.0,  # CVaR(5%) >= -5%
            min_trades=100,
            max_reconciliation_error_pct=0.1,  # 0.1% max reconciliation error
            required_flags_clear=['risk_alert', 'data_issue', 'broker_desync']
        )
        
        # Per-asset criteria
        self.criteria = {
            'spot': PromotionCriteria(
                min_profit_factor=promotion_config.get('spot', {}).get('min_profit_factor', 1.5),
                min_sharpe_ratio=promotion_config.get('spot', {}).get('min_sharpe_ratio', 2.0),
                max_drawdown_pct=promotion_config.get('spot', {}).get('max_drawdown_pct', 15.0),
                min_cvar_pct=promotion_config.get('spot', {}).get('min_cvar_pct', -5.0),
                min_trades=promotion_config.get('spot', {}).get('min_trades', 100),
                max_reconciliation_error_pct=promotion_config.get('spot', {}).get('max_reconciliation_error_pct', 0.1),
                required_flags_clear=promotion_config.get('spot', {}).get('required_flags_clear', default_criteria.required_flags_clear)
            ),
            'futures': PromotionCriteria(
                min_profit_factor=promotion_config.get('futures', {}).get('min_profit_factor', 1.8),
                min_sharpe_ratio=promotion_config.get('futures', {}).get('min_sharpe_ratio', 2.5),
                max_drawdown_pct=promotion_config.get('futures', {}).get('max_drawdown_pct', 12.0),
                min_cvar_pct=promotion_config.get('futures', {}).get('min_cvar_pct', -4.0),
                min_trades=promotion_config.get('futures', {}).get('min_trades', 100),
                max_reconciliation_error_pct=promotion_config.get('futures', {}).get('max_reconciliation_error_pct', 0.05),
                required_flags_clear=promotion_config.get('futures', {}).get('required_flags_clear', default_criteria.required_flags_clear)
            ),
            'forex': PromotionCriteria(
                min_profit_factor=promotion_config.get('forex', {}).get('min_profit_factor', 1.6),
                min_sharpe_ratio=promotion_config.get('forex', {}).get('min_sharpe_ratio', 2.2),
                max_drawdown_pct=promotion_config.get('forex', {}).get('max_drawdown_pct', 10.0),
                min_cvar_pct=promotion_config.get('forex', {}).get('min_cvar_pct', -3.0),
                min_trades=promotion_config.get('forex', {}).get('min_trades', 100),
                max_reconciliation_error_pct=promotion_config.get('forex', {}).get('max_reconciliation_error_pct', 0.05),
                required_flags_clear=promotion_config.get('forex', {}).get('required_flags_clear', default_criteria.required_flags_clear)
            ),
            'options': PromotionCriteria(
                min_profit_factor=promotion_config.get('options', {}).get('min_profit_factor', 2.0),
                min_sharpe_ratio=promotion_config.get('options', {}).get('min_sharpe_ratio', 3.0),
                max_drawdown_pct=promotion_config.get('options', {}).get('max_drawdown_pct', 8.0),
                min_cvar_pct=promotion_config.get('options', {}).get('min_cvar_pct', -2.0),
                min_trades=promotion_config.get('options', {}).get('min_trades', 100),
                max_reconciliation_error_pct=promotion_config.get('options', {}).get('max_reconciliation_error_pct', 0.01),
                required_flags_clear=promotion_config.get('options', {}).get('required_flags_clear', default_criteria.required_flags_clear)
            )
        }
    
    async def evaluate(self, strategy_id: str, validation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate strategy for promotion to live trading.
        
        Args:
            strategy_id: Strategy to evaluate
            validation_metrics: Complete validation metrics from all validation stages
            
        Returns:
            {
                'approved': bool,
                'reasons': List[str],
                'criteria_results': Dict[str, Any],
                'recommendation': str
            }
        """
        
        strategy = strategy_registry.get_strategy(strategy_id)
        if not strategy:
            return {
                'approved': False,
                'reasons': ['Strategy not found'],
                'criteria_results': {},
                'recommendation': 'Strategy not found in registry'
            }
        
        if strategy.state != StrategyState.VALIDATING:
            return {
                'approved': False,
                'reasons': [f'Strategy not in validating state: {strategy.state.value}'],
                'criteria_results': {},
                'recommendation': 'Strategy must be in validating state'
            }
        
        # Get criteria for this asset type
        criteria = self.criteria.get(strategy.asset_type)
        if not criteria:
            return {
                'approved': False,
                'reasons': [f'No criteria defined for asset type: {strategy.asset_type}'],
                'criteria_results': {},
                'recommendation': 'Asset type not supported'
            }
        
        # Evaluate each criterion
        results = await self._evaluate_criteria(strategy, criteria, validation_metrics)
        
        # Make promotion decision
        approved = all(result['pass'] for result in results.values())
        failed_criteria = [name for name, result in results.items() if not result['pass']]
        
        # Generate recommendation
        if approved:
            recommendation = f"APPROVE: All criteria met for {strategy.asset_type} strategy"
        else:
            recommendation = f"REJECT: Failed {len(failed_criteria)} criteria: {', '.join(failed_criteria)}"
        
        evaluation_result = {
            'approved': approved,
            'reasons': failed_criteria,
            'criteria_results': results,
            'recommendation': recommendation
        }
        
        # Record promotion decision
        self._record_promotion_decision(strategy_id, evaluation_result)
        
        # Execute promotion or rejection
        if approved:
            await self._promote_strategy(strategy_id)
            self.log.info(f"Strategy {strategy_id} APPROVED for live trading")
        else:
            await self._reject_strategy(strategy_id, failed_criteria)
            self.log.warning(f"Strategy {strategy_id} REJECTED: {', '.join(failed_criteria)}")
        
        return evaluation_result
    
    async def _evaluate_criteria(self, strategy, criteria: PromotionCriteria, 
                               validation_metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Evaluate all promotion criteria"""
        
        results = {}
        
        # 1. Profit Factor
        pf = validation_metrics.get('profit_factor', strategy.metrics_last_24h.get('pf', 0))
        results['profit_factor'] = {
            'pass': pf >= criteria.min_profit_factor,
            'value': pf,
            'threshold': criteria.min_profit_factor,
            'description': f"Profit Factor: {pf:.2f} >= {criteria.min_profit_factor:.2f}"
        }
        
        # 2. Sharpe Ratio
        sharpe = validation_metrics.get('sharpe_ratio', strategy.metrics_last_24h.get('sharpe', 0))
        results['sharpe_ratio'] = {
            'pass': sharpe >= criteria.min_sharpe_ratio,
            'value': sharpe,
            'threshold': criteria.min_sharpe_ratio,
            'description': f"Sharpe Ratio: {sharpe:.2f} >= {criteria.min_sharpe_ratio:.2f}"
        }
        
        # 3. Maximum Drawdown
        max_dd = validation_metrics.get('max_drawdown_pct', strategy.metrics_last_24h.get('max_dd', 100))
        results['max_drawdown'] = {
            'pass': max_dd <= criteria.max_drawdown_pct,
            'value': max_dd,
            'threshold': criteria.max_drawdown_pct,
            'description': f"Max Drawdown: {max_dd:.1f}% <= {criteria.max_drawdown_pct:.1f}%"
        }
        
        # 4. CVaR (5%)
        cvar_5 = validation_metrics.get('cvar_5_pct', strategy.metrics_last_24h.get('cvar_5', -100))
        results['cvar_5'] = {
            'pass': cvar_5 >= criteria.min_cvar_pct,
            'value': cvar_5,
            'threshold': criteria.min_cvar_pct,
            'description': f"CVaR(5%): {cvar_5:.1f}% >= {criteria.min_cvar_pct:.1f}%"
        }
        
        # 5. Minimum Trades
        trades_count = strategy.counters.get('paper_trades_closed', 0)
        results['min_trades'] = {
            'pass': trades_count >= criteria.min_trades,
            'value': trades_count,
            'threshold': criteria.min_trades,
            'description': f"Trade Count: {trades_count} >= {criteria.min_trades}"
        }
        
        # 6. Reconciliation Error
        reconciliation_error = validation_metrics.get('reconciliation_error_pct', 0)
        results['reconciliation_error'] = {
            'pass': reconciliation_error <= criteria.max_reconciliation_error_pct,
            'value': reconciliation_error,
            'threshold': criteria.max_reconciliation_error_pct,
            'description': f"Reconciliation Error: {reconciliation_error:.3f}% <= {criteria.max_reconciliation_error_pct:.3f}%"
        }
        
        # 7. Required Flags Clear
        active_flags = [flag.value for flag in strategy.flags]
        blocked_flags = [flag for flag in criteria.required_flags_clear if flag in active_flags]
        results['flags_clear'] = {
            'pass': len(blocked_flags) == 0,
            'value': active_flags,
            'threshold': criteria.required_flags_clear,
            'description': f"Blocked Flags: {blocked_flags if blocked_flags else 'None'}"
        }
        
        return results
    
    def decide(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Public interface for promotion decision.
        
        Args:
            metrics: Validation metrics dictionary
            
        Returns:
            {'pass': bool, 'reasons': List[str]}
        """
        
        # This is a simplified version for external calls
        # Full evaluation requires strategy context
        
        failures = []
        
        # Basic threshold checks
        if metrics.get('profit_factor', 0) < 1.5:
            failures.append('Profit Factor below 1.5')
        
        if metrics.get('sharpe_ratio', 0) < 2.0:
            failures.append('Sharpe Ratio below 2.0')
        
        if metrics.get('max_drawdown_pct', 100) > 15.0:
            failures.append('Max Drawdown exceeds 15%')
        
        return {
            'pass': len(failures) == 0,
            'reasons': failures
        }
    
    async def _promote_strategy(self, strategy_id: str):
        """Promote strategy to approved_live state"""
        
        success = strategy_registry.set_state(
            strategy_id,
            StrategyState.APPROVED_LIVE,
            "Passed all promotion gate criteria"
        )
        
        if success:
            # Update timestamps
            strategy = strategy_registry.get_strategy(strategy_id)
            if strategy:
                strategy.timestamps['approved'] = datetime.now().isoformat()
    
    async def _reject_strategy(self, strategy_id: str, reasons: List[str]):
        """Reject strategy"""
        
        reason_str = f"Failed promotion gate: {'; '.join(reasons)}"
        
        success = strategy_registry.set_state(
            strategy_id,
            StrategyState.REJECTED,
            reason_str
        )
        
        if success:
            # Add to reasons history
            strategy = strategy_registry.get_strategy(strategy_id)
            if strategy:
                strategy.reasons.append(f"{datetime.now().isoformat()}: REJECTED - {reason_str}")
    
    def _record_promotion_decision(self, strategy_id: str, evaluation_result: Dict[str, Any]):
        """Record promotion decision in history"""
        
        record = {
            'strategy_id': strategy_id,
            'timestamp': datetime.now().isoformat(),
            'approved': evaluation_result['approved'],
            'criteria_results': evaluation_result['criteria_results'],
            'recommendation': evaluation_result['recommendation']
        }
        
        self.promotion_history.append(record)
        
        # Keep history limited
        if len(self.promotion_history) > 1000:
            self.promotion_history = self.promotion_history[-500:]
    
    def get_promotion_stats(self) -> Dict[str, Any]:
        """Get promotion gate statistics"""
        
        total_evaluations = len(self.promotion_history)
        
        if total_evaluations == 0:
            return {
                'total_evaluations': 0,
                'approval_rate': 0,
                'common_failures': []
            }
        
        approved_count = len([r for r in self.promotion_history if r['approved']])
        
        # Find common failure reasons
        all_failures = []
        for record in self.promotion_history:
            if not record['approved']:
                for criterion, result in record['criteria_results'].items():
                    if not result['pass']:
                        all_failures.append(criterion)
        
        failure_counts = {}
        for failure in all_failures:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
        
        common_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_evaluations': total_evaluations,
            'approved_count': approved_count,
            'rejected_count': total_evaluations - approved_count,
            'approval_rate': approved_count / total_evaluations,
            'common_failures': common_failures
        }


# Module initialization
promotion_gate = PromotionGate()