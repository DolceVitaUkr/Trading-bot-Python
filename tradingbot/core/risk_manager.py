# file: tradingbot/core/risk_manager.py
# module_version: v1.00

"""
Risk Manager - Per-trade risk checks and position sizing.
This is the ONLY module that performs risk calculations.
All risk checks MUST go through this module.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .configmanager import config_manager
from .loggerconfig import get_logger


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RiskMetrics:
    """Current risk metrics"""
    daily_loss: float = 0.0
    daily_trades: int = 0
    open_risk: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    violations: List[str] = None


class RiskManager:
    """
    Centralized risk management for all trading operations.
    Enforces position sizing, stop-loss validation, and risk limits.
    """
    
    def __init__(self):
        self.log = get_logger("risk_manager")
        self.config = config_manager
        
        # Risk metrics per asset
        self.metrics: Dict[str, RiskMetrics] = {
            'spot': RiskMetrics(),
            'futures': RiskMetrics(),
            'forex': RiskMetrics(),
            'options': RiskMetrics()
        }
        
        # Daily tracking (resets at UTC midnight)
        self.daily_reset_time = None
        self._reset_daily_metrics()
        
        # Risk parameters from config
        self._load_risk_parameters()
        
        # Dependencies (will be injected)
        self.exposure_manager = None
        
        self.log.info("Risk Manager initialized - All risk checks route through here")
    
    def _load_risk_parameters(self):
        """Load risk parameters from config"""
        risk_config = self.config.config.get('risk', {})
        
        # Default risk parameters per asset type
        self.risk_params = {
            'spot': {
                'risk_fraction_min': risk_config.get('spot', {}).get('risk_fraction_min', 0.0025),
                'risk_fraction_max': risk_config.get('spot', {}).get('risk_fraction_max', 0.0075),
                'daily_loss_cap_pct': risk_config.get('spot', {}).get('daily_loss_cap_pct', 0.03),
                'max_concurrent': risk_config.get('spot', {}).get('max_concurrent', 6),
                'max_sl_distance_pct': 0.15  # 15% max SL distance
            },
            'futures': {
                'risk_fraction_min': risk_config.get('futures', {}).get('risk_fraction_min', 0.0015),
                'risk_fraction_max': risk_config.get('futures', {}).get('risk_fraction_max', 0.0050),
                'daily_loss_cap_pct': risk_config.get('futures', {}).get('daily_loss_cap_pct', 0.02),
                'max_concurrent': risk_config.get('futures', {}).get('max_concurrent', 3),
                'leverage_cap': risk_config.get('futures', {}).get('leverage_cap', 5),
                'max_sl_distance_pct': 0.10  # 10% max SL distance for futures
            },
            'forex': {
                'risk_fraction_min': risk_config.get('forex', {}).get('risk_fraction_min', 0.0010),
                'risk_fraction_max': risk_config.get('forex', {}).get('risk_fraction_max', 0.0040),
                'daily_loss_cap_pct': risk_config.get('forex', {}).get('daily_loss_cap_pct', 0.02),
                'max_concurrent': risk_config.get('forex', {}).get('max_concurrent', 4),
                'max_sl_distance_pips': 100  # Max 100 pips SL
            },
            'options': {
                'max_premium_per_trade_usd': risk_config.get('options', {}).get('max_premium_per_trade_usd', 50),
                'theta_cap_usd_per_day': risk_config.get('options', {}).get('theta_cap_usd_per_day', 20),
                'daily_loss_cap_pct': risk_config.get('options', {}).get('daily_loss_cap_pct', 0.02),
                'max_concurrent': risk_config.get('options', {}).get('max_concurrent', 3)
            }
        }
    
    def set_dependencies(self, exposure_manager=None):
        """Inject dependencies"""
        if exposure_manager:
            self.exposure_manager = exposure_manager
    
    async def pretrade_check(self, order_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive pretrade risk checks.
        
        Args:
            order_context: {
                'asset_type': str,
                'symbol': str,
                'side': str,
                'quantity': float,
                'price': float,
                'sl_price': float,
                'tp_price': float,
                'signal_weight': float,
                'strategy_id': str
            }
        
        Returns:
            {
                'pass': bool,
                'reason': str,
                'adjusted_qty': float,  # Risk-adjusted quantity
                'risk_metrics': dict
            }
        """
        
        asset_type = order_context.get('asset_type')
        
        # Reset daily metrics if needed
        self._check_daily_reset()
        
        violations = []
        
        # 1. Check daily loss cap
        daily_loss_check = self._check_daily_loss_cap(asset_type)
        if not daily_loss_check['pass']:
            violations.append(daily_loss_check['reason'])
        
        # 2. Validate stop-loss distance
        sl_check = self._validate_sl_distance(order_context)
        if not sl_check['pass']:
            violations.append(sl_check['reason'])
        
        # 3. Check leverage (futures only)
        if asset_type == 'futures':
            leverage_check = self._check_leverage(order_context)
            if not leverage_check['pass']:
                violations.append(leverage_check['reason'])
        
        # 4. Check options premium (options only)
        if asset_type == 'options':
            premium_check = self._check_options_premium(order_context)
            if not premium_check['pass']:
                violations.append(premium_check['reason'])
        
        # 5. Calculate risk-adjusted position size
        adjusted_qty = await self.compute_size(order_context)
        
        # 6. Check if adjusted size meets minimum requirements
        if adjusted_qty < order_context.get('min_trade_size', 0):
            violations.append(f"Adjusted size {adjusted_qty} below minimum")
        
        # Compile result
        if violations:
            self.log.warning(f"Pretrade check failed: {violations}")
            return {
                'pass': False,
                'reason': '; '.join(violations),
                'adjusted_qty': 0,
                'risk_metrics': self._get_current_metrics(asset_type)
            }
        
        # Update metrics
        self.metrics[asset_type].daily_trades += 1
        
        self.log.info(f"Pretrade check passed for {order_context.get('symbol')}, "
                     f"adjusted qty: {adjusted_qty}")
        
        return {
            'pass': True,
            'reason': 'All risk checks passed',
            'adjusted_qty': adjusted_qty,
            'risk_metrics': self._get_current_metrics(asset_type)
        }
    
    async def compute_size(self, order_context: Dict[str, Any]) -> float:
        """
        Compute risk-adjusted position size.
        
        Formula:
        position_size = (equity * risk_fraction * signal_weight) / (sl_distance * price)
        
        Where:
        - risk_fraction is tiered by drawdown
        - signal_weight adjusts for signal strength
        - sl_distance is the stop-loss distance
        """
        
        asset_type = order_context.get('asset_type')
        params = self.risk_params[asset_type]
        
        # Get available equity (from budget_manager in real implementation)
        equity = order_context.get('available_equity', 1000)
        
        # Get risk fraction based on current drawdown
        risk_fraction = self._get_risk_fraction(asset_type)
        
        # Signal weight (weak=0.5, base=1.0, strong=1.5)
        signal_weight = order_context.get('signal_weight', 1.0)
        
        # Calculate SL distance
        entry_price = order_context.get('price', 0)
        sl_price = order_context.get('sl_price', 0)
        
        if entry_price == 0 or sl_price == 0:
            self.log.error("Invalid prices for size calculation")
            return 0
        
        sl_distance = abs(entry_price - sl_price) / entry_price
        
        if sl_distance == 0:
            self.log.error("Zero SL distance")
            return 0
        
        # Core sizing formula
        position_size = (equity * risk_fraction * signal_weight) / (sl_distance * entry_price)
        
        # Apply asset-specific adjustments
        if asset_type == 'futures':
            # Adjust for leverage
            leverage = min(order_context.get('leverage', 1), params['leverage_cap'])
            position_size = position_size / leverage
        
        elif asset_type == 'forex':
            # Convert to lots
            lot_size = 100000  # Standard lot
            position_size = position_size / lot_size
        
        elif asset_type == 'options':
            # Options sizing by premium
            max_premium = params['max_premium_per_trade_usd']
            option_price = order_context.get('option_price', 1)
            max_contracts = max_premium / (option_price * 100)  # 100 shares per contract
            position_size = min(position_size, max_contracts)
        
        # Round to appropriate precision
        position_size = round(position_size, 8)
        
        self.log.debug(f"Computed size: {position_size} for {order_context.get('symbol')}")
        
        return position_size
    
    def _check_daily_loss_cap(self, asset_type: str) -> Dict[str, Any]:
        """Check if daily loss cap is breached"""
        params = self.risk_params[asset_type]
        metrics = self.metrics[asset_type]
        
        daily_loss_cap = params['daily_loss_cap_pct']
        
        # For now, use a placeholder equity value
        # In production, this would come from budget_manager
        equity = 1000
        
        max_daily_loss = equity * daily_loss_cap
        
        if abs(metrics.daily_loss) >= max_daily_loss:
            return {
                'pass': False,
                'reason': f"Daily loss cap reached: ${abs(metrics.daily_loss):.2f} >= ${max_daily_loss:.2f}"
            }
        
        return {'pass': True}
    
    def _validate_sl_distance(self, order_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate stop-loss distance is within acceptable range"""
        asset_type = order_context.get('asset_type')
        params = self.risk_params[asset_type]
        
        entry_price = order_context.get('price', 0)
        sl_price = order_context.get('sl_price', 0)
        
        if sl_price == 0:
            return {'pass': False, 'reason': 'No stop-loss provided'}
        
        sl_distance_pct = abs(entry_price - sl_price) / entry_price * 100
        
        # Check maximum SL distance
        if asset_type in ['spot', 'futures']:
            max_sl = params['max_sl_distance_pct'] * 100
            if sl_distance_pct > max_sl:
                return {
                    'pass': False,
                    'reason': f"SL distance {sl_distance_pct:.1f}% exceeds max {max_sl:.1f}%"
                }
        
        elif asset_type == 'forex':
            # Convert to pips
            pip_size = 0.0001 if 'JPY' not in order_context.get('symbol', '') else 0.01
            sl_distance_pips = abs(entry_price - sl_price) / pip_size
            
            if sl_distance_pips > params.get('max_sl_distance_pips', 100):
                return {
                    'pass': False,
                    'reason': f"SL distance {sl_distance_pips:.0f} pips exceeds max"
                }
        
        return {'pass': True}
    
    def _check_leverage(self, order_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check leverage limits for futures"""
        params = self.risk_params['futures']
        leverage = order_context.get('leverage', 1)
        
        if leverage > params['leverage_cap']:
            return {
                'pass': False,
                'reason': f"Leverage {leverage}x exceeds cap {params['leverage_cap']}x"
            }
        
        return {'pass': True}
    
    def _check_options_premium(self, order_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check options premium limits"""
        params = self.risk_params['options']
        
        option_price = order_context.get('option_price', 0)
        quantity = order_context.get('quantity', 0)
        total_premium = option_price * quantity * 100  # 100 shares per contract
        
        if total_premium > params['max_premium_per_trade_usd']:
            return {
                'pass': False,
                'reason': f"Premium ${total_premium:.2f} exceeds max ${params['max_premium_per_trade_usd']:.2f}"
            }
        
        return {'pass': True}
    
    def _get_risk_fraction(self, asset_type: str) -> float:
        """
        Get risk fraction based on current drawdown.
        Implements drawdown-based throttling.
        """
        params = self.risk_params[asset_type]
        metrics = self.metrics[asset_type]
        
        min_risk = params['risk_fraction_min']
        max_risk = params['risk_fraction_max']
        
        # Drawdown-based throttling
        drawdown_pct = metrics.current_drawdown
        
        if drawdown_pct < 5:
            # Low drawdown: use max risk
            risk_fraction = max_risk
        elif drawdown_pct < 10:
            # Medium drawdown: scale down
            scale = 1 - (drawdown_pct - 5) / 5 * 0.5
            risk_fraction = min_risk + (max_risk - min_risk) * scale
        else:
            # High drawdown: use min risk
            risk_fraction = min_risk
        
        return risk_fraction
    
    def _check_daily_reset(self):
        """Reset daily metrics at UTC midnight"""
        now = datetime.utcnow()
        
        if self.daily_reset_time is None or now.date() > self.daily_reset_time.date():
            self._reset_daily_metrics()
            self.daily_reset_time = now
    
    def _reset_daily_metrics(self):
        """Reset daily tracking metrics"""
        for metrics in self.metrics.values():
            metrics.daily_loss = 0
            metrics.daily_trades = 0
        
        self.log.info("Daily risk metrics reset")
    
    def _get_current_metrics(self, asset_type: str) -> Dict[str, Any]:
        """Get current risk metrics for reporting"""
        metrics = self.metrics[asset_type]
        params = self.risk_params[asset_type]
        
        return {
            'daily_loss': metrics.daily_loss,
            'daily_trades': metrics.daily_trades,
            'open_risk': metrics.open_risk,
            'current_drawdown': metrics.current_drawdown,
            'risk_level': metrics.risk_level.value,
            'risk_fraction': self._get_risk_fraction(asset_type),
            'daily_loss_cap': params['daily_loss_cap_pct']
        }
    
    async def update_metrics(self, asset_type: str, pnl: float):
        """Update risk metrics after trade completion"""
        metrics = self.metrics[asset_type]
        
        # Update daily loss
        if pnl < 0:
            metrics.daily_loss += abs(pnl)
        
        # Update drawdown
        # This would typically track equity curve in production
        if pnl < 0:
            metrics.current_drawdown = min(metrics.current_drawdown + abs(pnl) / 1000 * 100, 100)
        else:
            # Recover slowly
            metrics.current_drawdown = max(metrics.current_drawdown - pnl / 1000 * 100 * 0.5, 0)
        
        # Update risk level
        if metrics.current_drawdown < 5:
            metrics.risk_level = RiskLevel.LOW
        elif metrics.current_drawdown < 10:
            metrics.risk_level = RiskLevel.MEDIUM
        elif metrics.current_drawdown < 20:
            metrics.risk_level = RiskLevel.HIGH
        else:
            metrics.risk_level = RiskLevel.EXTREME
        
        self.log.info(f"Risk metrics updated for {asset_type}: DD={metrics.current_drawdown:.1f}%, "
                     f"Daily Loss=${metrics.daily_loss:.2f}")


# Module initialization
risk_manager = RiskManager()