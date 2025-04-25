# modules/risk_management.py
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
from dataclasses import dataclass
from modules.error_handler import RiskViolationError

logger = logging.getLogger(__name__)

# ================== SIMULATION FUNCTIONS ==================
def calculate_stop_loss(entry_price: float, risk_percentage: float = 0.02) -> float:
    """Basic stop loss calculation for simulations"""
    try:
        return entry_price * (1 - risk_percentage)
    except Exception as e:
        logger.error(f"Stop loss calculation failed: {str(e)}")
        return entry_price * 0.98  # Fallback to 2% stop loss

def dynamic_adjustment(entry_price: float, 
                     current_price: float, 
                     stop_loss: float, 
                     take_profit: float) -> Tuple[float, float]:
    """Basic dynamic adjustment for simulations"""
    try:
        price_change = (current_price - entry_price) / entry_price
        new_stop = max(stop_loss, current_price * 0.99)  # Simple trailing stop
        new_tp = take_profit * (1 + abs(price_change) * 0.5)
        return round(new_stop, 4), round(new_tp, 4)
    except Exception as e:
        logger.error(f"Dynamic adjustment failed: {str(e)}")
        return stop_loss, take_profit

# ================== PRODUCTION CLASSES ==================
@dataclass
class PositionRisk:
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_reward_ratio: float
    risk_percent: float
    dollar_risk: float

class RiskManager:
    def __init__(self, account_balance: float, max_portfolio_risk: float = 0.05):
        self.account_balance = account_balance
        self.max_portfolio_risk = max_portfolio_risk
        self.open_positions = {}
        self.max_leverage = 10
        self.risk_free_rate = 0.02
        self.min_risk_reward = 1.5
        self.atr_period = 14
        self.volatility_factor = 2.0

    def calculate_position_size(self, entry_price: float, stop_price: float, 
                              risk_percent: float = 0.01) -> float:
        self._validate_prices(entry_price, stop_price)
        price_risk = abs(entry_price - stop_price)
        if price_risk <= 0:
            raise RiskViolationError("Invalid price risk calculation")
        dollar_risk = self.account_balance * risk_percent
        position_size = dollar_risk / price_risk
        return self._apply_leverage_limits(position_size, entry_price)

    def dynamic_stop_management(self, current_price: float, position: PositionRisk) -> PositionRisk:
        price_change_pct = (current_price - position.entry_price) / position.entry_price
        if price_change_pct > 0.05:
            new_stop = max(position.stop_loss, position.entry_price)
        elif price_change_pct > 0.02:
            new_stop = position.stop_loss * 1.01
        else:
            new_stop = position.stop_loss
        atr = self._calculate_atr(position.entry_price)
        new_stop = max(new_stop, current_price - self.volatility_factor * atr)
        risk = position.entry_price - new_stop
        new_take_profit = position.entry_price + (risk * position.risk_reward_ratio)
        return PositionRisk(
            entry_price=position.entry_price,
            stop_loss=new_stop,
            take_profit=new_take_profit,
            position_size=position.position_size,
            risk_reward_ratio=position.risk_reward_ratio,
            risk_percent=position.risk_percent,
            dollar_risk=position.dollar_risk
        )

    def portfolio_risk_assessment(self) -> Dict:
        total_risk = sum(pos.dollar_risk for pos in self.open_positions.values())
        margin_used = sum(pos.position_size * pos.entry_price for pos in self.open_positions.values())
        return {
            'total_dollar_risk': total_risk,
            'portfolio_risk_ratio': total_risk / self.account_balance,
            'margin_utilization': margin_used / self.account_balance,
            'leverage_ratio': margin_used / self.account_balance,
            'max_drawdown': self._calculate_max_drawdown()
        }

    def _calculate_atr(self, current_price: float, lookback: int = 14) -> float:
        return current_price * 0.02  # Simplified implementation

    def _calculate_max_drawdown(self) -> float:
        return sum(pos.dollar_risk for pos in self.open_positions.values()) / self.account_balance

    def _validate_prices(self, entry: float, stop: float):
        if entry <= 0 or stop <= 0:
            raise RiskViolationError("Invalid price values")
        if (entry < stop and entry < 0) or (entry > stop and entry > 0):
            raise RiskViolationError("Stop loss must be on correct side of entry")

    def _apply_leverage_limits(self, position_size: float, entry_price: float) -> float:
        margin_required = position_size * entry_price / self.max_leverage
        if margin_required > self.account_balance:
            raise RiskViolationError("Insufficient margin for position size")
        return position_size

if __name__ == "__main__":
    # Test both implementations
    print("Simulation stop loss:", calculate_stop_loss(100.0))
    print("Dynamic adjustment:", dynamic_adjustment(100, 105, 95, 110))
    
    risk_mgr = RiskManager(100000)
    position_size = risk_mgr.calculate_position_size(100.0, 95.0)
    print("Production position size:", position_size)