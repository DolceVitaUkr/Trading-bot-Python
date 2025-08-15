# modules/reward_system.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class RewardParams:
    """
    Parameters for the reward system.
    """
    base_per_trade: float = 0.0
    profit_multiplier: float = 1.0
    risk_adjusted_boost: float = 1.15
    sl_penalty: float = 0.5        # if stop loss triggered
    time_bonus_per_hour: float = 0.0


class RewardSystem:
    """
    Simple additive reward scoring used by SelfLearningBot.
    """
    def __init__(self, params: Optional[RewardParams] = None):
        """
        Initializes the RewardSystem.
        """
        self.params = params or RewardParams()
        self.total_points: float = 0.0

    def calculate_reward(
        self,
        entry_price: float,
        exit_price: float,
        position_size: float,
        entry_time: datetime,
        exit_time: datetime,
        max_drawdown: float,
        volatility: float,
        stop_loss_triggered: bool = False,
    ) -> float:
        """
        Monetary reward proxy: net PnL in quote currency.
        """
        pnl = (exit_price - entry_price) * position_size
        return float(pnl)

    def add_points(
        self,
        profit_pct: float,
        entry_time: datetime,
        exit_time: datetime,
        stop_loss_triggered: bool = False,
        risk_adjusted: bool = True
    ) -> float:
        """
        Calculates and adds points to the total score.
        """
        pts = calculate_points(
            profit=profit_pct,
            entry_time=entry_time,
            exit_time=exit_time,
            stop_loss_triggered=stop_loss_triggered,
            risk_adjusted=risk_adjusted
        )
        self.total_points += pts
        return pts


def calculate_points(
    profit: float,
    entry_time: datetime,
    exit_time: datetime,
    stop_loss_triggered: bool,
    risk_adjusted: bool = True
) -> float:
    """
    Points are loosely tied to profitability; can be risk-adjusted if requested.
    - profit: percentage (or ROI%) in simulation usage
    """
    base = profit
    if risk_adjusted:
        base *= 0.9  # mild haircut to discourage over-leverage

    if stop_loss_triggered:
        base *= 0.5  # penalty

    # very small time factor
    hold_hours = max(0.0, (exit_time - entry_time).total_seconds() / 3600.0)
    base += min(hold_hours * 0.01, 0.1)  # cap tiny bonus

    return float(base)
