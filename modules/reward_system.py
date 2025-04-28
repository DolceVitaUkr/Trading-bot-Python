# modules/reward_system.py

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def calculate_points(
    profit: float,
    entry_time: datetime,
    exit_time: datetime,
    stop_loss_triggered: bool,
    risk_adjusted: bool = True
) -> float:
    """
    Basic reward‐point calculation for simulations.

    - base_points: profit * 100  
    - time_bonus: max(0, 50 − 10 * hours_held)  
    - risk_penalty: 100 if stop_loss_triggered else 0  
    - risk_adjusted: if True, divides total_points by (risk_penalty + 1)

    Returns a rounded float.
    """
    try:
        base_points = profit * 100
        duration_h = (exit_time - entry_time).total_seconds() / 3600.0
        time_bonus = max(0, 50 - (duration_h * 10))
        risk_penalty = 100 if stop_loss_triggered else 0
        total = base_points + time_bonus - risk_penalty
        if risk_adjusted:
            return round(total / (risk_penalty + 1), 2)
        else:
            return round(total, 2)
    except Exception as e:
        logger.error(f"calculate_points error: {e}")
        return 0.0


class RewardSystem:
    """
    Multi‐factor trade reward calculator for RL agents.

    Currently, calculate_reward simply returns raw P&L:
      (exit_price - entry_price) * position_size

    TODO (future):
      - Incorporate time‐based bonuses from calculate_points
      - Penalize drawdowns and volatility
      - Apply heavier penalties for stop‐loss triggers
      - Expose toggles or weights for each component
    """

    def calculate_reward(
        self,
        entry_price: float,
        exit_price: float,
        position_size: float,
        entry_time: datetime,
        exit_time: datetime,
        max_drawdown: float = 0.0,
        volatility: float = 0.0,
        stop_loss_triggered: bool = False
    ) -> float:
        """
        Compute the reward for one trade.

        Parameters:
          entry_price (float): the price at which the trade was opened
          exit_price  (float): the price at which the trade was closed
          position_size (float): size of the position
          entry_time (datetime): timestamp of entry
          exit_time  (datetime): timestamp of exit
          max_drawdown (float): observed max drawdown during the trade (currently unused)
          volatility   (float): observed volatility during the trade (currently unused)
          stop_loss_triggered (bool): whether the stop‐loss was hit (currently unused)

        Returns:
          float: raw P&L = (exit_price − entry_price) × position_size
        """
        try:
            profit = (exit_price - entry_price) * position_size
            return profit
        except Exception as e:
            logger.error(f"calculate_reward error: {e}")
            return 0.0
