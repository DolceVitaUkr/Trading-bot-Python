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
    risk_adjusted: bool = True,
    penalty_multiplier: float = 3.0
) -> float:
    """
    Calculate reward points based on profit, trade duration, and stop-loss events.

    - Base points: profit * 100 (rewards) or profit * 100 * penalty_multiplier (penalties)
    - Time bonus: max(0, 50 - 10 * duration_hours)
    - Risk penalty: 100 if stop_loss_triggered else 0
    - Overholding penalty: if loss and held >7 days, subtract 10 points per extra day
    - If risk_adjusted: divide total points by (risk_penalty + 1)

    Penalties are amplified by penalty_multiplier to be more severe than rewards.
    """
    try:
        # Calculate durations
        duration_hours = (exit_time - entry_time).total_seconds() / 3600.0
        duration_days = duration_hours / 24.0
        # Time bonus diminishes as duration increases
        time_bonus = max(0.0, 50.0 - (duration_hours * 10.0))
        # Penalty if stop-loss was triggered
        risk_penalty = 100.0 if stop_loss_triggered else 0.0

        # Base points: linear for gains, amplified for losses
        if profit >= 0:
            base_points = profit * 100.0
        else:
            base_points = profit * 100.0 * penalty_multiplier

        total = base_points + time_bonus - risk_penalty

        # Overholding penalty for holding a losing position > 7 days
        if profit < 0 and duration_days > 7.0:
            extra_days = duration_days - 7.0
            overhold_penalty = extra_days * 10.0
            total -= overhold_penalty

        # Risk-adjustment divides by (penalty + 1)
        if risk_adjusted:
            total = total / (risk_penalty + 1.0)

        return round(total, 2)
    except Exception as e:
        logger.error(f"calculate_points error: {e}")
        return 0.0


class RewardSystem:
    """
    Calculates a reward metric for completed trades with progressive penalties.
    """
    def calculate_reward(
        self,
        entry_price: float,
        exit_price: float,
        position_size: float,
        entry_time: datetime,
        exit_time: datetime,
        max_drawdown: float,
        volatility: float,
        stop_loss_triggered: bool,
        penalty_multiplier: float = 3.0
    ) -> float:
        """
        Basic reward: net profit * position_size,
        with amplified penalties for losses.
        """
        try:
            profit = exit_price - entry_price
            if profit >= 0:
                return profit * position_size
            else:
                return profit * position_size * penalty_multiplier
        except Exception as e:
            logger.error(f"calculate_reward error: {e}")
            return 0.0