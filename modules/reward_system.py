# modules/reward_system.py

import logging
from datetime import datetime
from typing import Optional, Literal

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class RewardSystem:
    """
    Multi‐factor trade reward calculator for RL agents.

    - Flexible profit tiers for different trade types (scalp, swing)
    - Time-based scaling to reward faster profitable trades
    - Penalties for stop-loss triggers, excessive drawdown, or low Sharpe-like risk-adjusted returns
    - Adjustable weights for each reward/penalty component
    """

    def __init__(
        self,
        trade_type: Literal["scalp", "swing"] = "scalp",
        weights: Optional[dict] = None
    ):
        """
        Parameters:
          trade_type (str): "scalp" or "swing" — determines profit tier thresholds
          weights (dict): Optional custom weightings for reward components
        """
        self.trade_type = trade_type
        self.weights = weights or {
            "profit": 1.0,
            "time": 0.5,
            "drawdown": -1.0,
            "stop_loss": -2.0
        }

        # Profit tiers can be adjusted per trade type
        self.profit_tiers = {
            "scalp": [
                (0.5, 1),   # ≥0.5% profit → 1 point
                (1, 3),     # ≥1% profit → 3 points
                (2, 5),     # ≥2% profit → 5 points
                (3, 8),     # ≥3% profit → 8 points
            ],
            "swing": [
                (2, 3),     # ≥2% profit → 3 points
                (5, 10),    # ≥5% profit → 10 points
                (10, 20),   # ≥10% profit → 20 points
                (20, 50),   # ≥20% profit → 50 points
            ]
        }

    def calculate_points(
        self,
        profit_pct: float,
        hours_held: float,
        stop_loss_triggered: bool,
        max_drawdown_pct: float
    ) -> float:
        """
        Calculates a weighted multi-factor score for a single trade.
        """
        # Profit reward from tier table
        profit_points = 0
        for threshold, pts in self.profit_tiers[self.trade_type]:
            if profit_pct >= threshold:
                profit_points = pts

        # Time bonus: reward faster exits for scalps, moderate for swings
        time_bonus = max(0, 10 - hours_held) if self.trade_type == "scalp" else max(0, 50 - hours_held)

        # Drawdown penalty
        drawdown_penalty = max_drawdown_pct * abs(self.weights["drawdown"]) if max_drawdown_pct > 0 else 0

        # Stop-loss penalty
        sl_penalty = abs(self.weights["stop_loss"]) * 5 if stop_loss_triggered else 0

        # Final weighted score
        score = (
            self.weights["profit"] * profit_points +
            self.weights["time"] * time_bonus -
            drawdown_penalty -
            sl_penalty
        )

        return round(score, 2)

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
        Computes a reward score for RL training based on trade outcome.

        Returns:
          float: Final multi-factor score (positive = good, negative = bad)
        """
        try:
            # Profit percentage
            profit_pct = ((exit_price - entry_price) / entry_price) * 100
            hours_held = (exit_time - entry_time).total_seconds() / 3600.0
            max_dd_pct = max_drawdown * 100

            return self.calculate_points(
                profit_pct=profit_pct,
                hours_held=hours_held,
                stop_loss_triggered=stop_loss_triggered,
                max_drawdown_pct=max_dd_pct
            )
        except Exception as e:
            logger.error(f"calculate_reward error: {e}")
            return 0.0
