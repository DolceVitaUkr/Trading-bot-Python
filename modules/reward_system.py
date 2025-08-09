# modules/reward_system.py

import logging
from datetime import datetime
from typing import Optional, Literal

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class RewardSystem:
    """
    Multi-factor trade reward calculator for RL agents.
    """

    def __init__(
        self,
        trade_type: Literal["scalp", "swing"] = "scalp",
        weights: Optional[dict] = None
    ):
        self.trade_type = trade_type
        self.weights = weights or {
            "profit": 1.0,
            "time": 0.5,
            "drawdown": -1.0,
            "stop_loss": -2.0
        }
        self.profit_tiers = {
            "scalp": [
                (0.5, 1),
                (1, 3),
                (2, 5),
                (3, 8),
            ],
            "swing": [
                (2, 3),
                (5, 10),
                (10, 20),
                (20, 50),
            ]
        }

    def calculate_points(
        self,
        profit_pct: float,
        hours_held: float,
        stop_loss_triggered: bool,
        max_drawdown_pct: float
    ) -> float:
        profit_points = 0
        for threshold, pts in self.profit_tiers[self.trade_type]:
            if profit_pct >= threshold:
                profit_points = pts

        time_bonus = max(0, 10 - hours_held) if self.trade_type == "scalp" else max(0, 50 - hours_held)
        drawdown_penalty = max_drawdown_pct * abs(self.weights["drawdown"]) if max_drawdown_pct > 0 else 0
        sl_penalty = abs(self.weights["stop_loss"]) * 5 if stop_loss_triggered else 0

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
        try:
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
