import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RewardSystem:
    """
    Tracks the bot's performance over time using net P&L.
    It maintains a simple wallet balance and a points system.
    """
    def __init__(self, starting_balance: float = 0.0, points_multiplier: float = 1.0):
        """
        Initializes the RewardSystem.

        Args:
            starting_balance (float, optional): The initial wallet balance. Defaults to 0.0.
            points_multiplier (float, optional): A multiplier to convert P&L to points. Defaults to 1.0.
        """
        self.wallet_balance: float = starting_balance
        self.total_points: float = 0.0
        self.points_multiplier = points_multiplier
        logger.info(f"RewardSystem initialized with starting balance: {starting_balance:.2f}")

    def update(self, pnl_net_usd: float, context: Dict[str, Any]):
        """
        Updates the wallet and points based on the net P&L of a completed trade.

        Args:
            pnl_net_usd (float): The net profit or loss from the trade.
            context (Dict[str, Any]): The context of the trade (asset, mode, etc.).
                                       This is for future enhancements.
        """
        # Update wallet balance
        self.wallet_balance += pnl_net_usd

        # Update points based on net P&L
        points_earned = pnl_net_usd * self.points_multiplier
        self.total_points += points_earned

        logger.info(
            f"RewardSystem updated: PnL={pnl_net_usd:.2f}, "
            f"Points Earned={points_earned:.2f}, "
            f"New Wallet Balance={self.wallet_balance:.2f}, "
            f"Total Points={self.total_points:.2f}"
        )

    def get_snapshot(self) -> Dict[str, float]:
        """
        Returns the current state of the reward system.

        Returns:
            Dict[str, float]: A snapshot of the current wallet balance and total points.
        """
        return {
            "wallet_balance": self.wallet_balance,
            "total_points": self.total_points
        }
