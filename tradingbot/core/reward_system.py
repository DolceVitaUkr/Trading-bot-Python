"""Simplified reward system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradeContext:
    pnl: float  # profit/loss after fees


class RewardSystem:
    def compute_reward(self, ctx: TradeContext) -> dict:
        """Return reward information for RL training.

        The real project applies drawdown penalties and bonuses.  This
        placeholder merely passes through the PnL value.
        """

        return {"reward": ctx.pnl, "points": ctx.pnl}


__all__ = ["TradeContext", "RewardSystem"]
