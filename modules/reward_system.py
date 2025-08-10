# modules/reward_system.py

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class RewardWeights:
    """
    Tunable weights for reward shaping.
    """
    pnl_weight: float = 1.0            # direct PnL contribution
    time_decay_half_life_min: float = 60.0  # decay half-life in minutes
    drawdown_penalty: float = 0.5      # penalty multiplier for max drawdown (0..1)
    vol_penalty: float = 0.25          # penalty multiplier for realized volatility (0..1)
    stop_loss_penalty: float = 0.6     # multiplicative penalty when SL triggers
    rr_bonus_weight: float = 0.15      # small positive for good R:R outcomes


class RewardSystem:
    """
    Converts trade outcomes into a continuous reward signal for RL and
    also supports 'points' scoring used for dashboards.

    Methods:
      - calculate_reward(...)
      - points_from_profit(...)
    """

    def __init__(self, weights: Optional[RewardWeights] = None):
        self.w = weights or RewardWeights()

    # ──────────────────────────────────────────────────────────────────────
    # RL-friendly scalar reward
    # ──────────────────────────────────────────────────────────────────────
    def calculate_reward(
        self,
        *,
        entry_price: float,
        exit_price: float,
        position_size: float,
        entry_time: datetime,
        exit_time: datetime,
        max_drawdown: float,
        volatility: float,
        stop_loss_triggered: bool,
        rr_achieved: Optional[float] = None,  # realized R multiple if known
    ) -> float:
        """
        Return a dense reward balancing profit, time, risk.

        - PnL: (exit - entry) * size
        - Time decay: exponential decay to discourage overholding unproductive trades
        - Drawdown & volatility penalties: scale down reward when risk is high
        - Stop loss penalty: multiplicative penalty to discourage frequent SL hits
        - RR bonus: small bonus when positive R:R achieved
        """
        pnl = (float(exit_price) - float(entry_price)) * float(position_size)

        # time decay factor (0..1)
        minutes = max(0.0, (exit_time - entry_time).total_seconds() / 60.0)
        if self.w.time_decay_half_life_min <= 0:
            time_factor = 1.0
        else:
            # decay to 0.5 every half-life minutes
            time_factor = 0.5 ** (minutes / self.w.time_decay_half_life_min)

        # penalties for risk
        dd_penalty = max(0.0, 1.0 - self.w.drawdown_penalty * max(0.0, float(max_drawdown)))
        vol_penalty = max(0.0, 1.0 - self.w.vol_penalty * max(0.0, float(volatility)))

        reward = pnl * self.w.pnl_weight * time_factor * dd_penalty * vol_penalty

        if stop_loss_triggered:
            reward *= max(0.0, self.w.stop_loss_penalty)

        # small positive reinforcement for realized RR > 1
        if rr_achieved is not None and rr_achieved > 1.0:
            reward += float(rr_achieved - 1.0) * self.w.rr_bonus_weight

        return float(reward)

    # ──────────────────────────────────────────────────────────────────────
    # Points used in dashboards/leaderboards
    # ──────────────────────────────────────────────────────────────────────
    def points_from_profit(
        self,
        *,
        profit_pct: float,
        minutes_in_trade: float,
        stop_loss_triggered: bool,
        risk_adjusted: bool = True,
    ) -> float:
        """
        Intuitive points:
          • Base: profit % * 100
          • Time penalty: divide by sqrt(minutes+1)
          • SL hit: subtract small fixed penalty
          • Risk-adjusted: dampen negative points (cap loss impact)
        """
        base = float(profit_pct) * 100.0
        time_scale = 1.0 / math.sqrt(max(1.0, minutes_in_trade))
        pts = base * time_scale

        if stop_loss_triggered:
            pts -= 5.0  # small penalty

        if risk_adjusted:
            # clamp negative points so one bad trade doesn't dominate
            pts = max(pts, -50.0)

        return float(pts)


# ────────────────────────────────────────────────────────────────────────────────
# Module-level helper for convenience (legacy imports use this)
# ────────────────────────────────────────────────────────────────────────────────
def calculate_points(
    *,
    profit: float,
    entry_time: datetime,
    exit_time: datetime,
    stop_loss_triggered: bool,
    risk_adjusted: bool = True,
) -> float:
    """
    Convert raw PnL (USD) into points considering holding time.
    We normalize by entry notional magnitude if available; since we don't
    have notional in this helper, approximate via time scaling only.

    Upstream callers typically pass profit % already; when raw USD is
    provided, treat it as a proxy (kept for backward compatibility).
    """
    minutes = max(0.0, (exit_time - entry_time).total_seconds() / 60.0)
    # Heuristic: 1 USD ~ 1 point for short trades, fades with time.
    base_pct = float(profit)  # assume this may already be pct-like from caller
    rs = RewardSystem()
    return rs.points_from_profit(
        profit_pct=base_pct,
        minutes_in_trade=minutes,
        stop_loss_triggered=stop_loss_triggered,
        risk_adjusted=risk_adjusted,
    )
