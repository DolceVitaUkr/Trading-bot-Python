"""Consolidated close-time reward system (industry-grade, per-asset, after-fee).
Rewards are computed ONLY when a trade is closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Deque
import math

@dataclass
class TradeContext:
    # Required (as provided by paper_trader close-time call)
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: float
    fees_paid: float
    holding_time_seconds: float
    current_equity: float
    open_exposure: float
    # Optional (if available)
    max_drawdown_pct: float = 0.0
    asset_type: Optional[str] = None  # "spot" | "futures" | "options" (if provided)

class RewardSystem:
    """Stateful reward system (keeps a small buffer of last 10 net-USD results)."""

    def __init__(self) -> None:
        self._last10: Deque[float] = deque(maxlen=10)

    # ---- Parameter profiles (starter, industry-grade) ----
    _ASSET_PROFILES = {
        "spot":    dict(alpha=0.18, beta=0.35, s=0.08, tau=3.0,  k=0.012, k_ramp=0.08, gamma=0.04, E0=0.50, r=0.50, b=0.20, w=0.20, clip=10.0, sl_penalty=(-8.0, 0.10)),
        "futures": dict(alpha=0.22, beta=0.50, s=0.10, tau=2.0,  k=0.018, k_ramp=0.10, gamma=0.06, E0=0.30, r=0.60, b=0.25, w=0.25, clip=10.0, sl_penalty=(-9.0, 0.10)),
        "options": dict(alpha=0.28, beta=0.60, s=0.12, tau=2.0,  k=0.020, k_ramp=0.12, gamma=0.08, E0=0.20, r=0.70, b=0.30, w=0.30, clip=10.0, sl_penalty=(-10.0, 0.10)),
    }
    _SL_THRESHOLD_PCT = 15.0  # hard alignment to your risk policy (15%)

    @staticmethod
    def _infer_asset_type(ctx: TradeContext) -> str:
        if ctx.asset_type:
            return ctx.asset_type.lower()
        # Heuristic: leverage>1 => futures; symbol hints for options can be added later if needed
        return "futures" if (ctx.leverage or 1.0) > 1.05 else "spot"

    def compute_reward(self, ctx: TradeContext) -> float:
        """Return a scalar reward for a CLOSED trade, using after-fee profit only."""
        # --- Derived quantities ---
        entry_value = float(ctx.entry_price) * float(ctx.quantity)
        exit_value  = float(ctx.exit_price)  * float(ctx.quantity)
        net_usd     = (exit_value - entry_value) - float(ctx.fees_paid)
        g           = 100.0 * (net_usd / entry_value) if entry_value > 0 else 0.0  # net % after fees
        T_hours     = float(ctx.holding_time_seconds or 0.0) / 3600.0
        E           = float(ctx.open_exposure) / max(float(ctx.current_equity), 1e-8)
        E           = max(0.0, min(E, 1.5))  # clip exposure
        DD          = max(0.0, float(ctx.max_drawdown_pct or 0.0))

        profile_key = self._infer_asset_type(ctx)
        prof = self._ASSET_PROFILES.get(profile_key, self._ASSET_PROFILES["spot"])

        alpha = prof["alpha"]; beta  = prof["beta"]
        s     = prof["s"];     tau   = prof["tau"]
        k     = prof["k"];     k_r   = prof["k_ramp"]
        gamma = prof["gamma"]; E0    = prof["E0"]; r = prof["r"]
        b     = prof["b"];     w     = prof["w"];  clipv = prof["clip"]
        sl_off, sl_mult = prof["sl_penalty"]  # (fixed negative offset, multiplier when beyond SL)

        # --- Core asymmetric exponential ---
        if g >= 0.0:
            core = math.exp(alpha * g) - 1.0
        else:
            core = - (math.exp(beta * abs(g)) - 1.0)

        # --- Quick scalp nudge ---
        speed = 1.0 + s * math.exp(-T_hours / max(tau, 1e-6))

        # --- Time penalty with Day-6 ramp ---
        base_decay = math.exp(-k * T_hours)
        ramp_decay = math.exp(-k_r * max(0.0, T_hours - 144.0))
        time_pen  = base_decay * ramp_decay

        # --- Drawdown penalty ---
        dd_pen = math.exp(-gamma * DD)

        # --- Risk usage modifier ---
        risk_mod = 1.0 - r * max(E - E0, 0.0)

        reward = core * speed * time_pen * dd_pen * max(0.0, risk_mod)  # don't invert sign via risk_mod

        # --- SL guard rail at -15% ---
        if g <= -self._SL_THRESHOLD_PCT:
            reward = reward * sl_mult + sl_off

        # --- Streak consistency (last 10 net USD) ---
        # Update buffer first, then compute bonus based on *previous* 9 + this trade
        self._last10.append(float(net_usd))
        last = list(self._last10)
        net10 = sum(last)
        if last:
            mean_abs10 = sum(abs(x) for x in last) / len(last)
        else:
            mean_abs10 = 0.0
        wins = sum(1 for x in last if x > 0.0)
        winrate10 = wins / len(last) if last else 0.0

        if net10 > 0.0 and mean_abs10 > 0.0:
            streak_mult = 1.0 + b * min(1.0, net10 / (10.0 * mean_abs10)) + w * max(0.0, winrate10 - 0.5)
            reward *= max(0.0, streak_mult)

        # --- Clip final value for stability ---
        reward = max(-clipv, min(clipv, reward))
        return float(reward)

__all__ = ["TradeContext", "RewardSystem"]