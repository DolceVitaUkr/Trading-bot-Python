# modules/rollout_manager.py

import logging
from dataclasses import dataclass
from typing import Literal, Dict, Any, List

import config

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, str(getattr(config, "LOG_LEVEL", "INFO")), logging.INFO)
                if isinstance(getattr(config, "LOG_LEVEL", "INFO"), str) else getattr(config, "LOG_LEVEL", logging.INFO))


@dataclass
class RolloutState:
    environment: Literal["simulation", "production"]
    stage: int
    can_trade_live: bool
    can_withdraw: bool
    risk_pct_max: float
    notes: str


class RolloutManager:
    """
    Central place to control rollout/risk gates.

    Stages (typical):
      1: Simulation only (paper). No live trading.
      2: Canary live on tiny size (still mostly paper). (Not used if you keep simulation-only)
      3+: Gradual ramp (ignored while ENVIRONMENT='simulation').

    This manager doesn’t place orders; it just tells the rest of the system what’s allowed.
    """

    def __init__(self):
        self.env = (config.ENVIRONMENT or "simulation").lower()
        self.stage = int(getattr(config, "ROLLOUT_STAGE", 1))

    def snapshot(self) -> RolloutState:
        if self.env == "simulation":
            return RolloutState(
                environment="simulation",
                stage=self.stage,
                can_trade_live=False,
                can_withdraw=False,
                risk_pct_max=0.0,
                notes="Simulation mode: all execution is paper; live endpoints disabled.",
            )

        # Production env (kept conservative — adjust if you start using this path)
        if self.stage <= 1:
            # technically live env but still blocked
            return RolloutState(
                environment="production",
                stage=self.stage,
                can_trade_live=False,
                can_withdraw=False,
                risk_pct_max=0.0,
                notes="Stage 1 in production: learning/monitoring only, no live orders.",
            )
        elif self.stage == 2:
            return RolloutState(
                environment="production",
                stage=2,
                can_trade_live=True,
                can_withdraw=False,
                risk_pct_max=max(0.01, min(0.02, config.TRADE_SIZE_PERCENT)),  # 1–2%
                notes="Canary: live orders allowed with tiny size; withdrawals disabled.",
            )
        else:
            return RolloutState(
                environment="production",
                stage=self.stage,
                can_trade_live=True,
                can_withdraw=True,
                risk_pct_max=max(0.02, min(0.05, config.TRADE_SIZE_PERCENT)),  # 2–5%
                notes="Ramp up per KPI guardrails.",
            )

    # Convenience gates
    def live_orders_allowed(self) -> bool:
        s = self.snapshot()
        return s.can_trade_live

    def max_risk_pct(self) -> float:
        return float(self.snapshot().risk_pct_max)

    def explain(self) -> str:
        s = self.snapshot()
        return (f"Env={s.environment}, Stage={s.stage}, "
                f"live_orders_allowed={s.can_trade_live}, withdraws={s.can_withdraw}, "
                f"max_risk_pct={s.risk_pct_max:.2%} | {s.notes}")
