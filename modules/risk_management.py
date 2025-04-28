# modules/risk_management.py

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from modules.error_handler import RiskViolationError

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@dataclass
class PositionRisk:
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_reward_ratio: float
    risk_percent: float
    dollar_risk: float

class RiskManager:
    """
    Encapsulates all risk logic:
      - position sizing
      - dynamic stop adjustments
      - portfolio‐level risk assessment
      - drawdown monitoring
    """

    def __init__(
        self,
        account_balance: float,
        max_portfolio_risk: float = 0.05,
        max_drawdown_limit: float  = 0.10
    ):
        self.account_balance     = account_balance
        self.max_portfolio_risk  = max_portfolio_risk
        self.open_positions      : Dict[str, PositionRisk] = {}
        self.max_leverage        = 10
        self.risk_free_rate      = 0.02
        self.min_risk_reward     = 1.5
        self.atr_period          = 14
        self.volatility_factor   = 2.0

        # --- drawdown tracking ---
        self.peak_equity         = account_balance
        self.current_equity      = account_balance
        self.max_drawdown_limit  = max_drawdown_limit

    def update_equity(self, equity: float) -> float:
        """
        Call this after each mark‐to‐market update.
        Raises RiskViolationError if drawdown exceeds the configured limit.
        Returns the current drawdown fraction.
        """
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown > self.max_drawdown_limit:
            logger.critical(
                f"Drawdown {drawdown:.2%} exceeds limit {self.max_drawdown_limit:.2%} "
                f"(peak={self.peak_equity:.2f}, current={equity:.2f})"
            )
            raise RiskViolationError(
                f"Maximum drawdown exceeded ({drawdown:.2%})",
                context={"peak_equity": self.peak_equity, "current_equity": equity}
            )
        return drawdown

    def calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        risk_percent: float = 0.01
    ) -> float:
        """
        Determine how many units to trade so that dollar risk = account_balance * risk_percent.
        Enforces max_leverage and max_portfolio_risk.
        """
        # validate inputs
        if entry_price <= 0 or stop_price <= 0:
            msg = f"Invalid prices: entry={entry_price}, stop={stop_price}"
            logger.error(msg)
            raise RiskViolationError(msg)

        price_risk = abs(entry_price - stop_price)
        if price_risk == 0:
            msg = f"Zero price risk (entry==stop=={entry_price})"
            logger.error(msg)
            raise RiskViolationError(msg)

        dollar_risk = self.account_balance * risk_percent
        position_size = dollar_risk / price_risk

        # enforce leverage
        margin_required = position_size * entry_price / self.max_leverage
        if margin_required > self.account_balance:
            msg = (
                f"Insufficient margin: required={margin_required:.2f} "
                f"> balance={self.account_balance:.2f}"
            )
            logger.error(msg)
            raise RiskViolationError(msg)

        # enforce portfolio risk
        total_risk = sum(p.dollar_risk for p in self.open_positions.values())
        if (total_risk + dollar_risk) / self.account_balance > self.max_portfolio_risk:
            msg = (
                f"Portfolio risk would exceed max: "
                f"current={total_risk/self.account_balance:.2%}, "
                f"new={(total_risk + dollar_risk)/self.account_balance:.2%}, "
                f"limit={self.max_portfolio_risk:.2%}"
            )
            logger.error(msg)
            raise RiskViolationError(msg)

        return position_size

    def dynamic_stop_management(
        self,
        current_price: float,
        position: PositionRisk
    ) -> PositionRisk:
        """
        Move the stop_loss up (never down) as price moves in your favor.
        """
        price_change_pct = (current_price - position.entry_price) / position.entry_price
        # Example trailing logic
        if price_change_pct > 0.05:
            new_stop = max(position.stop_loss, position.entry_price)
        elif price_change_pct > 0.02:
            new_stop = position.stop_loss * 1.01
        else:
            new_stop = position.stop_loss

        # also use ATR‐based floor
        atr = self._calculate_atr(position.entry_price)
        new_stop = max(new_stop, current_price - self.volatility_factor * atr)

        # recompute take_profit
        risk = position.entry_price - new_stop
        new_tp = position.entry_price + (risk * position.risk_reward_ratio)

        return PositionRisk(
            entry_price=position.entry_price,
            stop_loss=new_stop,
            take_profit=new_tp,
            position_size=position.position_size,
            risk_reward_ratio=position.risk_reward_ratio,
            risk_percent=position.risk_percent,
            dollar_risk=position.dollar_risk
        )

    def portfolio_risk_assessment(self) -> Dict[str, float]:
        """
        Summarize total risk, portfolio leverage, drawdown, etc.
        """
        total_dollar_risk = sum(p.dollar_risk for p in self.open_positions.values())
        margin_used       = sum(p.position_size * p.entry_price for p in self.open_positions.values())
        return {
            "total_dollar_risk": total_dollar_risk,
            "portfolio_risk_ratio": total_dollar_risk / self.account_balance,
            "margin_utilization": margin_used / self.account_balance,
            "leverage_ratio": margin_used / self.account_balance,
            "max_drawdown": (self.peak_equity - self.current_equity) / self.peak_equity,
        }

    def _calculate_atr(self, current_price: float, lookback: int = 14) -> float:
        """Placeholder ATR; replace with real history if desired."""
        return current_price * 0.02

