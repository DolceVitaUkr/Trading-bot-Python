# file: core/riskmanager.py
"""Comprehensive risk management utilities.

This module implements a light‑weight yet extensible risk manager that
covers the behaviour described in the project ``Modules.md``.  It still
keeps the public surface area small so the rest of the codebase and the
unit tests can interact with it easily.  The class focuses on three core
areas:

* order validation (mandatory SL, optional TP, exposure checks)
* position sizing based on account risk percentage
* running breach checks for daily loss and exposure caps

The implementation does not aim to be production ready but provides a
solid foundation for future expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class OrderProposal:
    """Representation of an order for validation.

    Parameters are intentionally optional so existing tests can construct the
    object with only what they need.  ``notional`` represents the total value
    of the order (price * quantity).
    """

    price: float
    stop_loss: float | None
    take_profit: float | None = None
    symbol: str = ""
    side: str = ""
    notional: float | None = None


@dataclass
class Position:
    """Minimal representation of an open position for trailing TP logic."""

    side: str
    entry: float
    take_profit: float | None = None


# ---------------------------------------------------------------------------
# Risk manager
# ---------------------------------------------------------------------------

class RiskManager:
    """Validate orders and monitor exposure limits."""

    def __init__(
        self,
        balance: float = 10_000.0,
        risk_per_trade: float = 0.01,
        max_stop_loss_pct: float = 0.15,
        max_daily_loss_pct: float = 0.2,
        max_exposure_pct: float = 0.5,
        require_take_profit: bool = False,
    ) -> None:
        self.balance = balance
        self.risk_per_trade = risk_per_trade
        self.max_stop_loss_pct = max_stop_loss_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_exposure_pct = max_exposure_pct
        self.require_take_profit = require_take_profit

        self.daily_start_equity = balance
        # ``peak_equity`` is used to determine drawdown based adjustments to
        # risk.  It starts at the initial balance and gets updated whenever the
        # balance makes a new high.
        self.peak_equity = balance
        self.open_exposure = 0.0

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def computesize(
        self,
        price: float,
        stop_loss: float,
        risk_pct: float | None = None,
        *,
        min_notional: float = 0.0,
        precision: float | None = None,
        signal_strength: str | float | None = None,
    ) -> float:
        """Return the notional amount allowed for a trade.

        The sizing logic follows the specification in ``modules.md``:

        * While equity is below 1,000 the bot trades a fixed 10 USD notional.
        * Above that, risk is a tiered percentage of equity (0.5–2%).
        * For every 5% of drawdown from the equity peak the risk percentage is
          reduced by 0.25 percentage points down to a 0.25% floor.
        * ``signal_strength`` applies a weighting of 0.5/1.0/1.5 for
          weak/normal/strong signals.
        * ``precision`` can be used to enforce exchange notional steps.
        """

        if price <= 0:
            return 0.0

        # fixed notional during learning phase
        if self.balance < 1_000:
            notional = 10.0
        else:
            distance_pct = abs(price - stop_loss) / price
            if distance_pct <= 0:
                return 0.0

            # determine base risk percentage
            equity = self.balance
            base_risk = self._tierriskpct(equity)

            # adjust for drawdown
            drawdown = max(0.0, (self.peak_equity - equity) / self.peak_equity)
            reduction_steps = int(drawdown / 0.05)
            adj_risk = base_risk - reduction_steps * 0.0025
            adj_risk = max(adj_risk, 0.0025)

            # optional override / weighting
            if risk_pct is not None:
                adj_risk = risk_pct

            weight = self._signalweight(signal_strength)
            risk_amount = equity * adj_risk * weight
            notional = risk_amount / distance_pct

        if precision and precision > 0:
            notional = (notional // precision) * precision
        return max(notional, min_notional)

    # ------------------------------------------------------------------
    def _tierriskpct(self, equity: float) -> float:
        """Return the base risk percentage based on equity tiers."""

        if equity < 10_000:
            return 0.005
        if equity < 50_000:
            return 0.01
        if equity < 100_000:
            return 0.015
        return 0.02

    def _signalweight(self, strength: str | float | None) -> float:
        """Translate ``signal_strength`` into a multiplier."""

        if isinstance(strength, (int, float)):
            return float(strength)
        mapping = {
            "weak": 0.5,
            "normal": 1.0,
            "strong": 1.5,
            "very_strong": 1.5,
        }
        return mapping.get(str(strength).lower(), 1.0)

    # ------------------------------------------------------------------
    # Order validation
    # ------------------------------------------------------------------
    def validateorder(self, proposal: OrderProposal) -> Tuple[bool, str]:
        """Validate an :class:`OrderProposal`.

        Checks for:
        * presence of stop loss (and optionally take profit)
        * maximum stop loss percentage
        * exposure / risk-per-trade limits if ``notional`` is supplied
        """

        if proposal.stop_loss is None:
            return False, "Stop loss required"
        if self.require_take_profit and proposal.take_profit is None:
            return False, "Take profit required"
        if proposal.price <= 0:
            return False, "Invalid price"

        distance_pct = abs(proposal.price - proposal.stop_loss) / proposal.price
        if distance_pct > self.max_stop_loss_pct:
            return False, "Stop loss exceeds max percentage"

        if proposal.notional is not None:
            # exposure check
            if proposal.notional + self.open_exposure > self.balance * self.max_exposure_pct:
                return False, "Exposure limit exceeded"
            # risk per trade check
            risk = distance_pct * proposal.notional
            if risk > self.balance * self.risk_per_trade:
                return False, "Risk per trade exceeded"

        return True, "OK"

    # ------------------------------------------------------------------
    # Trailing take profit
    # ------------------------------------------------------------------
    def applytrailingtp(
        self, position: Position, current_price: float, trail: float
    ) -> None:
        """Adjust a position's take profit based on the current price.

        ``trail`` can be interpreted as a percentage (e.g. 0.02 == 2%).  For a
        long position the TP will only ever increase; for a short position it
        will only ever decrease.  The function mutates ``position`` in place.
        """

        if trail <= 0:
            return
        if position.side.lower() == "buy":
            target = current_price * (1 + trail)
            if position.take_profit is None or target > position.take_profit:
                position.take_profit = target
        else:
            target = current_price * (1 - trail)
            if position.take_profit is None or target < position.take_profit:
                position.take_profit = target

    # ------------------------------------------------------------------
    # Exposure tracking helpers
    # ------------------------------------------------------------------
    def registeropen(self, notional: float) -> None:
        self.open_exposure += notional

    def registerclose(self, notional: float, pnl: float) -> None:
        self.open_exposure = max(0.0, self.open_exposure - notional)
        self.balance += pnl
        if self.balance > self.peak_equity:
            self.peak_equity = self.balance

    # ------------------------------------------------------------------
    def riskbreachcheck(self) -> List[str]:
        """Return a list of risk rule identifiers that are currently breached."""

        breaches: List[str] = []
        if self.balance <= self.daily_start_equity * (1 - self.max_daily_loss_pct):
            breaches.append("DAILY_LOSS")
        if self.open_exposure > self.balance * self.max_exposure_pct:
            breaches.append("EXPOSURE")
        return breaches

    # Backwards compatibility aliases ----------------------------------
    compute_size = computesize
    validate_order = validateorder
    apply_trailing_tp = applytrailingtp
    register_open = registeropen
    register_close = registerclose
    risk_breach_check = riskbreachcheck


__all__ = ["OrderProposal", "Position", "RiskManager"]
