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
        self.open_exposure = 0.0

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def compute_size(
        self,
        price: float,
        stop_loss: float,
        risk_pct: float | None = None,
        min_notional: float = 0.0,
    ) -> float:
        """Return the notional amount allowed for a trade.

        The size is derived from the configured ``risk_per_trade`` and the
        distance between ``price`` and ``stop_loss``.  ``min_notional`` acts as
        a floor required by some exchanges.
        """

        if price <= 0:
            return 0.0
        risk_pct = risk_pct if risk_pct is not None else self.risk_per_trade
        risk_amount = self.balance * risk_pct
        sl_distance = abs(price - stop_loss)
        if sl_distance <= 0:
            return 0.0
        distance_pct = sl_distance / price
        if distance_pct <= 0:
            return 0.0
        notional = risk_amount / distance_pct
        return max(notional, min_notional)

    # ------------------------------------------------------------------
    # Order validation
    # ------------------------------------------------------------------
    def validate_order(self, proposal: OrderProposal) -> Tuple[bool, str]:
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
    def apply_trailing_tp(
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
    def register_open(self, notional: float) -> None:
        self.open_exposure += notional

    def register_close(self, notional: float, pnl: float) -> None:
        self.open_exposure = max(0.0, self.open_exposure - notional)
        self.balance += pnl

    # ------------------------------------------------------------------
    def risk_breach_check(self) -> List[str]:
        """Return a list of risk rule identifiers that are currently breached."""

        breaches: List[str] = []
        if self.balance <= self.daily_start_equity * (1 - self.max_daily_loss_pct):
            breaches.append("DAILY_LOSS")
        if self.open_exposure > self.balance * self.max_exposure_pct:
            breaches.append("EXPOSURE")
        return breaches


__all__ = ["OrderProposal", "Position", "RiskManager"]
