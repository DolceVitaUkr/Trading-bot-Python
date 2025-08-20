"""Simplified risk management utilities.

This module enforces basic risk rules used in tests.  It is **not** a
complete implementation of the architecture described in the README but
provides the minimal API required by the unit tests and for future
expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class OrderProposal:
    """Light‑weight representation of an order for validation."""

    price: float
    stop_loss: float | None
    take_profit: float | None = None


class RiskManager:
    """Validate orders against basic risk constraints."""

    def __init__(self, balance: float = 10_000.0, risk_per_trade: float = 0.01,
                 max_stop_loss_pct: float = 0.15) -> None:
        self.balance = balance
        self.risk_per_trade = risk_per_trade
        self.max_stop_loss_pct = max_stop_loss_pct

    # ------------------------------------------------------------------
    def validate_order(self, proposal: OrderProposal) -> Tuple[bool, str]:
        """Validate an :class:`OrderProposal`.

        Checks that a stop loss is present and that it does not exceed the
        maximum configured distance from the entry price.  The method returns a
        tuple ``(allowed, reason)`` where ``reason`` is ``"OK"`` when the order
        is allowed.
        """

        if proposal.stop_loss is None:
            return False, "Stop loss required"
        if proposal.price <= 0:
            return False, "Invalid price"
        distance_pct = abs(proposal.price - proposal.stop_loss) / proposal.price
        if distance_pct > self.max_stop_loss_pct:
            return False, "Stop loss exceeds max percentage"
        return True, "OK"


__all__ = ["OrderProposal", "RiskManager"]
