"""Dynamic trading pair selection (very small placeholder)."""

from __future__ import annotations

from typing import List


class PairManager:
    """Return a static list of pairs.

    A real implementation would analyse volatility, volume and other
    metrics.  For the unit tests we simply return a fixed universe.
    """

    def __init__(self, default: List[str] | None = None) -> None:
        self._default = default or ["BTCUSDT", "ETHUSDT"]

    def current_universe(self) -> List[str]:
        return list(self._default)


__all__ = ["PairManager"]
