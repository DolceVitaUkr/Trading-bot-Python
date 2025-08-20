"""Dynamic trading pair selection (very small placeholder)."""

from __future__ import annotations

from typing import List, Dict

import pandas as pd


class PairManager:
    """Return a static list of pairs.

    A real implementation would analyse volatility, volume and other
    metrics.  For the unit tests we simply return a fixed universe.
    """

    def __init__(self, default: List[str] | None = None) -> None:
        self._default = default or ["BTCUSDT", "ETHUSDT"]

    def current_universe(self) -> List[str]:
        return list(self._default)

    # ------------------------------------------------------------------
    def tag_regimes(self, df: pd.DataFrame) -> Dict[str, str]:
        """Classify simple market regimes from price data.

        Parameters
        ----------
        df:
            DataFrame with ``close`` and ``volume`` columns.

        Returns
        -------
        dict
            Mapping of regime type (``volatility``, ``trend``, ``liquidity``) to
            a qualitative label.  The logic is intentionally lightweight but
            deterministic so unit tests can exercise the behaviour.
        """

        if df.empty:
            return {"volatility": "LOW", "trend": "SIDEWAYS", "liquidity": "LOW"}

        returns = df["close"].pct_change().dropna()
        vol = returns.std()
        volatility = "HIGH" if vol > 0.02 else "LOW"

        sma = df["close"].rolling(20, min_periods=1).mean()
        last = df["close"].iloc[-1]
        base = sma.iloc[-1]
        if last > base * 1.01:
            trend = "UP"
        elif last < base * 0.99:
            trend = "DOWN"
        else:
            trend = "SIDEWAYS"

        liquidity = "HIGH" if df.get("volume", pd.Series()).mean() > 1_000_000 else "LOW"
        return {"volatility": volatility, "trend": trend, "liquidity": liquidity}


__all__ = ["PairManager"]
