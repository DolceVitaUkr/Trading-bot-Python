# file: core/pair_manager.py

"""Dynamic trading pair selection (very small placeholder)."""

from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd


class PairManager:
    """Manage trading pair universe and simple regime tags.

    The class is purposely lightweight but exposes a small API that mimics the
    behaviour described in the project plan.  ``refresh_universe`` produces a
    deterministic ranking so unit tests can make assertions on the returned
    ordering without relying on external data.
    """

    def __init__(self, default: List[str] | None = None) -> None:
        self._default = default or ["BTCUSDT", "ETHUSDT"]
        self._sentiment: Callable[[str], float] | None = None
        self._ranked: List[str] = list(self._default)

    # ------------------------------------------------------------------
    def set_sentiment(self, providerfn: Callable[[str], float] | None) -> None:
        """Register an optional sentiment provider.

        The callable should return a numeric sentiment score in the range
        ``[-1, 1]`` for a given symbol.  Positive numbers boost the ranking
        produced by :meth:`refresh_universe`.
        """

        self._sentiment = providerfn

    # ------------------------------------------------------------------
    def refresh_universe(self) -> Dict[str, List[str]]:
        """Return a ranked universe grouped by asset class.

        A very small deterministic scoring function is used: the base score is
        derived from the symbol's character codes, and sentiment (if available)
        is added on top.  Rankings are stored for subsequent calls to
        :meth:`get_top` and :meth:`current_universe`.
        """

        scores: Dict[str, float] = {}
        for sym in self._default:
            base = sum(ord(c) for c in sym) / 1000.0
            sent = self._sentiment(sym) if self._sentiment else 0.0
            scores[sym] = base + sent

        # sort descending by score
        self._ranked = [s for s, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        return {"crypto": list(self._ranked)}

    # ------------------------------------------------------------------
    def current_universe(self) -> List[str]:
        """Return the last computed ranking (refreshing if needed)."""

        if not self._ranked:
            self.refresh_universe()
        return list(self._ranked)

    # ------------------------------------------------------------------
    def get_top(self, count: int, asset: str) -> List[str]:
        """Return the ``count`` highest ranked symbols for ``asset``.

        Only the ``"crypto"`` asset class is supported in this simplified
        implementation, but the signature mirrors the project specification.
        """

        if not self._ranked:
            self.refresh_universe()
        return self._ranked[:count]

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
