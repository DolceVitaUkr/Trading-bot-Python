# file: core/datamanager.py
"""DataManager wrapper exposing underscore-free helper methods."""

from __future__ import annotations

import os
from functools import reduce
from typing import Callable, Dict, Any

import pandas as pd

from .data_manager import DataManager as _DataManager, _build_paths


class DataManager(_DataManager):
    """Provide a simplified API with new naming conventions."""

    def fetchklines(
        self,
        symbol: str,
        timeframe: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch cached klines for ``symbol`` and ``timeframe``."""
        df = self.load_historical_data(symbol, timeframe, incremental=False)
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return df

    def savelocal(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Persist ``df`` under the standard cache location."""
        csv_path, _ = _build_paths(symbol, timeframe, self.exchange.client.id)
        df.to_csv(csv_path)

    def loadlocal(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load cached klines if available."""
        csv_path, _ = _build_paths(symbol, timeframe, self.exchange.client.id)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return self._normalize_df(df)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def jointimeframes(self, spec: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        """Join multiple timeframes into a single DataFrame.

        ``spec`` maps a label to ``{"symbol": str, "timeframe": str}``.
        Columns are prefixed by the label.
        """
        frames = []
        for label, params in spec.items():
            df = self.fetchklines(params["symbol"], params["timeframe"])
            frames.append(df.add_prefix(f"{label}_"))
        if not frames:
            return pd.DataFrame()
        return reduce(lambda l, r: l.join(r, how="outer"), frames)

    def subscribelive(self, symbol: str, timeframe: str, callback: Callable[[pd.DataFrame], Any]) -> None:
        """Immediately invoke ``callback`` with the latest klines.

        Real-time streaming is out of scope for this lightweight wrapper but the
        signature mirrors the intended interface.
        """
        df = self.fetchklines(symbol, timeframe)
        callback(df)


__all__ = ["DataManager"]
