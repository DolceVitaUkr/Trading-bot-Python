"""Technical indicator helpers.

Only a couple of lightweight indicators are implemented for testing
purposes.  They rely solely on :mod:`pandas` and avoid heavy numerical
libraries.
"""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def apply_indicators(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Return a copy of ``df`` enriched with the requested indicators."""

    result = df.copy()
    for name, params in spec.items():
        if name == "sma":
            result[f"sma_{params['period']}"] = sma(result[params.get("column", "close")], params["period"])
        elif name == "ema":
            result[f"ema_{params['period']}"] = ema(result[params.get("column", "close")], params["period"])
        elif name == "rsi":
            result[f"rsi_{params.get('period',14)}"] = rsi(result[params.get("column", "close")], params.get("period", 14))
    return result


__all__ = ["sma", "ema", "rsi", "apply_indicators"]
