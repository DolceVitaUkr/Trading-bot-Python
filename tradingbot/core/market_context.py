"""
Market context helpers: sessions (ASIA/EU/US) and regimes (trend/range/vol).
Sessions are UTC-based rough windows:
  - ASIA: 23:00–08:00 UTC
  - EU  : 07:00–16:00 UTC
  - US  : 12:00–21:00 UTC
Regime classification uses simple indicators (EMA slope, ADX proxy via directional movement, Bollinger width).
If data fetch fails, returns ("UNKNOWN", {"reason": ...}).
"""
from __future__ import annotations
import datetime as _dt
from typing import Tuple, Dict, Any, Optional

def session_now(now_utc: Optional[_dt.datetime] = None) -> str:
    now = now_utc or _dt.datetime.utcnow()
    h = now.hour
    # windows overlap; pick precedence US > EU > ASIA
    if 12 <= h < 21:  # US
        return "US"
    if 7 <= h < 16:   # EU
        return "EU"
    # Asia spans 23:00..24:00 and 0..8
    if h >= 23 or h < 8:
        return "ASIA"
    return "OFF"

def _ema(vals, span=14):
    k = 2/(span+1)
    ema = None
    out = []
    for v in vals:
        ema = v if ema is None else (v*k + ema*(1-k))
        out.append(ema)
    return out

def _bb_width(vals, n=20, k=2.0):
    import math
    if len(vals) < n: return 0.0
    m = sum(vals[-n:]) / n
    var = sum((x-m)**2 for x in vals[-n:]) / n
    sd = math.sqrt(max(0.0, var))
    upper = m + k*sd
    lower = m - k*sd
    if m == 0: return 0.0
    return (upper - lower) / m

def classify_regime_from_series(prices: list[float]) -> Tuple[str, Dict[str, Any]]:
    if not prices or len(prices) < 30:
        return "UNKNOWN", {"reason": "insufficient_data"}
    ema = _ema(prices, span=21)
    slope = ema[-1] - ema[-5] if len(ema) >= 5 else ema[-1] - ema[0]
    bbw = _bb_width(prices, n=20, k=2.0)
    # thresholds (heuristic)
    if abs(slope) < 1e-9:
        grad = 0.0
    else:
        grad = slope / max(1e-9, prices[-1])
    if bbw > 0.08:  # high volatility band
        regime = "HIGH_VOL"
    elif grad > 0.002:
        regime = "TREND_UP"
    elif grad < -0.002:
        regime = "TREND_DOWN"
    else:
        regime = "RANGE"
    return regime, {"grad": grad, "bbw": bbw}

def regime_now(symbol: str, prices: Optional[list[float]] = None) -> Tuple[str, Dict[str, Any]]:
    """If prices not provided, caller should inject recent close prices. Returns (REGIME, meta)."""
    if prices is None or len(prices) < 30:
        return "UNKNOWN", {"reason": "no_prices"}
    return classify_regime_from_series(prices)