# forex/forex_strategy.py

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Optional, Dict, Any, List

from modules.technical_indicators import sma, rsi  # using function API per your module


Side = Literal["long", "short"]


@dataclass
class TradeDecision:
    symbol: str
    side: Side
    confidence: float
    entry: Optional[Decimal]
    ttl_s: int
    attach_tp: Optional[Decimal]
    attach_sl: Optional[Decimal]
    is_exploration: bool = False


class ForexStrategyAdapter:
    """
    Tiny example strategy:
      - Uses 20/50 SMA cross + RSI(14) filter.
      - Returns a TradeDecision or None.

    This is intentionally simple—just enough to exercise the pipeline.
    """

    def __init__(self, min_confidence: float = 0.55):
        self.min_conf = float(min_confidence)

    def decide(self, bars: "pandas.DataFrame") -> Optional[TradeDecision]:
        """
        bars: DataFrame with columns ['open','high','low','close','volume']
        index: datetime or int timestamps (ignored here)
        """
        try:
            close = bars["close"].tolist()
            if len(close) < 60:
                return None

            sma20 = sma(close, 20)
            sma50 = sma(close, 50)
            rsi14 = rsi(close, 14)

            if sma20 is None or sma50 is None or rsi14 is None:
                return None

            px = Decimal(str(close[-1]))
            conf: float = 0.5
            side: Optional[Side] = None

            if sma20 > sma50 and 45 <= rsi14 <= 70:
                side = "long"
                conf = 0.6
            elif sma20 < sma50 and 30 <= rsi14 <= 55:
                side = "short"
                conf = 0.6

            if side is None or conf < self.min_conf:
                return None

            # naive TP/SL: 50–80 pips depending on side (for demo)
            pip = Decimal("0.0001")
            tp_pips = Decimal("0.0080") if side == "long" else Decimal("0.0080")
            sl_pips = Decimal("0.0050")

            attach_tp = (px + tp_pips) if side == "long" else (px - tp_pips)
            attach_sl = (px - sl_pips) if side == "long" else (px + sl_pips)

            return TradeDecision(
                symbol="N/A",  # fill by caller
                side=side,
                confidence=conf,
                entry=px,
                ttl_s=60 * 30,
                attach_tp=attach_tp,
                attach_sl=attach_sl,
                is_exploration=False,
            )
        except Exception:
            return None
