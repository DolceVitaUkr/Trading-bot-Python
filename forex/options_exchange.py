# options/options_exchange.py

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Literal


@dataclass
class OptionContract:
    instrument_id: str      # e.g., "EURUSD-2025-09-19-C-1.1200"
    symbol: str             # underlying
    expiry: str             # YYYY-MM-DD
    right: Literal["C", "P"]
    strike: Decimal
    bid: Decimal
    ask: Decimal


class OptionsExchange:
    """
    Minimal options adapter (stub).
    - Simulated chain with static greeks/prices derived from underlying spot.
    - Async API so you can wire it later to a real venue.

    Public async methods:
      - get_chain_async(symbol, expiry) -> list[dict]
      - create_order_async(symbol, instrument_id, side, qty, price?) -> dict
    """

    def __init__(self):
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._oid = 0
        self._last_spot: Dict[str, Decimal] = {}

    def _new_id(self) -> str:
        self._oid += 1
        return f"opt_{self._oid}"

    async def _spot(self, symbol: str) -> Decimal:
        # dummy last spot (you can inject a real provider later)
        return self._last_spot.get(symbol, Decimal("1.1000"))

    async def get_chain_async(self, symbol: str, expiry: str) -> List[Dict[str, Any]]:
        """
        Return a simulated options chain around spot: 5 strikes up/down for calls/puts.
        """
        spot = await self._spot(symbol)
        # Make strikes around spot in 50-pip increments (for FX)
        increments = [Decimal(i) * Decimal("0.0050") for i in range(-5, 6)]
        strikes = [spot + inc for inc in increments]

        chain: List[Dict[str, Any]] = []
        for strike in strikes:
            # toy pricing: bid/ask = intrinsic + small premium
            for right in ("C", "P"):
                intrinsic = max(Decimal("0"), (spot - strike)) if right == "C" else max(Decimal("0"), (strike - spot))
                premium = Decimal("0.0015")
                bid = intrinsic + premium
                ask = bid + Decimal("0.0005")
                iid = f"{symbol}-{expiry}-{right}-{strike:.4f}"
                chain.append(
                    {
                        "instrument_id": iid,
                        "symbol": symbol,
                        "expiry": expiry,
                        "right": right,
                        "strike": float(strike),
                        "bid": float(bid),
                        "ask": float(ask),
                    }
                )
        return chain

    async def create_order_async(
        self,
        *,
        symbol: str,
        instrument_id: str,
        side: Literal["buy", "sell"],
        qty: Decimal,
        price: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """
        Simulated options order: fills at provided price or midpoint of bid/ask from chain lookup.
        """
        # Parse instrument for a quick sanity check
        try:
            _, expiry, right, strike_txt = instrument_id.split("-")
            _ = (expiry, right, strike_txt)  # noqa
        except Exception:
            pass

        fill_price = Decimal(str(price)) if price is not None else Decimal("0.0020")
        oid = self._new_id()
        ts = int(time.time() * 1000)
        order = {
            "id": oid,
            "status": "filled",
            "symbol": symbol,
            "instrument_id": instrument_id,
            "side": side,
            "qty": float(qty),
            "price": float(fill_price),
            "time": ts,
        }
        self._orders[oid] = order
        await asyncio.sleep(0)  # keep it async
        return order
