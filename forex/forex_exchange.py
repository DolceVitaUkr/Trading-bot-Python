# forex/forex_exchange.py

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional

try:
    import config  # optional; sane fallbacks below
except Exception:  # pragma: no cover
    config = None


Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]


@dataclass
class _FxOrder:
    id: str
    symbol: str
    side: Side
    qty: Decimal
    order_type: OrderType
    price: Optional[Decimal]
    reduce_only: bool
    ts_ms: int
    status: str = "open"


@dataclass
class _FxPosition:
    symbol: str
    side: Literal["long", "short"]
    qty: Decimal
    entry_price: Decimal
    ts_ms: int


class ForexExchange:
    """
    Minimal async Forex exchange adapter.

    - **Simulation-first** (default): in-memory cash (USD), positions, and fills.
    - Live hookups can be added later; method signatures are stable.

    Public async API (matches Modules.md sketch):
      - get_price_async(symbol) -> Decimal
      - get_balance_async(asset) -> Decimal
      - create_order_async(...) -> dict
      - cancel_order_async(symbol, order_id) -> dict
      - get_open_positions_async(symbol?) -> list[dict]
      - reconcile_async() -> None
    """

    def __init__(self, broker: str = "sim", testnet: bool = True):
        self.broker = broker
        self.testnet = bool(testnet)

        # Sim wallet & state
        self._cash_usd: Decimal = Decimal(
            str(getattr(config, "FOREX_SIM_START_BALANCE", 10_000))
        )
        self._positions: Dict[str, _FxPosition] = {}
        self._orders: Dict[str, _FxOrder] = {}
        self._last_price: Dict[str, Decimal] = {}

        # Price hook (can be fed by forex_data adapter)
        self._price_provider: Optional[callable] = None

        # Simple ID counter
        self._oid = 0

    # ─────────────────────────── wiring ─────────────────────────── #

    def bind_price_provider(self, func: callable):
        """
        Attach an async price provider: async def f(symbol) -> Decimal
        """
        self._price_provider = func

    # ─────────────────────────── helpers ─────────────────────────── #

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def _new_id(self) -> str:
        self._oid += 1
        return f"fx_{self._oid}"

    # ─────────────────────────── public async api ─────────────────────────── #

    async def get_price_async(self, symbol: str) -> Decimal:
        """
        Return last known or provider price. In sim, if unknown, seed at 1.0000.
        """
        if self._price_provider:
            px = Decimal(str(await self._price_provider(symbol)))
            self._last_price[symbol] = px
            return px
        # fallback to cache or seed
        return self._last_price.get(symbol, Decimal("1.0000"))

    async def get_balance_async(self, asset: str = "USD") -> Decimal:
        """
        Only USD tracking for sim; live would query broker.
        """
        if asset.upper() != "USD":
            return Decimal("0")
        return self._cash_usd

    async def create_order_async(
        self,
        *,
        symbol: str,
        side: Side,
        qty: Decimal,
        order_type: OrderType = "market",
        price: Optional[Decimal] = None,
        tp: Optional[Decimal] = None,
        sl: Optional[Decimal] = None,
        reduce_only: bool = False,
        client_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sim execution model (no leverage by default):
          - 'buy' opens/adds LONG, 'sell' opens/adds SHORT (netted per symbol).
          - reduce_only closes position in the opposite direction up to qty.
          - market fills at current price; limit fills at given price (sim).
        """
        sym = symbol.upper()
        now = self._now_ms()
        px = price if price is not None else await self.get_price_async(sym)
        px = Decimal(str(px))

        oid = self._new_id()
        order = _FxOrder(
            id=oid,
            symbol=sym,
            side=side,
            qty=Decimal(str(qty)),
            order_type=order_type,
            price=px if order_type == "limit" else px,
            reduce_only=bool(reduce_only),
            ts_ms=now,
            status="open",
        )

        pos = self._positions.get(sym)
        opened = False
        closed_qty = Decimal("0")
        pnl = Decimal("0")

        if order.reduce_only:
            if not pos or pos.qty == 0:
                order.status = "no_position"
            else:
                # close in direction of position
                close_qty = min(order.qty, pos.qty)
                if pos.side == "long":
                    pnl = close_qty * (px - pos.entry_price)
                else:
                    pnl = close_qty * (pos.entry_price - px)

                # cash settles to USD
                self._cash_usd += pnl
                pos.qty -= close_qty
                closed_qty = close_qty
                order.status = "closed_partial" if pos.qty > 0 else "closed"
                if pos.qty <= 0:
                    self._positions.pop(sym, None)
        else:
            # opening / adding
            if side == "buy":
                if pos and pos.side == "long":
                    new_qty = pos.qty + order.qty
                    pos.entry_price = (pos.entry_price * pos.qty +
                                       px * order.qty) / new_qty
                    pos.qty = new_qty
                elif pos and pos.side == "short":
                    # netting against short
                    close_qty = min(order.qty, pos.qty)
                    pnl = close_qty * (pos.entry_price - px)
                    self._cash_usd += pnl
                    pos.qty -= close_qty
                    if pos.qty <= 0:
                        self._positions.pop(sym, None)
                    # residual becomes new long
                    residual = order.qty - close_qty
                    if residual > 0:
                        self._positions[sym] = _FxPosition(
                            sym, "long", residual, px, now)
                else:
                    self._positions[sym] = _FxPosition(
                        sym, "long", order.qty, px, now)
                opened = True
            else:  # sell → short/open or net against long
                if pos and pos.side == "short":
                    new_qty = pos.qty + order.qty
                    pos.entry_price = (pos.entry_price * pos.qty +
                                       px * order.qty) / new_qty
                    pos.qty = new_qty
                elif pos and pos.side == "long":
                    close_qty = min(order.qty, pos.qty)
                    pnl = close_qty * (px - pos.entry_price)
                    self._cash_usd += pnl
                    pos.qty -= close_qty
                    if pos.qty <= 0:
                        self._positions.pop(sym, None)
                    residual = order.qty - close_qty
                    if residual > 0:
                        self._positions[sym] = _FxPosition(
                            sym, "short", residual, px, now)
                else:
                    self._positions[sym] = _FxPosition(
                        sym, "short", order.qty, px, now)
                opened = True

            order.status = "open" if opened else order.status

        self._orders[oid] = order
        return {
            "id": oid,
            "status": order.status,
            "symbol": sym,
            "side": side,
            "amount": float(order.qty),
            "price": float(px),
            "pnl": float(pnl),
            "closed_qty": float(closed_qty),
            "time": now,
            "tp": float(tp) if tp is not None else None,
            "sl": float(sl) if sl is not None else None,
            "client_id": client_id,
        }

    async def cancel_order_async(self,
                                 symbol: str,
                                 order_id: str) -> Dict[str, Any]:
        o = self._orders.get(order_id)
        if not o:
            return {"id": order_id, "status": "not_found"}
        # In sim we don't keep resting orders; mark canceled
        o.status = "canceled"
        return {"id": order_id, "status": "canceled"}

    async def get_open_positions_async(self,
                                       symbol: Optional[str] = None
                                       ) -> List[Dict[str, Any]]:
        if symbol:
            pos = self._positions.get(symbol.upper())
            return [pos.__dict__] if pos else []
        return [p.__dict__ for p in self._positions.values()]

    async def reconcile_async(self) -> None:
        """
        No-op for sim. Live should re-attach SL/TP & refresh positions.
        """
        await asyncio.sleep(0)
