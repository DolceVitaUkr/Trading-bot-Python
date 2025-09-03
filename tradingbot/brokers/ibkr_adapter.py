"""
IBKR live adapter using ib_insync.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
from ..core.rate_limits import get_bucket
from ..core.fx_converter import convert_value

try:
    from ib_insync import IB, util
except Exception:
    IB = None

class IBKRAdapter:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, client_id: Optional[int] = None):
        if IB is None:
            raise RuntimeError("ib_insync not installed")
        self.host = host or os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("IBKR_PORT", "7497"))
        self.client_id = int(client_id or os.getenv("IBKR_CLIENT_ID", "2"))
        self.ib = IB()
        if not self.ib.isConnected():
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=5)

    def wallet(self) -> Dict[str, float]:
        try:
            summary = {s.tag: float(s.value) for s in self.ib.accountSummary()}
        except Exception:
            summary = {}
        return {
            "total": float(summary.get("NetLiquidation", 0.0)),
            "available": float(summary.get("AvailableFunds", 0.0)),
            "used": float(summary.get("InitMarginReq", 0.0)),
            "unrealized_pnl": float(summary.get("UnrealizedPnL", 0.0)),
        }

    def positions(self) -> List[Dict[str, Any]]:
        try:
            positions = self.ib.positions()
        except Exception:
            return []
        rows = []
        for p in positions:
            con = p.contract
            sym = getattr(con, "symbol", None) or getattr(con, "localSymbol", None) or getattr(con, "secType", "TICKER")
            secType = getattr(con, "secType", "").lower()
            side = "LONG" if p.position >= 0 else "SHORT"
            rows.append({
                "id": f"ibkr_{sym}_{secType}",
                "symbol": sym,
                "side": side,
                "amount": float(abs(p.position)),
                "value_usd": float(p.marketPrice * abs(p.position)) if getattr(p, "marketPrice", None) else 0.0,
                "pnl": float(getattr(p, "unrealPnl", 0.0)),
                "open_time": None,
            })
        return rows

    async def place_order(self, contract: Any, side: str, qty: float, price: Optional[float], **extra) -> Dict[str, Any]:
        await get_bucket("ibkr","trade").acquire(1)
        resp = await self._place_internal(contract, side, qty, price, **extra)
        tp = extra.get("tp") or extra.get("take_profit")
        sl = extra.get("sl") or extra.get("stop_loss")
        px = resp.get("avg_price") or resp.get("price") or price or 0.0
        notional = (float(px) * float(qty)) if px else 0.0
        return {
            "order_id": resp.get("order_id") or resp.get("permId") or resp.get("id"),
            "symbol": extra.get("symbol") or getattr(contract, "symbol", None),
            "side": side.upper(),
            "type": (extra.get("type") or "LIMIT").upper(),
            "time_in_force": (extra.get("time_in_force") or "GTC").upper(),
            "status": resp.get("status","NEW").upper(),
            "price": float(resp.get("price", price or 0.0)),
            "avg_price": float(resp.get("avg_price", px or 0.0)),
            "qty": float(qty),
            "filled_qty": float(resp.get("filled_qty", 0.0)),
            "tp_price": float(tp) if tp is not None else None,
            "sl_price": float(sl) if sl is not None else None,
            "notional_usd": float(notional),
            "ts": resp.get("ts"),
            "exchange_ts": resp.get("exchange_ts"),
            "client_order_id": extra.get("client_order_id"),
            "parent_order_id": resp.get("parent_order_id"),
            "bracket_id": resp.get("bracket_id"),
        }

    async def open_orders(self) -> List[Dict[str, Any]]:
        await get_bucket("ibkr","read").acquire(1)
        data = await self._fetch_open_orders()
        out = []
        for o in data:
            out.append({
                "order_id": o.get("orderId") or o.get("permId") or o.get("id"),
                "symbol": o.get("symbol"),
                "side": o.get("action","").upper(),
                "type": o.get("orderType","").upper(),
                "time_in_force": o.get("tif","GTC").upper(),
                "status": o.get("status","NEW").upper(),
                "price": float(o.get("lmtPrice", o.get("price", 0.0)) or 0.0),
                "avg_price": float(o.get("avgFillPrice", 0.0)),
                "qty": float(o.get("totalQuantity", 0.0)),
                "filled_qty": float(o.get("filled", 0.0)),
                "tp_price": float(o.get("takeProfitPrice", 0.0)) or None,
                "sl_price": float(o.get("stopLossPrice", 0.0)) or None,
                "notional_usd": None,
                "ts": o.get("transmitTime") or o.get("ts")
            })
        return out

    async def positions(self) -> List[Dict[str, Any]]:
        await get_bucket("ibkr","read").acquire(1)
        pos = await self._fetch_positions()
        out = []
        for p in pos:
            px = float(p.get("avgCost", p.get("avg_price", 0.0)) or 0.0)
            qty = float(p.get("position", p.get("qty", 0.0)) or 0.0)
            notional = px * abs(qty) if px else None
            out.append({
                "symbol": p.get("symbol"),
                "side": "LONG" if qty >= 0 else "SHORT",
                "qty": qty,
                "avg_price": px,
                "unrealized_pnl": float(p.get("unrealizedPNL", p.get("unrealized_pnl", 0.0)) or 0.0),
                "leverage": float(p.get("leverage", 0.0) or 0.0),
                "notional_usd": notional,
                "ts": p.get("ts")
            })
        return out