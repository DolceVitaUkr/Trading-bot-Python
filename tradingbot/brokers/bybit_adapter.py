"""
Bybit live adapter: fetch wallet & positions using ccxt.bybit (mainnet).
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import math
from ..core.rate_limits import get_bucket
from ..core.fx_converter import convert_value

try:
    import ccxt
except Exception:
    ccxt = None

class BybitAdapter:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        if ccxt is None:
            raise RuntimeError("ccxt not installed")
        self.ex = ccxt.bybit({
            "apiKey": api_key or os.getenv("BYBIT_API_KEY"),
            "secret": api_secret or os.getenv("BYBIT_API_SECRET"),
            "enableRateLimit": True,
            "options": {"defaultType": "spot"}  # we will query both spot & contract
        })
        self.ex.set_sandbox_mode(False)  # mainnet as requested

    def wallet(self) -> Dict[str, float]:
        # Aggregate balances to USDT (approx). ccxt returns free/used/total per currency
        b = self.ex.fetch_balance(params={})
        total = float(b.get("USDT", {}).get("total") or b.get("total", {}).get("USDT") or 0.0)
        free = float(b.get("USDT", {}).get("free") or b.get("free", {}).get("USDT") or 0.0)
        used = float(b.get("USDT", {}).get("used") or b.get("used", {}).get("USDT") or (total - free))
        # Basic unrealized PnL from positions if available
        upnl = 0.0
        try:
            for p in self.positions_contract():
                upnl += float(p.get("pnl") or 0.0)
        except Exception:
            pass
        return {"total": total, "available": free, "used": used, "unrealized_pnl": upnl}

    def positions_spot(self) -> List[Dict[str, Any]]:
        # ccxt has no unified "spot positions"; we approximate from balances > 0 minus USDT
        rows = []
        b = self.ex.fetch_balance(params={})
        for sym, info in (b.get("total") or {}).items():
            try:
                amt = float(info)
            except Exception:
                continue
            if sym in ("USDT", "USD", "USDC") or amt <= 0:
                continue
            rows.append({
                "id": f"bybit_spot_{sym}",
                "symbol": f"{sym}/USDT",
                "side": "LONG",
                "amount": amt,
                "value_usd": None,  # UI can price via ticker if needed
                "pnl": 0.0,
                "open_time": None,
            })
        return rows

    def positions_contract(self) -> List[Dict[str, Any]]:
        try:
            data = self.ex.fetch_positions()
        except Exception:
            return []
        rows = []
        for p in data:
            amt = float(p.get("contracts") or p.get("contractSize") or 0.0)
            if amt == 0.0 and float(p.get("info", {}).get("size") or 0) == 0:
                continue
            side = "LONG" if (p.get("side") or "").lower() in ("long", "buy") else "SHORT"
            rows.append({
                "id": p.get("id") or p.get("symbol") or "bybit_pos",
                "symbol": p.get("symbol"),
                "side": side,
                "amount": float(p.get("contracts") or p.get("info", {}).get("size") or 0.0),
                "value_usd": float(p.get("notional") or p.get("info", {}).get("positionValue") or 0.0),
                "pnl": float(p.get("unrealizedPnl") or p.get("info", {}).get("unrealisedPnl") or 0.0),
                "open_time": None,
            })
        return rows

    def positions(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "spot": self.positions_spot(),
            "futures": self.positions_contract()
        }

    async def submit_order(self, symbol: str, side: str, qty: float, price: Optional[float], **extra) -> Dict[str, Any]:
        """Submit order to Bybit. Wrap existing internal submit; normalize fields."""
        await get_bucket("bybit","trade").acquire(1)
        # Existing internal call (example): self._client.create_order(...)
        resp = await self._submit_internal(symbol, side, qty, price, **extra)  # must exist in your code
        tp = extra.get("tp") or extra.get("take_profit")
        sl = extra.get("sl") or extra.get("stop_loss")
        px = resp.get("avg_price") or resp.get("price") or price or 0.0
        notional = (float(px) * float(qty)) if px else 0.0
        return {
            "order_id": resp.get("order_id") or resp.get("id"),
            "symbol": symbol,
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
            "base_ccy": extra.get("base_ccy"),
            "quote_ccy": extra.get("quote_ccy","USDT"),
            "ts": resp.get("ts") or resp.get("transactTime"),
            "exchange_ts": resp.get("exchange_ts"),
            "client_order_id": extra.get("client_order_id"),
            "parent_order_id": resp.get("parent_order_id"),
            "bracket_id": resp.get("bracket_id"),
        }

    async def open_orders(self) -> List[Dict[str, Any]]:
        await get_bucket("bybit","read").acquire(1)
        data = await self._fetch_open_orders()
        out = []
        for o in data:
            out.append({
                "order_id": o.get("orderId") or o.get("order_id"),
                "symbol": o.get("symbol"),
                "side": o.get("side","").upper(),
                "type": o.get("type","").upper(),
                "time_in_force": o.get("timeInForce","GTC").upper(),
                "status": o.get("status","NEW").upper(),
                "price": float(o.get("price", 0.0)),
                "avg_price": float(o.get("avgPrice", 0.0)),
                "qty": float(o.get("qty", o.get("origQty", 0.0))),
                "filled_qty": float(o.get("executedQty", 0.0)),
                "tp_price": float(o.get("takeProfit", 0.0)) or None,
                "sl_price": float(o.get("stopLoss", 0.0)) or None,
                "notional_usd": float(o.get("price", 0.0)) * float(o.get("origQty", 0.0)) if o.get("price") else None,
                "ts": o.get("time") or o.get("ts")
            })
        return out

    async def positions(self) -> List[Dict[str, Any]]:
        await get_bucket("bybit","read").acquire(1)
        pos = await self._fetch_positions()
        out = []
        for p in pos:
            px = float(p.get("avgPrice", p.get("entryPrice", 0.0)) or 0.0)
            qty = float(p.get("qty", p.get("size", 0.0)) or 0.0)
            notional = px * abs(qty) if px else None
            out.append({
                "symbol": p.get("symbol"),
                "side": "LONG" if qty >= 0 else "SHORT",
                "qty": qty,
                "avg_price": px,
                "unrealized_pnl": float(p.get("unrealizedPnl", 0.0)),
                "leverage": float(p.get("leverage", 0.0) or 0.0),
                "notional_usd": notional,
                "ts": p.get("updateTime") or p.get("ts")
            })
        return out