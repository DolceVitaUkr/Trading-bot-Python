"""
Bybit live adapter: fetch wallet & positions using ccxt.bybit (mainnet).
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import math

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