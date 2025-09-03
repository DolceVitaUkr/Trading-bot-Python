"""
IBKR live adapter using ib_insync.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os

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