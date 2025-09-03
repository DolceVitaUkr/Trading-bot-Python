from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Query

try:
    from tradingbot.core import history_store as _hist
except Exception:
    _hist = None

router = APIRouter()

@router.get("/history/{asset}")
async def history(asset: str, mode: Optional[str] = Query(None), limit: int = Query(500)):
    """Return normalized trade history rows (paper/live) for an asset."""
    try:
        if _hist is None:
            return {"ok": False, "reason": "history_store_unavailable"}
        rows = _hist.read_history(asset, mode=mode, since=None, limit=int(limit))
        return {"ok": True, "rows": rows}
    except Exception as e:
        return {"ok": False, "reason": str(e)}

@router.get("/equity/{asset}")
async def equity(asset: str, mode: Optional[str] = Query(None), limit: int = Query(10000)):
    """Return equity curve from realized PnL of closed trades for Chart.js."""
    try:
        if _hist is None:
            return {"ok": False, "reason": "history_store_unavailable"}
        rows = _hist.read_history(asset, mode=mode, since=None, limit=int(limit))
        def _ts(r):
            return r.get("closed_at") or r.get("opened_at") or ""
        rows = sorted(rows, key=_ts)
        eq = 0.0
        series = []
        for r in rows:
            try:
                eq += float(r.get("realized_pnl") or 0.0)
            except Exception:
                pass
            series.append([_ts(r), eq])
        return {"ok": True, "series": series}
    except Exception as e:
        return {"ok": False, "reason": str(e)}