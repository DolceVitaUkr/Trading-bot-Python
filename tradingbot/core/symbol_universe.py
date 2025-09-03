from __future__ import annotations
from typing import Dict, List, Any
from .rate_limits import get_bucket

async def bybit_top_symbols(adapter, top24_limit: int = 20, top2h_limit: int = 10) -> Dict[str, List[str]]:
    """Lightweight universe selection using adapter calls only.

    - Top 24h: from adapter.tickers_linear_usdt() (USDT perps)

    - Top 2h: compute over 5m klines for last 2h only for the Top-50 from 24h
    """
    await get_bucket("bybit","read").acquire(1)
    tickers = await adapter.tickers_linear_usdt()  # adapter wrapper of /v5/market/tickers
    ranked = sorted([t for t in tickers if t.get("symbol","").endswith("USDT")],
                    key=lambda x: float(x.get("turnover24h", x.get("volume24h", 0.0))), reverse=True)
    top24 = [t["symbol"] for t in ranked[:max(1, top24_limit*2)]]  # fetch extra to refine
    candidates = top24[:50]
    vol2h: List[tuple[str, float]] = []
    for sym in candidates:
        await get_bucket("bybit","read").acquire(1)
        kl = await adapter.klines_5m(sym, lookback_minutes=120)  # wrapper of /v5/market/kline
        s = 0.0
        for row in kl:
            s += float(row.get("volume", 0.0))
        vol2h.append((sym, s))
    top2h = [s for s, _ in sorted([(s, v) for s, v in vol2h], key=lambda x: x[1], reverse=True)[:top2h_limit]]
    return {"top24h": top24[:top24_limit], "top2h": top2h}