"""
Virtual execution engine (paper trading) that uses live prices where possible.
Returns broker-like response dicts: {"success": True, "order_id": "<id>"} or {"success": False, "reason": "..."}
"""
from __future__ import annotations
import uuid
from typing import Optional, Tuple

from . import paper_state

def _get_best_prices(bybit_client, ibkr_client, symbol: str, asset_type: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Try multiple ways to get bid/ask/last from injected clients.
    """
    bid = ask = last = None
    # Bybit attempts
    c = bybit_client
    if c:
        for attr in ("get_best_bid_ask", "best_bid_ask", "get_orderbook", "get_ticker", "ticker"):
            fn = getattr(c, attr, None)
            if callable(fn):
                try:
                    ob = fn(symbol)  # allow various shapes
                    # normalize common shapes
                    if isinstance(ob, dict):
                        bid = bid or ob.get("bid") or ob.get("bestBid") or ob.get("b")
                        ask = ask or ob.get("ask") or ob.get("bestAsk") or ob.get("a")
                        last = last or ob.get("last") or ob.get("lastPrice") or ob.get("l")
                        if bid or ask or last:
                            break
                    elif isinstance(ob, (list, tuple)) and ob:
                        try:
                            # assume [(price,size,isBid), ...]
                            bids = [x for x in ob if len(x) >= 2 and (len(x) < 3 or x[2] is True)]
                            asks = [x for x in ob if len(x) >= 2 and (len(x) < 3 or x[2] is False)]
                            if bids: bid = bid or float(bids[0][0])
                            if asks: ask = ask or float(asks[0][0])
                            break
                        except Exception:
                            pass
                except Exception:
                    pass
    # IBKR attempts (very minimal)
    c = ibkr_client
    if c and (bid is None or ask is None):
        for attr in ("get_best_bid_ask", "best_bid_ask", "get_snapshot"):
            fn = getattr(c, attr, None)
            if callable(fn):
                try:
                    ob = fn(symbol)
                    if isinstance(ob, dict):
                        bid = bid or ob.get("bid")
                        ask = ask or ob.get("ask")
                        last = last or ob.get("last")
                        break
                except Exception:
                    pass
    return (bid, ask, last)

def simulate_order(asset_ui_key: str, asset_type: str, symbol: str, side: str, quantity: float, 
                  order_type: str = "market", price: float = None, bybit_client=None, ibkr_client=None):
    bid, ask, last = _get_best_prices(bybit_client, ibkr_client, symbol, asset_type)
    if order_type == "market":
        if side.lower() == "buy":
            fill_price = (ask or last or bid)
        else:
            fill_price = (bid or last or ask)
    else:  # limit
        if price is None:
            return {"success": False, "reason": "limit price required"}
        # basic marketability check
        mkt_bid, mkt_ask = bid, ask
        if side.lower() == "buy":
            if mkt_ask and price >= mkt_ask:
                fill_price = min(price, mkt_ask)
            else:
                return {"success": False, "reason": "limit not marketable"}
        else:
            if mkt_bid and price <= mkt_bid:
                fill_price = max(price, mkt_bid)
            else:
                return {"success": False, "reason": "limit not marketable"}
    if fill_price is None:
        return {"success": False, "reason": "no market price available"}

    # simple taker fee assumption (can be refined per venue)
    fee_rate = 0.0006
    fee = fee_rate * float(quantity) * float(fill_price)

    # apply to paper state
    paper_state.apply_fill(asset_ui_key, symbol, side, float(quantity), float(fill_price), float(fee))

    return {"success": True, "order_id": f"paper_{uuid.uuid4().hex[:10]}", "fill_price": float(fill_price), "fee": float(fee)}