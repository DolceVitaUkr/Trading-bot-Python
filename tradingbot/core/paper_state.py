"""
Paper state store: positions, wallet, and trade history for paper mode per asset.
Files under tradingbot/state/paper/:
  - wallet_{asset}.json
  - positions_{asset}.json
  - trades_{asset}.jsonl
"""
from __future__ import annotations
import json
import pathlib
import uuid
from datetime import datetime
from typing import Dict, Any, List

BASE = pathlib.Path("tradingbot/state/paper")
BASE.mkdir(parents=True, exist_ok=True)

def _wallet_path(asset: str) -> pathlib.Path:
    return BASE / f"wallet_{asset}.json"

def _positions_path(asset: str) -> pathlib.Path:
    return BASE / f"positions_{asset}.json"

def _trades_path(asset: str) -> pathlib.Path:
    return BASE / f"trades_{asset}.jsonl"

def _read_json(path: pathlib.Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def _write_json(path: pathlib.Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def get_wallet(asset: str) -> Dict[str, float]:
    w = _read_json(_wallet_path(asset), {"total": 0.0, "available": 0.0, "used": 0.0, "unrealized_pnl": 0.0})
    # keep fields
    for k in ("total", "available", "used", "unrealized_pnl"):
        w.setdefault(k, 0.0)
    return w

def set_wallet(asset: str, w: Dict[str, float]) -> None:
    _write_json(_wallet_path(asset), w)

def list_positions(asset: str) -> List[Dict[str, Any]]:
    return _read_json(_positions_path(asset), [])

def save_positions(asset: str, rows: List[Dict[str, Any]]) -> None:
    _write_json(_positions_path(asset), rows)

def append_trade(asset: str, trade: Dict[str, Any]) -> None:
    tp = _trades_path(asset)
    tp.parent.mkdir(parents=True, exist_ok=True)
    with tp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trade) + "\n")

def apply_fill(asset: str, symbol: str, side: str, qty: float, price: float, fee: float) -> Dict[str, Any]:
    """
    Very lightweight netting behaviour:
      - BUY: increases long exposure (aggregated per symbol)
      - SELL: decreases long exposure; if goes negative we track as short (for futures/forex)
    Wallet: total stays as baseline; we move 'used' up by notional for simplicity.
    """
    now = datetime.utcnow().isoformat() + "Z"
    positions = list_positions(asset)
    # find existing position by symbol
    pos = next((p for p in positions if p.get("symbol") == symbol), None)
    notional = float(qty) * float(price)
    if pos is None:
        pos = {
            "id": f"pos_{uuid.uuid4().hex[:8]}",
            "symbol": symbol,
            "side": "LONG" if side.lower() == "buy" else "SHORT",
            "amount": float(qty) if side.lower() == "buy" else -float(qty),
            "value_usd": notional if side.lower() == "buy" else -notional,
            "pnl": 0.0,
            "open_time": now
        }
        positions.append(pos)
    else:
        # adjust position
        delta = float(qty) if side.lower() == "buy" else -float(qty)
        pos["amount"] = float(pos.get("amount", 0.0)) + delta
        pos["value_usd"] = float(pos.get("value_usd", 0.0)) + (notional if side.lower() == "buy" else -notional)
        pos["side"] = "LONG" if pos["amount"] >= 0 else "SHORT"
        if not pos.get("open_time"):
            pos["open_time"] = now
    save_positions(asset, positions)

    # naive wallet update
    w = get_wallet(asset)
    w["used"] = float(w.get("used", 0.0)) + abs(notional) * 0.01  # placeholder exposure
    # keep total/available if user wants; this does not change cash reserve in this simple model
    set_wallet(asset, w)

    trade = {
        "id": f"paper_{uuid.uuid4().hex[:8]}",
        "ts_open": now,
        "ts_close": now,
        "symbol": symbol,
        "side": side.upper(),
        "amount": float(qty),
        "avg_price": float(price),
        "pnl": 0.0,
        "fees": float(fee),
        "mode": "paper"
    }
    append_trade(asset, trade)
    return trade