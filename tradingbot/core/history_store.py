"""
Unified trade history reader.
Currently sources:
  - Paper: tradingbot/state/paper/trades_{asset}.jsonl
Future:
  - Live: Bybit (ccxt) and IBKR (ib_insync) executions
"""
from __future__ import annotations
import json, pathlib
from datetime import datetime
from typing import Dict, Any, List, Optional

PAPER_DIR = pathlib.Path("tradingbot/state/paper")
PAPER_DIR.mkdir(parents=True, exist_ok=True)

def _paper_path(asset: str) -> pathlib.Path:
    return PAPER_DIR / f"trades_{asset}.jsonl"

def _parse_iso(ts: str) -> datetime:
    t = ts[:-1] if ts and ts.endswith("Z") else ts
    return datetime.fromisoformat(t) if t else datetime.min

def _read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def _norm_row(row: Dict[str, Any], mode_hint: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": row.get("id") or row.get("order_id"),
        "ts_open": row.get("ts_open") or row.get("timestamp") or row.get("time"),
        "ts_close": row.get("ts_close") or row.get("close_time"),
        "symbol": row.get("symbol"),
        "side": (row.get("side") or "").upper() if row.get("side") else None,
        "amount": row.get("amount") or row.get("qty") or row.get("size"),
        "avg_price": row.get("avg_price") or row.get("price"),
        "pnl": row.get("pnl") or 0.0,
        "fees": row.get("fees") or row.get("fee") or 0.0,
        "mode": row.get("mode") or mode_hint,
    }

def read_history(asset: str, mode: Optional[str] = None, since: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
    mode = (mode or "").lower().strip() or None
    # Paper
    paper_rows = _read_jsonl(_paper_path(asset))
    paper_rows = [_norm_row(r, "paper") for r in paper_rows]
    # TODO: Live brokers can be added later
    live_rows: List[Dict[str, Any]] = []
    rows = paper_rows + live_rows
    # Filter by mode
    if mode in ("paper","live"):
        rows = [r for r in rows if (r.get("mode")==mode)]
    # Filter since
    if since:
        try:
            cutoff = _parse_iso(since)
            rows = [r for r in rows if (_parse_iso(r.get("ts_open")) >= cutoff)]
        except Exception:
            pass
    # Sort desc by ts_open
    rows.sort(key=lambda r: _parse_iso(r.get("ts_open")), reverse=True)
    if limit and limit > 0:
        rows = rows[:int(limit)]
    return rows

# Import validation_manager lazily to avoid circular imports
try:
    from tradingbot.training import validation_manager
except Exception:
    validation_manager = None

def append_paper(asset: str, trade_row: dict) -> None:
    """Append a normalized paper trade and notify validator if closed."""
    with _paper_path(asset).open("a", encoding="utf-8") as f:
        f.write(json.dumps(trade_row) + "\n")
    try:
        if validation_manager and (trade_row.get("status","closed").lower() in ("closed","filled","exit") or trade_row.get("realized_pnl") is not None):
            sid = str(trade_row.get("strategy_id") or "")
            if sid:
                validation_manager.on_trade_closed(asset_ui_key=asset, strategy_id=sid, mode="paper", trade_row=trade_row)
    except Exception:
        pass