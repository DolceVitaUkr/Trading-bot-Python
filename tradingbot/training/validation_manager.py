"""
Validation Manager v1
- Computes rolling metrics per strategy (Sharpe, MaxDD, WinRate, Expectancy)
- Promotes Paper -> Candidate -> Live based on gates
- Degrades Live on persistent underperformance
- Persists status to tradingbot/state/strategies/strategies.json

This module is tolerant to missing files and can be imported anywhere.
"""
from __future__ import annotations
import json, pathlib, math
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

STATE_DIR = pathlib.Path("tradingbot/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
STRAT_PATH = STATE_DIR / "strategies" / "strategies.json"
STRAT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- thresholds ---
PAPER_MIN_TRADES = 100
PAPER_SHARPE_MIN = 1.0
PAPER_MAXDD_PCT_MAX = 0.08  # 8%
PAPER_WIN_MIN = 0.52

LIVE_MIN_TRADES = 50
LIVE_SHARPE_MIN = 0.8
LIVE_MAXDD_PCT_MAX = 0.12  # 12%
LIVE_WIN_MIN = 0.50

def _load_json(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_json(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _get_strat_entry(asset_ui_key: str, strategy_id: str) -> dict:
    m = _load_json(STRAT_PATH)
    m.setdefault(asset_ui_key, {})
    s = m[asset_ui_key].get(strategy_id) or {}
    return s

def _put_strat_entry(asset_ui_key: str, strategy_id: str, entry: dict) -> None:
    m = _load_json(STRAT_PATH)
    m.setdefault(asset_ui_key, {})
    m[asset_ui_key][strategy_id] = entry
    _save_json(STRAT_PATH, m)

# importing read_history lazily to avoid circulars
def _read_history(asset_ui_key: str, mode: Optional[str]) -> List[Dict[str, Any]]:
    try:
        from tradingbot.core import history_store
    except Exception:
        return []
    if not hasattr(history_store, "read_history"):
        return []
    return history_store.read_history(asset_ui_key, mode=mode, since=None, limit=100000)

def _closed_rows(rows: List[Dict[str, Any]], strategy_id: str) -> List[Dict[str, Any]]:
    out = []
    for r in rows or []:
        sid = str(r.get("strategy_id") or "")
        if sid != str(strategy_id):
            continue
        status = (r.get("status") or "closed").lower()
        if status in ("closed","filled","exit") or (r.get("realized_pnl") is not None):
            out.append(r)
    return out

def _pnl_series(rows: List[Dict[str, Any]]) -> List[float]:
    pnl = []
    for r in rows:
        try:
            pnl.append(float(r.get("realized_pnl") or 0.0))
        except Exception:
            pnl.append(0.0)
    return pnl

def _equity_curve(pnl: List[float]) -> List[float]:
    eq = []
    s = 0.0
    for x in pnl:
        s += x
        eq.append(s)
    return eq

def _max_drawdown_pct(eq: List[float]) -> float:
    if not eq: return 0.0
    peak = eq[0]
    max_dd = 0.0
    for x in eq:
        peak = max(peak, x)
        dd = peak - x  # drawdown in USD units
        if peak != 0:
            dd_pct = dd / abs(peak) if peak != 0 else 0.0
        else:
            dd_pct = 0.0
        max_dd = max(max_dd, dd_pct)
    return float(max_dd)

def _sharpe(pnl: List[float]) -> float:
    if not pnl or len(pnl) < 2:
        return 0.0
    mu = sum(pnl)/len(pnl)
    var = sum((x-mu)**2 for x in pnl) / (len(pnl)-1)
    sd = math.sqrt(max(1e-12, var))
    # per-trade Sharpe; scale by sqrt(n) for comparability
    return float((mu / sd) * math.sqrt(len(pnl))) if sd > 0 else 0.0

def _win_rate(pnl: List[float]) -> float:
    if not pnl: return 0.0
    wins = sum(1 for x in pnl if x > 0)
    return float(wins/len(pnl))

def _expectancy(pnl: List[float]) -> float:
    if not pnl: return 0.0
    return float(sum(pnl)/len(pnl))

def _metrics(rows: List[Dict[str, Any]]) -> dict:
    pnl = _pnl_series(rows)
    eq = _equity_curve(pnl)
    return {
        "trades": len(rows),
        "win_rate": _win_rate(pnl),
        "sharpe": _sharpe(pnl),
        "max_dd_pct": _max_drawdown_pct(eq),
        "expectancy": _expectancy(pnl),
        "last_closed_at": (rows[0].get("closed_at") if rows else None)
    }

def evaluate(asset_ui_key: str, strategy_id: str) -> dict:
    paper_rows = _closed_rows(_read_history(asset_ui_key, "paper"), strategy_id)
    live_rows  = _closed_rows(_read_history(asset_ui_key, "live"), strategy_id)
    paper_m = _metrics(paper_rows)
    live_m  = _metrics(live_rows)

    entry = _get_strat_entry(asset_ui_key, strategy_id)
    status = entry.get("approval_status") or "paper"

    # Promotion logic
    if status in ("paper","review","candidate"):
        if paper_m["trades"] >= PAPER_MIN_TRADES and paper_m["sharpe"] >= PAPER_SHARPE_MIN and paper_m["max_dd_pct"] <= PAPER_MAXDD_PCT_MAX and paper_m["win_rate"] >= PAPER_WIN_MIN:
            status = "candidate"
    if status in ("candidate","live"):
        # require at least LIVE_MIN_TRADES to consider "live-ok"
        if live_m["trades"] >= LIVE_MIN_TRADES and live_m["sharpe"] >= LIVE_SHARPE_MIN and live_m["max_dd_pct"] <= LIVE_MAXDD_PCT_MAX and live_m["win_rate"] >= LIVE_WIN_MIN:
            status = "live"

    # Degrade if live performance deteriorates badly (handled more in Patch 17)
    if status == "live":
        if live_m["trades"] >= LIVE_MIN_TRADES and (live_m["sharpe"] < 0.0 or live_m["max_dd_pct"] > (2*LIVE_MAXDD_PCT_MAX)):
            status = "review"

    entry.update({
        "approval_status": status,
        "metrics": {
            "paper": paper_m,
            "live": live_m
        },
        "trades_used": {
            "paper": paper_m["trades"],
            "live": live_m["trades"]
        }
    })
    _put_strat_entry(asset_ui_key, strategy_id, entry)
    return entry

def on_trade_closed(asset_ui_key: str, strategy_id: str, mode: str, trade_row: dict) -> None:
    """Call on each close to refresh gates."""
    try:
        evaluate(asset_ui_key, strategy_id)
    except Exception:
        pass