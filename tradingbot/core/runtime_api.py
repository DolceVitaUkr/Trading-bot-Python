"""
Runtime API: per-asset paper/live toggles with simple persistent state.
This is a lightweight shim used by the UI endpoints.
"""
from __future__ import annotations
import json
import pathlib
from typing import Dict, Any

try:
    from . import paper_state
except ImportError:
    paper_state = None

try:
    from tradingbot.core.retry import retry_call
except Exception:
    def retry_call(fn,*a,**k):
        return fn(*a,**k)

try:
    from tradingbot.core import history_store
except Exception:
    history_store = None

# Live adapters (lazy)
try:
    from tradingbot.brokers.bybit_adapter import BybitAdapter
except Exception:
    BybitAdapter = None
try:
    from tradingbot.brokers.ibkr_adapter import IBKRAdapter
except Exception:
    IBKRAdapter = None

STATE_DIR = pathlib.Path("tradingbot/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "runtime.json"

ASSETS = ["crypto_spot", "crypto_futures", "forex", "options"]

_BYBIT = None
_IBKR = None

def _bybit():
    global _BYBIT
    if _BYBIT is None and BybitAdapter:
        _BYBIT = BybitAdapter()
    return _BYBIT

def _ibkr():
    global _IBKR
    if _IBKR is None and IBKRAdapter:
        _IBKR = IBKRAdapter()
    return _IBKR

def _load() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    st = {"paper": {a: False for a in ASSETS}, "live": {a: False for a in ASSETS}}
    STATE_FILE.write_text(json.dumps(st, indent=2), encoding="utf-8")
    return st

def _save(st: Dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(st, indent=2), encoding="utf-8")

def _norm(asset: str) -> str:
    a = (asset or "").strip().lower().replace(" ", "_")
    # accept aliases
    aliases = {"spot": "crypto_spot", "futures": "crypto_futures"}
    return aliases.get(a, a)

# Public API used by ui/app.py
def enable_paper(asset: str) -> Dict[str, Any]:
    a = _norm(asset)
    st = _load()
    st.setdefault("paper", {})[a] = True
    _save(st)
    return {"asset": a, "paper": True}

def disable_paper(asset: str) -> Dict[str, Any]:
    a = _norm(asset)
    st = _load()
    st.setdefault("paper", {})[a] = False
    _save(st)
    return {"asset": a, "paper": False}

def paper_enabled(asset: str) -> bool:
    a = _norm(asset)
    st = _load()
    return bool(st.get("paper", {}).get(a, False))

def live_enabled(asset: str) -> bool:
    a = _norm(asset)
    st = _load()
    return bool(st.get("live", {}).get(a, False))

# Stubs for future patches (no-ops here)
def enable_live(asset: str): 
    a = _norm(asset)
    st = _load()
    st.setdefault("live", {})[a] = True
    _save(st)
    return {"asset": a, "live": True}

def disable_live(asset: str):
    a = _norm(asset)
    st = _load()
    st.setdefault("live", {})[a] = False
    _save(st)
    return {"asset": a, "live": False}

def aggregate_status(asset: str) -> Dict[str, Any]:
    a = _norm(asset)
    # paper
    if paper_state:
        pw = paper_state.get_wallet(a)
        paper = {
            "total": float(pw.get("total", 0.0)),
            "available": float(pw.get("available", 0.0)),
            "used": float(pw.get("used", 0.0)),
            "unrealized_pnl": float(pw.get("unrealized_pnl", 0.0)),
        }
    else:
        paper = {"total": 0, "available": 0, "used": 0, "unrealized_pnl": 0}
    
    # live with retry
    live = {"total": 0.0, "available": 0.0, "used": 0.0, "unrealized_pnl": 0.0}
    try:
        if a.startswith("crypto_"):
            by = _bybit()
            if by:
                lw = retry_call(by.wallet, retries=2, backoff=0.4)
                live = {k: float(lw.get(k, 0.0)) for k in ("total", "available", "used", "unrealized_pnl")}
        else:
            ib = _ibkr()
            if ib:
                lw = retry_call(ib.wallet, retries=2, backoff=0.4)
                live = {k: float(lw.get(k, 0.0)) for k in ("total", "available", "used", "unrealized_pnl")}
    except Exception:
        pass
    
    return {"asset": a, "paper": paper, "live": live}

def aggregate_positions(asset: str) -> Dict[str, Any]:
    a = _norm(asset)
    paper = paper_state.list_positions(a) if paper_state else []
    live = []
    try:
        if a.startswith("crypto_"):
            by = _bybit()
            if by:
                pos = retry_call(by.positions, retries=2, backoff=0.4)
                live = (pos.get("futures") or []) + (pos.get("spot") or [])
        else:
            ib = _ibkr()
            if ib:
                live = retry_call(ib.positions, retries=2, backoff=0.4)
    except Exception:
        pass
    return {"paper": paper, "live": live}

def health() -> Dict[str, Any]:
    out = {}
    # Bybit
    try:
        by = _bybit()
        if by:
            ok = True; err = None
            try:
                retry_call(by.wallet, retries=1, backoff=0.2)
            except Exception as e:
                ok = False; err = str(e)
            out["bybit"] = {"ok": ok, "error": err}
    except Exception as e:
        out["bybit"] = {"ok": False, "error": str(e)}
    # IBKR
    try:
        ib = _ibkr()
        if ib:
            ok = True; err = None
            try:
                retry_call(ib.wallet, retries=1, backoff=0.2)
            except Exception as e:
                ok = False; err = str(e)
            out["ibkr"] = {"ok": ok, "error": err}
    except Exception as e:
        out["ibkr"] = {"ok": False, "error": str(e)}
    return out

def read_trade_history(asset: str, mode: str | None = None, limit: int = 500, since: str | None = None):
    if history_store is None:
        return []
    return history_store.read_history(asset, mode=mode, since=since, limit=limit)