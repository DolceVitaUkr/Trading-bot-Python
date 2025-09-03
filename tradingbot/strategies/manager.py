"""
Strategy manager: lists strategies and toggles status (start/stop).
Backed by tradingbot/state/strategies/strategies.json.
"""
from __future__ import annotations
import json
import pathlib
from datetime import datetime
from typing import Dict, Any, List

STATE_FILE = pathlib.Path("tradingbot/state/strategies/strategies.json")
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

def _load() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save(obj: Dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _asset_match(rec: Dict[str, Any], asset_ui: str) -> bool:
    t = (rec or {}).get("asset_type", "").lower()
    if asset_ui == "crypto_spot":
        return t in ("spot", "crypto", "crypto_spot")
    if asset_ui == "crypto_futures":
        return t in ("futures", "perp", "contract", "crypto_futures")
    if asset_ui == "forex":
        return t in ("forex", "fx", "currency")
    if asset_ui == "options":
        return t in ("options", "opt")
    return True

def list(asset: str) -> List[Dict[str, Any]]:
    data = _load()
    out: List[Dict[str, Any]] = []
    for key, rec in data.items():
        if not isinstance(rec, dict):
            continue
        if not _asset_match(rec, asset):
            continue
        metrics = rec.get("metrics") or {}
        params = rec.get("params") or {}
        out.append({
            "id": rec.get("strategy_id") or key,
            "name": rec.get("name") or rec.get("strategy_id") or key,
            "status": rec.get("status") or "idle",
            "params": params,
            "performance": {
                "sharpe": metrics.get("sharpe_ratio"),
                "win_rate": metrics.get("win_rate"),
                "avg_trade": metrics.get("avg_trade_pnl") or metrics.get("avg_trade"),
            }
        })
    # deterministic order
    out.sort(key=lambda x: x["id"])
    # If empty, synthesize generic 'Strategy 1' placeholder (no mock trades, just display)
    if not out:
        out = [{
            "id": "strategy-1",
            "name": "Strategy 1",
            "status": "idle",
            "params": {},
            "performance": {}
        }]
    return out

def start(asset: str, sid: str) -> bool:
    data = _load()
    rec = data.get(sid) or next((v for v in data.values() if (v.get("strategy_id") == sid)), None)
    if rec is None:
        # create a new record with minimal fields
        rec = {"strategy_id": sid, "asset_type": asset.replace("crypto_", ""), "status": "running", "metrics": {}}
        data[sid] = rec
    rec["status"] = "running"
    rec["last_started"] = datetime.utcnow().isoformat() + "Z"
    _save(data)
    return True

def stop(asset: str, sid: str) -> bool:
    data = _load()
    rec = data.get(sid) or next((v for v in data.values() if (v.get("strategy_id") == sid)), None)
    if rec is None:
        rec = {"strategy_id": sid, "asset_type": asset.replace("crypto_", ""), "status": "idle", "metrics": {}}
        data[sid] = rec
    rec["status"] = "idle"
    rec["last_stopped"] = datetime.utcnow().isoformat() + "Z"
    _save(data)
    return True

# --- Patch 15: sessions/regimes fields & getter ---
def _with_defaults(s: dict) -> dict:
    s = dict(s or {})
    s.setdefault("allowed_sessions", ["ASIA","EU","US"])  # allow all by default
    s.setdefault("allowed_regimes", ["TREND_UP","TREND_DOWN","RANGE","HIGH_VOL"])
    s.setdefault("disallowed_regimes", [])
    s.setdefault("observe_only", False)
    return s

def get_strategy(asset: str, sid: str) -> dict:
    try:
        data = _load()
        # existing storage might be JSON loaded elsewhere; keep duck-typed access
        if isinstance(data, dict):
            rec = data.get(sid) or next((v for v in data.values() if (v.get("strategy_id") == sid)), None)
            if rec:
                return _with_defaults(rec)
        return _with_defaults({})
    except Exception:
        return _with_defaults({})