from __future__ import annotations
import json
import pathlib
from datetime import datetime, timedelta
from typing import Dict, List

STATE_DIR = pathlib.Path("tradingbot/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)

def _file(asset: str, mode: str) -> pathlib.Path:
    a = asset.replace("/", "_").replace("-", "_")
    m = mode.replace("/", "_").replace("-", "_")
    return STATE_DIR / f"equity_{a}_{m}.jsonl"

def append_equity_point(asset: str, mode: str, equity: float, ts: datetime | None = None) -> None:
    if equity is None: return
    ts = ts or datetime.utcnow()
    p = _file(asset, mode)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": ts.isoformat() + "Z", "equity": float(equity)}) + "\n")

def read_equity_series(asset: str, mode: str, window: str = "7d") -> Dict[str, List[Dict]]:
    p = _file(asset, mode)
    if not p.exists(): return {"series": []}
    days = 7
    if isinstance(window, str) and window.endswith("d"):
        try: days = int(window[:-1])
        except: pass
    cutoff = datetime.utcnow() - timedelta(days=days)
    series = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line.strip())
                ts = o.get("ts")
                eq = o.get("equity")
                if ts is None or eq is None: continue
                t = ts[:-1] if ts.endswith("Z") else ts
                dt = datetime.fromisoformat(t)
                if dt >= cutoff: series.append({"ts": ts, "equity": float(eq)})
            except Exception: pass
    return {"series": series}