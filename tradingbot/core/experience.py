from __future__ import annotations
import json, pathlib, time
from typing import Dict, Any

EXP_DIR = pathlib.Path("tradingbot/state/experience")
EXP_DIR.mkdir(parents=True, exist_ok=True)

def append(asset: str, row: Dict[str, Any]) -> None:
    path = EXP_DIR / f"{asset}.jsonl"
    row = dict(row or {})
    row.setdefault("ts", int(time.time()))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")