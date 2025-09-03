import json, time
from pathlib import Path
from typing import Dict, Any
from .loggerconfig import get_logger
log = get_logger(__name__)
def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
def emit_event(base_dir: Path, mode: str, asset: str, event: Dict[str, Any]) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    row = {"ts": ts, "mode": mode, "asset": asset, **event}
    if mode == "paper":
        path = base_dir / "state" / "paper" / f"trades_{asset}.jsonl"
    else:
        venue = event.get("venue", "live")
        path = base_dir / "state" / "live" / venue / f"trades_{asset}.jsonl"
    _append_jsonl(path, row)