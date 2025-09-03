import json
from pathlib import Path
from typing import Dict, Any

RUNTIME_PATH = Path("tradingbot/state/runtime.json")

def _read_runtime() -> Dict[str, Any]:
    if not RUNTIME_PATH.exists():
        return {"kill_switch": False, "close_only": False, "live_overrides": {}}
    return json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))

def _write_runtime(data: Dict[str, Any]) -> None:
    RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

def get_flags() -> Dict[str, Any]:
    return _read_runtime()

def set_kill_switch(on: bool) -> Dict[str, Any]:
    data = _read_runtime()
    data["kill_switch"] = bool(on)
    _write_runtime(data)
    return data

def set_close_only(on: bool) -> Dict[str, Any]:
    data = _read_runtime()
    data["close_only"] = bool(on)
    _write_runtime(data)
    return data