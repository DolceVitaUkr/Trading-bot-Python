import json, time, asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from .loggerconfig import get_logger
log = get_logger(__name__)

LOG_PATH = Path("tradingbot/logs/reconciler.jsonl")

def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")

async def reconcile_once(bybit_adapter=None, ibkr_adapter=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "bybit": {}, "ibkr": {}}
    try:
        if bybit_adapter:
            out["bybit"]["open_orders"] = await bybit_adapter.open_orders()
            out["bybit"]["positions"] = await bybit_adapter.positions()
    except Exception as e:
        out["bybit"]["error"] = str(e)
        log.error(f"Bybit reconcile failed: {e}")
    try:
        if ibkr_adapter:
            out["ibkr"]["open_orders"] = await ibkr_adapter.open_orders()
            out["ibkr"]["positions"] = await ibkr_adapter.positions()
    except Exception as e:
        out["ibkr"]["error"] = str(e)
        log.error(f"IBKR reconcile failed: {e}")
    _append_jsonl(LOG_PATH, out)
    return out

async def reconcile_periodic(bybit_adapter=None, ibkr_adapter=None, interval_sec: int = 60):
    while True:
        try:
            await reconcile_once(bybit_adapter, ibkr_adapter)
        except Exception as e:
            log.error(f"reconcile_periodic error: {e}")
        await asyncio.sleep(interval_sec)