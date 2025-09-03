import datetime as dt, json
from pathlib import Path
from typing import Tuple, List, Dict, Any
from .loggerconfig import get_logger
log = get_logger(__name__)

BASE = Path("tradingbot/config/calendars")

def _load_calendar(key: str) -> Dict[str, Any]:
    filename = {
        "CME-ES": "cme_es.json",
        "IBKR-FX-OPT": "ibkr_fx_opt.json",
        "BYBIT-PERP": "bybit_perp.json",
    }.get(key)
    if not filename:
        return {}
    p = BASE / filename
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def is_session_open(calendar_key: str, now_utc: dt.datetime | None = None) -> bool:
    now_utc = now_utc or dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    cfg = _load_calendar(calendar_key)
    if not cfg:
        return True
    # Holidays
    today = now_utc.date().isoformat()
    for h in cfg.get("holidays_utc", []):
        if h == today:
            return False
    # Windows
    t = now_utc.time()
    windows = cfg.get("windows_utc", [])
    if not windows:
        return True
    for w in windows:
        s = dt.time.fromisoformat(w["start"])  # HH:MM
        e = dt.time.fromisoformat(w["end"])    # HH:MM
        if s <= e and s <= t <= e:
            return True
        if s > e and (t >= s or t <= e):
            return True
    return False