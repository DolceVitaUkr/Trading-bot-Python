"""
Budget & guardrails manager.

Config precedence:
1) tradingbot/state/budgets.json (persisted overrides)
2) Environment variables (BUDGET_SPOT_USD, BUDGET_FUTURES_USD, BUDGET_FOREX_USD, BUDGET_OPTIONS_USD,
   MAX_CONCURRENT_POS, DAILY_LOSS_CAP_PCT, COOL_DOWN_SECONDS)
3) Defaults

Exposed helpers:
- can_place_order(asset_ui_key, mode, symbol, side, quantity, price, asset_type, broker) -> (ok, reason)
- count_open_positions(asset_ui_key, mode) -> int
- get_alloc(asset_ui_key) -> float
- set_cooldown(asset_ui_key, seconds)
"""
from __future__ import annotations
import os, json, pathlib, time
from typing import Dict, Any, Tuple

STATE_DIR = pathlib.Path("tradingbot/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)

BUDGETS_FILE = STATE_DIR / "budgets.json"
RISK_FILE    = STATE_DIR / "risk_state.json"

DEFAULTS = {
    "alloc": {
        "crypto_spot":    1000.0,
        "crypto_futures": 1500.0,
        "forex":          5000.0,
        "options":        3000.0,
    },
    "max_concurrent": 5,
    "daily_loss_cap_pct": 0.03,
    "cooldown_seconds": 900,
}

def _load_json(path: pathlib.Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def _save_json(path: pathlib.Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

def _merge_config() -> Dict[str, Any]:
    cfg = DEFAULTS.copy()
    cfg["alloc"] = cfg["alloc"].copy()
    file_cfg = _load_json(BUDGETS_FILE, {})
    # env overrides
    env_map = {
        "crypto_spot":    _env_float("BUDGET_SPOT_USD",    cfg["alloc"]["crypto_spot"]),
        "crypto_futures": _env_float("BUDGET_FUTURES_USD", cfg["alloc"]["crypto_futures"]),
        "forex":          _env_float("BUDGET_FOREX_USD",   cfg["alloc"]["forex"]),
        "options":        _env_float("BUDGET_OPTIONS_USD", cfg["alloc"]["options"]),
    }
    cfg["alloc"].update(env_map)
    if isinstance(file_cfg, dict):
        cfg["alloc"].update(file_cfg.get("alloc", {}))
        cfg["max_concurrent"] = int(file_cfg.get("max_concurrent", cfg["max_concurrent"]))
        cfg["daily_loss_cap_pct"] = float(file_cfg.get("daily_loss_cap_pct", cfg["daily_loss_cap_pct"]))
        cfg["cooldown_seconds"]   = int(file_cfg.get("cooldown_seconds", cfg["cooldown_seconds"]))
    # allow env to override caps & counts too
    cfg["max_concurrent"]   = _env_int("MAX_CONCURRENT_POS", cfg["max_concurrent"])
    cfg["daily_loss_cap_pct"] = _env_float("DAILY_LOSS_CAP_PCT", cfg["daily_loss_cap_pct"])
    cfg["cooldown_seconds"] = _env_int("COOL_DOWN_SECONDS", cfg["cooldown_seconds"])
    return cfg

# Runtime API to count positions and available cash
try:
    from tradingbot.core.runtime_api import aggregate_positions, aggregate_status
except Exception:
    aggregate_positions = None
    aggregate_status = None

def _risk_state() -> Dict[str, Any]:
    return _load_json(RISK_FILE, {"cooldown_until": {}})

def set_cooldown(asset_ui_key: str, seconds: int | None = None) -> None:
    rs = _risk_state()
    until = int(time.time()) + int(seconds if seconds is not None else _merge_config()["cooldown_seconds"])
    (rs["cooldown_until"]).update({asset_ui_key: until})
    _save_json(RISK_FILE, rs)

def in_cooldown(asset_ui_key: str) -> bool:
    rs = _risk_state()
    until = int((rs.get("cooldown_until") or {}).get(asset_ui_key, 0))
    return int(time.time()) < until

def count_open_positions(asset_ui_key: str, mode: str) -> int:
    if not aggregate_positions:
        return 0
    try:
        pos = aggregate_positions(asset_ui_key)
        rows = pos.get(mode, []) if isinstance(pos, dict) else []
        return len([r for r in rows if r.get("amount",0)!=0])
    except Exception:
        return 0

def get_alloc(asset_ui_key: str) -> float:
    return float(_merge_config()["alloc"].get(asset_ui_key, 0.0))

def _available_cash_live(asset_ui_key: str) -> float:
    if not aggregate_status:
        return 0.0
    try:
        st = aggregate_status(asset_ui_key)
        live = st.get("live") or {}
        return float(live.get("available", 0.0))
    except Exception:
        return 0.0

def can_place_order(asset_ui_key: str, mode: str, symbol: str, side: str, quantity: float, price: float | None, asset_type: str, broker: str | None = None) -> tuple[bool, str]:
    cfg = _merge_config()
    if in_cooldown(asset_ui_key):
        return False, "asset in cooldown"
    # basic per-trade notional limit
    if price is None or float(price) <= 0:
        # require a price for market orders
        return False, "price required for budget check"
    notional = abs(float(quantity) * float(price))
    alloc = float(cfg["alloc"].get(asset_ui_key, 0.0))
    if notional > alloc:
        return False, f"notional ${notional:.2f} exceeds allocation ${alloc:.2f}"
    # concurrent positions
    if count_open_positions(asset_ui_key, mode) >= int(cfg["max_concurrent"]):
        return False, "max concurrent positions reached"
    # IBKR equities cash-only: for non-crypto live ensure available cash covers notional
    if mode == "live" and not asset_ui_key.startswith("crypto_"):
        avail = _available_cash_live(asset_ui_key)
        if notional > avail:
            return False, f"notional ${notional:.2f} exceeds IBKR available cash ${avail:.2f}"
    return True, "ok"

try:
    from tradingbot.core import history_store
except Exception:
    history_store = None
try:
    from tradingbot.core.runtime_api import aggregate_status
except Exception:
    aggregate_status = None


# ----- Wallet-aware allocations & dynamic sizing -----
def _budgets() -> Dict[str, Any]:
    cfg = _merge_config()
    cfg.setdefault("alloc_mode", "absolute")
    cfg.setdefault("per_trade_risk_pct", {
        "crypto_spot": 0.02, "crypto_futures": 0.01, "forex": 0.005, "options": 0.01
    })
    cfg.setdefault("scale", {
        "enabled": True,
        "equity_curve_window_days": 7,
        "ladder": [{"profit_usd": 200, "risk_multiplier": 1.2},
                   {"profit_usd": 500, "risk_multiplier": 1.5},
                   {"profit_usd": 1000, "risk_multiplier": 2.0}]
    })
    return cfg

def _wallet_total_for(asset_ui_key: str) -> float:
    if not aggregate_status: return 0.0
    try:
        st = aggregate_status(asset_ui_key); live = st.get("live") or {}
        return float(live.get("total", 0.0))
    except Exception: return 0.0

def get_alloc_v2(asset_ui_key: str) -> float:
    cfg = _budgets()
    alloc = cfg["alloc"].get(asset_ui_key, 0.0)
    if str(cfg.get("alloc_mode","absolute")).lower() == "percent":
        base = _wallet_total_for(asset_ui_key)
        try: pct = float(alloc)
        except Exception: pct = 0.0
        return max(0.0, float(base) * max(0.0, min(1.0, pct)))
    try: return float(alloc)
    except Exception: return 0.0

def _realized_profit(asset_ui_key: str, days: int) -> float:
    if not history_store: return 0.0
    try:
        from datetime import datetime, timedelta
        since = (datetime.utcnow() - timedelta(days=int(days))).isoformat() + "Z"
        rows = (history_store.read_history(asset_ui_key, mode=None, since=since, limit=5000) or [])
        return float(sum(float(r.get("pnl") or 0.0) for r in rows))
    except Exception: return 0.0

def _risk_multiplier(asset_ui_key: str) -> float:
    cfg = _budgets().get("scale") or {}
    if not cfg.get("enabled", True): return 1.0
    ladder = cfg.get("ladder") or []
    lookback = int(cfg.get("equity_curve_window_days", 7))
    profit = _realized_profit(asset_ui_key, lookback)
    mult = 1.0
    try:
        for step in sorted(ladder, key=lambda x: float(x.get("profit_usd",0))):
            if profit >= float(step.get("profit_usd",0)):
                mult = float(step.get("risk_multiplier", 1.0))
    except Exception: pass
    return max(0.5, min(5.0, mult))

def position_size_cap(asset_ui_key: str, price: float) -> tuple[float, float]:
    cfg = _budgets()
    alloc_usd = float(get_alloc_v2(asset_ui_key))
    risk_pct = float((cfg.get("per_trade_risk_pct") or {}).get(asset_ui_key, 0.01))
    mult = _risk_multiplier(asset_ui_key)
    cap_notional = max(0.0, alloc_usd * risk_pct * mult)
    if not price or float(price) <= 0: return (cap_notional, 0.0)
    cap_qty = cap_notional / float(price)
    cap_qty = float(f"{cap_qty:.8f}")
    return (cap_notional, cap_qty)

def suggest_position_size(asset_ui_key: str, price: float, min_qty: float | None = None, step: float | None = None) -> float:
    _, cap_qty = position_size_cap(asset_ui_key, price)
    q = cap_qty
    try:
        if min_qty: q = max(q, float(min_qty))
        if step and step > 0:
            import math; q = math.floor(q / step) * step
    except Exception: pass
    return max(0.0, float(f"{q:.8f}"))