import json
from pathlib import Path
from typing import Dict, Any

_G = None
_T = None

def _load_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def load_configs(base: Path = Path("tradingbot/config")):
    global _G, _T
    _G = _load_json(base / "promotion_gates.json")
    _T = _load_json(base / "tripwires.json")

def can_go_live(strategy_metrics: Dict[str, Any]) -> bool:
    if _G is None: load_configs()
    if not _G:
        return False
    # Minimal check: require APPROVED flag or thresholds met
    state = strategy_metrics.get("state")
    if state in ("APPROVED", "LIVE_TESTING", "VALIDATED"):
        return True
    # thresholds example
    p = _G.get("paper", {})
    if strategy_metrics.get("trades", 0) >= p.get("min_trades", 1000) and strategy_metrics.get("sharpe", 0) >= p.get("sharpe_min", 1.3):
        return True
    return False

def tripwire_violation(asset_metrics: Dict[str, Any]) -> str | None:
    if _T is None: load_configs()
    if not _T:
        return None
    if asset_metrics.get("daily_loss_pct", 0) > _T.get("daily_loss_cap_pct", 0.75):
        return "DAILY_LOSS_CAP"
    if asset_metrics.get("max_drawdown_pct", 0) > _T.get("max_drawdown_cap_pct", 15.0):
        return "MAX_DRAWDOWN_CAP"
    if asset_metrics.get("reject_rate_pct", 0) > _T.get("max_reject_rate_pct", 5.0):
        return "REJECT_RATE_CAP"
    return None