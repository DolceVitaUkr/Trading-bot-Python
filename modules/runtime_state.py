# modules/runtime_state.py
import json
import os
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Optional

STATE_DIR = os.getenv("STATE_DIR", os.path.join("data", "_state"))
STATE_FILE = os.path.join(STATE_DIR, "runtime.json")
STATE_BACKUP_FILE = os.path.join(STATE_DIR, "runtime.backup.json")

# Schema versioning so we can migrate state in the future if fields change.
SCHEMA_VERSION = 1

_default_state: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "created_ts": None,
    "updated_ts": None,

    # Rollout stage (1..5). We persist what the bot *was on*, but stage changes
    # are controlled via rollout_manager and the UI gate checks.
    "rollout_stage": 1,

    # Domain toggles (runtime view). Forex/Options always boot OFF unless explicitly
    # enabled after startup — rollout_manager will enforce that on load().
    "domains": {
        "crypto": {"profile": "spot", "live": False},
        "perp":   {"live": False},
        "forex":  {"enabled": False, "live": False},
        "options":{"enabled": False, "live": False}
    },

    # Exploration (paper “exploration trades” share). Manager enforces min.
    "exploration": {
        "enabled": True,
        "target_rate": 0.25,
        "min_rate": 0.10,
        "window": {"lookback_trades": 200, "actual_rate": 0.0, "count_explore": 0, "count_total": 0}
    },

    # Canary sizing/ramp tracking (applies when a domain is newly live).
    "canary": {
        "active": False,
        "domain": None,               # "crypto_spot" | "perp" | "forex"
        "schedule": [0.02, 0.03, 0.05],
        "step_index": 0,
        "started_ts": None,
        "max_days": 7,
        "max_trades": 50,
        "trade_count": 0
    },

    # Exposure snapshot & open positions (ids or symbols) so we can reconcile on restart.
    "positions": {
        "crypto_spot": {},
        "perp": {},
        "forex": {},
        "options": {}
    },

    # KPI moving snapshot per domain (used for gates & monitoring)
    "kpi": {
        "crypto_spot": {"win_rate": None, "sharpe": None, "avg_profit_swing": None, "avg_profit_scalp": None, "max_dd": None},
        "perp":        {"win_rate": None, "sharpe": None, "avg_profit_swing": None, "avg_profit_scalp": None, "max_dd": None},
        "forex":       {"win_rate": None, "sharpe": None, "avg_profit_swing": None, "avg_profit_scalp": None, "max_dd": None},
        "options":     {"win_rate": None, "sharpe": None, "avg_profit_swing": None, "avg_profit_scalp": None, "max_dd": None}
    },

    # Wallet views for paper accounts (segregated), persist across runs unless reset via UI.
    "paper_wallets": {
        "Crypto_Paper": {"balance": 1000.0, "created_ts": None},
        "Perps_Paper":  None,  # becomes {"balance": 1000.0, ...} when Stage 3 starts
        "Forex_Paper":  None,  # becomes active at Stage 3
        "ForexOptions_Paper": None  # becomes active at Stage 4
    }
}


@dataclass
class RuntimeState:
    path: str = STATE_FILE
    backup_path: str = STATE_BACKUP_FILE
    _state: Dict[str, Any] = field(default_factory=lambda: json.loads(json.dumps(_default_state)))
    _lock: RLock = field(default_factory=RLock)

    # ───────────────────────────
    # Core I/O
    # ───────────────────────────
    def load(self) -> None:
        with self._lock:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

            # Enforce safe-boot semantics BEFORE loading persisted toggles:
            # Forex/Options trading switches must come up disabled for NEW orders,
            # but we will still reconcile any open positions later.
            self._apply_boot_defaults()

            if not os.path.isfile(self.path):
                # First run: stamp created_ts
                now = time.time()
                self._state["created_ts"] = now
                self._state["updated_ts"] = now
                self._write_atomic(self._state)
                return

            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    disk = json.load(f)
                self._state = self._migrate_if_needed(disk)
                # After load, apply boot rule for FX/Options toggles.
                self._apply_boot_defaults()
            except Exception:
                # If state is corrupt, attempt backup; otherwise restore defaults.
                if os.path.isfile(self.backup_path):
                    with open(self.backup_path, "r", encoding="utf-8") as f:
                        disk = json.load(f)
                    self._state = self._migrate_if_needed(disk)
                    self._apply_boot_defaults()
                else:
                    self._state = json.loads(json.dumps(_default_state))
                    self._state["created_ts"] = time.time()
                    self._state["updated_ts"] = self._state["created_ts"]
                    self._apply_boot_defaults()
                # Always rewrite after recovering
                self._write_atomic(self._state)

    def save(self) -> None:
        with self._lock:
            self._state["updated_ts"] = time.time()
            self._write_atomic(self._state)

    def _write_atomic(self, state: Dict[str, Any]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=False)
        # Rotate backup then replace
        try:
            if os.path.isfile(self.path):
                # copy current to backup
                with open(self.path, "r", encoding="utf-8") as src, open(self.backup_path, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
        except Exception:
            # Non-fatal
            pass
        os.replace(tmp, self.path)

    # ───────────────────────────
    # Boot rules & migration
    # ───────────────────────────
    def _apply_boot_defaults(self) -> None:
        """Enforce safe restart semantics:
        - Forex/Options 'enabled' flags are forced False on boot (no new orders),
          but we keep any position snapshots for reconciliation by the domain adapters.
        """
        d = self._state.get("domains", {})
        forex = d.get("forex", {"enabled": False, "live": False})
        options = d.get("options", {"enabled": False, "live": False})

        forex["enabled"] = False
        options["enabled"] = False

        d["forex"] = forex
        d["options"] = options
        self._state["domains"] = d

    def _migrate_if_needed(self, disk: Dict[str, Any]) -> Dict[str, Any]:
        schema = disk.get("schema_version", 0)
        if schema == SCHEMA_VERSION:
            # Fill any missing keys from defaults without overwriting existing
            return _deep_merge_defaults(disk, _default_state)
        # Future migrations go here
        # Example: if schema == 0: migrate... and set to SCHEMA_VERSION
        disk = _deep_merge_defaults(disk, _default_state)
        disk["schema_version"] = SCHEMA_VERSION
        return disk

    # ───────────────────────────
    # Accessors / Mutators
    # ───────────────────────────
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(key, default)

    def whole(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def set_rollout_stage(self, stage: int) -> None:
        with self._lock:
            self._state["rollout_stage"] = int(stage)
            self.save()

    def set_domain(self, domain: str, key: str, value: Any) -> None:
        with self._lock:
            if "domains" not in self._state:
                self._state["domains"] = {}
            if domain not in self._state["domains"]:
                self._state["domains"][domain] = {}
            self._state["domains"][domain][key] = value
            self.save()

    def start_canary(self, domain: str, schedule: Optional[list] = None, max_days: int = 7, max_trades: int = 50) -> None:
        with self._lock:
            c = self._state["canary"]
            c.update({
                "active": True,
                "domain": domain,
                "schedule": schedule or c.get("schedule") or [0.02, 0.03, 0.05],
                "step_index": 0,
                "started_ts": time.time(),
                "max_days": max_days,
                "max_trades": max_trades,
                "trade_count": 0
            })
            self.save()

    def canary_step_pct(self) -> Optional[float]:
        with self._lock:
            c = self._state["canary"]
            if not c.get("active"):
                return None
            idx = c.get("step_index", 0)
            sch = c.get("schedule") or []
            return sch[idx] if 0 <= idx < len(sch) else None

    def increment_canary_trade(self) -> None:
        with self._lock:
            c = self._state["canary"]
            if not c.get("active"):
                return
            c["trade_count"] = int(c.get("trade_count", 0)) + 1
            self.save()

    def advance_canary(self) -> None:
        with self._lock:
            c = self._state["canary"]
            if not c.get("active"):
                return
            c["step_index"] += 1
            # Auto-complete if we outgrow schedule
            if c["step_index"] >= len(c.get("schedule") or []):
                c["active"] = False
                c["domain"] = None
            self.save()

    def stop_canary(self) -> None:
        with self._lock:
            c = self._state["canary"]
            c.update({"active": False, "domain": None})
            self.save()

    def record_position_snapshot(self, domain: str, symbol: str, snapshot: Dict[str, Any]) -> None:
        with self._lock:
            self._state["positions"].setdefault(domain, {})
            self._state["positions"][domain][symbol] = snapshot
            self.save()

    def clear_position_snapshot(self, domain: str, symbol: str) -> None:
        with self._lock:
            if symbol in self._state["positions"].get(domain, {}):
                del self._state["positions"][domain][symbol]
                self.save()

    def record_exploration_trade(self, is_exploration: bool) -> None:
        with self._lock:
            w = self._state["exploration"]["window"]
            w["count_total"] += 1
            if is_exploration:
                w["count_explore"] += 1
            w["actual_rate"] = (w["count_explore"] / max(1, w["count_total"]))
            self.save()

    def set_kpi(self, domain: str, **kpis: Any) -> None:
        with self._lock:
            self._state["kpi"].setdefault(domain, {})
            self._state["kpi"][domain].update({k: v for k, v in kpis.items() if v is not None})
            self.save()

    def ensure_paper_wallet(self, name: str, initial: float = 1000.0) -> None:
        with self._lock:
            pw = self._state["paper_wallets"].get(name)
            if pw is None:
                self._state["paper_wallets"][name] = {
                    "balance": float(initial),
                    "created_ts": time.time()
                }
                self.save()

    def set_paper_balance(self, name: str, balance: float) -> None:
        with self._lock:
            if name not in self._state["paper_wallets"]:
                self.ensure_paper_wallet(name, balance)
            else:
                self._state["paper_wallets"][name]["balance"] = float(balance)
                self.save()


def _deep_merge_defaults(current: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Merge keys from defaults into current without overwriting existing values."""
    for k, v in defaults.items():
        if k not in current:
            current[k] = json.loads(json.dumps(v))
        else:
            if isinstance(v, dict) and isinstance(current[k], dict):
                _deep_merge_defaults(current[k], v)
    return current
