# state/runtime_state.py
import json
import os
import time
import threading
from typing import Any, Dict, Optional, List

try:
    # Prefer config values if present
    from . import config
    STATE_DIR = getattr(config, "STATE_DIR", os.path.join("data", "_state"))
    RUNTIME_STATE_FILE = getattr(
        config,
        "RUNTIME_STATE_FILE",
        os.path.join(STATE_DIR, "runtime.json"))
    EXPLORATION_MIN_RATE = getattr(config, "EXPLORATION_MIN_RATE", 0.10)
    EXPLORATION_RATE_TARGET = getattr(
        config, "EXPLORATION_RATE_TARGET", 0.25)
except Exception:
    # Safe fallbacks
    STATE_DIR = os.path.join("data", "_state")
    RUNTIME_STATE_FILE = os.path.join(STATE_DIR, "runtime.json")
    EXPLORATION_MIN_RATE = 0.10
    EXPLORATION_RATE_TARGET = 0.25


def _now() -> float:
    return time.time()


class RuntimeState:
    """
    JSON-backed runtime state.
    """

    _LOCK = threading.Lock()

    # Domains used across the bot
    DOMAINS = ("crypto", "perps", "forex", "options")

    # Paper wallets keyed in the file
    PAPER_KEYS = (
        "Crypto_Paper", "Perps_Paper", "Forex_Paper", "ForexOptions_Paper")

    SCHEMA_VERSION = 1

    def __init__(self, path: Optional[str] = None):
        """
        Initializes the RuntimeState.
        """
        self.path = path or RUNTIME_STATE_FILE
        self._ensure_dirs()
        self.state: Dict[str, Any] = {}
        self.load()

    # ────────────────────────────────────────────────────────────────────────────
    # Filesystem
    # ────────────────────────────────────────────────────────────────────────────
    def _ensure_dirs(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def _default_state(self) -> Dict[str, Any]:
        ts = _now()
        return {
            "schema_version": self.SCHEMA_VERSION,
            "created_at": ts,
            "updated_at": ts,

            # Rollout & flags
            "rollout_stage": 1,
            "exchange_profile": "spot",
            "forex_enabled": 0,
            "options_enabled": 0,
            "online_learning_enabled": 1,
            "exploration_rate": EXPLORATION_MIN_RATE,

            # Canary (per domain)
            "canary": {
                # domain -> dict
            },

            # Per-domain live state
            "domains": {
                "crypto": {"live": 0},
                "perps":  {"live": 0},
                "forex":  {"live": 0},
                "options": {"live": 0},
            },

            # Paper wallets (persist across boots)
            "paper_wallets": {
                "Crypto_Paper": 0.0,
                "Perps_Paper": 0.0,
                "Forex_Paper": 0.0,
                "ForexOptions_Paper": 0.0,
            },

            # Open positions snapshot by domain
            "open_positions": {
            },

            # Last seen balances on venues
            "last_seen_balances": {
            },

            # KPI snapshots (rolling window)
            "kpi_history": {
            },

            # Audit / breadcrumbs
            "events": [],
        }

    # ────────────────────────────────────────────────────────────────────────────
    # Load / Save
    # ────────────────────────────────────────────────────────────────────────────
    def load(self) -> None:
        """
        Loads the state from the JSON file.
        """
        with self._LOCK:
            if not os.path.exists(self.path):
                self.state = self._default_state()
                self._write()
                return
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Simple schema guard / merge defaults for missing keys
                defaults = self._default_state()
                merged = {**defaults, **data}
                # Deep merge for nested dicts we care about
                for k in ("canary", "domains", "paper_wallets",
                          "open_positions", "last_seen_balances",
                          "kpi_history"):
                    merged[k] = {**defaults.get(k, {}), **data.get(k, {})}
                self.state = merged
            except Exception:
                # If corrupted, rotate and recreate minimal valid file
                backup = f"{self.path}.corrupt.{int(_now())}.bak"
                try:
                    os.replace(self.path, backup)
                except Exception:
                    pass
                self.state = self._default_state()
                self._write()

    def save(self) -> None:
        """
        Saves the state to the JSON file.
        """
        with self._LOCK:
            self.state["updated_at"] = _now()
            self._write()

    def _write(self) -> None:
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, sort_keys=False)
        os.replace(tmp, self.path)

    # ────────────────────────────────────────────────────────────────────────────
    # Rollout & toggles
    # ────────────────────────────────────────────────────────────────────────────
    def get_stage(self) -> int:
        """
        Gets the current rollout stage.
        """
        return int(self.state.get("rollout_stage", 1))

    def set_stage(self, stage: int) -> None:
        """
        Sets the rollout stage.
        """
        self.state["rollout_stage"] = int(stage)
        self._event("stage.set", f"rollout_stage={stage}")
        self.save()

    def get_exchange_profile(self) -> str:
        """
        Gets the exchange profile.
        """
        return str(self.state.get("exchange_profile", "spot"))

    def set_exchange_profile(self, profile: str) -> None:
        """
        Sets the exchange profile.
        """
        profile = profile.lower()
        assert profile in ("spot", "perp", "spot+perp")
        self.state["exchange_profile"] = profile
        self._event("exchange_profile.set", profile)
        self.save()

    def get_flag(self, name: str) -> Any:
        """
        Gets a flag from the state.
        """
        return self.state.get(name)

    def set_flag(self, name: str, value: Any) -> None:
        """
        Sets a flag in the state.
        """
        self.state[name] = value
        self._event("flag.set", f"{name}={value}")
        self.save()

    def set_domain_live(self, domain: str, live: bool) -> None:
        """
        Sets a domain to live or paper mode.
        """
        assert domain in self.DOMAINS
        self.state["domains"].setdefault(domain, {})
        self.state["domains"][domain]["live"] = 1 if live else 0
        self._event("domain.live.set", f"{domain}={live}")
        self.save()

    def get_domain_live(self, domain: str) -> bool:
        """
        Gets whether a domain is live.
        """
        assert domain in self.DOMAINS
        return bool(
            self.state.get("domains", {}).get(domain, {}).get("live", 0))

    # ────────────────────────────────────────────────────────────────────────────
    # Exploration / learning
    # ────────────────────────────────────────────────────────────────────────────
    def get_exploration_rate(self) -> float:
        """
        Gets the exploration rate.
        """
        return float(
            self.state.get("exploration_rate", EXPLORATION_MIN_RATE))

    def set_exploration_rate(self, rate: float) -> None:
        """
        Sets the exploration rate.
        """
        rate = float(max(0.0, min(1.0, rate)))
        self.state["exploration_rate"] = rate
        self._event("exploration_rate.set", f"{rate:.4f}")
        self.save()

    def get_online_learning_enabled(self) -> bool:
        """
        Gets whether online learning is enabled.
        """
        return bool(self.state.get("online_learning_enabled", 1))

    def set_online_learning_enabled(self, enabled: bool) -> None:
        """
        Sets whether online learning is enabled.
        """
        self.state["online_learning_enabled"] = 1 if enabled else 0
        self._event("online_learning.set", str(enabled))
        self.save()

    # ────────────────────────────────────────────────────────────────────────────
    # Canary controls
    # ────────────────────────────────────────────────────────────────────────────
    def get_canary(self, domain: str) -> Dict[str, Any]:
        """
        Gets the canary status for a domain.
        """
        assert domain in self.DOMAINS
        can = self.state["canary"].get(domain)
        if not can:
            can = {"active": 0, "start_ts": 0.0, "trade_count": 0}
            self.state["canary"][domain] = can
            self.save()
        return can

    def set_canary(
            self, domain: str, active: bool, reset: bool = False) -> None:
        """
        Sets the canary status for a domain.
        """
        assert domain in self.DOMAINS
        can = self.get_canary(domain)
        can["active"] = 1 if active else 0
        if reset or (active and can["start_ts"] == 0.0):
            can["start_ts"] = _now()
            can["trade_count"] = 0
        self.state["canary"][domain] = can
        self._event("canary.set", f"{domain} active={active} reset={reset}")
        self.save()

    def canary_mark_trade(self, domain: str) -> int:
        """
        Marks a trade for the canary.
        """
        can = self.get_canary(domain)
        can["trade_count"] = int(can.get("trade_count", 0)) + 1
        self.state["canary"][domain] = can
        self._event("canary.trade", f"{domain} count={can['trade_count']}")
        self.save()
        return can["trade_count"]

    # ────────────────────────────────────────────────────────────────────────────
    # Paper wallets
    # ────────────────────────────────────────────────────────────────────────────
    def get_paper_wallet(self, key: str) -> float:
        """
        Gets the balance of a paper wallet.
        """
        assert key in self.PAPER_KEYS, f"Unknown paper wallet key: {key}"
        return float(self.state.get("paper_wallets", {}).get(key, 0.0))

    def set_paper_wallet(self, key: str, balance: float) -> None:
        """
        Sets the balance of a paper wallet.
        """
        assert key in self.PAPER_KEYS, f"Unknown paper wallet key: {key}"
        self.state["paper_wallets"][key] = float(balance)
        self._event("paper_wallet.set", f"{key}={balance:.2f}")
        self.save()

    def add_paper_wallet(self, key: str, delta: float) -> float:
        """
        Adds a delta to a paper wallet.
        """
        bal = self.get_paper_wallet(key) + float(delta)
        self.set_paper_wallet(key, bal)
        return bal

    # ────────────────────────────────────────────────────────────────────────────
    # Open positions snapshot
    # ────────────────────────────────────────────────────────────────────────────
    def upsert_open_position(
            self, domain: str, symbol: str, position: Dict[str, Any]) -> None:
        """
        Upserts an open position.
        """
        assert domain in self.DOMAINS
        self.state["open_positions"].setdefault(domain, {})
        self.state["open_positions"][domain][symbol] = position
        self._event("position.upsert", f"{domain}:{symbol}")
        self.save()

    def remove_open_position(self, domain: str, symbol: str) -> None:
        """
        Removes an open position.
        """
        assert domain in self.DOMAINS
        domain_book = self.state.get("open_positions", {}).get(domain, {})
        if symbol in domain_book:
            del domain_book[symbol]
            self._event("position.remove", f"{domain}:{symbol}")
            self.save()

    def get_open_positions(self, domain: str) -> Dict[str, Any]:
        """
        Gets all open positions for a domain.
        """
        assert domain in self.DOMAINS
        return dict(self.state.get("open_positions", {}).get(domain, {}))

    def clear_open_positions(self, domain: str) -> None:
        """
        Clears all open positions for a domain.
        """
        assert domain in self.DOMAINS
        self.state["open_positions"][domain] = {}
        self._event("positions.clear", domain)
        self.save()

    # ────────────────────────────────────────────────────────────────────────────
    # Last seen balances (live venues)
    # ────────────────────────────────────────────────────────────────────────────
    def set_last_seen_balance(
            self, domain: str, key: str, value: float) -> None:
        """
        Sets the last seen balance for a domain.
        """
        self.state["last_seen_balances"].setdefault(domain, {})
        self.state["last_seen_balances"][domain][key] = float(value)
        self._event("balance.set", f"{domain}:{key}={value:.2f}")
        self.save()

    def get_last_seen_balance(
            self, domain: str, key: str) -> Optional[float]:
        """
        Gets the last seen balance for a domain.
        """
        return self.state.get(
            "last_seen_balances", {}).get(domain, {}).get(key)

    # ────────────────────────────────────────────────────────────────────────────
    # KPI snapshots
    # ────────────────────────────────────────────────────────────────────────────
    def record_kpi(
            self, domain: str,
            snapshot: Dict[str, Any],
            keep_last: int = 200) -> None:
        """
        Records a KPI snapshot.
        """
        assert domain in self.DOMAINS
        self.state["kpi_history"].setdefault(domain, [])
        snap = {"ts": _now(), **snapshot}
        self.state["kpi_history"][domain].append(snap)
        # keep tail small
        if len(self.state["kpi_history"][domain]) > keep_last:
            self.state["kpi_history"][domain] = self.state[
                "kpi_history"][domain][-keep_last:]
        self._event("kpi.record", domain)
        self.save()

    def get_kpis(
            self, domain: str,
            last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Gets KPI snapshots for a domain.
        """
        assert domain in self.DOMAINS
        arr = list(self.state.get("kpi_history", {}).get(domain, []))
        if last_n is not None:
            return arr[-int(last_n):]
        return arr

    # ────────────────────────────────────────────────────────────────────────────
    # Events / Audit
    # ────────────────────────────────────────────────────────────────────────────
    def _event(self, typ: str, msg: str) -> None:
        ev = {"ts": _now(), "type": typ, "msg": msg}
        self.state.setdefault("events", []).append(ev)
        # Trim long audit lists silently (don’t spam the file)
        if len(self.state["events"]) > 1000:
            self.state["events"] = self.state["events"][-500:]
