# state/runtime_state.py

import json
import os
import time
import threading
from typing import Any, Dict, Optional

try:
    import config  # optional; default fallbacks used if missing
except Exception:
    config = None


class RuntimeState:
    """
    Minimal persistent runtime state with thread-safety.
    Backed by a JSON file on disk.

    Layout:
    {
      "version": 1,
      "stage": int,
      "paper_wallets": { "<name>": float },
      "last_seen_balances": { "<domain>": { "<account>": float } },
      "open_positions": { "<domain>": { "<symbol>": {..raw exchange pos..} } },
      "events": [ { "ts": int_ms, "tag": str, "message": str } ],
      "last_heartbeat": int_ms
    }
    """

    def __init__(self, path: Optional[str] = None):
        self._lock = threading.RLock()
        self.path = path or os.path.join("state", "runtime_state.json")
        self._ensure_dir(os.path.dirname(self.path))

        # defaults
        default_stage = getattr(config, "ROLLOUT_STAGE", 1) if config else 1
        self._data: Dict[str, Any] = {
            "version": 1,
            "stage": int(default_stage),
            "paper_wallets": {},
            "last_seen_balances": {},
            "open_positions": {},
            "events": [],
            "last_heartbeat": None,
        }

        self._load()

    # ──────────────────────────── basic io ────────────────────────────

    def _ensure_dir(self, d: str) -> None:
        if not d:
            return
        os.makedirs(d, exist_ok=True)

    def _load(self) -> None:
        with self._lock:
            if not os.path.exists(self.path):
                self._save_locked()
                return
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    # Shallow merge with defaults to tolerate new fields
                    merged = dict(self._data)
                    merged.update(loaded)
                    self._data = merged
            except Exception:
                # Corrupt file → rotate and start clean
                bad = f"{self.path}.corrupt.{int(time.time())}.json"
                try:
                    os.replace(self.path, bad)
                except Exception:
                    pass
                self._save_locked()

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    # ──────────────────────────── getters/setters ────────────────────────────

    def get_stage(self) -> int:
        with self._lock:
            return int(self._data.get("stage", 1))

    def set_stage(self, stage: int) -> None:
        with self._lock:
            self._data["stage"] = int(stage)
            self._save_locked()

    def get_paper_wallet(self, name: str) -> float:
        with self._lock:
            return float(self._data.get("paper_wallets", {}).get(name, 0.0))

    def set_paper_wallet(self, name: str, amount: float) -> None:
        with self._lock:
            self._data.setdefault("paper_wallets", {})[name] = float(amount)
            self._save_locked()

    def set_last_seen_balance(self, domain: str, account: str, amount: float) -> None:
        with self._lock:
            d = self._data.setdefault("last_seen_balances", {})
            d.setdefault(domain, {})[account] = float(amount)
            self._save_locked()

    def get_last_seen_balance(self, domain: str, account: str) -> float:
        with self._lock:
            return float(self._data.get("last_seen_balances", {}).get(domain, {}).get(account, 0.0))

    def upsert_open_position(self, domain: str, symbol: str, position: Dict[str, Any]) -> None:
        """
        Store/replace the last known open position snapshot for a domain+symbol.
        """
        with self._lock:
            d = self._data.setdefault("open_positions", {})
            d.setdefault(domain, {})[symbol] = position
            self._save_locked()

    def get_open_position(self, domain: str, symbol: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._data.get("open_positions", {}).get(domain, {}).get(symbol)

    def clear_open_position(self, domain: str, symbol: str) -> None:
        with self._lock:
            d = self._data.setdefault("open_positions", {}).setdefault(domain, {})
            if symbol in d:
                del d[symbol]
                self._save_locked()

    def clear_all_open_positions(self, domain: Optional[str] = None) -> None:
        with self._lock:
            if domain is None:
                self._data["open_positions"] = {}
            else:
                self._data.setdefault("open_positions", {})[domain] = {}
            self._save_locked()

    def heartbeat(self) -> None:
        """
        Record a heartbeat (ms since epoch). Caller can update periodically to indicate liveness.
        """
        with self._lock:
            self._data["last_heartbeat"] = self._now_ms()
            # Writes are cheap; persist
            self._save_locked()

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    # ──────────────────────────── events ────────────────────────────

    def _event(self, tag: str, message: str) -> None:
        """
        Append a small event (kept in memory and persisted). Truncated to the last N entries.
        """
        with self._lock:
            ev = {"ts": self._now_ms(), "tag": str(tag), "message": str(message)}
            self._data.setdefault("events", []).append(ev)
            # Keep last 500 to avoid runaway growth
            if len(self._data["events"]) > 500:
                self._data["events"] = self._data["events"][-500:]
            self._save_locked()

    # ──────────────────────────── export ────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data))  # deep copy via JSON
