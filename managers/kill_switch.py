"""
Kill Switch Manager to halt trading based on risk rules.
"""
import datetime
import orjson
import os
from typing import Tuple, Dict, Any

from core.schemas import KillEvent

KILL_SWITCH_FILE = "state/kill_switch.jsonl"
# Time in hours to automatically re-arm the kill switch
AUTO_REARM_HOURS = 24

class KillSwitch:
    """
    Monitors trading activity and can halt new orders if risk limits are breached.
    """

    def __init__(self, kill_switch_file: str = KILL_SWITCH_FILE):
        self.kill_switch_file = kill_switch_file
        self.active_scopes = {} # scope_key -> KillEvent
        self._load_state()

    def _get_scope_key(self, asset_class: str, venue: str) -> str:
        return f"{asset_class.upper()}:{venue.upper()}"

    def _ensure_file_exists(self):
        """Ensures the kill switch JSONL file exists."""
        os.makedirs(os.path.dirname(self.kill_switch_file), exist_ok=True)
        with open(self.kill_switch_file, "a"):
            pass

    def _load_state(self):
        """Loads the last known state of kill switches from the log file."""
        self._ensure_file_exists()
        now = datetime.datetime.now(datetime.timezone.utc)
        last_events = {}
        with open(self.kill_switch_file, "rb") as f:
            for line in f:
                event = KillEvent.parse_raw(line)
                key = self._get_scope_key(event.scope['asset_class'], event.scope['venue'])
                last_events[key] = event

        for key, event in last_events.items():
            if now < event.auto_rearm_at:
                self.active_scopes[key] = event
                print(f"Kill switch for {key} is active from previous state.")

    def _trigger_kill_switch(self, scope_key: str, rule: str, note: str):
        """Triggers the kill switch for a given scope."""
        now = datetime.datetime.now(datetime.timezone.utc)
        asset_class, venue = scope_key.split(':')

        event = KillEvent(
            ts=now,
            scope={"asset_class": asset_class, "venue": venue},
            rule=rule,
            action={"block_new_orders": True, "allow_closes": True, "allow_paper": True},
            auto_rearm_at=now + datetime.timedelta(hours=AUTO_REARM_HOURS),
            note=note
        )

        self.active_scopes[scope_key] = event
        with open(self.kill_switch_file, "ab") as f:
            f.write(orjson.dumps(event.dict(by_alias=True)))
            f.write(b"\n")

        print(f"!!! KILL SWITCH TRIGGERED for {scope_key} due to rule '{rule}' !!!")

    def check_and_update(self, asset_class: str, venue: str, current_pnl: float, num_positions: int):
        """
        Checks all rules for a given scope and updates the state.
        In a real system, this would be called with real-time data.
        """
        scope_key = self._get_scope_key(asset_class, venue)

        # --- Simulate Rule Checks ---
        # A real implementation would get these values from a portfolio manager.

        # Rule 1: Daily Loss Limit
        daily_loss_limit = -5000 # e.g., $5000 loss
        if current_pnl < daily_loss_limit:
            if not self.is_trading_blocked(asset_class, venue)[0]:
                self._trigger_kill_switch(scope_key, "DAILY_LOSS_LIMIT", f"PnL {current_pnl} exceeded limit {daily_loss_limit}")

        # Rule 2: Max Concurrent Positions
        max_positions = 10
        if num_positions > max_positions:
            if not self.is_trading_blocked(asset_class, venue)[0]:
                self._trigger_kill_switch(scope_key, "MAX_CONCURRENT_POSITIONS", f"Positions {num_positions} exceeded limit {max_positions}")

    def is_trading_blocked(self, asset_class: str, venue: str) -> Tuple[bool, str]:
        """
        Checks if new orders are blocked for the given scope.
        """
        scope_key = self._get_scope_key(asset_class, venue)
        event = self.active_scopes.get(scope_key)

        if not event:
            return False, ""

        # Check if the re-arm time has passed
        now = datetime.datetime.now(datetime.timezone.utc)
        if now >= event.auto_rearm_at:
            print(f"Auto-rearming kill switch for {scope_key}.")
            del self.active_scopes[scope_key]
            return False, ""

        return True, event.rule
